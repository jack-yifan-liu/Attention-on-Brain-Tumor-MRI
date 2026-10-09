"""Evaluation metrics and analysis helpers shared by all four classifiers."""

import math
import hashlib
import statistics
import time

import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.nn import functional as neural_functional
from torch.utils.flop_counter import FlopCounterMode
from torchvision.transforms import functional as image_functional

from src.training import classification_metrics


def _binary_auc(labels, scores):
    """Mann–Whitney AUC; equal scores receive half credit."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    position = 0
    while position < len(ranked):
        end = position + 1
        while end < len(ranked) and ranked[end][0] == ranked[position][0]:
            end += 1
        average_rank = (position + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ranked[position:end])
        position = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluation_metrics(targets, predictions, probabilities, num_classes):
    """Compute fixed-class classification metrics from ordered per-image probabilities."""
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("Expected at least two mapped classes")
    if len(probabilities) != len(targets):
        raise ValueError("Probabilities and labels must align")
    for guess, row in zip(predictions, probabilities):
        if (len(row) != num_classes or any(not math.isfinite(value) or value < 0 or value > 1
                                            for value in row)
                or not math.isclose(sum(row), 1.0, abs_tol=1e-4)
                or max(range(num_classes), key=row.__getitem__) != guess):
            raise ValueError("Invalid probability row or predicted label")
    metrics = classification_metrics(targets, predictions, num_classes)
    matrix = metrics["confusion_matrix"]
    precision, recall = [], []
    for label in range(num_classes):
        tp = matrix[label][label]
        predicted_count = sum(row[label] for row in matrix)
        actual_count = sum(matrix[label])
        # Same zero-denominator rule as the saved Macro-F1.
        precision.append(tp / predicted_count if predicted_count else 0.0)
        recall.append(tp / actual_count if actual_count else 0.0)
    auc = [_binary_auc([int(truth == label) for truth in targets],
                       [row[label] for row in probabilities]) for label in range(num_classes)]
    metrics.update({"per_class_precision": precision, "per_class_recall": recall,
                    "per_class_ovr_auroc": auc,
                    "macro_ovr_auroc": sum(auc) / num_classes if all(value is not None for value in auc)
                    else None})
    return metrics


def corruption_input(image, sample_id, condition, config, mean, std):
    """Perturb the resized [0, 1] MRI before the shared ImageNet normalization."""
    if condition == "clean":
        return image
    family, severity = condition.split("_", maxsplit=1)
    if family not in config or severity not in config[family]:
        raise ValueError(f"Unknown corruption condition: {condition}")
    gray = (image[0:1] * std[0] + mean[0]).clamp(0, 1)
    # One random draw per image and corruption family, shared across severities: only the magnitude changes.
    digest = hashlib.sha256(f"{config['seed']}|{sample_id}|{family}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big"))
    if family == "rician":
        sigma = config[family][severity]
        n_real = torch.randn(gray.shape, generator=generator, dtype=gray.dtype)
        n_imag = torch.randn(gray.shape, generator=generator, dtype=gray.dtype)
        changed = torch.sqrt((gray + sigma * n_real).square() + (sigma * n_imag).square())
    elif family == "blur":
        sigma = config[family][severity]
        kernel = 2 * math.ceil(3 * sigma) + 1
        changed = image_functional.gaussian_blur(gray, [kernel, kernel], [sigma, sigma])
    elif family == "intensity":
        lower, upper = config[family][severity]
        factor = lower + (upper - lower) * torch.rand((), generator=generator).item()
        changed = gray * factor
    else:
        raise ValueError(f"Unknown corruption family: {family}")
    changed = changed.clamp(0, 1)
    return torch.cat([(changed - mean[channel]) / std[channel] for channel in range(3)])


def model_cost(model, device, config):
    """Measure forward-only operator FLOPs and synchronized batch-one latency."""
    example = torch.zeros((config["batch_size"], 3, *config["image_size"]),
                          device=device, dtype=torch.float32)
    unsupported = {}
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            original_count = counter._count_flops

            def count_and_record(operation, out, args, kwargs):
                if operation not in counter.flop_registry:
                    name = str(operation)
                    unsupported[name] = unsupported.get(name, 0) + 1
                return original_count(operation, out, args, kwargs)

            counter._count_flops = count_and_record
            output = model(example)
        if output.shape[0] != config["batch_size"] or not torch.isfinite(output).all():
            raise ValueError("Invalid model output during FLOP count")
        for _ in range(config["warmup"]):
            model(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations = []
        for _ in range(config["repetitions"]):
            started = time.perf_counter()
            model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(1000 * (time.perf_counter() - started))
    ordered = sorted(durations)
    return {"total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                        if parameter.requires_grad),
            "counted_forward_flops": counter.get_total_flops(),
            "counted_operators": {str(operator): value for operator, value in
                                  counter.get_flop_counts()["Global"].items()},
            "unregistered_operators": dict(sorted(unsupported.items())),
            "latency_ms_median": statistics.median(durations),
            "latency_ms_p25": ordered[len(ordered) // 4],
            "latency_ms_p75": ordered[3 * len(ordered) // 4],
            "timing_samples": len(durations)}


def grad_cam(model, image, class_index, *, layer, size=None):
    """Grad-CAM for one model and one confirmed target class at a shared layer.

    The map is upsampled to ``size`` (default: the model input size). Because the
    input is the full image resized without cropping, the 512 x 512 source grid is
    geometrically aligned with the 224 x 224 input.
    """
    activation = {}

    def capture(_module, _inputs, output):
        activation["feature"] = output
        output.register_hook(lambda gradient: activation.__setitem__("gradient", gradient))

    handle = layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(image.unsqueeze(0))
            if not torch.isfinite(logits).all() or not 0 <= class_index < logits.shape[1]:
                raise ValueError("Invalid Grad-CAM output or target")
            logits[0, class_index].backward()
            features, gradients = activation["feature"], activation["gradient"]
            if (features.shape[2:] != (7, 7) or not torch.isfinite(features).all()
                    or not torch.isfinite(gradients).all()):
                raise ValueError("Unexpected Grad-CAM layer shape or gradient")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * features).sum(dim=1, keepdim=True))
            heatmap = neural_functional.interpolate(heatmap, size=tuple(size or image.shape[-2:]),
                                                    mode="bilinear", align_corners=False)[0, 0]
            maximum = heatmap.detach().max().item()
            heatmap = heatmap / maximum if maximum > 0 else torch.zeros_like(heatmap)
            return heatmap.detach().cpu(), {"feature_shape": list(features.shape),
                                            "target_class": class_index,
                                            "zero_heatmap": maximum == 0}
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def load_binary_mask(path, *, threshold, size):
    """Read one original tumour mask and binarize it without any geometric resampling."""
    with Image.open(path) as image:
        if image.mode != "L" or image.size != tuple(size):
            raise ValueError(f"Unexpected tumour mask format: {path}")
        # 15 masks have intermediate grey values; binarise at a fixed threshold and leave the files untouched.
        mask = np.asarray(image) >= threshold
    if not mask.any():
        raise ValueError(f"Empty tumour mask after thresholding: {path}")
    return mask


def cam_mask_overlap(heatmap, mask):
    """Share of Grad-CAM energy inside the tumour mask, and its ratio to the mask area.

    This measures overlap between an explanation map and the annotated tumour; it is
    not a segmentation or localisation accuracy. Returns None for an all-zero map.
    """
    cam = np.asarray(heatmap, dtype=np.float64)
    if cam.shape != mask.shape or not np.isfinite(cam).all() or cam.min() < 0:
        raise ValueError("Grad-CAM map and mask must share one non-negative finite grid")
    total = cam.sum()
    if total <= 0:
        return None
    energy = float(cam[mask].sum() / total)
    area = float(mask.mean())
    return {"energy_in_mask": energy, "mask_area_fraction": area, "enrichment": energy / area}


def macro_f1_from_counts(counts, num_classes):
    """Vectorised Macro-F1 from flattened confusion counts (..., C*C); same zero rule as training."""
    matrix = np.asarray(counts, dtype=np.float64).reshape(*np.shape(counts)[:-1], num_classes, num_classes)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    fp = matrix.sum(axis=-2) - tp
    fn = matrix.sum(axis=-1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1.mean(axis=-1)


def paired_group_bootstrap(records, *, reference, comparisons, seeds, num_classes,
                           replicates, seed, level):
    """Pre-specified paired comparison of Macro-F1 against the reference model.

    ``records`` maps (model, training_seed) to per-image rows with sample_id,
    patient_group_id, true_label and predicted_label. Patient groups are resampled
    with replacement within each true class; every run is evaluated on the same
    resampled groups, so the comparison stays paired by image and by training seed.
    """
    keys = [(model, training_seed) for model in (reference, *comparisons) for training_seed in seeds]
    if any(key not in records for key in keys):
        raise ValueError("Paired bootstrap needs every model for every requested seed")
    first = sorted(records[keys[0]], key=lambda row: row["sample_id"])
    identity = [(row["sample_id"], row["patient_group_id"], int(row["true_label"])) for row in first]
    group_class = {}
    for _sample, group, label in identity:
        if group_class.setdefault(group, label) != label:
            raise ValueError(f"Patient group spans classes: {group}")
    groups = sorted(group_class)
    position = {group: index for index, group in enumerate(groups)}
    counts = {}
    for key in keys:
        rows = sorted(records[key], key=lambda row: row["sample_id"])
        if [(row["sample_id"], row["patient_group_id"], int(row["true_label"])) for row in rows] != identity:
            raise ValueError(f"Runs do not share the same evaluated images: {key}")
        table = np.zeros((len(groups), num_classes * num_classes), dtype=np.int64)
        for row in rows:
            table[position[row["patient_group_id"]],
                  int(row["true_label"]) * num_classes + int(row["predicted_label"])] += 1
        counts[key] = table
    # Resample groups within each true class so every replicate contains all classes.
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = np.zeros((replicates, len(groups)), dtype=np.int64)
    for label in sorted(set(group_class.values())):
        members = np.array([position[group] for group in groups if group_class[group] == label])
        draws = members[rng.integers(0, len(members), size=(replicates, len(members)))]
        np.add.at(weights, (np.repeat(np.arange(replicates), len(members)), draws.ravel()), 1)
    observed = {key: float(macro_f1_from_counts(table.sum(axis=0), num_classes)) for key, table in counts.items()}
    boot = {key: macro_f1_from_counts(weights @ table, num_classes) for key, table in counts.items()}
    tail = (1 - level) / 2
    results = {}
    for model in comparisons:
        per_seed = {str(s): observed[(model, s)] - observed[(reference, s)] for s in seeds}
        mean_boot = np.mean([boot[(model, s)] - boot[(reference, s)] for s in seeds], axis=0)
        lower, upper = (float(value) for value in np.quantile(mean_boot, [tail, 1 - tail]))
        differences = list(per_seed.values())
        if all(value > 0 for value in differences) and lower > 0:
            decision = "consistent_benefit"
        elif all(value < 0 for value in differences) and upper < 0:
            decision = "consistent_harm"
        else:
            decision = "no_consistent_difference"
        results[model] = {"reference": reference, "per_seed_macro_f1_difference": per_seed,
                          "mean_macro_f1_difference": float(np.mean(differences)),
                          "ci_lower": lower, "ci_upper": upper, "ci_level": level,
                          "bootstrap_sd": float(mean_boot.std(ddof=1)), "decision": decision}
    return {"observed_macro_f1": {f"{model}|seed{s}": value for (model, s), value in observed.items()},
            "comparisons": results, "seeds": list(seeds), "patient_groups": len(groups),
            "images": len(identity), "replicates": replicates, "bootstrap_seed": seed,
            "stratified_by_class_groups": {str(label): sum(1 for value in group_class.values() if value == label)
                                           for label in sorted(set(group_class.values()))}}


def save_cam_panel(path, image, mask_path, heatmaps, mean, std):
    """Save aligned 224-pixel MRI, optional mask contour, and model overlays."""
    gray = ((image[0] * std[0] + mean[0]).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    base = np.stack([gray] * 3, axis=-1)
    panels = [("MRI", base)]
    if mask_path is not None:
        with Image.open(mask_path) as original:
            if original.size != (512, 512):
                raise ValueError("Mask/MRI geometry mismatch")
            resized = original.convert("L").resize((gray.shape[1], gray.shape[0]),
                                                     resample=Image.Resampling.NEAREST)
        inside = np.asarray(resized) > 0
        interior = inside.copy()
        interior[1:, :] &= inside[:-1, :]
        interior[:-1, :] &= inside[1:, :]
        interior[:, 1:] &= inside[:, :-1]
        interior[:, :-1] &= inside[:, 1:]
        marked = base.copy()
        marked[inside & ~interior] = (0, 255, 0)
        panels.append(("Mask outline", marked))
    for name, heatmap in heatmaps.items():
        intensity = heatmap.numpy()[..., None]
        if heatmap.shape != gray.shape or not np.isfinite(intensity).all():
            raise ValueError("Grad-CAM overlay does not align with the MRI")
        overlay = base.astype(np.float32)
        overlay[..., 0] += intensity[..., 0] * (255 - overlay[..., 0]) * 0.65
        overlay[..., 1:] *= 1 - intensity * 0.55
        panels.append((name, overlay.clip(0, 255).astype(np.uint8)))
    height, width = gray.shape
    canvas = Image.new("RGB", (width * len(panels), height + 24), color="white")
    draw = ImageDraw.Draw(canvas)
    for position, (title, pixels) in enumerate(panels):
        draw.text((position * width + 5, 4), title, fill="black")
        canvas.paste(Image.fromarray(pixels), (position * width, 24))
    canvas.save(path)


def save_corruption_panel(path, images, mean, std):
    """Render the same MRI under clean and nine confirmed pilot conditions."""
    if len(images) != 10:
        raise ValueError("Expected clean plus nine corruption conditions")
    canvas = Image.new("RGB", (224 * 5, 248 * 2), color="white")
    draw = ImageDraw.Draw(canvas)
    for position, (condition, image) in enumerate(images.items()):
        pixels = ((image[0] * std[0] + mean[0]).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        if pixels.shape != (224, 224):
            raise ValueError("Corruption preview geometry changed")
        left, top = (position % 5) * 224, (position // 5) * 248
        draw.text((left + 5, top + 4), condition, fill="black")
        canvas.paste(Image.fromarray(pixels).convert("RGB"), (left, top + 24))
    canvas.save(path)


def save_severity_curves(path, model_results, *, scope):
    """Macro-F1 curves with an explicit split/stage label and fixed axes."""
    families = ("rician", "blur", "intensity")
    names = {"resnet50": "Baseline", "resnet50_se_terminal": "SE",
             "resnet50_psanet_terminal": "PSA", "resnet50_cbam_terminal": "CBAM"}
    colors = {"resnet50": "#1f77b4", "resnet50_se_terminal": "#ff7f0e",
              "resnet50_psanet_terminal": "#2ca02c", "resnet50_cbam_terminal": "#d62728"}
    canvas = Image.new("RGB", (1200, 330), "white")
    draw = ImageDraw.Draw(canvas)
    for panel, family in enumerate(families):
        left, top, right, bottom = panel * 400 + 42, 32, panel * 400 + 376, 258
        draw.text((left, 8), family + " / " + scope, fill="black")
        for value in (0.0, 0.5, 1.0):
            y = round(bottom - value * (bottom - top))
            draw.line((left, y, right, y), fill="#dddddd")
            draw.text((left - 30, y - 6), str(value), fill="black")
        for position, label in enumerate(("clean", "mild", "moderate", "severe")):
            x = round(left + position * (right - left) / 3)
            draw.text((x - 16, bottom + 5), label, fill="black")
        for model_name, result in model_results.items():
            scores = [result["conditions"]["clean"]["metrics"]["macro_f1"],
                      *(result["conditions"][f"{family}_{severity}"]["metrics"]["macro_f1"]
                        for severity in ("mild", "moderate", "severe"))]
            points = [(round(left + i * (right - left) / 3),
                       round(bottom - score * (bottom - top))) for i, score in enumerate(scores)]
            draw.line(points, fill=colors[model_name], width=2)
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[model_name])
    for position, (model_name, label) in enumerate(names.items()):
        x = 70 + position * 275
        draw.line((x, 303, x + 20, 303), fill=colors[model_name], width=3)
        draw.text((x + 28, 296), label, fill="black")
    canvas.save(path)


def save_confusion_matrix(path, matrix, class_names, *, title):
    """Display raw counts; rows are true labels and columns are predictions."""
    if len(matrix) != len(class_names) or any(len(row) != len(class_names) for row in matrix):
        raise ValueError("Confusion matrix and class names must align")
    cell, left, top = 145, 160, 90
    canvas = Image.new("RGB", (left + cell * len(class_names) + 20,
                               top + cell * len(class_names) + 20), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black")
    draw.text((left, 35), "Predicted class (columns)", fill="black")
    draw.text((12, 55), "True class (rows)", fill="black")
    maximum = max(max(row) for row in matrix) or 1
    for column, name in enumerate(class_names):
        draw.text((left + column * cell + 5, top - 22), name, fill="black")
    for row, name in enumerate(class_names):
        draw.text((8, top + row * cell + cell // 2), name, fill="black")
        for column, count in enumerate(matrix[row]):
            x, y = left + column * cell, top + row * cell
            shade = int(255 - 170 * count / maximum)
            draw.rectangle((x, y, x + cell, y + cell), fill=(shade, shade, 255), outline="white")
            draw.text((x + cell // 2 - 8, y + cell // 2), str(count), fill="black")
    canvas.save(path)
