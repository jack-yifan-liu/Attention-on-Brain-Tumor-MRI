"""Frozen formal analysis: validation preparation (cost, Grad-CAM, CAM localisation) and one-time test evaluation."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torchvision.models import ResNet50_Weights

from src.dataset import (MRIDataset, PROJECT_ROOT, build_transform, file_sha256,
                         load_data_index, seed_data_pipeline)
from src.evaluation import (cam_mask_overlap, corruption_input, evaluation_metrics, grad_cam,
                            load_binary_mask, model_cost, paired_group_bootstrap, save_cam_panel,
                            save_corruption_panel, save_severity_curves, save_confusion_matrix)
from src.models.resnet import build_model


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_formal_runs(formal, mode, index, formal_path):
    """Validate saved best checkpoints and their clean validation predictions."""
    paths = read_json(PROJECT_ROOT / formal["paths_config"])
    candidates = formal["runs"] if mode == "formal" else [
        run for run in formal["runs"] if run["seed"] == formal["training"]["seeds"][0]]
    validation = {row["sample_id"]: row for row in index["rows"] if row["split"] == "validation"}
    details = {}
    reference_sources = None
    for run in candidates:
        directory = PROJECT_ROOT / paths["results_dir"] / mode / run["run_name"]
        record, summary, saved_config = (read_json(directory / name) for name in
                                         ("run.json", "summary.json", "config.json"))
        checkpoint_path = PROJECT_ROOT / summary["checkpoint_path"]
        if (record["mode"] != mode or summary["mode"] != mode
                or saved_config["effective_mode"] != mode or saved_config["run"] != run
                or saved_config["formal"] != formal
                or record["formal_config_sha256"] != file_sha256(formal_path)
                or record["metadata_sha256"] != index["metadata_sha256"]
                or record["class_to_idx"] != index["class_to_idx"]
                or record["test_images_loaded"] != 0 or summary["test_images_loaded"] != 0
                or file_sha256(checkpoint_path) != summary["checkpoint_sha256"]
                or file_sha256(directory / "samples.csv") != record["samples_sha256"]):
            raise ValueError(f"Formal run provenance or checkpoint changed: {run['run_name']}")
        training_sources = {name.replace("\\", "/"): digest
                            for name, digest in record["source_sha256"].items()
                            if name.replace("\\", "/").startswith("src/") or name == "train.py"}
        if reference_sources is not None and mode == "formal" and training_sources != reference_sources:
            raise ValueError("Formal training runs used different source versions")
        reference_sources = training_sources
        rows = [row for row in read_rows(directory / "validation_predictions.csv")
                if int(row["epoch"]) == summary["best_epoch"]]
        expected_count = formal["split_counts"]["validation"]["images"] if mode == "formal" else 18
        if len(rows) != expected_count or len({row["sample_id"] for row in rows}) != expected_count:
            raise ValueError(f"Best validation predictions incomplete: {run['run_name']}")
        predictions = {}
        for row in rows:
            original = validation.get(row["sample_id"])
            if (original is None or original["patient_id"] != row["patient_id"]
                    or int(original["class_idx"]) != int(row["true_label"])):
                raise ValueError("Formal validation prediction/metadata mismatch")
            probabilities = [float(row[f"prob_{label}"]) for label in range(len(index["class_to_idx"]))]
            predicted = int(row["predicted_label"])
            if max(range(len(probabilities)), key=probabilities.__getitem__) != predicted:
                raise ValueError("Saved best validation probability/label mismatch")
            predictions[row["sample_id"]] = {"true_label": int(row["true_label"]),
                                             "predicted_label": predicted,
                                             "probabilities": probabilities}
        metrics = evaluation_metrics([int(row["true_label"]) for row in rows],
                                     [int(row["predicted_label"]) for row in rows],
                                     [[float(row[f"prob_{label}"]) for label in range(len(index["class_to_idx"]))]
                                      for row in rows], len(index["class_to_idx"]))
        if (metrics["confusion_matrix"] != summary["best_validation"]["confusion_matrix"]
                or not math.isclose(metrics["macro_f1"], summary["best_validation"]["macro_f1"],
                                    abs_tol=1e-12)):
            raise ValueError("Saved formal validation metrics do not reproduce")
        details[run["run_name"]] = {"run": run, "record": record, "summary": summary,
                                   "predictions": predictions}
    if len(details) != len(candidates):
        raise ValueError("Not every frozen run is complete")
    return details


def select_formal_cases(formal, details, class_mapping):
    """Within each training seed, pick the first case per true class and category."""
    protocol = formal["evaluation"]["grad_cam"]
    if (protocol["selection_split"] != "validation" or protocol["target"] != "true_class"
            or protocol["layer"] != "layer4"
            or protocol["per_true_class"] != "first_lexicographic_sample_id_for_each_category"):
        raise ValueError("Unsupported frozen Grad-CAM case rule")
    cases = {}
    for seed in formal["training"]["seeds"]:
        group = [detail for detail in details.values() if detail["run"]["seed"] == seed]
        if not group:
            continue
        if len(group) != 4 or len({detail["run"]["model"] for detail in group}) != 4:
            raise ValueError("Each case-selection seed needs four complete models")
        ids = set(group[0]["predictions"])
        if any(set(detail["predictions"]) != ids for detail in group):
            raise ValueError("Models did not share the same validation cases")
        cases[str(seed)] = {}
        for class_name, label in sorted(class_mapping.items(), key=lambda item: item[1]):
            eligible = sorted(sample_id for sample_id in ids
                              if group[0]["predictions"][sample_id]["true_label"] == label)
            cases[str(seed)][class_name] = {}
            for category in protocol["case_categories"]:
                chosen = None
                for sample_id in eligible:
                    votes = [detail["predictions"][sample_id]["predicted_label"]
                             for detail in group]
                    if ((category == "all_correct" and all(vote == label for vote in votes))
                            or (category == "all_incorrect" and all(vote != label for vote in votes))
                            or (category == "disagreement" and len(set(votes)) > 1)):
                        chosen = sample_id
                        break
                cases[str(seed)][class_name][category] = chosen
    return cases


def describe(values):
    """Mean and sample SD that stay defined for a single preflight run."""
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None, "n": len(values)}


def sample_standard(values):
    if any(value is None for value in values):
        return {"mean": None, "std": None, "n": len(values)}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "n": len(values)}


def aggregate_formal(formal, results):
    """Aggregate only matching test conditions across the three planned seeds."""
    aggregation = {}
    names = [model["name"] for model in formal["models"]]
    conditions = next(iter(results.values()))["conditions"]
    scalar = ("accuracy", "macro_f1", "macro_ovr_auroc")
    per_class = ("per_class_precision", "per_class_recall", "per_class_f1", "per_class_ovr_auroc")
    for name in names:
        group = [results[run["run_name"]] for run in formal["runs"] if run["model"] == name]
        if len(group) != 3:
            raise ValueError("Three seeds are required for formal aggregation")
        aggregation[name] = {"conditions": {},
                             "inference_latency_ms_median": sample_standard(
                                 [item["cost"]["latency_ms_median"] for item in group]),
                             "total_parameters": group[0]["cost"]["total_parameters"],
                             "counted_forward_flops": group[0]["cost"]["counted_forward_flops"]}
        if any((item["cost"]["total_parameters"] != aggregation[name]["total_parameters"]
                or item["cost"]["counted_forward_flops"] != aggregation[name]["counted_forward_flops"])
               for item in group):
            raise ValueError("Same model produced inconsistent parameter or counted FLOP totals")
        for condition in conditions:
            metrics = [item["conditions"][condition]["metrics"] for item in group]
            aggregation[name]["conditions"][condition] = {
                key: sample_standard([row[key] for row in metrics]) for key in scalar}
            aggregation[name]["conditions"][condition]["macro_f1_absolute_drop_from_clean"] = (
                sample_standard([item["conditions"][condition]["macro_f1_absolute_drop_from_clean"]
                                 for item in group]))
            aggregation[name]["conditions"][condition].update({
                key: [sample_standard([row[key][label] for row in metrics]) for label in range(3)]
                for key in per_class})
    return aggregation


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def analysis_source_hashes():
    names = ("analyze.py", "src/evaluation.py", "src/training.py", "src/dataset.py",
             "src/patient_groups.py", "src/models/resnet.py", "src/models/se.py",
             "src/models/psanet.py", "src/models/cbam.py")
    return {name: file_sha256(PROJECT_ROOT / name) for name in names}


def formal_analysis_main():
    parser = argparse.ArgumentParser(description="Frozen formal validation preparation and final test evaluation")
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--analysis-name", required=True)
    parser.add_argument("--stage", choices=("preflight", "prepare", "finalize"), required=True)
    args = parser.parse_args()
    formal_path = args.formal_config.resolve()
    if formal_path != (PROJECT_ROOT / "configs/formal_experiment_pid_numeric.json").resolve():
        raise ValueError("Use configs/formal_experiment_pid_numeric.json; exact-PID analysis is historical")
    if (not args.analysis_name.isascii() or not args.analysis_name.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("Invalid analysis name")
    formal = read_json(formal_path)
    current_analysis_sources = analysis_source_hashes()
    if current_analysis_sources != formal["analysis_source_sha256"]:
        raise ValueError("Frozen analysis implementation changed; review before evaluation")
    if formal["mode"] != "formal" or formal["status"] != "frozen_not_run":
        raise ValueError("Formal experiment configuration is not frozen")
    evaluation = formal["evaluation"]
    localization, comparison = evaluation["grad_cam_localization"], evaluation["statistical_comparison"]
    model_names = [item["name"] for item in formal["models"]]
    if (evaluation["grad_cam"]["layer"] != "layer4" or localization["layer"] != "layer4"
            or localization["split"] != "validation" or localization["target"] != "true_class"
            or localization["condition"] != "clean" or localization["grid"] != [512, 512]
            or comparison["primary_metric"] != "macro_f1" or comparison["condition"] != "clean"
            or comparison["pairing"] != "same_training_seed"
            or [comparison["reference_model"], *comparison["comparison_models"]] != model_names
            or comparison["bootstrap"]["unit"] != "patient_group_id"
            or comparison["bootstrap"]["stratify_by"] != "true_class"
            or comparison["bootstrap"]["interval"] != "percentile"):
        raise ValueError("Unsupported frozen Grad-CAM localisation or statistical comparison protocol")
    # Same deterministic cuBLAS setting as training; must be set before any CUDA call.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = formal["training"]["random_control"]["cublas_workspace_config"]
    paths = read_json(PROJECT_ROOT / formal["paths_config"])
    index = load_data_index(PROJECT_ROOT / formal["paths_config"])
    if (index["class_to_idx"] != formal["class_to_idx"]
            or index["split_seed"] != formal["split_seed"]
            or index["metadata_sha256"] != {key: formal["expected_sha256"][key]
                                            for key in ("manifest", "data_split", "class_mapping")}
            or file_sha256(PROJECT_ROOT / formal["data_config"]) != formal["expected_sha256"]["data_config"]):
        raise ValueError("Frozen split, labels or preprocessing changed")
    mode = "preflight" if args.stage == "preflight" else "formal"
    details = load_formal_runs(formal, mode, index, formal_path)
    output = (PROJECT_ROOT / paths["results_dir"] / "analysis" / args.analysis_name).resolve()
    raw_root = (PROJECT_ROOT / paths["data_root"]).resolve()
    if output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError("Analysis output must be separate from raw data")
    if args.stage == "finalize":
        prepared = read_json(output / "prepare.json")
        if (prepared["stage"] != "prepare" or prepared["formal_config_sha256"] != file_sha256(formal_path)
                or prepared["analysis_source_sha256"] != current_analysis_sources
                or prepared["samples"] != formal["split_counts"]["validation"]["images"]
                or prepared["test_images_loaded"] != 0
                or len(prepared["run_results"]) != len(formal["runs"])
                or prepared["checkpoint_sha256"] != {name: item["summary"]["checkpoint_sha256"]
                                                        for name, item in details.items()}
                or (output / "test.json").exists() or (output / "test_predictions.csv").exists()):
            raise ValueError("Prepared analysis changed or final test has already run")
    elif output.exists():
        raise ValueError("Analysis output already exists")
    data_config = read_json(PROJECT_ROOT / formal["data_config"])
    preset = ResNet50_Weights[data_config["weights"]].transforms()
    mean, std = preset.mean, preset.std
    torch_device = torch.device(formal["training"]["device"])
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA device unavailable")
    seed_data_pipeline(formal["evaluation"]["corruption"]["seed"],
                       deterministic_algorithms=formal["training"]["random_control"]["deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split = "test" if args.stage == "finalize" else "validation"
    dataset = MRIDataset(PROJECT_ROOT / formal["paths_config"], split,
                         build_transform(data_config, training=False))
    if len(dataset) != formal["split_counts"][split]["images"]:
        raise ValueError("Frozen evaluation split count changed")
    selected_ids = (set(next(iter(details.values()))["predictions"])
                    if args.stage == "preflight" else None)
    samples = [(row["sample_id"], dataset[position], row)
               for position, row in enumerate(dataset.rows)
               if selected_ids is None or row["sample_id"] in selected_ids]
    if selected_ids is not None and {sample_id for sample_id, _item, _row in samples} != selected_ids:
        raise ValueError("Preflight validation examples changed")
    if args.stage == "preflight" and len(samples) != 18:
        raise ValueError("Unexpected preflight sample count")
    conditions = ["clean", *(f"{family}_{severity}" for family in ("rician", "blur", "intensity")
                              for severity in ("mild", "moderate", "severe"))]
    corruption = formal["evaluation"]["corruption"]
    transformed = {condition: [corruption_input(item["image"], sample_id, condition,
                                                corruption, mean, std)
                               for sample_id, item, _row in samples]
                   for condition in conditions}
    input_hashes = {}
    for condition, images in transformed.items():
        digest = hashlib.sha256()
        for image in images:
            digest.update(image.numpy().tobytes())
        input_hashes[condition] = digest.hexdigest()
    sample_positions = {sample_id: position for position, (sample_id, _item, _row) in enumerate(samples)}
    cases = (select_formal_cases(formal, details, index["class_to_idx"])
             if args.stage != "finalize" else prepared["case_selection"])
    cam_maps, cam_checks = {}, {}
    results, prediction_rows, localization_rows = {}, [], []
    masks = ({sample_id: load_binary_mask(dataset.data_root / row["mask_path"],
                                          threshold=localization["mask_threshold"],
                                          size=localization["grid"])
              for sample_id, _item, row in samples} if args.stage != "finalize" else {})
    model_specs = {item["name"]: item for item in formal["models"]}
    for run_name, detail in details.items():
        spec, summary = detail["run"], detail["summary"]
        checkpoint = torch.load(PROJECT_ROOT / summary["checkpoint_path"], map_location="cpu",
                                weights_only=True)
        if (checkpoint["formal_config_sha256"] != file_sha256(formal_path)
                or checkpoint["run"] != spec or checkpoint["class_to_idx"] != index["class_to_idx"]
                or checkpoint["metadata_sha256"] != index["metadata_sha256"]):
            raise ValueError(f"Checkpoint changed during formal analysis: {run_name}")
        model = build_model(len(index["class_to_idx"]), model_name=spec["model"],
                            attention_config=model_specs[spec["model"]]["attention"]).to(torch_device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        cost = (model_cost(model, torch_device, formal["evaluation"]["cost"])
                if args.stage != "finalize" else prepared["run_results"][run_name]["cost"])
        result = {"model": spec["model"], "seed": spec["seed"], "best_epoch": summary["best_epoch"],
                  "cost": cost, "conditions": {}}
        clean_f1 = None
        for condition in conditions:
            probabilities = []
            with torch.inference_mode():
                batch_size = formal["training"]["validation_batch_size"]
                for start in range(0, len(samples), batch_size):
                    batch = torch.stack(transformed[condition][start:start + batch_size])
                    logits = model(batch.to(torch_device))
                    if (logits.shape != (len(batch), len(index["class_to_idx"]))
                            or not torch.isfinite(logits).all()):
                        raise ValueError("Non-finite or wrong-shape formal evaluation logits")
                    probabilities.extend(logits.softmax(dim=1).cpu().tolist())
            targets = [item["label"] for _sample_id, item, _row in samples]
            guesses = [max(range(len(probability)), key=probability.__getitem__)
                       for probability in probabilities]
            metrics = evaluation_metrics(targets, guesses, probabilities, len(index["class_to_idx"]))
            if condition == "clean":
                clean_f1 = metrics["macro_f1"]
                if split == "validation":
                    for (sample_id, _item, _row), prob, guess in zip(samples, probabilities, guesses):
                        saved = detail["predictions"][sample_id]
                        if (guess != saved["predicted_label"] or any(
                                not math.isclose(prob[label], saved["probabilities"][label], abs_tol=1e-5)
                                for label in range(len(prob)))):
                            raise ValueError(f"Best checkpoint validation prediction changed: {run_name}/{sample_id}")
            result["conditions"][condition] = {
                "metrics": metrics, "macro_f1_absolute_drop_from_clean": clean_f1 - metrics["macro_f1"]}
            for (sample_id, item, _row), prob, guess in zip(samples, probabilities, guesses):
                prediction_rows.append({"run_name": run_name, "model": spec["model"],
                                        "seed": spec["seed"], "condition": condition,
                                        "sample_id": sample_id, "patient_id": item["patient_id"],
                                        "patient_group_id": item["patient_group_id"],
                                        "true_label": item["label"], "predicted_label": guess,
                                        **{f"prob_{label}": prob[label] for label in range(len(prob))}})
        if args.stage != "finalize":
            for class_name, categories in cases[str(spec["seed"])].items():
                for category, sample_id in categories.items():
                    if sample_id is None:
                        continue
                    position = sample_positions[sample_id]
                    for condition in formal["evaluation"]["grad_cam"]["visual_conditions"]:
                        key = (spec["seed"], class_name, category, condition)
                        image = transformed[condition][position]
                        heatmap, check = grad_cam(model, image.to(torch_device),
                                                  samples[position][1]["label"],
                                                  layer=model.layer4)
                        cam_maps.setdefault(key, {})[spec["model"]] = heatmap
                        cam_checks.setdefault(str(key), {})[spec["model"]] = check
            # Grad-CAM / tumour-mask overlap on every evaluated image: an explanation analysis, not segmentation or localisation accuracy.
            overlaps, zero_maps = [], 0
            for sample_id, item, _row in samples:
                heatmap, _check = grad_cam(model, item["image"].to(torch_device), item["label"],
                                           layer=model.layer4, size=localization["grid"])
                overlap = cam_mask_overlap(heatmap.numpy(), masks[sample_id])
                zero_maps += overlap is None
                overlaps.append(overlap)
                localization_rows.append({"run_name": run_name, "model": spec["model"], "seed": spec["seed"],
                                          "sample_id": sample_id, "patient_group_id": item["patient_group_id"],
                                          "true_label": item["label"],
                                          **(overlap or {"energy_in_mask": None, "mask_area_fraction": None,
                                                         "enrichment": None})})
            valid = [overlap for overlap in overlaps if overlap is not None]
            result["cam_localization"] = {
                "images": len(overlaps), "zero_heatmaps_excluded": zero_maps,
                "energy_in_mask_mean": statistics.mean(o["energy_in_mask"] for o in valid) if valid else None,
                "energy_in_mask_median": statistics.median(o["energy_in_mask"] for o in valid) if valid else None,
                "enrichment_mean": statistics.mean(o["enrichment"] for o in valid) if valid else None,
                "enrichment_median": statistics.median(o["enrichment"] for o in valid) if valid else None}
        results[run_name] = result
        del model, checkpoint
        torch.cuda.empty_cache()
        print(json.dumps({"stage": args.stage, "run_name": run_name,
                          "clean_macro_f1": clean_f1, "conditions": len(conditions)}), flush=True)
    if args.stage != "finalize":
        output.mkdir(parents=True)
    fields = ["run_name", "model", "seed", "condition", "sample_id", "patient_id", "patient_group_id",
              "true_label", "predicted_label", *(f"prob_{label}" for label in range(len(index["class_to_idx"])))]
    prefix = "test" if args.stage == "finalize" else "validation"
    with (output / f"{prefix}_predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    with (output / f"{prefix}_severity_curves.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run_name", "model", "seed", "family",
                                                   "clean_macro_f1", "mild_macro_f1",
                                                   "moderate_macro_f1", "severe_macro_f1"))
        writer.writeheader()
        for run_name, result in results.items():
            for family in ("rician", "blur", "intensity"):
                writer.writerow({"run_name": run_name, "model": result["model"], "seed": result["seed"],
                                 "family": family,
                                 "clean_macro_f1": result["conditions"]["clean"]["metrics"]["macro_f1"],
                                 **{f"{severity}_macro_f1":
                                    result["conditions"][f"{family}_{severity}"]["metrics"]["macro_f1"]
                                    for severity in ("mild", "moderate", "severe")}})
    if args.stage != "finalize":
        with (output / f"{prefix}_cam_localization.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("run_name", "model", "seed", "sample_id",
                                                       "patient_group_id", "true_label", "energy_in_mask",
                                                       "mask_area_fraction", "enrichment"))
            writer.writeheader()
            writer.writerows(localization_rows)
        preview_position = sample_positions[min(sample_positions)]
        save_corruption_panel(output / f"{prefix}_corruption_preview.png",
                              {condition: transformed[condition][preview_position] for condition in conditions},
                              mean, std)
        for (seed, class_name, category, condition), heatmaps in cam_maps.items():
            sample_id = cases[str(seed)][class_name][category]
            position = sample_positions[sample_id]
            row = samples[position][2]
            save_cam_panel(output / f"gradcam_seed{seed}_class{index['class_to_idx'][class_name]}_"
                            f"{category}_{condition}.png", transformed[condition][position],
                           dataset.data_root / row["mask_path"], heatmaps, mean, std)
    for seed in {detail["run"]["seed"] for detail in details.values()}:
        by_model = {result["model"]: result for result in results.values() if result["seed"] == seed}
        scope = f"{split} preflight" if args.stage == "preflight" else f"formal {split}"
        save_severity_curves(output / f"{prefix}_severity_curves_seed{seed}.png", by_model, scope=scope)
    class_names = sorted(index["class_to_idx"], key=index["class_to_idx"].get)
    for run_name, result in results.items():
        save_confusion_matrix(output / f"{prefix}_confusion_{run_name}.png",
                              result["conditions"]["clean"]["metrics"]["confusion_matrix"],
                              class_names, title=f"{split} clean | {run_name}")
    # Pre-registered paired comparison with a class-stratified patient-group bootstrap; on validation it is only a code check.
    clean_records = {}
    for row in prediction_rows:
        if row["condition"] == "clean":
            clean_records.setdefault((row["model"], row["seed"]), []).append(row)
    seeds_present = sorted({detail["run"]["seed"] for detail in details.values()})
    if args.stage == "finalize" and seeds_present != formal["training"]["seeds"]:
        raise ValueError("Final statistical comparison requires every frozen training seed")
    bootstrap = comparison["bootstrap"]
    statistical_comparison = paired_group_bootstrap(
        clean_records, reference=comparison["reference_model"],
        comparisons=comparison["comparison_models"], seeds=seeds_present,
        num_classes=len(index["class_to_idx"]), replicates=bootstrap["replicates"],
        seed=bootstrap["seed"], level=bootstrap["level"])
    statistical_comparison["split"] = split
    statistical_comparison["interpretation"] = (
        "Pre-specified test comparison." if split == "test" else
        "Validation code check only: validation selected the checkpoints, so this is not inferential evidence.")
    result_report = {
        "stage": args.stage, "split": split, "formal_config_sha256": file_sha256(formal_path),
        "checkpoint_sha256": {name: item["summary"]["checkpoint_sha256"]
                              for name, item in details.items()},
        "class_to_idx": index["class_to_idx"], "metadata_sha256": index["metadata_sha256"],
        "samples": len(samples), "test_images_loaded": len(samples) if split == "test" else 0,
        "input_sha256_by_condition": input_hashes, "case_selection": cases,
        "grad_cam_checks": cam_checks, "run_results": results,
        "statistical_comparison": statistical_comparison,
        "environment": {"interpreter": sys.executable, "python": platform.python_version(),
                        "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "numpy": np.__version__, "pillow": pillow_version,
                        "gpu": torch.cuda.get_device_name(torch_device),
                        "cuda_build": torch.version.cuda,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "analysis_source_sha256": current_analysis_sources}
    if args.stage != "finalize":
        result_report["cam_localization_by_model"] = {
            name: {key: describe([result["cam_localization"][key] for result in results.values()
                                  if result["model"] == name])
                   for key in ("energy_in_mask_mean", "enrichment_mean")}
            for name in model_names}
    if args.stage == "finalize":
        result_report["mean_and_sample_std_by_model"] = aggregate_formal(formal, results)
        write_new_json(output / "test.json", result_report)
    else:
        write_new_json(output / ("preflight.json" if args.stage == "preflight" else "prepare.json"),
                       result_report)
    print(json.dumps({"stage": args.stage, "output": str(output), "runs": len(results),
                      "samples": len(samples), "test_images_loaded": result_report["test_images_loaded"]}),
          flush=True)


if __name__ == "__main__":
    formal_analysis_main()
