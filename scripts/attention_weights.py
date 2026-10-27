"""Post-hoc look at the attention weights learned by the SE and CBAM models (validation only).

Run after the formal analysis. For the SE and CBAM checkpoints of every training seed,
the script records the weights that each terminal module applies to the 544 clean
validation images and summarises them. It opens no test image, computes no test metric
and changes no saved result; the outputs are labelled post hoc.

The module sits after layer4 and is followed only by global average pooling and the
linear head. A channel weight that is the same for every image is therefore a fixed
rescaling that the baseline's linear head can represent just as well, so the summaries
ask how much the weights vary between images and where the CBAM spatial weights peak:
- channel weights (SE; CBAM channel step): mean, share above 0.9 and below 0.1, share
  of their variance explained by channel identity alone (1 = identical for every
  image), median spread across images within a channel, share of the image-to-image
  variation explained by the true class, and the lowest correlation between the
  class-average weight vectors;
- spatial weights (CBAM): range within each 7 x 7 map; enrichment inside the tumour
  mask (mean weight inside the mask / mean weight over the image, after the same
  bilinear upsampling to the 512 grid as Grad-CAM); and a pointing test: how often the
  highest-weighted of the 49 positions overlaps the tumour mask, against the rate
  expected for a randomly chosen position.
Spatial-weight examples for the figure: for each class, the validation image classified
correctly by all three CBAM seeds whose tumour-mask area is closest to the class median.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset import MRIDataset, build_transform, file_sha256, seed_data_pipeline
from src.evaluation import cam_mask_overlap, load_binary_mask
from src.models.resnet import build_model

MODELS = ("resnet50_se_terminal", "resnet50_cbam_terminal")
SLUG = {"resnet50_se_terminal": "resnet50-se-terminal", "resnet50_cbam_terminal": "resnet50-cbam-terminal"}
SEEDS = (0, 1, 2)
EDGES = np.round(np.linspace(0, 512, 8)).astype(int)  # the 7 x 7 feature cells on the original 512 grid


def channel_summary(gates, labels):
    g = gates.astype(np.float64)
    channel_means = g.mean(axis=0)
    class_means = np.vstack([g[labels == k].mean(axis=0) for k in range(3)])
    within = ((g - channel_means) ** 2).sum()
    between = sum((labels == k).sum() * ((class_means[k] - channel_means) ** 2).sum() for k in range(3))
    correlation = np.corrcoef(class_means)
    return {"mean_weight": g.mean(), "share_above_0.9": (g > 0.9).mean(), "share_below_0.1": (g < 0.1).mean(),
            "variance_share_channel_identity": channel_means.var() / g.var(),
            "median_sd_across_images": np.median(g.std(axis=0)),
            "class_share_of_image_variation": between / within,
            "min_correlation_between_class_means": min(correlation[0, 1], correlation[0, 2], correlation[1, 2])}


def spatial_summary(maps, masks, cells):
    enrichment, hits = [], []
    for weights, mask, cell in zip(maps, masks, cells):
        grid = F.interpolate(torch.from_numpy(weights)[None, None], size=mask.shape, mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        enrichment.append(cam_mask_overlap(grid, mask)["enrichment"])
        hits.append(bool(cell.flat[weights.argmax()]))
    enrichment, hits = np.array(enrichment), np.array(hits)
    chance = np.array([cell.mean() for cell in cells])
    ranges = maps.max(axis=(1, 2)) - maps.min(axis=(1, 2))
    summary = {"mean_weight": maps.mean(), "median_range_within_map": np.median(ranges),
               "median_enrichment_in_mask": np.median(enrichment), "share_enrichment_above_1": (enrichment > 1).mean(),
               "pointing_hit_rate": hits.mean(), "pointing_chance_rate": chance.mean()}
    return summary, enrichment, hits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formal-config", type=Path,
                        default=PROJECT_ROOT / "configs/formal_experiment_pid_numeric.json")
    parser.add_argument("--analysis-dir", type=Path,
                        default=PROJECT_ROOT / "results/analysis/pid-numeric-v2-final")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/figures/attention")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "validation_attention.json").exists():
        raise ValueError("Output already exists; use a new folder")
    formal = json.loads(args.formal_config.read_text(encoding="utf-8"))
    data_config = json.loads((PROJECT_ROOT / formal["data_config"]).read_text(encoding="utf-8"))
    control = formal["training"]["random_control"]
    # Same deterministic GPU settings and batching as analyze.py, so the saved validation predictions can be matched.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = control["cublas_workspace_config"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the saved validation predictions")
    device = torch.device("cuda")
    seed_data_pipeline(formal["evaluation"]["corruption"]["seed"],
                       deterministic_algorithms=control["deterministic_algorithms"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    localization = formal["evaluation"]["grad_cam_localization"]
    specs = {item["name"]: item["attention"] for item in formal["models"]}
    dataset = MRIDataset(PROJECT_ROOT / formal["paths_config"], "validation",
                         build_transform(data_config, training=False))
    rows = dataset.rows
    labels = np.array([int(row["class_idx"]) for row in rows])
    masks = [load_binary_mask(dataset.data_root / row["mask_path"], threshold=localization["mask_threshold"],
                              size=localization["grid"]) for row in rows]
    cells = [np.array([[mask[EDGES[i]:EDGES[i + 1], EDGES[j]:EDGES[j + 1]].any() for j in range(7)]
                       for i in range(7)]) for mask in masks]
    saved = {}
    with (args.analysis_dir / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["condition"] == "clean":
                saved[(item["run_name"], item["sample_id"])] = (
                    int(item["predicted_label"]), [float(item[f"prob_{k}"]) for k in range(3)])

    batch_size = formal["training"]["validation_batch_size"]
    runs, arrays, checks, digests, correct = {}, {}, {}, {}, {}
    for model_name in MODELS:
        for seed in SEEDS:
            run = f"formal-v2-{SLUG[model_name]}-seed{seed}"
            checkpoint = PROJECT_ROOT / f"checkpoints/formal/{run}/best.pt"
            model = build_model(3, model_name=model_name, attention_config=specs[model_name]).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"],
                                  strict=True)
            model.eval()
            captured = {"channel": [], "spatial": []}

            def keep(_module, _inputs, output, captured=captured):
                # The CBAM sigmoid runs twice per forward pass: channel weights (N, C), then spatial (N, 1, 7, 7).
                captured["channel" if output.ndim == 2 else "spatial"].append(output.detach().cpu())

            handle = model.avgpool[0].sigmoid.register_forward_hook(keep)
            probabilities = []
            with torch.inference_mode():
                for start in range(0, len(rows), batch_size):
                    batch = torch.stack([dataset[i]["image"] for i in range(start, min(start + batch_size, len(rows)))])
                    probabilities.extend(model(batch.to(device)).softmax(dim=1).cpu().tolist())
            handle.remove()
            predicted = [max(range(3), key=p.__getitem__) for p in probabilities]
            reference = [saved[(run, row["sample_id"])] for row in rows]
            if predicted != [label for label, _ in reference]:
                raise ValueError(f"Recomputed validation predictions differ from the saved ones: {run}")
            checks[run] = {"labels_identical": True, "max_probability_difference": max(
                abs(a - b) for p, (_, q) in zip(probabilities, reference) for a, b in zip(p, q))}
            correct[run] = np.array(predicted) == labels
            digests[run] = file_sha256(checkpoint)

            channel = torch.cat(captured["channel"]).numpy()
            if channel.shape != (len(rows), 2048):
                raise ValueError(f"Unexpected channel-weight shape: {channel.shape}")
            summary = {"channel": channel_summary(channel, labels)}
            arrays[f"{run}|channel_mean"] = channel.mean(axis=0)
            arrays[f"{run}|channel_p05"] = np.percentile(channel, 5, axis=0)
            arrays[f"{run}|channel_p95"] = np.percentile(channel, 95, axis=0)
            arrays[f"{run}|channel_class_means"] = np.vstack([channel[labels == k].mean(axis=0) for k in range(3)])
            if model_name == "resnet50_cbam_terminal":
                spatial = torch.cat(captured["spatial"]).numpy()
                if spatial.shape != (len(rows), 1, 7, 7):
                    raise ValueError(f"Unexpected spatial-weight shape: {spatial.shape}")
                spatial = spatial[:, 0]
                summary["spatial"], enrichment, hits = spatial_summary(spatial, masks, cells)
                arrays[f"{run}|spatial"] = spatial
                arrays[f"{run}|spatial_enrichment"] = enrichment
                arrays[f"{run}|pointing_hit"] = hits
            elif captured["spatial"]:
                raise ValueError("SE must not produce spatial weights")
            runs[run] = {"model": model_name, "seed": seed,
                         **{part: {key: float(value) for key, value in values.items()}
                            for part, values in summary.items()}}

    area = np.array([mask.mean() for mask in masks])
    cbam_runs = [f"formal-v2-{SLUG['resnet50_cbam_terminal']}-seed{seed}" for seed in SEEDS]
    examples = []
    for k in range(3):
        members = np.flatnonzero(labels == k)
        median = np.median(area[members])
        eligible = [i for i in members if all(correct[run][i] for run in cbam_runs)]
        examples.append(rows[min(eligible, key=lambda i: (abs(area[i] - median), rows[i]["sample_id"]))]["sample_id"])
    by_model = {}
    for model_name in MODELS:
        model_runs = [r for r in runs.values() if r["model"] == model_name]
        by_model[model_name] = {part: {key: {"mean": statistics.mean(r[part][key] for r in model_runs),
                                             "sd": statistics.stdev(r[part][key] for r in model_runs)}
                                       for key in model_runs[0][part]}
                                for part in ("channel", "spatial") if part in model_runs[0]}

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "validation_attention.npz", sample_ids=np.array([row["sample_id"] for row in rows]),
                        labels=labels, mask_area=area, **arrays)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-hoc, validation only; descriptive summaries of learned attention weights; no test image opened",
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "validation_images": len(rows), "checkpoint_sha256": digests, "prediction_check": checks,
        "method": __doc__.split("\n\n", 1)[1].strip(), "examples": examples,
        "runs": runs, "mean_and_sample_sd_by_model": by_model,
    }
    (output / "validation_attention.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for model_name, parts in by_model.items():
        for part, values in parts.items():
            print(model_name, part, {key: f"{v['mean']:.3f} ({v['sd']:.3f})" for key, v in values.items()})
    print("examples", examples)
    print("max probability difference", max(c["max_probability_difference"] for c in checks.values()))


if __name__ == "__main__":
    main()
