"""Post-hoc Grad-CAM for a few rule-selected test images (illustration only).

Run after the one-time test evaluation. The script reads the saved test
predictions, selects cases by fixed rules, and computes Grad-CAM maps with the
seed-0 checkpoints of the baseline, SE and CBAM models. It computes no metric and
changes no result; outputs are labelled post-hoc.

Selection rules (clean test images, 9 runs = 3 models x seeds 0-2):
- correct cases: for each class, images classified correctly by all 9 runs; the
  one whose tumour-mask area is closest to the class median is shown.
- error cases: for each error type (glioma -> meningioma, pituitary -> glioma,
  meningioma -> pituitary), the image with that error in the most runs; ties
  prefer images where all three seed-0 models make the error, then sample ID.
"""

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset import MRIDataset, build_transform, file_sha256
from src.evaluation import grad_cam
from src.models.resnet import build_model

MODELS = ("resnet50", "resnet50_se_terminal", "resnet50_cbam_terminal")
RUN_SLUG = {"resnet50": "resnet50", "resnet50_se_terminal": "resnet50-se-terminal",
            "resnet50_cbam_terminal": "resnet50-cbam-terminal"}
ERROR_TYPES = {"glioma_as_meningioma": (0, 1), "pituitary_as_glioma": (2, 0),
               "meningioma_as_pituitary": (1, 2)}


def mask_area(data_root, row):
    with Image.open(data_root / row["mask_path"]) as image:
        return float((np.asarray(image) >= 128).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formal-config", type=Path,
                        default=PROJECT_ROOT / "configs/formal_experiment_pid_numeric.json")
    parser.add_argument("--analysis-dir", type=Path,
                        default=PROJECT_ROOT / "results/analysis/pid-numeric-v2-final")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/figures/gradcam")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "test_cases.json").exists():
        raise ValueError("Output already exists; use a new folder")
    formal = json.loads(args.formal_config.read_text(encoding="utf-8"))
    data_config = json.loads((PROJECT_ROOT / formal["data_config"]).read_text(encoding="utf-8"))
    specs = {item["name"]: item["attention"] for item in formal["models"]}
    dataset = MRIDataset(PROJECT_ROOT / formal["paths_config"], "test", build_transform(data_config, training=False))
    rows = {row["sample_id"]: (index, row) for index, row in enumerate(dataset.rows)}

    predictions = {}
    with (args.analysis_dir / "test_predictions.csv").open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            if item["condition"] == "clean" and item["model"] in MODELS:
                predictions.setdefault(item["sample_id"], {})[(item["model"], int(item["seed"]))] = int(item["predicted_label"])
    if any(len(runs) != 9 for runs in predictions.values()) or set(predictions) != set(rows):
        raise ValueError("Expected nine saved predictions for every test image")

    areas = {sample_id: mask_area(dataset.data_root, row) for sample_id, (_, row) in rows.items()}
    cases = []
    for label in range(3):
        correct = [s for s, runs in predictions.items()
                   if int(rows[s][1]["class_idx"]) == label and all(p == label for p in runs.values())]
        median = statistics.median(areas[s] for s in rows if int(rows[s][1]["class_idx"]) == label)
        chosen = min(correct, key=lambda s: (abs(areas[s] - median), s))
        cases.append({"case": f"correct_class{label}", "kind": "correct", "sample_id": chosen,
                      "runs_with_shown_outcome": 9})
    for name, (true, wrong) in ERROR_TYPES.items():
        scored = []
        for s, runs in predictions.items():
            if int(rows[s][1]["class_idx"]) != true:
                continue
            count = sum(p == wrong for p in runs.values())
            seed0 = all(runs[(m, 0)] == wrong for m in MODELS)
            if count:
                scored.append((-count, not seed0, s, count))
        _, _, chosen, count = min(scored)
        cases.append({"case": name, "kind": "error", "sample_id": chosen, "runs_with_shown_outcome": count})

    models = {}
    for name in MODELS:
        checkpoint = PROJECT_ROOT / f"checkpoints/formal/formal-v2-{RUN_SLUG[name]}-seed0/best.pt"
        model = build_model(3, model_name=name, attention_config=specs[name])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
        models[name] = (model.eval(), file_sha256(checkpoint))

    heatmaps = {}
    for case in cases:
        index, row = rows[case["sample_id"]]
        item = dataset[index]
        case.update(true_label=int(row["class_idx"]), class_name=row["class_name"], patient_id=row["patient_id"],
                    patient_group_id=row["patient_group_id"], mri_path=row["mri_path"], mask_path=row["mask_path"],
                    code_format="MR-prefixed" if row["patient_id"].startswith("MR") else "numeric",
                    mask_area_fraction=areas[case["sample_id"]], seed0={})
        for name, (model, _) in models.items():
            with torch.no_grad():
                probabilities = model(item["image"].unsqueeze(0)).softmax(dim=1)[0].tolist()
            predicted = max(range(3), key=probabilities.__getitem__)
            if predicted != predictions[case["sample_id"]][(name, 0)]:
                raise ValueError(f"Recomputed prediction differs from the saved one: {name} {case['sample_id']}")
            # Predicted-class map: shows what supported the decision, including a wrong one.
            for target, key in ((predicted, "predicted"), (case["true_label"], "true")):
                heatmap, _ = grad_cam(model, item["image"], target, layer=model.layer4, size=(512, 512))
                heatmaps[f"{case['case']}|{name}|{key}"] = heatmap.numpy().astype(np.float16)
            case["seed0"][name] = {"predicted_label": predicted, "probabilities": probabilities}

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "test_cases_heatmaps.npz", **heatmaps)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-hoc illustration after the one-time test evaluation; no metric computed",
        "grad_cam": {"layer": "layer4", "targets": ["predicted", "true"], "grid": [512, 512],
                     "checkpoints": "seed 0", "device": "cpu"},
        "checkpoint_sha256": {name: digest for name, (_, digest) in models.items()},
        "selection_rules": __doc__.split("Selection rules", 1)[1].strip(),
        "cases": cases,
    }
    (output / "test_cases.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(case["case"], case["sample_id"], case["class_name"], case["code_format"], case["patient_id"],
              f"{case['runs_with_shown_outcome']}/9",
              {m: v["predicted_label"] for m, v in case["seed0"].items()})


if __name__ == "__main__":
    main()
