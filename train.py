"""Train one frozen formal MRI classifier run (or its short preflight) without opening test images."""

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torch import nn
from torch.utils.data import Subset

from src.dataset import (MRIDataset, PROJECT_ROOT, build_transform, file_sha256,
                         make_loader, seed_data_pipeline)
from src.models.resnet import build_model
from src.evaluation import evaluation_metrics
from src.training import is_better_checkpoint, run_epoch, write_predictions
from src.patient_groups import patient_group


class PreflightSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.split = dataset.split


def select_preflight_subset(dataset, per_class, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = []
    for name in sorted(dataset.class_to_idx, key=dataset.class_to_idx.get):
        patients = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            if row["class_name"] == name:
                patients[patient_group(row)].append(index)
        if len(patients) < per_class:
            raise ValueError(f"Not enough distinct patients for {dataset.split}/{name}")
        # Preflight subset: one image per distinct patient group and class, drawn inside the saved split.
        selected = rng.choice(sorted(patients), size=per_class, replace=False).tolist()
        for pid in selected:
            candidates = sorted(patients[pid], key=lambda i: int(dataset.rows[i]["image_id"]))
            indices.append(candidates[int(rng.integers(len(candidates)))])
    indices.sort(key=lambda i: int(dataset.rows[i]["image_id"]))
    return PreflightSubset(dataset, indices)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_optimizer(model, config):
    options = dict(config["optimizer"])
    if options.pop("name") != "AdamW":
        raise ValueError("The confirmed training protocol supports AdamW")
    options["betas"] = tuple(options["betas"])
    return torch.optim.AdamW(model.parameters(), **options)


def formal_main():
    parser = argparse.ArgumentParser(description="Run one frozen formal configuration or its short preflight")
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    formal_path = args.formal_config.resolve()
    if formal_path != (PROJECT_ROOT / "configs/formal_experiment_pid_numeric.json").resolve():
        raise ValueError("Use configs/formal_experiment_pid_numeric.json; exact-PID runs are historical")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    training = formal["training"]
    selection = training["checkpoint_selection"]
    if (formal["mode"] != "formal" or formal["status"] != "frozen_not_run"
            or training["precision"] != "float32" or training["loss"] != "cross_entropy"
            or training["freeze_backbone"] or training["class_weights"] is not None
            or training["scheduler"] is not None or training["early_stopping"] is not None
            or not training["random_control"]["deterministic_algorithms"]
            or training["random_control"]["allow_tf32"]
            or training["random_control"]["cublas_workspace_config"] != ":4096:8"
            or training["train_shuffle"] is not True
            or training["validation_shuffle"] is not False or training["drop_last"] is not False
            or selection != {"metric": "validation_macro_f1", "direction": "maximize",
                             "tie_break": "lower_validation_loss", "final_tie": "earlier_epoch"}
            or formal["output_subdirectory"] != "formal"):
        raise ValueError("Unsupported or unfrozen formal training protocol")
    run = next((row for row in formal["runs"] if row["run_name"] == args.run_name), None)
    if run is None or run["seed"] not in training["seeds"]:
        raise ValueError("Run name and seed must come from the frozen 12-run manifest")
    model_spec = next((row for row in formal["models"] if row["name"] == run["model"]), None)
    if model_spec is None:
        raise ValueError("Run model missing from the frozen model list")
    paths_config = PROJECT_ROOT / formal["paths_config"]
    data_config_path = PROJECT_ROOT / formal["data_config"]
    paths = json.loads(paths_config.read_text(encoding="utf-8"))
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    if data_config["weights"] != "IMAGENET1K_V2":
        raise ValueError("Formal preprocessing/weights version changed")
    expected = formal["expected_sha256"]
    if file_sha256(data_config_path) != expected["data_config"]:
        raise ValueError("Frozen preprocessing configuration changed")
    pretrained_path = PROJECT_ROOT / formal["pretrained_path"]
    if file_sha256(pretrained_path) != expected["pretrained"]:
        raise ValueError("Frozen V2 pretrained file changed")
    for name, digest in formal["training_source_sha256"].items():
        if file_sha256(PROJECT_ROOT / name) != digest:
            raise ValueError(f"Frozen training implementation changed: {name}")
    mode = "preflight" if args.preflight else "formal"
    output = (PROJECT_ROOT / paths["results_dir"] / mode / args.run_name).resolve()
    checkpoint_dir = (PROJECT_ROOT / paths["checkpoints_dir"] / mode / args.run_name).resolve()
    raw_root = (PROJECT_ROOT / paths["data_root"]).resolve()
    for directory in (output, checkpoint_dir):
        if directory.exists() or directory.is_relative_to(raw_root) or raw_root.is_relative_to(directory):
            raise ValueError(f"Run output must be new and separate from raw data: {directory}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        raise ValueError("Invalid run name")
    control = training["random_control"]
    # cuBLAS reads this when its handle is created, so set it before any CUDA call (overriding the environment).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = control["cublas_workspace_config"]
    device = torch.device(training["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA device unavailable; no silent CPU fallback")
    seed = run["seed"]
    seed_data_pipeline(seed, deterministic_algorithms=control["deterministic_algorithms"])
    torch.backends.cudnn.deterministic = control["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = control["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = control["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = control["allow_tf32"]
    datasets = {split: MRIDataset(paths_config, split,
                                  build_transform(data_config, training=split == "train"))
                for split in ("train", "validation")}
    if (datasets["train"].metadata_sha256 != {k: expected[k] for k in
                                               ("manifest", "data_split", "class_mapping")}
            or datasets["train"].class_to_idx != formal["class_to_idx"]
            or datasets["validation"].class_to_idx != formal["class_to_idx"]
            or any(len(dataset) != formal["split_counts"][split]["images"]
                   for split, dataset in datasets.items())):
        raise ValueError("Frozen split, labels or sample counts changed")
    split_seed = json.loads((PROJECT_ROOT / paths["data_split"]).with_name("summary.json")
                            .read_text(encoding="utf-8"))["split_seed"]
    if split_seed != formal["split_seed"]:
        raise ValueError("Frozen patient-group split seed changed")
    if args.preflight:
        # Exercise the formal code path on a small distinct-patient subset; not a formal result.
        subsets = {split: select_preflight_subset(dataset, per_class, 0)
                   for split, dataset, per_class in (("train", datasets["train"], 12),
                                                     ("validation", datasets["validation"], 6))}
        selected_rows = [datasets[split].rows[index] for split, subset in subsets.items()
                         for index in subset.indices]
        epochs = 1
    else:
        subsets = datasets
        selected_rows = [*datasets["train"].rows, *datasets["validation"].rows]
        epochs = training["epochs"]
    patients = {split: {patient_group(row) for row in dataset.rows}
                for split, dataset in datasets.items()}
    if patients["train"] & patients["validation"]:
        raise ValueError("Patient crossed the saved train/validation split")
    loaders = {split: make_loader(subset,
                                  batch_size=training["batch_size"] if split == "train"
                                  else training["validation_batch_size"],
                                  num_workers=training["num_workers"], seed=seed)
               for split, subset in subsets.items()}
    num_classes = len(formal["class_to_idx"])
    model = build_model(num_classes, model_name=run["model"],
                        attention_config=model_spec["attention"],
                        pretrained_path=pretrained_path).to(device)
    optimizer = make_optimizer(model, training)
    criterion = nn.CrossEntropyLoss()
    output.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    write_json(output / "config.json", {"formal": formal, "run": run,
                                         "effective_mode": mode, "effective_epochs": epochs,
                                         "data": data_config})
    with (output / "samples.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True)
    training_sources = ("src/dataset.py", "src/patient_groups.py", "src/training.py", "src/evaluation.py",
                        "src/models/resnet.py", "src/models/se.py", "src/models/psanet.py",
                        "src/models/cbam.py")
    sources = (formal_path, paths_config, data_config_path, Path(__file__),
               *(PROJECT_ROOT / name for name in training_sources))
    provenance = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "run_name": args.run_name,
        "mode": mode, "command_arguments": sys.argv[1:], "interpreter": sys.executable,
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": np.__version__,
        "pillow": pillow_version, "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()),
        "source_sha256": {str(path.resolve().relative_to(PROJECT_ROOT)): file_sha256(path)
                          for path in sources},
        "formal_config_sha256": file_sha256(formal_path),
        "metadata_sha256": datasets["train"].metadata_sha256,
        "samples_sha256": file_sha256(output / "samples.csv"),
        "pretrained_sha256": file_sha256(pretrained_path), "class_to_idx": formal["class_to_idx"],
        "model": {"name": run["model"], "attention": model_spec["attention"],
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "all_parameters_trainable": all(p.requires_grad for p in model.parameters())},
        "random_control": {"training_seed": seed, "split_seed": split_seed,
                           "dataloader_seed": seed, "num_workers": training["num_workers"],
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "samples": {split: len(subset) for split, subset in subsets.items()},
        "patient_overlap": 0, "test_images_loaded": 0,
    }
    write_json(output / "run.json", provenance)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path = checkpoint_dir / "best.pt"
    best, best_epoch, best_predictions, history, all_predictions = None, None, None, [], []
    # Re-seed after model construction so attention-module initialisation cannot shift the augmentation stream.
    seed_data_pipeline(seed, deterministic_algorithms=control["deterministic_algorithms"])
    for epoch in range(1, epochs + 1):
        synchronize(device)
        started = time.perf_counter()
        train_metrics, _ = run_epoch(model, loaders["train"], criterion, device, num_classes,
                                     optimizer=optimizer, validate_updates=args.preflight)
        synchronize(device)
        train_seconds = time.perf_counter() - started
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("Non-finite updated parameter")
        started = time.perf_counter()
        validation, predictions = run_epoch(model, loaders["validation"], criterion, device, num_classes)
        synchronize(device)
        val_seconds = time.perf_counter() - started
        extended = evaluation_metrics([row["true_label"] for row in predictions],
                                      [row["predicted_label"] for row in predictions],
                                      [[row[f"prob_{label}"] for label in range(num_classes)]
                                       for row in predictions], num_classes)
        validation.update(extended)
        epoch_record = {"epoch": epoch, "train": train_metrics, "validation": validation,
                        "training_seconds": train_seconds, "validation_seconds": val_seconds}
        history.append(epoch_record)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        all_predictions.extend({"epoch": epoch, **record} for record in predictions)
        if is_better_checkpoint(validation, best,
                                tie_break=training["checkpoint_selection"]["tie_break"]):
            best, best_epoch, best_predictions = validation, epoch, predictions
            temporary = checkpoint_dir / "best.pt.tmp"
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "validation_metrics": validation,
                        "formal_config_sha256": provenance["formal_config_sha256"],
                        "run": run, "data_config": data_config,
                        "class_to_idx": formal["class_to_idx"],
                        "metadata_sha256": provenance["metadata_sha256"],
                        "samples_sha256": provenance["samples_sha256"]}, temporary)
            temporary.replace(checkpoint_path)
        print(json.dumps({"run_name": args.run_name, "mode": mode, "epoch": epoch,
                          "training_seconds": round(train_seconds, 3),
                          "validation_macro_f1": validation["macro_f1"],
                          "validation_loss": validation["loss"], "best_epoch": best_epoch}), flush=True)
    write_json(output / "history.json", history)
    write_predictions(output / "validation_predictions.csv", all_predictions, num_classes)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del model, optimizer
    torch.cuda.empty_cache()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["formal_config_sha256"] != provenance["formal_config_sha256"]
            or checkpoint["run"] != run or checkpoint["class_to_idx"] != formal["class_to_idx"]
            or checkpoint["metadata_sha256"] != provenance["metadata_sha256"]
            or checkpoint["samples_sha256"] != provenance["samples_sha256"]):
        raise ValueError("Best checkpoint differs from this frozen run")
    restored = build_model(num_classes, model_name=run["model"],
                           attention_config=model_spec["attention"]).to(device)
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    restored_optimizer = make_optimizer(restored, training)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if len(restored_optimizer.state) != sum(1 for p in restored.parameters() if p.requires_grad):
        raise ValueError("Optimizer state did not restore every trainable parameter")
    restored_metrics, restored_predictions = run_epoch(restored, loaders["validation"], criterion,
                                                       device, num_classes)
    reference = torch.tensor([[row[f"prob_{label}"] for label in range(num_classes)]
                              for row in best_predictions])
    recovered = torch.tensor([[row[f"prob_{label}"] for label in range(num_classes)]
                              for row in restored_predictions])
    if (not torch.allclose(reference, recovered, atol=1e-7, rtol=1e-6)
            or [row["sample_id"] for row in best_predictions]
            != [row["sample_id"] for row in restored_predictions]
            or [row["predicted_label"] for row in best_predictions]
            != [row["predicted_label"] for row in restored_predictions]):
        raise ValueError("Best checkpoint reload predictions disagree")
    summary = {"mode": mode, "best_epoch": best_epoch, "best_validation": best,
               "restored_validation": restored_metrics,
               "checkpoint_selection": training["checkpoint_selection"],
               "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
               "checkpoint_sha256": file_sha256(checkpoint_path),
               "checkpoint_reload": {"max_probability_absolute_error":
                                     float((reference - recovered).abs().max()),
                                     "labels_identical": True, "optimizer_state_loaded": True},
               "gpu_peak_allocated_bytes": peak_allocated,
               "gpu_peak_reserved_bytes": peak_reserved,
               "training_seconds": sum(row["training_seconds"] for row in history),
               "validation_seconds": sum(row["validation_seconds"] for row in history),
               "test_images_loaded": 0}
    write_json(output / "summary.json", summary)
    print(json.dumps({"run_name": args.run_name, "mode": mode, "best_epoch": best_epoch,
                      "checkpoint_reload": summary["checkpoint_reload"],
                      "results": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    formal_main()
