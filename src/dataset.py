"""Shared MRI classification data pipeline; labels come from saved metadata."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from src.patient_groups import numeric_pid, patient_group


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_transform(data_config, *, training):
    preprocessing = data_config["preprocessing"]
    weights = ResNet50_Weights[data_config["weights"]]
    preset = weights.transforms()
    if (preprocessing["resize_mode"] != "full_image" or preprocessing["crop"] != "none"
            or preprocessing["remove_blank_edges"]
            or preprocessing["interpolation"] != "bilinear"
            or preprocessing["normalization"] != "weight_preset"):
        raise ValueError("Unsupported preprocessing; do not silently change the confirmed protocol")
    operations = [transforms.Resize(tuple(preprocessing["image_size"]),
                                   interpolation=transforms.InterpolationMode.BILINEAR,
                                   antialias=preprocessing["antialias"])]
    if training:
        augmentation = data_config["training_augmentation"]
        if augmentation["interpolation"] != "bilinear":
            raise ValueError("Training affine interpolation must be explicit bilinear")
        operations.extend([
            transforms.RandomAffine(degrees=augmentation["rotation_degrees"],
                                    translate=tuple(augmentation["translate"]),
                                    scale=None, shear=None,
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=augmentation["fill"]),
            transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip_probability"]),
        ])
    # Use the weights' mean/std but keep the full MRI: no resize-to-232 and centre crop from the preset.
    operations.extend([transforms.ToTensor(), transforms.Normalize(preset.mean, preset.std)])
    return transforms.Compose(operations)


def load_data_index(paths_config):
    paths_config = Path(paths_config)
    paths = json.loads(paths_config.read_text(encoding="utf-8"))
    resolved = {key: (PROJECT_ROOT / paths[key]).resolve()
                for key in ("data_root", "manifest", "data_split", "class_mapping")}
    evidence_path = resolved["data_split"].parent / "summary.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # Fail on missing or changed metadata; never rebuild the split or fall back to folder labels.
    for key, name in (("manifest", "manifest.csv"), ("data_split", "data_split.csv"),
                      ("class_mapping", "class_to_idx.json")):
        if file_sha256(resolved[key]) != evidence["output_sha256"][name]:
            raise ValueError(f"Saved metadata hash mismatch: {resolved[key]}")
    class_mapping = json.loads(resolved["class_mapping"].read_text(encoding="utf-8"))
    if sorted(class_mapping.values()) != list(range(len(class_mapping))):
        raise ValueError("Class indices must be unique and contiguous from zero")
    manifest = read_csv(resolved["manifest"])
    by_id = {row["sample_id"]: row for row in manifest}
    rows = read_csv(resolved["data_split"])
    if len(by_id) != len(manifest) or len(rows) != len(manifest):
        raise ValueError("Duplicate manifest ID or incomplete split")
    seen = set()
    patient_splits, patient_classes = {}, {}
    group_splits, group_classes = {}, {}
    for row in rows:
        sample, pid, name, split = (row[key] for key in
                                    ("sample_id", "patient_id", "class_name", "split"))
        if (not sample or not pid or pid.strip() != pid or sample in seen
                or split not in SPLITS or name not in class_mapping
                or int(row["class_idx"]) != class_mapping[name]):
            raise ValueError(f"Invalid sample, split, PID or label: {sample}")
        seen.add(sample)
        if {key: value for key, value in row.items() if key != "split"} != by_id.get(sample):
            raise ValueError(f"Split/manifest disagreement: {sample}")
        if (patient_splits.setdefault(pid, split) != split
                or patient_classes.setdefault(pid, name) != name):
            raise ValueError(f"Cross-split patient or inconsistent patient class: {pid}")
        group = patient_group(row)
        expected_group = numeric_pid(pid) if evidence.get("patient_grouping") == "numeric" else pid
        if (group != expected_group or group_splits.setdefault(group, split) != split
                or group_classes.setdefault(group, name) != name):
            raise ValueError(f"Invalid grouping key, cross-split group or conflicting group class: {pid}")
        relative = Path(row["mri_path"])
        path = (resolved["data_root"] / relative).resolve()
        if (relative.is_absolute() or not path.is_relative_to(resolved["data_root"])
                or path.name.endswith("_mask.png") or not path.is_file()):
            raise ValueError(f"Invalid classification MRI path: {row['mri_path']}")
    if seen != set(by_id) or len(rows) != evidence["samples"]:
        raise ValueError("Saved split does not cover the verified manifest")
    for split in SPLITS:
        if {row["class_name"] for row in rows if row["split"] == split} != set(class_mapping):
            raise ValueError(f"Missing class in {split}")
    return {"data_root": resolved["data_root"], "rows": rows,
            "class_to_idx": class_mapping, "split_seed": evidence["split_seed"],
            "metadata_sha256": {key: file_sha256(resolved[key]) for key in
                                ("manifest", "data_split", "class_mapping")}}


class MRIDataset(Dataset):
    def __init__(self, paths_config, split, transform):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        if not callable(transform):
            raise TypeError("An explicit image transform is required")
        index = load_data_index(paths_config)
        self.data_root = index["data_root"]
        self.class_to_idx = index["class_to_idx"]
        self.metadata_sha256 = index["metadata_sha256"]
        self.split = split
        self.rows = [row for row in index["rows"] if row["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        # Open the MRI only; folder names and masks never feed the classifier input or its labels.
        with Image.open(self.data_root / row["mri_path"]) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"Unexpected raw MRI format: {row['sample_id']}")
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "label": int(row["class_idx"]),
                "sample_id": row["sample_id"], "patient_id": row["patient_id"],
                "patient_group_id": patient_group(row)}


def seed_data_pipeline(seed, *, deterministic_algorithms):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic_algorithms)


def seed_worker(worker_id):
    # DataLoader seeds torch per worker; derive the NumPy and Python seeds from it.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, *, batch_size, num_workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.split == "train",
                      num_workers=num_workers, drop_last=False, pin_memory=False,
                      worker_init_fn=seed_worker, generator=generator)
