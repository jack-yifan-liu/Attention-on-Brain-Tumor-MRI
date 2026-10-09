"""Build author-labelled metadata and a fixed, stratified patient split."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.patient_groups import numeric_pid, patient_group
ORIGINAL_CLASSES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary tumor"}
SPLITS = ("train", "validation", "test")
FIELDS = (
    "sample_id", "image_id", "patient_id", "patient_group_id", "class_name", "class_idx",
    "mri_path", "mask_path", "directory_class", "original_label",
    "source_archive", "source_mat",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def corrected_rows(mapping, inspection, class_mapping, root):
    inspected = {row["sample_id"]: row for row in inspection}
    if len(inspected) != len(inspection):
        raise ValueError("Duplicate sample_id in inspection manifest")
    rows = []
    seen = {key: set() for key in ("sample_id", "image_id", "source_mat")}
    patient_classes = {}
    for source in mapping:
        for key, values in seen.items():
            if not source[key] or source[key] in values:
                raise ValueError(f"Missing or duplicate {key}: {source[key]}")
            values.add(source[key])
        old = inspected.get(source["sample_id"])
        if old is None or old["status"] != "ok":
            raise ValueError(f"Sample not verified by inspection: {source['sample_id']}")
        for key in ("image_id", "class_name", "class_idx", "mri_path", "mask_path"):
            if source[key] != old[key]:
                raise ValueError(f"Mapping/inspection disagree: {source['sample_id']} {key}")
        label = int(source["original_label"])
        name = ORIGINAL_CLASSES[label]
        if name != source["original_class"] or source["mask_exact"] != "True":
            raise ValueError(f"Unverified original label or mask: {source['sample_id']}")
        pid = source["patient_id"]
        # PID is the author's original string (suffix kept); never infer patient or label from PNG numbers or folder names.
        if not pid or pid.strip() != pid or patient_classes.setdefault(pid, name) != name:
            raise ValueError(f"Missing PID or conflicting patient label: {pid}")
        row = {key: source[key] for key in FIELDS if key in source}
        row.update(class_name=name, class_idx=class_mapping[name],
                   directory_class=source["class_name"], original_label=label)
        for key in ("mri_path", "mask_path"):
            relative = Path(row[key])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Invalid or missing raw path: {row[key]}")
        if (row["mri_path"].endswith("_mask.png")
                or row["mask_path"] != row["mri_path"][:-4] + "_mask.png"):
            raise ValueError(f"Invalid MRI/mask pair: {row['sample_id']}")
        rows.append(row)
    if seen["sample_id"] != set(inspected):
        raise ValueError("Mapping does not cover the complete inspection manifest")
    return sorted(rows, key=lambda row: int(row["image_id"]))


def allocate_counts(total, ratios):
    # Counts are patient groups; largest-remainder rounding, with ties going to train, then validation, then test.
    exact = [total * value for value in ratios]
    counts = [value.numerator // value.denominator for value in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    if min(counts) < 1:
        raise ValueError("Each class must have at least one patient in each split")
    return counts


def assign_splits(rows, class_mapping, ratios, seed):
    patients = defaultdict(set)
    patient_classes = {}
    for row in rows:
        pid, name = patient_group(row), row["class_name"]
        if patient_classes.setdefault(pid, name) != name:
            raise ValueError(f"Conflicting patient label: {pid}")
        patients[name].add(pid)
    rng = np.random.Generator(np.random.PCG64(seed))
    assignment = {}
    for name in sorted(class_mapping, key=class_mapping.get):
        # Sort before shuffling so the input row order cannot change the split.
        shuffled = rng.permutation(sorted(patients[name])).tolist()
        offset = 0
        for split, count in zip(SPLITS, allocate_counts(len(shuffled), ratios)):
            for pid in shuffled[offset:offset + count]:
                assignment[pid] = split
            offset += count
    return [{**row, "split": assignment[patient_group(row)]} for row in rows]


def validate_split(rows, class_mapping):
    samples, patients = {}, {}
    counts = {split: {} for split in SPLITS}
    for row in rows:
        sample, pid, split = row["sample_id"], patient_group(row), row["split"]
        if sample in samples or patients.setdefault(pid, split) != split:
            raise ValueError(f"Duplicate sample or patient overlap: {sample}, {pid}")
        samples[sample] = split
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        for name in class_mapping:
            selected = [row for row in subset if row["class_name"] == name]
            counts[split][name] = {
                "images": len(selected), "patient_groups": len({patient_group(r) for r in selected}),
                "original_pids": len({r["patient_id"] for r in selected}),
            }
            if not selected:
                raise ValueError(f"Empty class: {split}/{name}")
        counts[split]["total"] = {
            "images": len(subset), "patient_groups": len({patient_group(r) for r in subset}),
            "original_pids": len({r["patient_id"] for r in subset}),
        }
    return counts


def check_raw_files(rows, root):
    fingerprint = hashlib.sha256()
    image_groups = defaultdict(list)
    for row in rows:
        for key in ("mri_path", "mask_path"):
            relative = row[key]
            fingerprint.update((relative + "\0" + sha256(root / relative) + "\n").encode())
        with Image.open(root / row["mri_path"]) as image:
            image.load()
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"Unexpected MRI format: {row['sample_id']}")
            image_groups[hashlib.sha256(image.tobytes()).hexdigest()].append(row)
    duplicates = [[row["sample_id"] for row in group]
                  for group in image_groups.values() if len(group) > 1]
    cross_split = [group for group in image_groups.values()
                   if len({row["split"] for row in group}) > 1]
    if cross_split:
        raise ValueError("Identical MRI pixels cross splits; investigate before saving")
    return fingerprint.hexdigest(), duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--inspection-dir", type=Path, required=True)
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs/paths_pid_numeric.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs=3, type=Fraction, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--patient-grouping", choices=("exact", "numeric"), default="exact")
    parser.add_argument("--previous-split", type=Path)
    args = parser.parse_args()
    if any(value <= 0 for value in args.ratios) or sum(args.ratios) != 1 or args.seed < 0:
        parser.error("Ratios must be positive and sum to 1; seed must be nonnegative")
    paths = json.loads(args.paths_config.read_text(encoding="utf-8"))
    root = (PROJECT_ROOT / paths["data_root"]).resolve()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be a new directory separate from raw data")
    mapping_path = args.mapping_dir / "patient_mapping.csv"
    inspection_path = args.inspection_dir / "manifest.csv"
    class_path = args.inspection_dir / "class_to_idx.json"
    evidence_path = args.mapping_dir / "summary.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (sha256(mapping_path) != evidence["mapping_sha256"]
            or sha256(inspection_path) != evidence["manifest_sha256"]):
        raise ValueError("Source CSV changed since the patient mapping was verified")
    class_mapping = json.loads(class_path.read_text(encoding="utf-8"))
    if class_mapping != {"Glioma": 0, "Meningioma": 1, "Pituitary tumor": 2}:
        raise ValueError("Unexpected class mapping; do not silently change label order")
    rows = corrected_rows(read_csv(mapping_path), read_csv(inspection_path), class_mapping, root)
    patient_counts = {name: len({r["patient_id"] for r in rows if r["class_name"] == name})
                      for name in class_mapping}
    if len(rows) != evidence["samples"] or patient_counts != evidence["original_patient_counts_by_class"]:
        raise ValueError("Sample or patient counts disagree with the verified mapping")
    for row in rows:
        row["patient_group_id"] = (numeric_pid(row["patient_id"]) if args.patient_grouping == "numeric"
                                   else row["patient_id"])
    group_counts = {name: len({patient_group(r) for r in rows if r["class_name"] == name})
                    for name in class_mapping}
    split_rows = assign_splits(rows, class_mapping, args.ratios, args.seed)
    counts = validate_split(split_rows, class_mapping)
    raw_fingerprint, duplicates = check_raw_files(split_rows, root)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "manifest.csv", rows, FIELDS)
    write_csv(output / "data_split.csv", split_rows, (*FIELDS, "split"))
    write_json(output / "class_to_idx.json", class_mapping)
    if args.previous_split:
        previous = {row["sample_id"]: row for row in read_csv(args.previous_split)}
        if set(previous) != {row["sample_id"] for row in rows}:
            raise ValueError("Previous split does not contain the same samples")
        changes = [{"sample_id": row["sample_id"], "patient_id": row["patient_id"],
                    "patient_group_id": row["patient_group_id"],
                    "old_split": previous[row["sample_id"]]["split"], "new_split": row["split"]}
                   for row in split_rows]
        write_csv(output / "split_changes.csv", changes, tuple(changes[0]))
    sources = {str(path.resolve().relative_to(PROJECT_ROOT)) if path.resolve().is_relative_to(PROJECT_ROOT)
               else str(path.resolve()): sha256(path)
               for path in (mapping_path, inspection_path, class_path, evidence_path, Path(__file__),
                            PROJECT_ROOT / "src/patient_groups.py")}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": sys.argv[1:], "python": platform.python_version(),
        "numpy": np.__version__, "pillow": pillow_version,
        "source_doi": evidence["source_doi"], "source_sha256": sources,
        "data_root": paths["data_root"], "samples": len(rows), "original_pids": sum(patient_counts.values()),
        "patient_groups": sum(group_counts.values()), "class_group_counts": group_counts,
        "patient_grouping": args.patient_grouping,
        "grouping_assumption": ("Same integer numeric PID component means the same patient; unverified conservative assumption authorized 2026-10-02. Original PID retained."
                                 if args.patient_grouping == "numeric" else "Exact original PID strings"),
        "class_image_counts": dict(Counter(r["class_name"] for r in rows)),
        "class_patient_counts": patient_counts,
        "corrected_directory_labels": sum(r["class_name"] != r["directory_class"] for r in rows),
        "excluded_samples": 0, "split_seed": args.seed,
        "training_seeds": "Configured separately from split seed in the experiment config",
        "requested_patient_ratios": dict(zip(SPLITS, map(str, args.ratios))),
        "algorithm": "Class-index order, lexicographically sorted patient_group_id strings, NumPy PCG64 permutation; per-class largest remainder counts; ties train/validation/test",
        "counts": counts, "patient_overlap": 0, "sample_overlap": 0,
        "identical_mri_pixel_groups": duplicates, "identical_mri_cross_split_groups": 0,
        "raw_files_fingerprint_sha256": raw_fingerprint,
        "raw_fingerprint_method": "SHA256 of image-id-sorted relative-path + NUL + file SHA256 + LF; MRI then mask",
        "limitations": "Grouping is an assumption, not identity verification or proof of no leakage. New test membership can overlap historical train/validation; old runs cannot be used for this split.",
        "output_sha256": {name: sha256(output / name) for name in
                          ("manifest.csv", "data_split.csv", "class_to_idx.json")},
    }
    if args.previous_split:
        summary["previous_split_sha256"] = sha256(args.previous_split)
        summary["transition_counts"] = dict(Counter(f"{r['old_split']} -> {r['new_split']}" for r in changes))
        summary["changed_split_samples"] = sum(r["old_split"] != r["new_split"] for r in changes)
        summary["new_test_previous_membership"] = dict(Counter(r["old_split"] for r in changes if r["new_split"] == "test"))
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in
                      ("samples", "original_pids", "patient_groups", "corrected_directory_labels", "counts",
                       "patient_overlap", "identical_mri_cross_split_groups")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
