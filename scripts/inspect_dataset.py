"""Read-only MRI/mask inspection; writes a new manifest and report directory."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Keep the project's existing label order instead of numbering classes by folder traversal order.
CLASS_TO_IDX = {"Glioma": 0, "Meningioma": 1, "Pituitary tumor": 2}


def read_image(path):
    with Image.open(path) as image:
        image.load()
        array = np.array(image)
        return array, image.mode, image.size


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_preview(root, rows, output):
    # Display only: no resizing or binarisation; red opacity is proportional to the raw mask value.
    selected = [next(r for r in rows if r["class_name"] == name and r["status"] == "ok")
                for name in CLASS_TO_IDX]
    canvas = Image.new("RGB", (512 * 3, 548 * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        mri, _, _ = read_image(root / row["mri_path"])
        mask, _, _ = read_image(root / row["mask_path"])
        if mri.shape != (512, 512) or mask.shape != (512, 512):
            raise ValueError("Preview requires verified 512 x 512 grayscale pairs")
        rgb = np.repeat(mri[..., None], 3, axis=-1).astype(np.float32)
        alpha = (mask.astype(np.float32) / 255 * 0.45)[..., None]
        overlay = np.clip(rgb * (1 - alpha) + np.array([255, 0, 0]) * alpha, 0, 255).astype(np.uint8)
        y = index * 548
        for column, (label, image) in enumerate([
                ("MRI", Image.fromarray(mri).convert("RGB")),
                ("Original mask", Image.fromarray(mask).convert("RGB")),
                ("Overlay (display only)", Image.fromarray(overlay))]):
            draw.text((column * 512 + 8, y + 10), f"{row['sample_id']} | {label}", fill="black")
            canvas.paste(image, (column * 512, y + 36))
    canvas.save(output / "examples.png")
    return [row["sample_id"] for row in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if not root.is_dir():
        parser.error("Data root does not exist")
    if output.exists():
        parser.error("Output directory already exists; use a new directory")
    if output.is_relative_to(root):
        parser.error("Outputs must be outside the read-only data root")
    dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if dirs != set(CLASS_TO_IDX):
        parser.error(f"Unexpected class directories: {sorted(dirs)}")

    rows, issues, nonbinary = [], [], []
    counts, modes, dimensions, dtypes = {}, Counter(), Counter(), Counter()
    file_count = 0
    for name, label in CLASS_TO_IDX.items():
        files = sorted((root / name).iterdir(), key=lambda p: p.name)
        mris = [p for p in files if p.is_file() and p.suffix.lower() == ".png" and not p.stem.endswith("_mask")]
        masks = [p for p in files if p.is_file() and p.suffix.lower() == ".png" and p.stem.endswith("_mask")]
        counts[name] = {"mri": len(mris), "mask": len(masks)}
        file_count += len(files)
        for path in files:
            if path not in mris and path not in masks:
                issues.append({"path": path.relative_to(root).as_posix(), "reason": "unexpected_entry"})
        for mask in masks:
            if not mask.with_name(mask.stem[:-5] + ".png").exists():
                issues.append({"path": mask.relative_to(root).as_posix(), "reason": "orphan_mask"})
        for mri in mris:
            mask = mri.with_name(mri.stem + "_mask.png")
            row = {"sample_id": f"{name}/{mri.stem}", "image_id": mri.stem,
                   "class_name": name, "class_idx": label,
                   "mri_path": mri.relative_to(root).as_posix(),
                   "mask_path": mask.relative_to(root).as_posix(), "status": "ok"}
            try:
                for kind, path in (("mri", mri), ("mask", mask)):
                    array, mode, size = read_image(path)
                    row.update({f"{kind}_width": size[0], f"{kind}_height": size[1],
                                f"{kind}_mode": mode, f"{kind}_dtype": str(array.dtype)})
                    modes[f"{kind}:{mode}"] += 1
                    dimensions[f"{kind}:{size[0]}x{size[1]}"] += 1
                    dtypes[f"{kind}:{array.dtype}"] += 1
                    if kind == "mask":
                        values = np.unique(array)
                        binary = set(values.tolist()).issubset({0, 255})
                        row["mask_binary_0_255"] = binary
                        row["mask_unique_values"] = len(values)
                        row["mask_nonzero_pixels"] = int(np.count_nonzero(array))
                        if not binary:
                            nonbinary.append({"sample_id": row["sample_id"], "mask_path": row["mask_path"],
                                              "unique_values": values.tolist()})
                if (row["mri_width"], row["mri_height"]) != (row["mask_width"], row["mask_height"]):
                    raise ValueError("MRI/mask size mismatch")
            except (OSError, ValueError) as exc:
                row["status"] = "error"
                issues.append({"path": row["mri_path"], "reason": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
    for path in root.iterdir():
        if not path.is_dir():
            issues.append({"path": path.name, "reason": "unexpected_root_file"})
    assert len({r["sample_id"] for r in rows}) == len(rows)
    output.mkdir(parents=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "manifest.csv", rows, fields)
    write_csv(output / "issues.csv", issues, ["path", "reason"])
    # Record problems only; never drop samples or change labels automatically at this stage.
    write_csv(output / "exclusions.csv", [], ["sample_id", "reason"])
    for filename, obj in (("class_to_idx.json", CLASS_TO_IDX), ("nonbinary_masks.json", nonbinary)):
        (output / filename).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    preview_ids = []
    if not issues:
        preview_ids = make_preview(root, rows, output)
    historical_counts = {"Glioma": 1426, "Meningioma": 708, "Pituitary tumor": 930}
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": args.data_root.as_posix(), "manifest_paths_relative_to": "data_root",
              "counts": counts, "entries_in_class_directories": file_count,
              "mri_count": len(rows), "issues_count": len(issues), "exclusions_count": 0,
              "modes": dict(modes), "dimensions": dict(dimensions), "dtypes": dict(dtypes),
              "nonbinary_masks_count": len(nonbinary), "binary_definition": "values subset of {0, 255}",
              "historical_class_counts_match": all(counts[c]['mri'] == n and counts[c]['mask'] == n for c,n in historical_counts.items()),
              "preview_sample_ids": preview_ids, "preview_selection": "first readable pair per class in filename lexicographic order",
              "manifest_sha256": hashlib.sha256((output / 'manifest.csv').read_bytes()).hexdigest(),
              "scope": "File readability, pairing, metadata, mask values; no split, no patient mapping or leakage audit; folder labels not clinically revalidated"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
