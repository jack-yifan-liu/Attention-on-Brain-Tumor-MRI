"""Audit ScienceDB directory labels against the matched original Figshare labels.

Outputs (written to a new directory):
- cross_table.csv: directory label x original label counts for all images (metadata only).
- directory_label_runs.csv: contiguous runs of directory labels in PNG image_id order (metadata only).
- anatomical_check.csv: tumour-mask centroid and area summaries by directory -> original label,
  computed from train and validation masks only; test images are not opened.
- summary.json: inputs, hashes, parameters and the tables above.
"""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset import load_data_index

CLASSES = ("Glioma", "Meningioma", "Pituitary tumor")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs/paths_pid_numeric.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--central-radius", type=float, default=0.12,
                        help="Centroid distance from the image centre, as a fraction of image width")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Output directory exists; use a new one")
    index = load_data_index(args.paths_config)  # also verifies the metadata hashes
    rows = sorted(index["rows"], key=lambda row: int(row["image_id"]))

    cross = Counter((row["directory_class"], row["class_name"]) for row in rows)
    cross_rows = [{"directory_label": d, "original_label": o, "images": cross[(d, o)]}
                  for d in CLASSES for o in CLASSES]

    runs = []
    for row in rows:
        if runs and runs[-1]["directory_label"] == row["directory_class"]:
            runs[-1]["last_image_id"] = int(row["image_id"])
            runs[-1]["images"] += 1
        else:
            runs.append({"directory_label": row["directory_class"], "first_image_id": int(row["image_id"]),
                         "last_image_id": int(row["image_id"]), "images": 1})

    # Anatomical plausibility: pituitary tumours sit at the sella, so their masks should be central and small. Train/validation masks only.
    measures = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            continue
        with Image.open(index["data_root"] / row["mask_path"]) as image:
            if image.mode != "L" or image.size != (512, 512):
                raise ValueError(f"Unexpected mask format: {row['sample_id']}")
            mask = np.asarray(image) >= args.mask_threshold
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            raise ValueError(f"Empty mask after thresholding: {row['sample_id']}")
        distance = float(np.hypot(ys.mean() / 512 - 0.5, xs.mean() / 512 - 0.5))
        measures[(row["directory_class"], row["class_name"])].append((distance, float(mask.mean())))
    anatomy_rows = []
    for (directory, original), values in sorted(measures.items()):
        distances = [value[0] for value in values]
        areas = [value[1] for value in values]
        anatomy_rows.append({
            "directory_label": directory, "original_label": original, "images": len(values),
            "centroid_distance_median": round(statistics.median(distances), 4),
            "fraction_central": round(sum(d < args.central_radius for d in distances) / len(values), 4),
            "mask_area_fraction_median": round(statistics.median(areas), 5)})

    output.mkdir(parents=True)
    write_csv(output / "cross_table.csv", cross_rows, ("directory_label", "original_label", "images"))
    write_csv(output / "directory_label_runs.csv", runs,
              ("directory_label", "first_image_id", "last_image_id", "images"))
    write_csv(output / "anatomical_check.csv", anatomy_rows,
              ("directory_label", "original_label", "images", "centroid_distance_median",
               "fraction_central", "mask_area_fraction_median"))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": sys.argv[1:],
        "metadata_sha256": index["metadata_sha256"],
        "images": len(rows),
        "directory_label_disagreements": sum(row["directory_class"] != row["class_name"] for row in rows),
        "directory_label_runs": len(runs),
        "anatomical_check_scope": "train and validation masks only; test masks not opened",
        "anatomical_check_images": sum(item["images"] for item in anatomy_rows),
        "mask_threshold": args.mask_threshold,
        "central_radius_fraction_of_width": args.central_radius,
        "interpretation": ("Descriptive plausibility check of label provenance. Pituitary tumours sit near the "
                           "sella, so their masks should be central and small; this does not re-diagnose images."),
        "cross_table": cross_rows, "anatomical_check": anatomy_rows,
        "outputs_sha256": {name: sha256(output / name) for name in
                           ("cross_table.csv", "directory_label_runs.csv", "anatomical_check.csv")},
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in ("images", "directory_label_disagreements",
                                                    "directory_label_runs", "anatomical_check_images")},
                     indent=2))
    for item in anatomy_rows:
        print(item)


if __name__ == "__main__":
    main()
