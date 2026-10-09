"""Screen for near-duplicate MRI images across train / validation / test (no model, no labels used).

Each MRI is reduced to a 32 x 32 area-averaged thumbnail and compared by cosine similarity of the
mean-centred pixels. The most similar cross-split pairs are then re-scored with full-resolution
Pearson correlation and saved as a review panel. Pairs from the same assumed patient group cannot
cross splits by construction; same-group pairs inside one split are reported as a reference for how
similar adjacent slices of one patient are.

This is a screening aid for human review. A high score is not proof of a duplicate or of the same
patient, and a low score does not prove independence.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.dataset import load_data_index


def load_pixels(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError(f"Unexpected MRI format: {path}")
        full = np.asarray(image, dtype=np.float32)
        thumb = np.asarray(image.resize((32, 32), Image.Resampling.BOX), dtype=np.float32)
    return full, thumb


def centred_unit(matrix):
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths-config", type=Path, default=PROJECT_ROOT / "configs/paths_pid_numeric.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30, help="Cross-split pairs re-scored at full resolution")
    parser.add_argument("--flag-threshold", type=float, default=0.99,
                        help="Full-resolution correlation at or above which a pair is flagged for review")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Output directory exists; use a new one")
    index = load_data_index(args.paths_config)
    rows = sorted(index["rows"], key=lambda row: row["sample_id"])
    thumbs, fulls = [], {}
    for position, row in enumerate(rows):
        full, thumb = load_pixels(index["data_root"] / row["mri_path"])
        thumbs.append(thumb.ravel())
        fulls[position] = full
    vectors = centred_unit(np.stack(thumbs))
    similarity = vectors @ vectors.T
    splits = np.array([row["split"] for row in rows])
    groups = np.array([row["patient_group_id"] for row in rows])
    upper = np.triu(np.ones(similarity.shape, dtype=bool), k=1)
    cross = upper & (splits[:, None] != splits[None, :])
    same_group = upper & (groups[:, None] == groups[None, :])
    other_group_same_split = upper & (groups[:, None] != groups[None, :]) & (splits[:, None] == splits[None, :])

    def quantiles(mask):
        values = similarity[mask]
        return {f"q{int(q * 1000) / 10:g}": round(float(np.quantile(values, q)), 4)
                for q in (0.5, 0.99, 0.999, 1.0)} | {"pairs": int(values.size)}

    order = np.argsort(similarity[cross])[::-1][:args.top]
    candidates = np.argwhere(cross)[order]
    pairs = []
    for i, j in candidates:
        a, b = rows[i], rows[j]
        pairs.append({"sample_a": a["sample_id"], "split_a": a["split"], "group_a": a["patient_group_id"],
                      "class_a": a["class_name"], "sample_b": b["sample_id"], "split_b": b["split"],
                      "group_b": b["patient_group_id"], "class_b": b["class_name"],
                      "thumbnail_cosine": round(float(similarity[i, j]), 5),
                      "full_resolution_pearson": round(pearson(fulls[i], fulls[j]), 5)})
    flagged = [pair for pair in pairs if pair["full_resolution_pearson"] >= args.flag_threshold]
    # Reference: full-resolution correlation of the most similar slice pairs within one assumed patient group.
    reference_order = np.argsort(similarity[same_group])[::-1][:args.top]
    reference = [round(pearson(fulls[i], fulls[j]), 5) for i, j in np.argwhere(same_group)[reference_order]]

    output.mkdir(parents=True)
    with (output / "top_cross_split_pairs.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    shown = pairs[:8]
    canvas = Image.new("L", (2 * 160 + 10, len(shown) * 175), color=255)
    draw = ImageDraw.Draw(canvas)
    position = {row["sample_id"]: k for k, row in enumerate(rows)}
    for k, pair in enumerate(shown):
        for column, key in enumerate(("sample_a", "sample_b")):
            tile = Image.fromarray(fulls[position[pair[key]]].astype(np.uint8)).resize((160, 160))
            canvas.paste(tile, (column * 170, k * 175 + 15))
            draw.text((column * 170 + 2, k * 175 + 1), f"{pair['split_' + key[-1]]} r={pair['full_resolution_pearson']:.3f}", fill=0)
    canvas.save(output / "top_cross_split_pairs.png")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "command_arguments": sys.argv[1:],
        "metadata_sha256": index["metadata_sha256"], "images": len(rows),
        "images_read_by_split": {split: int((splits == split).sum()) for split in ("train", "validation", "test")},
        "uses_model_or_labels_for_scoring": False,
        "thumbnail": "32x32 area average, mean-centred cosine similarity",
        "cosine_quantiles": {"cross_split": quantiles(cross), "same_group_same_split": quantiles(same_group),
                             "different_group_same_split": quantiles(other_group_same_split)},
        "top_cross_split_full_resolution_pearson_max": max(pair["full_resolution_pearson"] for pair in pairs),
        "top_same_group_full_resolution_pearson": reference[:10],
        "flag_threshold": args.flag_threshold, "flagged_pairs": len(flagged), "flagged": flagged,
        "interpretation": "Screening aid only; flagged pairs need visual review before any split decision.",
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in ("images", "cosine_quantiles",
                                                    "top_cross_split_full_resolution_pearson_max",
                                                    "top_same_group_full_resolution_pearson", "flagged_pairs")},
                     indent=2))


if __name__ == "__main__":
    main()
