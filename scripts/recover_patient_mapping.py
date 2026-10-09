"""Fetch original Figshare MAT archives and verify patient IDs against local PNGs."""

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

API_URL = "https://api.figshare.com/v2/articles/1512427/versions/8"
LABEL_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary tumor"}


def file_hash(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def fetch_sources(folder):
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "figshare-metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(API_URL, timeout=60) as response:
            metadata = json.load(response)
        with metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
    files = [f for f in metadata["files"] if f["name"].endswith(".zip") or f["name"].startswith("README")]
    for item in files:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unexpected remote filename")
        path = folder / item["name"]
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {item['name']} ({item['size']} bytes)", flush=True)
            with urllib.request.urlopen(item["download_url"], timeout=60) as response:
                with partial.open("xb") as stream:
                    total, last = 0, 0
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        total += len(chunk)
                        if total - last >= 32 * 1024 * 1024:
                            print(f"  {total // (1024 * 1024)} MiB", flush=True)
                            last = total
            if partial.stat().st_size != item["size"] or file_hash(partial, "md5") != item["computed_md5"]:
                raise ValueError(f"Incomplete or corrupt download: {partial}")
            partial.rename(path)
        if path.stat().st_size != item["size"] or file_hash(path, "md5") != item["computed_md5"]:
            raise ValueError(f"Source hash mismatch: {path}")
        print(f"Verified {path.name}", flush=True)
    return metadata


def read_mat(raw):
    with io.BytesIO(raw) as stream:
        with h5py.File(stream, "r") as mat:
            group = mat["cjdata"]
            codes = group["PID"][()].reshape(-1)
            pid = "".join(chr(int(c)) for c in codes)
            label = int(group["label"][()].item())
            # MATLAB stores arrays column-major, so 2-D arrays read through h5py must be transposed.
            image = group["image"][()].T
            mask = group["tumorMask"][()].T
    if not pid or "\x00" in pid:
        raise ValueError("Empty or malformed PID")
    return pid, label, image, mask


def double_bilinear(array):
    """Half-pixel bilinear sampling, 2x scale, border replication, no quantization."""
    height, width = array.shape
    x = np.arange(width * 2) / 2 - 0.25
    y = np.arange(height * 2) / 2 - 0.25
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    horizontal = array[:, np.clip(x0, 0, width-1)] * (1-wx) + array[:, np.clip(x0+1, 0, width-1)] * wx
    return horizontal[np.clip(y0, 0, height-1)] * (1-wy[:, None]) + horizontal[np.clip(y0+1, 0, height-1)] * wy[:, None]


def inspect_mapping(folder, data_root, manifest, output):
    if output.exists():
        raise ValueError("Output directory exists; use a new one")
    if output.resolve().is_relative_to(data_root.resolve()):
        raise ValueError("Outputs must be outside the raw PNG directory")
    metadata = json.loads((folder / "figshare-metadata.json").read_text(encoding="utf-8"))
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # PNG numbers are not MAT numbers: link records by pixel content, never by number.
    png_index, mask_index = defaultdict(list), defaultdict(list)
    for row in rows:
        with Image.open(data_root / row["mri_path"]) as im:
            array = np.array(im)
        png_index[hashlib.sha256(array.tobytes()).hexdigest()].append(row)
        with Image.open(data_root / row["mask_path"]) as im:
            array = np.array(im)
        mask_index[hashlib.sha256(array.tobytes()).hexdigest()].append(row)
    mapped, seen, unmatched_sources = [], set(), []
    for item in metadata["files"]:
        if not item["name"].endswith(".zip"):
            continue
        archive_path = folder / item["name"]
        if file_hash(archive_path, "md5") != item["computed_md5"]:
            raise ValueError(f"Source archive mismatch: {archive_path}")
        print(f"Reading {archive_path.name}", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if Path(member.filename).suffix != ".mat":
                    raise ValueError(f"Unexpected archive member: {member.filename}")
                image_id = Path(member.filename).stem
                if image_id in seen:
                    raise ValueError(f"Duplicate MAT image ID: {image_id}")
                seen.add(image_id)
                pid, label, original, mask = read_mat(archive.read(member))
                scaled = (original.astype(np.float64) - original.min()) * (255. / (original.max() - original.min()))
                candidates = {"minmax_floor": np.clip(scaled, 0, 255).astype(np.uint8),
                              "minmax_round": np.clip(np.floor(scaled + 0.5), 0, 255).astype(np.uint8)}
                matches = {}
                for transform, array in candidates.items():
                    for transpose in (False, True):
                        oriented = array.T if transpose else array
                        key = hashlib.sha256(oriented.tobytes()).hexdigest()
                        for row in png_index.get(key, []):
                            matches[row['sample_id']] = (row, transform, transpose, 0, False)
                # Fallback: locate the PNG by an identical mask; the MRI must stay within the per-pixel quantisation bound.
                if not matches:
                    for transpose in (False, True):
                        oriented_mask = mask.T if transpose else mask
                        expected_mask = oriented_mask.astype(np.float64) * 255
                        resized = original.shape == (256, 256)
                        if resized:
                            expected_mask = double_bilinear(expected_mask)
                        expected_mask = np.clip(np.rint(expected_mask), 0, 255).astype(np.uint8)
                        for row in mask_index.get(hashlib.sha256(expected_mask.tobytes()).hexdigest(), []):
                            with Image.open(data_root / row['mri_path']) as im:
                                png = np.array(im)
                            for transform, candidate in candidates.items():
                                candidate = candidate.T if transpose else candidate
                                if resized:
                                    candidate = double_bilinear(candidate)
                                delta = np.abs(png.astype(np.float64) - candidate)
                                max_error = float(delta.max())
                                limit = 2 if resized else 1
                                if max_error <= limit:
                                    matches[row['sample_id']] = (row, transform + ('_bilinear_2x' if resized else '') + '_quantization_tolerance', transpose, max_error, resized)
                if len(matches) != 1:
                    unmatched_sources.append({"source_mat": member.filename, "patient_id": pid,
                                              "original_label": label, "shape": list(original.shape),
                                              "matches_count": len(matches)})
                    continue
                row, transform, transpose, max_error, resized = next(iter(matches.values()))
                with Image.open(data_root / row["mask_path"]) as im:
                    png_mask = np.array(im)
                oriented_mask = mask.T if transpose else mask
                expected_mask = oriented_mask.astype(np.float64) * 255
                if resized:
                    expected_mask = double_bilinear(expected_mask)
                expected_mask = np.clip(np.rint(expected_mask), 0, 255).astype(np.uint8)
                mask_exact = np.array_equal(png_mask, expected_mask)
                mapped.append({"sample_id": row["sample_id"], "image_id": row["image_id"],
                               "patient_id": pid, "class_name": row["class_name"], "class_idx": row["class_idx"],
                               "original_label": label, "original_class": LABEL_NAMES[label],
                               "class_matches": row['class_name'] == LABEL_NAMES[label],
                               "mri_path": row["mri_path"], "mask_path": row["mask_path"],
                               "source_archive": item["name"], "source_mat": member.filename,
                               "mat_shape": str(original.shape), "image_exact": max_error == 0,
                               "mask_exact": mask_exact, "image_transform": transform,
                               "transpose_after_matlab_orientation": transpose,
                               "image_max_absolute_error": max_error, "resized_2x": resized})
    if len({r['sample_id'] for r in mapped}) != len(mapped):
        raise ValueError("Multiple source images map to the same PNG")
    mapped.sort(key=lambda row: int(row["image_id"]))
    patient_classes, original_patient_classes = defaultdict(set), defaultdict(set)
    for row in mapped:
        patient_classes[row['patient_id']].add(row['class_name'])
        original_patient_classes[row['patient_id']].add(row['original_class'])
    output.mkdir(parents=True)
    verified = len(mapped) == len(rows) == len(seen) and all(r['mask_exact'] for r in mapped)
    with (output / ("patient_mapping.csv" if verified else "patient_mapping_candidates.csv")).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapped[0]))
        writer.writeheader()
        writer.writerows(mapped)
    conflicts = {pid: sorted(classes) for pid, classes in patient_classes.items() if len(classes) > 1}
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "source_api": API_URL,
              "source_doi": metadata["doi"], "source_version": metadata["version"],
              "samples": len(mapped), "png_samples": len(rows), "mat_samples": len(seen),
              "patients": len(patient_classes), "class_matches_count": sum(r['class_matches'] for r in mapped),
              "class_cross_table": dict(Counter(f"{r['class_name']} -> {r['original_class']}" for r in mapped)),
              "unmatched_source_images": unmatched_sources,
              "unmatched_png_images": [r['sample_id'] for r in rows if r['sample_id'] not in {m['sample_id'] for m in mapped}],
              "current_folder_consistent_patient_counts": dict(Counter(next(iter(classes)) for classes in patient_classes.values() if len(classes)==1)),
              "original_patient_counts_by_class": dict(Counter(next(iter(classes)) for classes in original_patient_classes.values() if len(classes)==1)),
              "original_patient_class_conflicts": {pid: sorted(classes) for pid,classes in original_patient_classes.items() if len(classes)>1},
              "patient_class_conflicts": conflicts,
              "image_exact_count": sum(r["image_exact"] for r in mapped),
              "mask_exact_count": sum(r["mask_exact"] for r in mapped),
              "shape_counts": dict(Counter(r["mat_shape"] for r in mapped)),
              "image_transform_counts": dict(Counter(r["image_transform"] for r in mapped)),
              "image_verification": "1555 initial exact image matches; otherwise exact unique mask plus per-pixel MRI error <=1 (native 512) or <=2 (256 resized 2x)",
              "manifest_sha256": file_hash(manifest, "sha256"),
              "archives": [{"name": f["name"], "md5": f["computed_md5"]} for f in metadata["files"] if f["name"].endswith('.zip')],
              "mapping_status": "verified_image_correspondence_with_label_conflicts" if verified else "candidates_pending_pixel_verification"}
    if verified:
        report["mapping_sha256"] = file_hash(output / "patient_mapping.csv", "sha256")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key,value in report.items() if key not in ('unmatched_source_images','unmatched_png_images','patient_class_conflicts')}
    compact['unmatched_sources_count'] = len(unmatched_sources)
    compact['current_patient_class_conflicts_count'] = len(conflicts)
    print(json.dumps(compact, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/downloads/figshare-original"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/sciencedb/data"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/data-inspection/2026-09-29/manifest.csv"))
    args = parser.parse_args()
    if args.download:
        fetch_sources(args.source_dir)
    if args.output_dir:
        inspect_mapping(args.source_dir, args.data_root, args.manifest, args.output_dir)
    if not args.download and not args.output_dir:
        parser.error("Specify --download and/or --output-dir")


if __name__ == "__main__":
    main()
