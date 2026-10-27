# ScienceDB data inspection (P1)

Inspection date: 2026-09-29. The downloaded archive was kept and extracted into a new, separate folder. Raw MRI images and masks were not modified. No split was created, no model was trained, and no patient mapping or leakage audit was part of this step.

## Source and file integrity

- Source: [ScienceDB Brain Tumor Dataset](https://www.scidb.cn/en/detail?dataSetId=faa44e0a12da4c11aeee91cc3c8ac11e), DOI `10.57760/sciencedb.06290`.
- Download link: `https://china.scidb.cn/download?fileId=636a0c5a72a25737e4ca6e51`.
- Archive kept at `data/downloads/data.zip`, 284,527,903 bytes.
- MD5 `a5bd11a085d31d8cc1fac9c8c9485a2b` (matches the project's earlier record); SHA-256 `84503fb454f614a8aca18ddfced8fcf23112901bb296f63bfa49057627cbbf85`.
- The ZIP holds 6,132 entries (6,128 files, 4 folders), 284,169,364 uncompressed bytes; `ZipFile.testzip()` returned None.
- Extracted with the Python standard-library `zipfile`, refusing path traversal, symbolic links, duplicate paths and existing targets.
- Extracted into `data/raw/sciencedb`. The archive contains its own `data/` folder, so the data root is `data/raw/sciencedb/data`.
- After extraction, the size and CRC of all 6,128 files matched the ZIP, and the archive MD5 was unchanged.
- An MD5 match shows the archive is the same as before. It says nothing about provenance or patient independence.

## Command

From the repository root:

```powershell
python -B scripts/inspect_dataset.py --data-root data/raw/sciencedb/data --output-dir docs/data-inspection/2026-09-29
```

Exit code 0. The script only reads data and refuses to overwrite an existing output folder.

## Results

| Class (directory) | Index | MRI | Mask |
| --- | ---: | ---: | ---: |
| Glioma | 0 | 1,426 | 1,426 |
| Meningioma | 1 | 708 | 708 |
| Pituitary tumor | 2 | 930 | 930 |
| Total | — | 3,064 | 3,064 |

- Every MRI and mask is 512 × 512, Pillow mode `L`, NumPy uint8.
- Every file decodes completely. There are no missing pairs, orphan masks or size mismatches between pairs.
- 3,049 masks contain only values in {0, 255}; 15 contain intermediate gray values (`nonbinary_masks.json`). Masks were not binarized.
- No sample was filtered or excluded: `issues.csv` and `exclusions.csv` contain only headers.
- `manifest.csv` has 3,064 rows, one per MRI; the mask path is stored as a column.
- `sample_id` is `class/filename-stem` and is unique for all 3,064 images. The numeric image ID is not a patient ID.
- Manifest paths are relative to the data root. `class_to_idx.json` keeps the project's class order.
- The union of MRI and mask paths in the manifest covers exactly the 6,128 PNG files.
- Manifest SHA-256: `a5be81f70dce8f7e02e03ac46f6cd334c1c9b86e74943038483bd6be415bbdb5`.

## Limits

A visual check of one MRI and mask overlay per class showed no obvious misalignment; it was not an image-by-image annotation review or a quantitative localization check. At this stage the class labels came from the archive's folder names and were not checked against the original source. Patient identity cannot be established from the PNG files or their names. This inspection does not show that the data are free of leakage. No resizing, cropping, augmentation or splitting was done here.

## Follow-up on the same day: original labels

After patient identifiers were recovered from the original Figshare release, all 3,064 images were mapped to 233 original patient IDs, and **1,768 directory labels were found to disagree with the original author labels**. See [`../patient-mapping-2026-09-29/`](../patient-mapping-2026-09-29/README.md) and [`../label-audit-2026-10-08/`](../label-audit-2026-10-08/README.md). The integrity, pairing and size checks above remain valid, but the directory labels in this manifest are not used for training. The raw PNG files and this manifest were left unchanged.
