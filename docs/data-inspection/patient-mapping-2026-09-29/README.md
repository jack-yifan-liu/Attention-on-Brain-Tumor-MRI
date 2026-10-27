# Patient ID recovery and label conflict check

Date: 2026-09-29. This step produced mapping evidence only. It did not change the ScienceDB PNG files, the classes in the earlier manifest, `class_to_idx.json` or any split.

## Original source

- Original author Jun Cheng: [Figshare dataset, version 8](https://doi.org/10.6084/m9.figshare.1512427.v8).
- File names, download URLs and MD5 values were read from the [public API metadata](https://api.figshare.com/v2/articles/1512427).
- The author's [README 2024.txt](https://ndownloader.figshare.com/files/51340418) states `cjdata.PID: patient ID`.
- Original labels: 1 = Meningioma, 2 = Glioma, 3 = Pituitary tumor. These are not the project's 0 / 1 / 2 indices.
- The four ZIP archives (879,493,326 bytes in total) and the README are stored in `data/downloads/figshare-original/`, which is not versioned. All four MD5 values match Figshare's `computed_md5` (see `summary.json`).
- Reading the MAT files requires h5py (3.15.1, listed in `requirements.txt`).

## Commands

From the repository root:

```powershell
python -B scripts/recover_patient_mapping.py --download
python -B scripts/recover_patient_mapping.py --output-dir docs/data-inspection/patient-mapping-2026-09-29
```

Both exited with code 0. The script refuses to overwrite an existing output folder. Rerunning into a new folder on 2026-10-08 reproduced `patient_mapping.csv` byte for byte.

## How images were matched

The numeric PNG file names do not follow the original MAT file numbering, so the two cannot be joined by number. Each original MRI and tumor mask was matched to a PNG by pixel content:

- The HDF5 arrays were transposed to MATLAB image orientation. Every match uses this orientation, with no further rotation or flip.
- 1,555 MRIs are pixel-identical after per-image min–max scaling to uint8.
- Another 1,494 images at 512 × 512 have identical masks, and their MRIs differ by at most one gray level per pixel (rounding differences).
- 15 original images are 256 × 256, all of them pituitary tumors. After bilinear 2× upsampling with half-pixel coordinates and edge replication, their masks are identical, and their MRIs differ by at most 2 gray levels per pixel (observed maximum 1.4375). These 15 images account for the 15 masks with intermediate gray values. This explains where those gray values come from, but it does not confirm the publisher's conversion code.
- Every match had to be unique. All 3,064 PNG images map one-to-one to the 3,064 MAT records, with no unmatched or ambiguous cases, and all 3,064 masks agree.
- The correspondence rests on this content check. PNG file names were never treated as patient numbers.

## Results and how to use the CSV

`patient_mapping.csv` has one row per PNG image, with `sample_id`, `patient_id`, `source_mat`, the directory class and index, the original label (`original_label`, `original_class`), `class_matches`, the MRI and mask paths, and the matching evidence.

- `class_name` / `class_idx` keep the **directory** values from the earlier manifest and must not be used as training labels.
- `original_class` is the author's label for that image.
- `patient_id` is the original `cjdata.PID` string, with letters and suffixes kept; identifiers were not merged at this step.

| Original class | Images | Distinct PIDs |
| --- | ---: | ---: |
| Glioma | 1,426 | 89 |
| Meningioma | 708 | 82 |
| Pituitary tumor | 930 | 62 |
| Total | 3,064 | 233 |

Every original PID has a single class. The ScienceDB directory class matches the original class for only 1,296 images; 1,768 disagree:

| ScienceDB directory | Original Glioma | Original Meningioma | Original Pituitary tumor |
| --- | ---: | ---: | ---: |
| Glioma | 589 | 617 | 220 |
| Meningioma | 705 | 0 | 3 |
| Pituitary tumor | 132 | 91 | 707 |

Examples:

| ScienceDB sample | Original MAT | PID | Directory class | Original class |
| --- | --- | --- | --- | --- |
| Glioma/767 | 1.mat | 100360 | Glioma | Meningioma |
| Glioma/768 | 10.mat | 101016 | Glioma | Meningioma |
| Meningioma/1 | 2299.mat | MR049453B | Meningioma | Glioma |

The class totals agree with the original release, but the per-image labels do not, and swapping two class names would not fix them. The step in publication or conversion where the error arose has not been verified; [`../label-audit-2026-10-08/`](../label-audit-2026-10-08/README.md) describes a pattern consistent with position-based assignment. When this check was made, no split, checkpoint or training result existed, so no model result had to be withdrawn.

## Additional checks and what happened next

An independent check of the CSV confirmed 3,064 unique `sample_id` and `source_mat` values, 233 PIDs, one class per original PID, and `mask_exact = True` for every row. Image counts, patient counts and error bounds match `summary.json`.

The project then adopted the original author labels and saved a patient-grouped split. The current split groups identifiers that share a numeric part; see [`data/metadata_pid_numeric/README.md`](../../../data/metadata_pid_numeric/README.md). Patient grouping addresses the known risk of one patient appearing in several splits, but it does not rule out every other form of leakage or establish external generalization.
