# Current split: conservative grouping by the numeric part of the PID (v2)

Adopted on 2026-10-02: original patient identifiers that share the same numeric part are treated as one patient group. **This is a conservative assumption that the original authors have not confirmed. 209 is the number of groups, not a re-verified number of patients.**

- `patient_id`: the original, complete `cjdata.PID` string, unchanged.
- `patient_group_id`: the single continuous run of digits in the PID, converted to an integer and back to a string, ignoring letters and leading zeros. For example, `MR049453B` and `MR049453` both become `49453`. PIDs with no digits or more than one run of digits are rejected.
- The original author labels, class order, all 3,064 MRI images and masks are kept. No sample is excluded.
- Within each class, group IDs are sorted as strings, shuffled with NumPy PCG64 (seed 42), and allocated 70 / 15 / 15 % to train / validation / test with the largest-remainder method; ties go to train, then validation, then test.
- No group spans two classes. The 233 original PIDs form 209 groups (Glioma 65, Meningioma 82, Pituitary tumor 62).

| Split | Glioma images / groups | Meningioma images / groups | Pituitary images / groups | Total images / groups | Original PIDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 922 / 45 | 519 / 58 | 658 / 44 | 2,099 / 147 | 160 |
| validation | 280 / 10 | 109 / 12 | 155 / 9 | 544 / 31 | 36 |
| test | 224 / 10 | 80 / 12 | 117 / 9 | 421 / 31 | 37 |

No original PID, numeric group or sample appears in more than one split, and no two MRIs are pixel-identical. A screen for near-duplicate images across splits found no pair needing action ([`docs/data-inspection/near-duplicates-2026-10-08/`](../../docs/data-inspection/near-duplicates-2026-10-08/README.md)). None of this guarantees that every other form of leakage is excluded.

## Relation to earlier development runs

An earlier split grouped exact PID strings only. It is kept locally and is not part of this repository. `split_changes.csv` records the old and new split of every image; 1,469 images changed split. Of the 421 current test images, 317 were in the old training set, 59 in the old validation set and 45 in the old test set. The current test set therefore cannot be described as never having been touched during development, although no model has been evaluated on it.

Two baseline runs, the four-model seed-3 preview and the smoke tests all used the old split. Their models must not be evaluated on the current test set or pooled with the formal results. Each of the 12 formal runs starts from the shared ImageNet V2 weights.

## How the split was generated

From the repository root:

```powershell
python -B scripts/prepare_data.py --mapping-dir docs/data-inspection/patient-mapping-2026-09-29 --inspection-dir docs/data-inspection/2026-09-29 --output-dir data/metadata_pid_numeric --ratios 0.70 0.15 0.15 --seed 42 --patient-grouping numeric --previous-split data/metadata/data_split.csv
```

The generation step reads every MRI and mask only for integrity, fingerprint and duplicate checks; no model is involved. `--previous-split` points to the old exact-PID split and is used only to write `split_changes.csv`. To reproduce the split, omit that argument and write to a new folder: `data_split.csv` and `manifest.csv` should then have the SHA-256 values recorded in `summary.json` here. This was confirmed on 2026-10-08.

The data path configuration is `configs/paths_pid_numeric.json` and the formal protocol is `configs/formal_experiment_pid_numeric.json`.
