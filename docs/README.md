# Documentation

Records of the data checks, the PSANet design and the results, grouped by stage. The date in each name is when the record was made. Per-run outputs that the records mention (`results/`, `checkpoints/`) are not versioned.

## Data checks

| Record | Contents | Evidence for |
| --- | --- | --- |
| [`data-inspection/2026-09-29/`](data-inspection/2026-09-29/README.md) | Archive integrity, image and mask counts, pairing and size checks, masks with gray values, the first manifest and the class mapping | Intact files; every MRI has a mask of the same size |
| [`data-inspection/patient-mapping-2026-09-29/`](data-inspection/patient-mapping-2026-09-29/README.md) | Match of every PNG to its original record by pixel content, with patient identifier and original label | Original labels and patient identifiers for every image |
| [`data-inspection/label-audit-2026-10-08/`](data-inspection/label-audit-2026-10-08/README.md) | Directory versus original labels, label runs in file order, anatomical check of the masks | Why training uses the original labels |
| [`data-inspection/near-duplicates-2026-10-08/`](data-inspection/near-duplicates-2026-10-08/README.md) | Similarity screen of all images across train, validation and test | No near-copies found across splits (a screen, not proof) |

## Model and pipeline

| Record | Contents | Evidence for |
| --- | --- | --- |
| [`psanet-design.md`](psanet-design.md) | Design of the PSANet adaptation, its sources and its differences from the original network | The design choices in `src/models/psanet.py` |
| [`preflight-2026-10-08.json`](preflight-2026-10-08.json) | Four-model preflight of the formal pipeline on a small subset, with environment and random-control settings | Reruns reproduce predictions exactly on the same hardware |

## Results

| Record | Contents | Evidence for |
| --- | --- | --- |
| [`formal-training-2026-10-10.md`](formal-training-2026-10-10.md) | The 12 formal runs: provenance checks and validation scores of the selected checkpoints | Validation Macro-F1 about 0.98 for every model |
| [`analysis-prepare-2026-10-11.md`](analysis-prepare-2026-10-11.md) | Validation stage: computational cost, robustness, Grad-CAM cases and their overlap with tumor masks, check of the statistics code | Parameter, FLOP and latency comparison |
| [`test-results-2026-10-11.md`](test-results-2026-10-11.md) | One-time test evaluation: clean metrics, comparison with the baseline, robustness, post-hoc error analysis | Main results, including PSANet |
| [`attention-weights-2026-10-20.md`](attention-weights-2026-10-20.md) | SE channel weights and CBAM spatial weights on validation images (post hoc) | SE acts mostly as a fixed rescaling; CBAM weights depend on the image |

The CSV and JSON files in `data-inspection/` are unedited script outputs. `data-inspection/2026-09-29/manifest.csv` and `class_to_idx.json` and `data-inspection/patient-mapping-2026-09-29/patient_mapping.csv` are inputs of `scripts/prepare_data.py`, which checks the manifest and the mapping by SHA-256, so keep these files byte-identical.
