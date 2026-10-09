# Label audit: ScienceDB directory labels versus original Figshare labels

Generated on 2026-10-08 with:

```powershell
python -B scripts/audit_labels.py --output-dir docs/data-inspection/label-audit-2026-10-08
```

All 3,064 ScienceDB PNG images were matched to the original Figshare v8 MAT records by pixel content (see `../patient-mapping-2026-09-29/`). This audit summarises how the ScienceDB directory labels relate to the original author labels that the project uses for training.

## Findings

1. **1,768 of 3,064 directory labels disagree with the original labels** (`cross_table.csv`). The class totals agree (1,426 / 708 / 930), so the disagreement is visible only image by image.
2. **The directory labels form 12 contiguous runs in PNG `image_id` order** (`directory_label_runs.csv`), for example 918–1840 Pituitary and 1841–3029 Glioma. PNG numbering does not follow the original MAT numbering, so this is consistent with labels having been assigned by position rather than per image. This is a hypothesis about how the error arose, not an established cause.
3. **The tumour masks are anatomically consistent with the original labels** (`anatomical_check.csv`). Pituitary tumours sit near the sella turcica, so their masks should lie near the image centre and be small. Only training and validation masks were opened; test masks were not.

| Directory label → original label | Images | Centroid within 0.12 of centre | Median mask area |
| --- | ---: | ---: | ---: |
| Pituitary → Pituitary | 620 | 0.75 | 0.0065 |
| Glioma → Pituitary | 190 | 0.91 | 0.0076 |
| Pituitary → Glioma | 111 | 0.28 | 0.0182 |
| Pituitary → Meningioma | 75 | 0.25 | 0.0181 |

Images that the original authors label as pituitary tumours behave like pituitary tumours even when the ScienceDB directory says otherwise, and the reverse also holds.

## Method notes

- Masks are binarised at intensity ≥ 128 on the original 512 × 512 grid. Fifteen masks contain intermediate grey values and are left unchanged on disk.
- "Centroid within 0.12 of centre" is the Euclidean distance from the mask centroid to the image centre, as a fraction of the image width.
- This is a descriptive plausibility check of label provenance. It does not re-diagnose any image.
