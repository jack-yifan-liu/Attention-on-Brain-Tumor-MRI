# Near-duplicate screening across train, validation and test

Generated on 2026-10-08 with:

```powershell
python -B scripts/check_near_duplicates.py --output-dir docs/data-inspection/near-duplicates-2026-10-08
```

The patient-group split already prevents the same original PID or numeric PID group from crossing splits, and no pixel-identical MRI pairs exist. This screen looks for a remaining risk: the same scan, or a near copy, appearing under different identifiers in different splits.

## Method

- Every MRI (all 3,064, including test) is reduced to a 32 × 32 area-averaged thumbnail and compared with every other image by cosine similarity of the mean-centred pixels. No model and no labels are used.
- The 30 most similar cross-split pairs are re-scored with full-resolution Pearson correlation. A pair at or above 0.99 would be flagged for visual review.
- As a reference, the most similar pairs within one assumed patient group (adjacent slices of the same patient, same split) are scored in the same way.

## Result

- **No pair was flagged.** The highest full-resolution correlation between images in different splits is 0.932. The most similar same-patient slice pairs score 0.92–0.95, so no cross-split pair is more alike than neighbouring slices of one patient.
- Cross-split thumbnail similarity (median 0.536, 99.9th percentile 0.891) matches the distribution for different patients within the same split (0.533, 0.882).
- `top_cross_split_pairs.png` shows the eight most similar cross-split pairs. They are different patients with different tumours at similar axial levels.

This is a screening aid. It cannot prove that two images come from different patients.
