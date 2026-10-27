# Test results: one-time finalize stage (2026-10-11)

Command (git `7c133e6`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`), run once after the prepare stage:

```powershell
python -B analyze.py --formal-config configs/formal_experiment_pid_numeric.json --analysis-name pid-numeric-v2-final --stage finalize
```

It ran from 11:03 to 11:05, evaluated the 12 selected checkpoints on all 421 test images (31 assumed patient groups) under clean input and nine corruptions, and wrote `results/analysis/pid-numeric-v2-final/test.json`, `test_predictions.csv`, severity curves and confusion matrices. The test set was not used for any earlier decision, and the analysis refuses to run again once `test.json` exists.

## Clean test performance (mean and sample SD over training seeds 0, 1, 2)

| Model | Macro-F1 | Accuracy | Macro AUROC | Recall G / M / P |
| --- | ---: | ---: | ---: | --- |
| ResNet-50 baseline | 0.898 (0.003) | 0.908 (0.003) | 0.978 (0.002) | 0.903 / 0.908 / 0.917 |
| + SE | 0.901 (0.009) | 0.909 (0.008) | 0.982 (0.003) | 0.893 / 0.942 / 0.917 |
| + PSANet adaptation | 0.891 (0.007) | 0.900 (0.007) | 0.977 (0.009) | 0.885 / 0.917 / 0.917 |
| + CBAM | 0.907 (0.015) | 0.918 (0.015) | 0.981 (0.001) | 0.915 / 0.904 / 0.934 |

G, M, P: glioma, meningioma, pituitary tumour. Each run misclassifies 29–45 of the 421 test images.

## Pre-registered comparison (primary result)

Paired by training seed, Macro-F1 difference from the baseline; 95 % percentile interval from 10,000 class-stratified bootstrap resamples of the 31 test patient groups.

| Model | Seed 0 | Seed 1 | Seed 2 | Mean difference | 95 % interval | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| + SE | +0.009 | +0.004 | −0.003 | +0.003 | −0.012 to +0.023 | No consistent difference |
| + PSANet adaptation | +0.000 | −0.016 | −0.004 | −0.007 | −0.025 to +0.013 | No consistent difference |
| + CBAM | +0.023 | +0.008 | −0.003 | +0.009 | −0.004 to +0.027 | No consistent difference |

Under the rule fixed before training, none of the terminal attention modules gave a consistent benefit or harm. The intervals are wide: with 31 test patient groups, differences of about ±0.02 Macro-F1 cannot be excluded or confirmed. CBAM had the highest mean and the narrowest lower bound, but one of its three seeds was below the baseline. The three comparisons were not adjusted for multiplicity.

## Robustness on test (Macro-F1, mean (SD) over seeds)

| Condition | Baseline | + SE | + PSANet | + CBAM |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.898 (0.003) | 0.901 (0.009) | 0.891 (0.007) | 0.907 (0.015) |
| Rician-like noise, mild | 0.750 (0.090) | 0.818 (0.123) | 0.834 (0.056) | 0.844 (0.028) |
| Rician-like noise, moderate | 0.526 (0.152) | 0.573 (0.245) | 0.548 (0.205) | 0.687 (0.158) |
| Rician-like noise, severe | 0.342 (0.130) | 0.158 (0.049) | 0.242 (0.013) | 0.327 (0.153) |
| Gaussian blur, mild | 0.899 (0.014) | 0.909 (0.022) | 0.894 (0.008) | 0.910 (0.009) |
| Gaussian blur, moderate | 0.892 (0.023) | 0.885 (0.054) | 0.902 (0.018) | 0.881 (0.001) |
| Gaussian blur, severe | 0.807 (0.050) | 0.757 (0.135) | 0.820 (0.046) | 0.820 (0.023) |
| Intensity scaling, mild | 0.896 (0.004) | 0.901 (0.008) | 0.891 (0.012) | 0.906 (0.013) |
| Intensity scaling, moderate | 0.896 (0.009) | 0.899 (0.010) | 0.887 (0.016) | 0.905 (0.007) |
| Intensity scaling, severe | 0.898 (0.008) | 0.896 (0.015) | 0.884 (0.011) | 0.902 (0.014) |

As on validation, intensity scaling has almost no effect, blur reduces performance mainly at the strongest level, and Rician-like noise causes large losses with very large seed-to-seed variation. The robustness comparison was planned as descriptive; with these standard deviations, no model can be called more robust.

## Post-hoc, descriptive: where the errors are

These analyses were done after seeing the test results, use only the saved predictions, and are hypothesis-generating.

- **Errors concentrate in a few patients.** Of 460 clean test errors summed over the 12 runs, five patient groups account for 91 %: two glioma groups whose images are misclassified 65 % and 34 % of the time, one pituitary group (52 %), one further glioma group (31 %) and one meningioma group (49 %). 16 of the 31 groups have no error in any run. The most common error is glioma predicted as meningioma (59 % of errors), followed by pituitary tumour predicted as glioma (24 %).
- **The glioma errors follow the patient-identifier format.** Glioma is the only class with two identifier formats (suffixed `MR…` codes and plain numeric codes). Pooled over the 12 runs, test gliomas with `MR` codes were misclassified in 0.4 % of cases (8 / 1,968) and those with numeric codes in 36.5 % (263 / 720), with the same pattern for every model (30.6–42.2 %). The validation set contained only 4 numeric-code glioma images from one group, compared with 60 images from four groups in the test set and 141 images from 17 groups in training, so validation could not reveal this.
- **Baseline, SE and CBAM only** (the nine runs discussed in the final report; added 2026-10-27 from the same saved predictions). The nine runs make 334 clean test errors. The same five groups account for 92 % of them, with 30–61 % of each group's images misclassified, and 19 of the 31 groups have no error in any run. Glioma predicted as meningioma makes up 58 % of the errors and pituitary tumour predicted as glioma 24 %. Numeric-code gliomas are misclassified in 34.6 % of cases (187 / 540; 30.6–37.8 % per model) and `MR`-code gliomas in 0.5 % (7 / 1,476).
- **Interpretation.** Meningioma and pituitary tumours carry numeric codes only. If the two code formats reflect different acquisition sources or periods, the models may partly rely on source-related image characteristics, a form of shortcut learning, which would make numeric-code gliomas resemble the other classes. This risk had already been noted on 8 October, before the test evaluation: MR-prefixed codes occur only in glioma and might reflect acquisition differences. The analysis by code format itself was done after the test results. Four test groups cannot establish the explanation, and clinical differences between the two glioma subsets are an alternative explanation. Added 2026-10-20 (post hoc, baseline, SE and CBAM only): leaving out the 60 numeric-code glioma images raises clean test Macro-F1 from 0.898–0.907 to 0.950–0.956, and leaving out all five high-error groups to 0.989–0.990, the validation level. The numeric-code gliomas therefore explain about half of the gap between validation (about 0.98) and test (about 0.90), and the five groups nearly all of it.
