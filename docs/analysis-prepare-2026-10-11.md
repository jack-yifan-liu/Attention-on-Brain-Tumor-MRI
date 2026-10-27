# Validation analysis: prepare stage (2026-10-11)

Command (git `7c133e6`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`):

```powershell
python -B analyze.py --formal-config configs/formal_experiment_pid_numeric.json --analysis-name pid-numeric-v2-final --stage prepare
```

The run took 5 minutes, used the 12 formal best checkpoints and the 544 validation images, and loaded no test images. For every run, clean validation predictions reproduced the predictions saved during training. Outputs are in `results/analysis/pid-numeric-v2-final/` (not versioned). All values below are validation results from checkpoints that were selected on the same validation set, so they describe development behaviour and are not the study's comparison.

## Computational cost

| Model | Parameters | Counted forward FLOPs | GPU latency, batch 1, FP32 (median of run medians) |
| --- | ---: | ---: | ---: |
| Baseline | 23,514,179 | 8.174 G | 5.09 ms |
| + SE | 24,040,643 | 8.175 G | 5.21 ms |
| + PSANet adaptation | 36,806,723 | 9.481 G | 5.87 ms |
| + CBAM | 24,040,743 | 8.176 G | 5.29 ms |

FLOPs come from `torch.utils.flop_counter.FlopCounterMode` (two operations per multiply–accumulate) and cover convolutions and matrix products. Element-wise and normalisation operators are not counted and are listed per run in `prepare.json` (batch normalisation, ReLU, residual addition, pooling, and for PSANet also softmax, concatenation and index selection). Latency is the median of 100 synchronized forward passes after 20 warm-up passes on the RTX 5060 Laptop GPU, excluding preprocessing and data transfer.

## Robustness on validation (Macro-F1, mean of three seeds)

| Condition | Baseline | + SE | + PSANet | + CBAM |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.986 | 0.982 | 0.982 | 0.982 |
| Rician-like noise σ 0.02 / 0.05 / 0.10 | 0.824 / 0.544 / 0.327 | 0.877 / 0.564 / 0.165 | 0.892 / 0.609 / 0.238 | 0.904 / 0.704 / 0.326 |
| Gaussian blur σ 0.5 / 1.0 / 2.0 px | 0.970 / 0.917 / 0.869 | 0.967 / 0.916 / 0.800 | 0.977 / 0.936 / 0.854 | 0.964 / 0.892 / 0.839 |
| Intensity scaling 0.9–1.1 / 0.75–1.25 / 0.6–1.4 | 0.986 / 0.984 / 0.980 | 0.981 / 0.979 / 0.975 | 0.981 / 0.978 / 0.971 | 0.981 / 0.979 / 0.972 |

- Intensity scaling barely matters, and blur causes moderate losses.
- Rician-like noise causes large losses for every model. The corruption preview shows the expected effect: the noise turns the black background into speckle, a pattern absent from the clean training images. Under severe noise, several runs score Macro-F1 ≈ 0.227, the value obtained when every image is assigned to one class.
- Seed-to-seed variation under noise is large (for example, moderate noise: baseline 0.387–0.710, CBAM 0.583–0.828, PSANet 0.463–0.824), so differences between model means are not reliable at this stage.

## Grad-CAM / tumour-mask overlap (all 544 validation images, layer4, true class)

| Model | Energy inside mask, mean per run | Mean of three runs | Enrichment over mask area |
| --- | --- | ---: | ---: |
| Baseline | 0.070, 0.077, 0.087 | 0.078 | 4.8× |
| + SE | 0.064, 0.082, 0.091 | 0.079 | 4.7× |
| + PSANet adaptation | 0.063, 0.066, 0.065 | 0.065 | 3.7× |
| + CBAM | 0.072, 0.102, 0.054 | 0.076 | 4.6× |

No heatmap was all zero. Tumour masks are small (median area 0.7–1.9 % of the image depending on class, from the label audit), so a mean energy share of 0.065–0.079 corresponds to roughly four to five times more weight on the annotated tumour than a uniform map would give. PSANet maps are more diffuse, consistent with its randomly initialized fusion layer. These measure overlap between explanations and annotations, not localisation accuracy.

## Statistical comparison: validation code check

The pre-registered paired, class-stratified patient-group bootstrap ran on validation (31 groups: 10 / 12 / 9 by class; 10,000 replicates). Mean paired Macro-F1 differences from the baseline were −0.0038 (SE), −0.0043 (PSANet) and −0.0033 (CBAM), with 95 % intervals of about ±0.01, and every comparison was classified as no consistent difference. This only confirms that the procedure runs; it is not evidence, because validation selected the checkpoints.

## Grad-CAM example cases

Cases were chosen per seed by the fixed rule (first validation image in identifier order where all models are correct, all are incorrect, or the models disagree). No image of glioma or pituitary tumour was misclassified by all four models in some seeds, and those categories are recorded as missing. Sample identifiers keep the ScienceDB folder name, which is not the label: for example `Glioma/2218` is a meningioma under the original labels, so figure captions must use the original class.
