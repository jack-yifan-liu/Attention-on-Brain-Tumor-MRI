# Learned attention weights on validation (post hoc, 2026-10-20)

Command (working tree at git `7c133e6` plus the new, not yet committed script; `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`):

```powershell
python -B scripts/attention_weights.py
```

The script loads the best checkpoints of the SE and CBAM models for training seeds 0, 1 and 2, runs them on the 544 clean validation images with the same GPU settings and batching as `analyze.py`, and records the weights that the terminal attention module applies to each image. It opens no test image and changes no saved result. For all six runs the recomputed validation probabilities equal the saved ones exactly (maximum difference 0). This analysis was added after the test results were known and is descriptive only.

Outputs: `results/figures/attention/validation_attention.json` (summaries, provenance, selection rule) and `validation_attention.npz` (per-channel summaries, all CBAM spatial maps); figure `results/figures/report/fig13_attention_weights.png`.

## Results (mean (SD) over three seeds)

| Measure | SE channel weights | CBAM channel weights | CBAM spatial weights |
| --- | ---: | ---: | ---: |
| Mean weight | 0.745 (0.011) | 0.633 (0.017) | 0.498 (0.001) |
| Weights above 0.9 | 55 % (4) | 59 % (2) | — |
| Weights below 0.1 | 9 % (2) | 32 % (2) | — |
| Variation explained by the channel alone | 85 % (5) | 60 % (1) | — |
| Median SD of one channel's weight across images | 0.054 (0.011) | 0.189 (0.020) | — |
| Image-to-image variation explained by the true class | 87 % (2) | 89 % (2) | — |
| Lowest correlation between class-average weight vectors | 0.72 (0.14) | 0.35 (0.03) | — |
| Median range within one 7 × 7 map | — | — | 0.80 (0.03) |
| Median enrichment inside the tumor mask | — | — | 1.42 (0.25) |
| Highest-weight cell overlaps the tumor mask | — | — | 50 % (5); 8 % by chance |

Per seed, the share of variation explained by the channel is 79–89 % for SE and 59–61 % for CBAM, and the pointing rate of the CBAM spatial weights is 46–56 %.

## Interpretation (descriptive)

- **SE behaves mostly like a fixed rescaling.** Most of the variation in its weights comes from which channel it is, not from the image, and over half of the weights are above 0.9. Because the module is followed only by global average pooling and the linear head, a weight that is the same for every image is equivalent to rescaling the head's weights, which the baseline can learn as well. This matches the observation of Hu et al. (Squeeze-and-Excitation Networks) that SE blocks in the last stage of SE-ResNet tend to saturate and can be removed with little loss.
- **CBAM's weights depend on the image.** About 40 % of the variation in its channel weights is image-specific, a third of the weights are below 0.1, and the class-average weight patterns differ between classes, so the module acts partly as a class-dependent channel selector. Its spatial weights are high over the brain and low in the background, and the highest-weight cell overlaps the tumor in about half of the validation images, six times the rate expected by chance, although the module never received tumor locations.
- **Learned attention did not translate into a consistent gain.** The pre-specified test comparison found no consistent difference from the baseline for either module (`docs/test-results-2026-10-11.md`). On validation all models were already near ceiling (Macro-F1 about 0.98), and on test the models agree closely outside five difficult patient groups.

## Definitions and limitations

- "Variation explained by the channel alone" is the between-channel share of the variance of all weights (images × 2,048 channels); 100 % would mean every image receives identical weights. "Explained by the true class" is the between-class share of the remaining, image-specific variation, pooled over channels.
- Enrichment is the mean spatial weight inside the tumor mask divided by the mean over the image, after the same bilinear upsampling to the 512 grid as Grad-CAM. The pointing test asks whether the highest of the 49 weights lies in a cell that contains any tumor-mask pixel; the chance rate is the share of such cells, averaged over images.
- The 7 × 7 grid is coarse: one cell covers about 73 × 73 pixels of the 512 × 512 image, larger than many tumors.
- Validation images also selected the checkpoints. The figure shows seed 0 only, on one rule-selected image per class (correct for all three CBAM seeds, tumor area closest to the class median).
