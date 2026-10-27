# Formal training runs (2026-10-10)

All 12 formal runs of `configs/formal_experiment_pid_numeric.json` (4 models × training seeds 0, 1, 2; 30 epochs each) completed on 2026-10-10 between 14:13 and 16:49 (2 h 36 min) on an NVIDIA RTX 5060 Laptop GPU. Runs were executed seed by seed with `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`; before launch, augmented inputs, initial weights, logits and weights after three training steps were confirmed bit-identical to the default 24-thread setting, so the thread limit does not affect results.

## Provenance checks (read from each run's `run.json` and `summary.json`)

- Every run records git head `7c133e6` with a clean working tree, the frozen config SHA-256 `e8cef959…`, identical metadata, sample-list and pretrained-weight hashes, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms on and TF32 off.
- Every run trained for 30 epochs on 2,099 training images and was validated on 544 validation images; `test_images_loaded` is 0 in every record.
- Reloading each best checkpoint reproduced its validation predictions exactly (maximum probability difference 0, identical labels, optimiser state restored).
- The training log contains no errors.

## Validation results of the selected checkpoints

These are validation scores at the epoch chosen by validation Macro-F1, so they are optimistically biased and are not the study's comparison. The pre-registered comparison uses the test set with a patient-group bootstrap.

| Model | Seed 0 | Seed 1 | Seed 2 | Mean (SD) | Best epochs | Epoch-30 mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| ResNet-50 baseline | 0.9828 | 0.9873 | 0.9871 | 0.9857 (0.0025) | 25, 17, 19 | 0.9783 |
| + SE | 0.9843 | 0.9827 | 0.9788 | 0.9819 (0.0028) | 29, 21, 28 | 0.9736 |
| + PSANet adaptation | 0.9870 | 0.9804 | 0.9771 | 0.9815 (0.0051) | 29, 19, 25 | 0.9671 |
| + CBAM | 0.9816 | 0.9814 | 0.9842 | 0.9824 (0.0015) | 30, 25, 12 | 0.9794 |

Validation Macro-F1; SD is the sample standard deviation over three seeds. Validation accuracy means: baseline 0.9877, SE 0.9847, PSANet 0.9841, CBAM 0.9859.

Observations, descriptive only:

- Each selected checkpoint misclassifies 6–11 of the 544 validation images. Most errors are meningioma predicted as glioma or glioma predicted as meningioma; pituitary tumors are almost always correct.
- Paired by seed, the attention models differ from the baseline by +0.004 to −0.010 Macro-F1, that is, by a few images. This is within the epoch-to-epoch variation seen inside single runs.
- The selected epoch scored on average 0.003–0.014 higher than the final epoch, which illustrates the optimism of best-of-30 selection.
