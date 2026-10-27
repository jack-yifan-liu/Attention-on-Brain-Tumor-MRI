# Attention in brain tumor MRI classification

This repository compares an ImageNet-pretrained ResNet-50 with three attention-augmented variants for three-class brain tumor MRI classification (glioma, meningioma, pituitary tumor):

| Model | Added module (one, after the full `layer4`, before global average pooling) |
| --- | --- |
| `resnet50` | none (baseline) |
| `resnet50_se_terminal` | Squeeze-and-Excitation, reduction ratio 16 |
| `resnet50_psanet_terminal` | Point-wise Spatial Attention adapted for classification (collect + distribute branches, 4096→2048 fusion) |
| `resnet50_cbam_terminal` | CBAM, channel then spatial attention |

All four models share the data split, preprocessing, training protocol and evaluation. The terminal modules are classification adaptations, not reproductions of SE-ResNet, CBAM-ResNet or the PSANet segmentation network.

**Status (2026-10-27):** the 12 formal training runs (4 models × seeds 0, 1, 2) finished on 2026-10-10, and the validation analysis and the one-time test evaluation on 2026-10-11. The post-hoc analyses and the report and presentation figures were completed afterwards. Results are summarized [below](#results).

## Data

- Images: ScienceDB Brain Tumor Dataset, DOI [10.57760/sciencedb.06290](https://doi.org/10.57760/sciencedb.06290), 3,064 contrast-enhanced T1-weighted 512 × 512 PNG slices with tumor masks.
- Labels and patient identifiers: original Figshare release by Cheng et al., version 8, DOI [10.6084/m9.figshare.1512427.v8](https://doi.org/10.6084/m9.figshare.1512427.v8).
- **1,768 ScienceDB directory labels disagree with the original labels.** Every PNG was matched to its original MAT record by pixel content, and training uses the original labels. See [`docs/data-inspection/label-audit-2026-10-08/`](docs/data-inspection/label-audit-2026-10-08/README.md).
- Split: 70 / 15 / 15 % of assumed patient groups within each class, split seed 42. Patient identifiers that share a numeric part (for example `MR049453` and `MR049453B`) are treated as one group. This is a conservative assumption, not a verified identity, and gives 209 groups from 233 original identifiers. Train / validation / test contain 2,099 / 544 / 421 images (147 / 31 / 31 groups). Details: [`data/metadata_pid_numeric/README.md`](data/metadata_pid_numeric/README.md).
- The saved split (`data/metadata_pid_numeric/`) is part of this repository. Training refuses to start if its SHA-256 hashes change; it never re-splits the data.

## Repository layout

```
train.py                     one frozen formal run (or its --preflight)
analyze.py                   validation preparation and the one-time test evaluation
configs/                     frozen formal protocol, data preprocessing, paths
src/dataset.py               metadata checks, preprocessing, augmentation, loaders
src/models/                  ResNet-50 builder, SE, PSANet adaptation, CBAM
src/training.py              shared epoch loop, metrics, checkpoint selection
src/evaluation.py            metrics, corruptions, Grad-CAM, cost, paired bootstrap
scripts/                     data preparation, label audit, duplicate screen, weights download,
                             post-hoc analyses, report and presentation figures
data/metadata_pid_numeric/   saved manifest, split and class mapping
docs/                        data inspection, PSANet design notes, preflight, training and result
                             records (index: docs/README.md)
```

## Setup

Requirements: Windows or Linux, Python 3.12, and an NVIDIA GPU with CUDA. The formal entry points refuse to fall back to the CPU. Versions used: PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.5.2, Pillow 12.3.0, h5py 3.15.1, on an RTX 5060 Laptop GPU. matplotlib 3.10.9 is used only for the report figures.

```powershell
conda create -n attention-mri python=3.12
conda activate attention-mri
pip install -r requirements.txt
```

1. **Images.** Download `data.zip` from ScienceDB (MD5 `a5bd11a085d31d8cc1fac9c8c9485a2b`) and extract it into `data/raw/sciencedb/`, so that the class folders are `data/raw/sciencedb/data/Glioma`, `.../Meningioma` and `.../Pituitary tumor`. Raw files are read-only inputs.
2. **Pretrained weights.** Download and verify the torchvision ResNet-50 `IMAGENET1K_V2` file:
   ```powershell
   python -B scripts/fetch_pretrained.py
   ```
3. **Optional: re-derive labels and split from the raw sources.** Only needed to check the saved metadata. This downloads the original Figshare archives (about 840 MB, MD5-verified), matches every PNG to its MAT record again, and rebuilds the split in new folders:
   ```powershell
   python -B scripts/recover_patient_mapping.py --download
   python -B scripts/recover_patient_mapping.py --output-dir results/mapping-check
   python -B scripts/prepare_data.py --mapping-dir results/mapping-check --inspection-dir docs/data-inspection/2026-09-29 --output-dir results/split-check --ratios 0.70 0.15 0.15 --seed 42 --patient-grouping numeric
   ```
   `results/mapping-check/patient_mapping.csv` should be byte-identical to `docs/data-inspection/patient-mapping-2026-09-29/patient_mapping.csv`, and `results/split-check/data_split.csv` and `manifest.csv` should have the SHA-256 values listed in `data/metadata_pid_numeric/summary.json`. This was checked on 2026-10-08.

## Running the experiment

Each command refuses to overwrite an existing output folder. Run them from the repository root.

```powershell
# 1. Preflight: 36 train / 18 validation images, 1 epoch, per model (about 1 minute each)
python -B train.py --formal-config configs/formal_experiment_pid_numeric.json --run-name formal-v2-resnet50-seed0 --preflight
# repeat for formal-v2-resnet50-se-terminal-seed0, -psanet-terminal-seed0, -cbam-terminal-seed0
python -B analyze.py --formal-config configs/formal_experiment_pid_numeric.json --analysis-name preflight-check --stage preflight

# 2. Formal training: 12 runs, about 13 minutes each on the GPU above
foreach ($model in "resnet50", "resnet50-se-terminal", "resnet50-psanet-terminal", "resnet50-cbam-terminal") {
  foreach ($seed in 0, 1, 2) {
    python -B train.py --formal-config configs/formal_experiment_pid_numeric.json --run-name "formal-v2-$model-seed$seed"
  }
}

# 3. Validation preparation: cost, Grad-CAM cases, Grad-CAM/mask overlap, validation code check of the statistics
python -B analyze.py --formal-config configs/formal_experiment_pid_numeric.json --analysis-name pid-numeric-v2-final --stage prepare

# 4. One-time test evaluation (clean and nine corruptions, pre-specified paired comparison)
python -B analyze.py --formal-config configs/formal_experiment_pid_numeric.json --analysis-name pid-numeric-v2-final --stage finalize
```

Outputs go to `results/<stage>/<run>/` (configuration, provenance, per-epoch history, per-image validation probabilities) and `checkpoints/<stage>/<run>/best.pt`. Neither folder is versioned.

**If a run fails part-way**, move its partial `results/formal/<run>` and `checkpoints/formal/<run>` folders into `results/failed/` with a note of the error, then rerun the same command. Do not delete them, and do not reuse a checkpoint from a failed run.

Post-hoc analyses and figures, run after step 4. They compute no new test metric and are labeled post hoc wherever they are used.

```powershell
python -B scripts/gradcam_test_cases.py   # Grad-CAM for six rule-selected test images (illustration only)
python -B scripts/attention_weights.py    # learned SE and CBAM weights on validation images
python -B scripts/make_figures.py         # report figures in results/figures/report/
python -B scripts/make_slide_images.py    # presentation image panels and comparison plot in results/figures/slides/
```

## Results

One-time test evaluation, clean images (421 images, 31 assumed patient groups). Macro-F1 is the mean (sample SD) over training seeds 0–2; the difference is paired by seed, with a 95 % interval from 10,000 class-stratified patient-group bootstrap resamples.

| Model | Macro-F1 | Difference from baseline | 95 % interval | Pre-specified decision |
| --- | ---: | ---: | --- | --- |
| ResNet-50 baseline | 0.898 (0.003) | — | — | — |
| + SE | 0.901 (0.009) | +0.003 | −0.012 to +0.023 | no consistent difference |
| + PSANet adaptation | 0.891 (0.007) | −0.007 | −0.025 to +0.013 | no consistent difference |
| + CBAM | 0.907 (0.015) | +0.009 | −0.004 to +0.027 | no consistent difference |

The final report discusses the baseline, SE and CBAM models. The PSANet adaptation was trained and evaluated under the same protocol and is reported here and in the result records. Details: [training](docs/formal-training-2026-10-10.md), [validation analysis](docs/analysis-prepare-2026-10-11.md), [test results, robustness and post-hoc error analysis](docs/test-results-2026-10-11.md), [learned attention weights](docs/attention-weights-2026-10-20.md).

## Protocol summary

- Preprocessing: the full image resized to 224 × 224 (bilinear, antialiased, no crop), gray replicated to three channels, ImageNet normalization. Training augmentation: rotation ±10°, translation up to 5 %, horizontal flip p = 0.5.
- Training: all parameters fine-tuned, AdamW (lr 1e-4, weight decay 1e-4), batch 32, 30 epochs, unweighted cross-entropy, FP32, no scheduler or early stopping. The checkpoint with the highest validation Macro-F1 is kept (ties: lower validation loss, then the earlier epoch). Only this best checkpoint is saved.
- Evaluation: Macro-F1 (primary), accuracy, per-class precision / recall / F1, one-vs-rest AUROC, confusion matrices; three corruption families at three severities; parameter count, counted FLOPs and GPU latency; Grad-CAM at the `layer4` output.
- Pre-specified comparison: for each attention model, the Macro-F1 difference from the baseline within each training seed, averaged over seeds, with a 95 % percentile interval from 10,000 class-stratified bootstrap resamples of patient groups. "Consistent benefit" requires every per-seed difference > 0 and a lower bound > 0. The full rule is in `configs/formal_experiment_pid_numeric.json`.
- Every run and analysis checks the SHA-256 of the configuration, metadata, pretrained weights and source code that were frozen before the first formal run.

## Reproducibility notes

- Python, NumPy, PyTorch and DataLoader seeds are fixed. Deterministic algorithms and deterministic cuDNN are on, cuDNN benchmark and TF32 are off, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set. On the same hardware and software, a rerun of the preflight reproduced every prediction exactly (`docs/preflight-2026-10-08.json`). Other GPUs, drivers or library versions may give slightly different numbers.
- `.gitattributes` disables line-ending conversion because the code checks byte-level hashes. Do not re-save metadata or configuration files in an editor that changes line endings.

## Limitations

The patient grouping is an assumption; 31 validation and 31 test groups give limited statistical precision. The test split includes images that were in the training or validation sets of earlier, discarded development runs; the test set itself was evaluated once, after the protocol was frozen. The analyses of where the errors occur, the test Grad-CAM examples and the attention-weight summaries were added after the test results and are descriptive. All data come from one public dataset, there is no external validation, and the results are image-level rather than patient-level. Grad-CAM maps are explanations, not segmentations.

## Data citation

Cheng J, Huang W, Cao S, Yang R, Yang W, Yun Z, et al. Enhanced performance of brain tumor classification via tumor region augmentation and partition. PLoS One. 2015;10(10):e0140381. See the ScienceDB and Figshare pages for the dataset license terms.
