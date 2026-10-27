# PSANet classification adaptation: confirmed design

Confirmed on 2026-09-29, together with the 1 × 1 projection after concatenation described below. The module has been implemented and passed GPU smoke tests; the development smoke records are kept in a local archive and are not part of this repository. Those tests were not performance experiments.

The insertion point is shared by all three attention models: one module after the complete `layer4` and before the original global average pooling, with input and output of shape `[B, 2048, 7, 7]`.

## Core configuration

- Two separate branches, **collect** and **distribute**. Collect gathers information according to the features at the receiving position; distribute spreads information according to the features at the sending position.
- 512 intermediate channels per branch. Reduction and attention-feature adaptation use 1 × 1 Conv / BN / ReLU, with convolution bias off and default BN eps and momentum.
- Non-compact relative-position mapping: the 7 × 7 feature map covers a 13 × 13 offset range, so 169 mask channels are predicted and the valid entries form a `[B, 49, 49]` position matrix.
- As in the authors' implementation, collect and distribute are mapped first and then normalized with a softmax over the source positions; `normalization_factor = 1.0`. This is not replaced by query–key dot-product attention.
- `shrink_factor = 1`, so the 7 × 7 resolution is kept. The authors' code supports this setting; its default is 2.
- The two branch outputs are concatenated to 1,024 channels and projected back to 2,048 with 1 × 1 Conv / BN / ReLU.

The mechanism follows [Sections 3.1–3.3 of the original paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper.pdf), and the channel and operator settings follow the [authors' PSA implementation](https://github.com/hszhao/semseg/blob/master/model/psanet.py) (commit `4f274c3f276778228bc14a4565822d46359f0cc8`). The attention "mask" here is a position weighting produced by the model. It is unrelated to the dataset's tumor masks, and classification still uses the MRI alone.

## Output fusion

The authors' concatenation of local and global features is kept, followed by a channel projection added for this project:

```text
X: 2048×7×7
  → collect / distribute, 512×7×7 each
  → concatenate 1024×7×7 → project G: 2048×7×7
  → concat(X, G): 4096×7×7
  → 1×1 Conv(4096, 2048, bias=False) + BN + ReLU
  → original avgpool → original Linear(2048, 3)
```

The final 1 × 1 fusion layer is the classification adaptation. The original segmentation head, auxiliary loss and backbone dilation / stride changes are not used. An earlier `X + G` candidate was rejected, and the fusion was not changed on the basis of smoke-test metrics.

Parameter counts: 4,899,840 in the core module plus 8,392,704 in the fusion layer, 13,292,544 in total; the full model has 36,806,723 parameters. These are not FLOP or speed measurements. The fusion parameters count as part of the attention cost, so any performance difference cannot be attributed to spatial information propagation alone.

## Implementation and checks

The relative-position indexing is implemented in plain, differentiable PyTorch; the original CUDA extension was not installed. On CPU and CUDA, the mapping outputs and their gradients were checked against an independent Python translation of the authors' CPU indexing code. The original extension was never compiled or run, so this is not a binary comparison with it.

The shared backbone and head are initialized first. New convolutions use PyTorch default initialization and BN uses weight 1 / bias 0, with no extra zero initialization or learnable residual coefficient. Note that at initialization the randomly initialized fusion layer replaces the pretrained `layer4` representation rather than refining it, which is relevant when interpreting results.

The 36 / 18-image, 2-epoch smoke protocol confirmed updates of all 20 parameter tensors in both branches and the fusion layer, checkpoint and optimizer restoration, and a maximum validation probability error of 0 after reloading; the test set was not used. The four-model preflight of the formal pipeline from 2026-10-08 is recorded in [`preflight-2026-10-08.json`](preflight-2026-10-08.json).
