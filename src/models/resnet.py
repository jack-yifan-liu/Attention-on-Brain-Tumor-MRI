"""ResNet-50 baseline: unchanged backbone, three-class linear head."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50

from src.dataset import file_sha256
from src.models.se import SEBlock
from src.models.psanet import PSANetAttention
from src.models.cbam import CBAMBlock


def build_baseline(num_classes, *, pretrained_path=None):
    model = resnet50(weights=None)
    if pretrained_path is not None:
        path = Path(pretrained_path)
        # The official V2 file name carries a SHA-256 prefix; reject any other initialisation.
        if not file_sha256(path).startswith("11ad3fa6"):
            raise ValueError("Pretrained file is not the verified ResNet-50 V2 checkpoint")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    # Load the full ImageNet weights, then replace only the head; backbone, BN and strides are unchanged.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(num_classes, *, model_name, attention_config=None, pretrained_path=None):
    if model_name == "resnet50":
        if attention_config is not None:
            raise ValueError("Baseline must not contain attention")
    elif model_name == "resnet50_se_terminal":
        confirmed = {"name": "se", "position": "after_layer4_before_avgpool",
                     "count": 1, "channels": 2048, "reduction_ratio": 16}
        if attention_config != confirmed:
            raise ValueError("SE configuration differs from the confirmed terminal r=16 design")
    elif model_name == "resnet50_psanet_terminal":
        confirmed = {"name": "psanet_adaptation", "position": "after_layer4_before_avgpool",
                     "count": 1, "channels": 2048, "mid_channels": 512,
                     "feature_size": [7, 7], "mask_size": [13, 13],
                     "branches": "collect_distribute", "compact": False, "shrink_factor": 1,
                     "softmax": True, "normalization_factor": 1.0,
                     "fusion": "concat_conv1x1_bn_relu"}
        if attention_config != confirmed:
            raise ValueError("PSANet configuration differs from the confirmed classification adaptation")
    elif model_name == "resnet50_cbam_terminal":
        confirmed = {"name": "cbam", "position": "after_layer4_before_avgpool",
                     "count": 1, "channels": 2048, "order": "channel_then_spatial",
                     "reduction_ratio": 16, "channel_pooling": ["avg", "max"],
                     "channel_mlp_bias": True, "spatial_pooling": ["max", "avg"],
                     "spatial_kernel_size": 7, "spatial_padding": 3,
                     "spatial_conv_bias": False, "spatial_bn_eps": 1e-5,
                     "spatial_bn_momentum": 0.01, "spatial_bn_weight_init": 1.0,
                     "spatial_bn_bias_init": 0.0}
        if attention_config != confirmed:
            raise ValueError("CBAM configuration differs from the confirmed terminal design")
    else:
        raise ValueError(f"Model not implemented: {model_name}")
    # Build the shared backbone and head first, so the head initialisation is identical across models for one seed.
    model = build_baseline(num_classes, pretrained_path=pretrained_path)
    if model_name == "resnet50_se_terminal":
        # torchvision calls avgpool right after layer4, so Sequential(module, avgpool) inserts the attention there;
        # all residual blocks, strides, BN and fc are kept, and attention parameters are named avgpool.0.*.
        model.avgpool = nn.Sequential(SEBlock(2048, reduction_ratio=16), model.avgpool)
    elif model_name == "resnet50_psanet_terminal":
        model.avgpool = nn.Sequential(PSANetAttention(2048, 512, (7, 7)), model.avgpool)
    elif model_name == "resnet50_cbam_terminal":
        model.avgpool = nn.Sequential(CBAMBlock(2048, 16, 7), model.avgpool)
    return model
