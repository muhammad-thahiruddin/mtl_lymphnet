"""Tests for MTL-UNet and single-task model variants (output shapes)."""
from __future__ import annotations

import torch

from mtl_lymphnet.models.mtl_unet import MTLUNet, count_parameters
from mtl_lymphnet.models.single_task import ClsOnlyNet, SegOnlyUNet

IMG_SIZE = 64
BACKBONE = "resnet18"  # fast, first-class smp encoder; no internet needed with pretrained=None


def test_mtlunet_output_shapes():
    model = MTLUNet(backbone=BACKBONE, pretrained=None)
    model.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    seg_logits, cls_logits = model(x)

    assert seg_logits.shape == (2, 1, IMG_SIZE, IMG_SIZE)
    assert cls_logits.shape == (2, 1)


def test_seg_only_and_cls_only_shapes():
    seg_model = SegOnlyUNet(backbone=BACKBONE, pretrained=None)
    cls_model = ClsOnlyNet(backbone=BACKBONE, pretrained=None)
    seg_model.eval()
    cls_model.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    seg_logits, cls_dummy = seg_model(x)
    assert seg_logits.shape == (2, 1, IMG_SIZE, IMG_SIZE)
    assert cls_dummy.shape == (2, 1)

    seg_dummy, cls_logits = cls_model(x)
    assert seg_dummy.shape == (2, 1, IMG_SIZE, IMG_SIZE)
    assert cls_logits.shape == (2, 1)


def test_count_parameters():
    model = MTLUNet(backbone=BACKBONE, pretrained=None)
    info = count_parameters(model)
    assert info["total"] > 0
    assert info["trainable"] == info["total"]  # nothing frozen by default
    assert info["size_mb"] > 0
