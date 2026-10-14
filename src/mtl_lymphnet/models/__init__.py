from .backbones import BACKBONES, PROPOSED_KEY, proposed_backbone, variant_backbones, variant_names
from .mtl_unet import MTLUNet, count_parameters
from .single_task import SegOnlyUNet, ClsOnlyNet

# Backward compatibility alias
BACKBONE_CHOICES = BACKBONES

__all__ = [
    "BACKBONES",
    "BACKBONE_CHOICES",
    "PROPOSED_KEY",
    "proposed_backbone",
    "variant_backbones",
    "variant_names",
    "MTLUNet",
    "count_parameters",
    "SegOnlyUNet",
    "ClsOnlyNet",
]
