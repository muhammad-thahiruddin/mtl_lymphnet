"""Global configuration for training / experiment runs.

`Config` holds only plain, YAML-serializable values (hyperparameters, paths,
mode). Environment resolution (device, Kaggle/Colab detection, GPU-based mode
auto-detection) lives in :mod:`mtl_lymphnet.env` and returns a populated
`Config` via `env.setup_environment(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Config:
    mode: str = "debug"  # 'debug' or 'full' -- controls the presets below

    # Data / model
    img_size: int = 256
    backbone: str = "tu-tiny_vit_5m_224.dist_in22k_ft_in1k"  # Must match backbones.PROPOSED_KEY
    pretrained: Optional[str] = "imagenet"

    # Training
    batch_size: int = 8
    num_epochs: int = 3
    lr: float = 5e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    num_workers: int = 0
    patience: int = 20
    label_smoothing: float = 0.05
    subset_fraction: float = 1.0

    # Cross-validation protocol (R1.1: val_fraction enables the inner
    # train/val split so the held-out test fold is never used for early
    # stopping, checkpoint selection, or threshold selection).
    n_folds: int = 5
    val_fraction: float = 0.2

    # Reproducibility
    seed: int = 42

    # Paths (resolved by env.setup_environment for the current platform)
    data_dir: str = "dataset"
    output_dir: str = "outputs_v2"

    # Populated by env.setup_environment(); safe defaults for pure-config use
    device: str = "cpu"
    multi_gpu: bool = False

    # --- Mode presets -----------------------------------------------------
    def apply_mode_defaults(self) -> "Config":
        """Return a copy with 'full' mode presets applied (mirrors the
        original notebook's Config.__post_init__ behaviour)."""
        if self.mode != "full":
            return self
        overrides = dict(
            img_size=256,
            batch_size=8,
            num_epochs=150,
            n_folds=5,
            use_amp=True,
            num_workers=0,
            subset_fraction=1.0,
            patience=20,
        )
        return self.replace(**overrides)

    # --- (de)serialization --------------------------------------------------
    def replace(self, **overrides: Any) -> "Config":
        data = asdict(self)
        data.update(overrides)
        return Config(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # allow nested {"training": {...}} style YAML files
        if "training" in data and isinstance(data["training"], dict):
            merged = {**data.get("training", {})}
            data = {**data, **merged}
        return cls.from_dict(data)
