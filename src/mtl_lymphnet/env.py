"""Environment detection, reproducibility, and path resolution.

Extracted from the original notebook's Cell 2 ("Environment Setup & Mode
Toggle"). Kept side-effect-free at import time: nothing here mounts drives,
copies datasets, or mutates global state until `setup_environment(...)` is
called explicitly.

Authentication & Token Handling
--------------------------------
If a Hugging Face token is required (e.g. for private model weights or gated
assets), set the ``HF_TOKEN`` environment variable outside of source control
(shell profile, Kaggle secrets, Colab userdata) and read it via
:func:`get_hf_token`. Never hardcode API keys or secrets in source code.
"""
from __future__ import annotations

import os
import platform
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config


@dataclass
class EnvironmentInfo:
    is_kaggle: bool
    is_colab: bool
    platform: str
    python_version: str
    torch_version: str
    torchvision_version: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_mem_gb: Optional[float] = None
    gpu_count: int = 0


def get_hf_token() -> Optional[str]:
    """Read HF_TOKEN from the environment. Never hardcode secrets in source."""
    return os.environ.get("HF_TOKEN")


def _quiet_expected_warnings() -> None:
    """Silence noisy-but-harmless warnings that show up on every run and
    were flagged as log clutter after the first full Kaggle run:

    - timm logs an "Unexpected keys ... found while loading pretrained
      weights" message every time a classification-pretrained checkpoint is
      loaded into our encoder-only usage (segmentation_models_pytorch strips
      the classifier/conv_head/bn2 layers). timm's own message says "this
      may be expected if model is being adapted" -- it is, by design, so we
      suppress it at the logger level rather than at each call site.
    """
    import logging

    logging.getLogger("timm").setLevel(logging.ERROR)


def _setup_hf_auth() -> None:
    """Log in to the HF Hub if HF_TOKEN is set; otherwise proceed anonymously
    and quiet the resulting "unauthenticated requests" warning, since for
    this project's public model weights that warning is expected noise, not
    an actionable problem.
    """
    import logging

    token = get_hf_token()
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    else:
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def seed_everything(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_environment() -> EnvironmentInfo:
    """Detect Kaggle / Colab / local platform and GPU availability."""
    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    is_colab = "COLAB_RELEASE_TAG" in os.environ
    cuda_available = torch.cuda.is_available()

    gpu_name = gpu_mem_gb = None
    gpu_count = 0
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / 1e9

    try:
        import torchvision

        tv_version = torchvision.__version__
    except ImportError:
        tv_version = "not installed"

    return EnvironmentInfo(
        is_kaggle=is_kaggle,
        is_colab=is_colab,
        platform=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        torchvision_version=tv_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_mem_gb=gpu_mem_gb,
        gpu_count=gpu_count,
    )


def detect_mode(env: EnvironmentInfo) -> str:
    """Auto-detect 'full' vs 'debug' mode based on available GPU."""
    if not env.cuda_available:
        return "debug"
    gpu_name = (env.gpu_name or "").lower()
    if any(g in gpu_name for g in ("t4", "p100", "v100", "a100", "l4")):
        return "full"
    return "full" if (env.gpu_mem_gb or 0) >= 8 else "debug"


def resolve_paths(
    env: EnvironmentInfo,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    colab_drive_dataset: str = "/content/drive/MyDrive/dataset",
) -> tuple[str, str]:
    """Resolve data/output directories for the current platform.

    On Colab, this will mount Google Drive and copy the dataset locally on
    first use (mirrors the original notebook behaviour). On Kaggle/local this
    is a pure path computation with no side effects.
    """
    if env.is_kaggle:
        d = data_dir or "/kaggle/input/datasets/muhammadthahiruddin/lymphus/lymphus-multicenter"
        o = output_dir or "/kaggle/working/outputs_v2"
    elif env.is_colab:
        local_dataset_path = "/content/dataset"
        if not os.path.exists(local_dataset_path):
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
            print(f"Copying dataset from Drive to {local_dataset_path} ...")
            shutil.copytree(colab_drive_dataset, local_dataset_path)
            print("Dataset copied to local Colab storage.")
        d = data_dir or local_dataset_path
        o = output_dir or "/content/outputs_v2"
    else:
        d = data_dir or str(Path(".") / "dataset")
        o = output_dir or str(Path(".") / "outputs_v2")

    os.makedirs(o, exist_ok=True)
    return d, o


def setup_environment(
    mode: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
    base_config: Optional[Config] = None,
    **overrides,
) -> tuple[Config, EnvironmentInfo]:
    """Detect environment, seed everything, and return a fully-resolved Config.

    Parameters
    ----------
    mode: force 'full' or 'debug'; if None, auto-detected from GPU.
    data_dir/output_dir: override auto-resolved paths.
    base_config: an existing Config (e.g. loaded from YAML) to layer on top of.
    overrides: any other Config field to override.
    """
    _quiet_expected_warnings()
    _setup_hf_auth()
    seed_everything(seed)
    env = detect_environment()

    resolved_mode = mode or detect_mode(env)
    d, o = resolve_paths(env, data_dir, output_dir)

    cfg = base_config or Config()
    cfg = cfg.replace(
        mode=resolved_mode,
        data_dir=d,
        output_dir=o,
        seed=seed,
        device="cuda" if env.cuda_available else "cpu",
        multi_gpu=env.gpu_count > 1,
        **overrides,
    )
    cfg = cfg.apply_mode_defaults()
    # apply_mode_defaults may have been computed before device/paths were
    # finalized above; re-apply device/paths/mode to be safe.
    cfg = cfg.replace(
        mode=resolved_mode,
        data_dir=d,
        output_dir=o,
        device="cuda" if env.cuda_available else "cpu",
        multi_gpu=env.gpu_count > 1,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    return cfg, env


def print_environment_summary(cfg: Config, env: EnvironmentInfo) -> None:
    print("=" * 65)
    print("  MTL-LymphNet -- Environment Summary")
    print("=" * 65)
    print(f"  Platform     : {env.platform}")
    print(f"  Python       : {env.python_version}")
    print(f"  PyTorch      : {env.torch_version}")
    print(f"  TorchVision  : {env.torchvision_version}")
    print(f"  CUDA avail   : {env.cuda_available}")
    if env.cuda_available:
        print(f"  GPU          : {env.gpu_name} ({env.gpu_mem_gb:.1f} GB)")
    print(f"  Multi-GPU    : {cfg.multi_gpu}")
    print("-" * 65)
    print(f"  MODE         : {cfg.mode.upper()}")
    print(f"  Image size   : {cfg.img_size}x{cfg.img_size}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Epochs       : {cfg.num_epochs}")
    print(f"  Folds        : {cfg.n_folds}  (val_fraction={cfg.val_fraction})")
    print(f"  AMP          : {cfg.use_amp}")
    print(f"  Subset       : {cfg.subset_fraction * 100:.0f}%")
    print(f"  Backbone     : {cfg.backbone}")
    print(f"  Data dir     : {cfg.data_dir}")
    print(f"  Output dir   : {cfg.output_dir}")
    print("=" * 65)
