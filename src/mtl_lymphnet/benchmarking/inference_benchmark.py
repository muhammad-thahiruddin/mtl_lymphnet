"""Standardized inference benchmarking (R1.11).

The manuscript reported CPU/GPU timings without hardware specs, batch size,
precision, warm-up iterations, repetition count, or whether preprocessing was
included. This module captures all of those fields alongside every timing
result so the numbers are reproducible and fully specified.
"""
from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HardwareInfo:
    platform: str
    processor: str
    cpu_count: int
    ram_gb: Optional[float]
    gpu_name: Optional[str]
    gpu_mem_gb: Optional[float]


@dataclass
class BenchmarkResult:
    device: str
    hardware: HardwareInfo
    batch_size: int
    precision: str
    n_warmup: int
    n_reps: int
    include_preprocessing: bool
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    throughput_ips: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def get_hardware_info(device: str) -> HardwareInfo:
    ram_gb = None
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass

    gpu_name = gpu_mem_gb = None
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / 1e9

    return HardwareInfo(
        platform=f"{platform.system()} {platform.release()}",
        processor=platform.processor() or platform.machine(),
        cpu_count=torch.get_num_threads() if device == "cpu" else (torch.get_num_threads()),
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_mem_gb=gpu_mem_gb,
    )


def benchmark_inference(
    model: nn.Module,
    input_size: tuple[int, int, int] = (3, 256, 256),
    device: str = "cpu",
    batch_size: int = 1,
    precision: str = "fp32",
    n_warmup: int = 10,
    n_reps: int = 100,
    preprocess_fn: Optional[Callable[[], torch.Tensor]] = None,
) -> BenchmarkResult:
    """Benchmark a single forward pass with a fully specified, reproducible
    protocol (hardware, batch size, precision, warm-up, repetitions).

    Parameters
    ----------
    preprocess_fn: if provided, called once per repetition INSIDE the timed
        region to produce the input tensor, so preprocessing cost is included
        in the reported latency. If None, a single dummy tensor is created
        once outside the timed region (preprocessing excluded).
    """
    model = model.to(device)
    model.eval()
    if precision == "fp16" and device == "cuda":
        model = model.half()

    dtype = torch.float16 if (precision == "fp16" and device == "cuda") else torch.float32
    static_input = torch.randn(batch_size, *input_size, device=device, dtype=dtype)

    def _get_input() -> torch.Tensor:
        return preprocess_fn() if preprocess_fn is not None else static_input

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(_get_input())
    if device == "cuda":
        torch.cuda.synchronize()

    timings_ms = np.empty(n_reps)
    with torch.no_grad():
        for i in range(n_reps):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(_get_input())
            if device == "cuda":
                torch.cuda.synchronize()
            timings_ms[i] = (time.perf_counter() - t0) * 1000

    hardware = get_hardware_info(device)
    mean_ms = float(timings_ms.mean())
    return BenchmarkResult(
        device=device,
        hardware=hardware,
        batch_size=batch_size,
        precision=precision,
        n_warmup=n_warmup,
        n_reps=n_reps,
        include_preprocessing=preprocess_fn is not None,
        mean_ms=mean_ms,
        std_ms=float(timings_ms.std()),
        p50_ms=float(np.percentile(timings_ms, 50)),
        p95_ms=float(np.percentile(timings_ms, 95)),
        throughput_ips=float(batch_size / (mean_ms / 1000)) if mean_ms > 0 else float("nan"),
    )
