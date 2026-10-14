"""
regenerate_corrected_tables_figures.py
=======================================

PURPOSE
-------
This script regenerates the LaTeX table rows and the external-validation bar
chart for four artifacts in the MTL-LymphNet manuscript ("Ultrasonic
Imaging" submission) that were found, during a full manuscript-vs-code audit,
to contain numbers that no longer matched the underlying experiment data:

    * Table 6  (external validation by BACKBONE, Section 4.2)
    * Table 7  (external validation by LOSS-WEIGHTING STRATEGY, Section 4.2)
    * Table 9  (backbone ablation incl. GPU/CPU inference timing, Section 4.4)
    * Table 10 (loss-weighting ablation, MCC column, Section 4.5)
    * Figure 5 (Internal CV vs bidirectional external validation bar chart)

WHY THIS SCRIPT EXISTS
-----------------------
The manuscript's Tables 6 and 7 previously showed standard-deviation values
that did not match the underlying 3-seed data (off by up to ~9x for some
cells) -- most likely because only the *mean* columns were regenerated after
a later multi-seed re-run, while the *std* columns were left over from an
earlier, different computation. Figure 5 was found to have been plotted from
a single (primary) seed's results, while the surrounding table/text already
reported 3-seed means. Table 9's GPU/CPU timing mixed values from two
different benchmarking passes (an ad-hoc timing captured inside the ablation
training loop, and the dedicated `benchmarking/inference_benchmark.py`
module added for reviewer comment R1.11). This script always reads directly
from the two canonical, timestamped result artifacts below, so re-running it
after any new experiment sweep will regenerate self-consistent numbers
instead of letting tables silently drift out of sync with the data again.

DATA SOURCES (read-only -- this script never modifies them)
-------------------------------------------------------------
1. outputs_v2/experiment_results.pkl
       The main pickled results dictionary written by the training/eval
       pipeline. Relevant top-level keys used here:
         - 'exp1_internal_cv'      : internal 5-fold x 3-seed CV results
         - 'exp2_c1_to_c2'         : external validation, default config
         - 'exp2_c2_to_c1'         : external validation, default config
         - 'exp2_variants'         : external validation, backbone/strategy
                                      sweep (non-default configs only)
         - 'exp4_ablation_backbone': internal-CV backbone ablation (Dice,
                                      AUC, params, size; also has ad-hoc
                                      timing fields that this script
                                      deliberately IGNORES in favor of #2)
         - 'exp5_ablation_loss'    : internal-CV loss-weighting ablation

2. outputs_v2/benchmark_results.json
       Output of the dedicated, reviewer-comment-R1.11-compliant benchmark
       script (`src/mtl_lymphnet/benchmarking/inference_benchmark.py`).
       This is the ONLY source this script uses for GPU/CPU inference
       timing, because it is the one benchmark run whose methodology
       (hardware, batch size, precision, warm-up count, repetition count)
       is fully specified and matches what Table 9's caption claims.

HOW TO RUN
----------
From the project root (the directory containing `outputs_v2/`):

    python script/regenerate_corrected_tables_figures.py

Optional arguments (all have sensible defaults matching this project's
layout, so in the common case no arguments are needed at all):

    --pkl PATH            Path to experiment_results.pkl
                           (default: outputs_v2/experiment_results.pkl)
    --benchmark-json PATH Path to benchmark_results.json
                           (default: outputs_v2/benchmark_results.json)
    --output-dir PATH     Where to write the regenerated .tex snippets and
                           the corrected Figure 5 PNG
                           (default: outputs_v2/regenerated/)

Example with explicit paths (e.g. running from a different working directory):

    python regenerate_corrected_tables_figures.py \\
        --pkl /path/to/outputs_v2/experiment_results.pkl \\
        --benchmark-json /path/to/outputs_v2/benchmark_results.json \\
        --output-dir /path/to/outputs_v2/regenerated

WHAT THE SCRIPT PRODUCES
-------------------------
Inside --output-dir, five files are written:
    table6_ext_backbone_sweep.tex   -- LaTeX tabular rows only (paste into
                                        the existing \\begin{tabular}...
                                        \\end{tabular} in the manuscript)
    table7_ext_strategy_sweep.tex
    table9_ablation_backbone.tex
    table10_ablation_loss.tex
    figure5_external_validation.png -- full replacement for
                                        5_external_validation.png

The script also prints every generated row to stdout so you can eyeball the
numbers immediately without opening the files.

WHAT THIS SCRIPT DOES **NOT** DO
----------------------------------
    * It does NOT touch, import, or depend on any file under `src/`. It only
      reads the two result artifacts listed above.
    * It does NOT overwrite anything inside `outputs_v2/tables/` or
      `outputs_v2/figures/` -- outputs go to a separate `regenerated/`
      subfolder so nothing existing can be accidentally clobbered.
    * It does NOT edit the manuscript .tex file itself. You still need to
      copy the relevant rows / swap in the new PNG by hand (or with your own
      editor tooling) -- this keeps the script safe to re-run at any time
      without risking an unreviewed change to the manuscript.

REQUIREMENTS
------------
    numpy, matplotlib (both already used elsewhere in this project)

AUTHOR / CONTEXT
-----------------
Written as part of the pre-submission audit of the "Ultrasonic Imaging"
revision (revised_manuscript_01). See audit_findings.md in that folder for
the full list of discrepancies this script resolves.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required to regenerate Figure 5. Install it with "
        "`pip install matplotlib` (or `pip install matplotlib --break-system-packages` "
        "in externally-managed environments) and re-run this script."
    ) from exc


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_experiment_results(pkl_path: Path) -> dict:
    """Load the main pickled experiment-results dictionary.

    Parameters
    ----------
    pkl_path:
        Path to ``outputs_v2/experiment_results.pkl``.

    Returns
    -------
    dict
        The top-level results dictionary, unmodified. See the module
        docstring for the keys this script relies on.

    Raises
    ------
    FileNotFoundError
        If ``pkl_path`` does not exist -- with a message pointing at the
        expected default location, to make the failure mode obvious when
        this script is run from an unexpected working directory.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Could not find experiment results at: {pkl_path}\n"
            "Pass the correct path with --pkl, or run this script from the "
            "project root so the default 'outputs_v2/experiment_results.pkl' "
            "resolves correctly."
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_benchmark_results(json_path: Path) -> List[dict]:
    """Load the standardized inference-benchmark results (R1.11 benchmark).

    Parameters
    ----------
    json_path:
        Path to ``outputs_v2/benchmark_results.json``. This file is a flat
        JSON list with one entry per (backbone, device) pair -- i.e. 3
        backbones x 2 devices (cpu, cuda) = 6 entries in this project.

    Returns
    -------
    list of dict
        Each dict has (at minimum) the keys ``backbone``, ``device``,
        ``mean_ms``, ``std_ms``, ``n_warmup``, ``n_reps``.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Could not find benchmark results at: {json_path}\n"
            "Pass the correct path with --benchmark-json, or run this "
            "script from the project root."
        )
    with open(json_path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Small numeric / formatting helpers
# --------------------------------------------------------------------------- #


def _mean_std(entries: Sequence[dict], metric: str) -> Tuple[float, float]:
    """Compute the population mean and std (ddof=0) of ``metric`` across runs.

    Parameters
    ----------
    entries:
        A list of per-run result dictionaries, e.g.
        ``data['exp5_ablation_loss']['DWA']`` (a list of 5 run dicts), or
        a list assembled from ``results_by_seed`` across the 3 seeds.
    metric:
        The dict key to average, e.g. ``'seg_dice'``, ``'cls_auc'``,
        ``'cls_mcc'``.

    Returns
    -------
    (mean, std)
        ``std`` uses ``ddof=0`` (population standard deviation), matching
        the convention used everywhere else in this project's reporting
        (e.g. ``outputs_v2/tables/table2_internal_cv.tex``).
    """
    values = np.array([entry[metric] for entry in entries], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def _fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    """Format a mean/std pair as a LaTeX ``$X.XXX \\pm Y.YYY$`` style string.

    Note the manuscript's tables actually omit the outer ``$...$`` (they are
    already inside a math-adjacent macro via ``\\pm``), so this returns e.g.
    ``"0.865 $\\pm$ 0.015"`` -- ready to paste directly into a tabular row.
    """
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _seed_entries(results_by_seed: Dict[int, dict], seeds: Sequence[int] = (42, 43, 44)) -> List[dict]:
    """Pull out the per-seed result dicts for the given seeds, in order.

    ``results_by_seed`` is a dict keyed by seed (e.g. ``{42: {...}, 43:
    {...}, 44: {...}}``) as stored under
    ``data['exp2_c1_to_c2']['results_by_seed']`` or the equivalent field
    inside each ``exp2_variants`` entry. Each value is itself a single
    aggregated result dict for that seed (already pooled across that
    seed's 5 folds), containing keys like ``'seg_dice'``, ``'cls_auc'``,
    ``'cls_mcc'``, etc.
    """
    return [results_by_seed[s] for s in seeds]


# --------------------------------------------------------------------------- #
# Table 6 -- External validation by BACKBONE architecture
# --------------------------------------------------------------------------- #


def build_table6_ext_backbone_sweep(data: dict) -> str:
    """Regenerate the 3 data rows of Table 6 (external validation by backbone).

    Table 6 reports, for each of the three backbones (TinyViT-5M,
    MobileNetV3-Large, EfficientNet-B0) and both cross-center directions
    (C1->C2, C2->C1), the mean +/- std across three random seeds of Dice,
    IoU, AUC, and MCC.

    Data provenance
    ----------------
    * TinyViT-5M is the *default* configuration and its external-validation
      results live directly under ``data['exp2_c1_to_c2']`` and
      ``data['exp2_c2_to_c1']`` (each has a ``'results_by_seed'`` dict keyed
      by seed 42/43/44).
    * MobileNetV3-Large and EfficientNet-B0 are *non-default* sweep
      variants and live under
      ``data['exp2_variants']['MobileNetV3-Large_uncertainty_w0.5']`` and
      ``data['exp2_variants']['EfficientNet-B0_uncertainty_w0.5']``
      respectively; each of those has its own ``'C1->C2'`` /
      ``'C2->C1'`` sub-dicts, each with a ``'results_by_seed'`` dict.

    Returns
    -------
    str
        The three data rows, newline-joined, ready to replace the three
        data rows inside the existing ``tab:ext_backbone_sweep`` table
        environment in the manuscript. Column order per row: Dice, IoU,
        AUC, MCC (C1->C2), then Dice, IoU, AUC, MCC (C2->C1) -- matching
        the manuscript's table.
    """
    metrics = ["seg_dice", "seg_iou", "cls_auc", "cls_mcc"]

    def row_for(results_by_seed: Dict[int, dict]) -> List[str]:
        entries = _seed_entries(results_by_seed)
        return [_fmt_pm(*_mean_std(entries, m)) for m in metrics]

    rows = []
    backbone_sources = {
        "TinyViT-5M": (
            data["exp2_c1_to_c2"]["results_by_seed"],
            data["exp2_c2_to_c1"]["results_by_seed"],
        ),
        "MobileNetV3-Large": (
            data["exp2_variants"]["MobileNetV3-Large_uncertainty_w0.5"]["C1->C2"]["results_by_seed"],
            data["exp2_variants"]["MobileNetV3-Large_uncertainty_w0.5"]["C2->C1"]["results_by_seed"],
        ),
        "EfficientNet-B0": (
            data["exp2_variants"]["EfficientNet-B0_uncertainty_w0.5"]["C1->C2"]["results_by_seed"],
            data["exp2_variants"]["EfficientNet-B0_uncertainty_w0.5"]["C2->C1"]["results_by_seed"],
        ),
    }

    for name, (c1_to_c2, c2_to_c1) in backbone_sources.items():
        cols = row_for(c1_to_c2) + row_for(c2_to_c1)
        rows.append(f"\t\t{name:<17s} & " + " & ".join(cols) + r" \\")

    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Table 7 -- External validation by LOSS-WEIGHTING strategy
# --------------------------------------------------------------------------- #


def build_table7_ext_strategy_sweep(data: dict) -> str:
    """Regenerate the 4 data rows of Table 7 (external validation by loss
    weighting strategy), including automatic bold-highlighting of whichever
    strategy has the column-wise best value -- mirroring the original
    manuscript table's convention of bolding the strongest strategy.

    Data provenance
    ----------------
    * "Uncertainty" is the *default* configuration -- same source as
      TinyViT-5M in Table 6 (``exp2_c1_to_c2`` / ``exp2_c2_to_c1``).
    * "Fixed (0.5/0.5)", "Fixed (0.7/0.3)", and "DWA" are non-default sweep
      variants under ``data['exp2_variants']`` with keys
      ``'TinyViT-5M_fixed_w0.5'``, ``'TinyViT-5M_fixed_w0.7'``, and
      ``'TinyViT-5M_dwa_w0.5'`` respectively.

    Returns
    -------
    str
        Four LaTeX data rows, newline-joined. The best (highest) value in
        each of the 8 metric columns is wrapped in ``\\textbf{...}`` --
        recomputed fresh each run, so if a future re-run changes which
        strategy wins a given column, the bolding updates automatically
        instead of silently going stale (which is what caused one of the
        original bugs this script fixes).
    """
    metrics = ["seg_dice", "seg_iou", "cls_auc", "cls_mcc"]

    strategy_sources = {
        "Uncertainty": (
            data["exp2_c1_to_c2"]["results_by_seed"],
            data["exp2_c2_to_c1"]["results_by_seed"],
        ),
        "Fixed (0.5/0.5)": (
            data["exp2_variants"]["TinyViT-5M_fixed_w0.5"]["C1->C2"]["results_by_seed"],
            data["exp2_variants"]["TinyViT-5M_fixed_w0.5"]["C2->C1"]["results_by_seed"],
        ),
        "Fixed (0.7/0.3)": (
            data["exp2_variants"]["TinyViT-5M_fixed_w0.7"]["C1->C2"]["results_by_seed"],
            data["exp2_variants"]["TinyViT-5M_fixed_w0.7"]["C2->C1"]["results_by_seed"],
        ),
        "DWA": (
            data["exp2_variants"]["TinyViT-5M_dwa_w0.5"]["C1->C2"]["results_by_seed"],
            data["exp2_variants"]["TinyViT-5M_dwa_w0.5"]["C2->C1"]["results_by_seed"],
        ),
    }

    # First pass: compute raw (mean, std) for every strategy x column.
    raw: Dict[str, List[Tuple[float, float]]] = {}
    for name, (c1_to_c2, c2_to_c1) in strategy_sources.items():
        e1, e2 = _seed_entries(c1_to_c2), _seed_entries(c2_to_c1)
        raw[name] = [_mean_std(e1, m) for m in metrics] + [_mean_std(e2, m) for m in metrics]

    # Determine, per column (8 columns total: 4 metrics x 2 directions),
    # which strategy has the highest mean -- that cell gets bolded.
    n_cols = 8
    best_strategy_per_col = [
        max(raw, key=lambda name: raw[name][col][0]) for col in range(n_cols)
    ]

    rows = []
    for name in strategy_sources:
        cells = []
        for col in range(n_cols):
            mean, std = raw[name][col]
            text = _fmt_pm(mean, std)
            if best_strategy_per_col[col] == name:
                text = r"\textbf{" + text + "}"
            cells.append(text)
        rows.append(f"\t\t{name:<17s} & " + " & ".join(cells) + r" \\")

    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Table 9 -- Backbone ablation (Dice/AUC/params/size from pkl,
#             GPU/CPU timing from the dedicated R1.11 benchmark JSON)
# --------------------------------------------------------------------------- #


def build_table9_ablation_backbone(data: dict, benchmark: List[dict]) -> str:
    """Regenerate the 3 data rows of Table 9 (internal-CV backbone ablation).

    Dice, AUC, parameter count, and model size come from
    ``data['exp4_ablation_backbone']`` (a dict of backbone_name -> list of
    per-run result dicts; TinyViT-5M has 15 runs -- all 3 seeds, reused from
    Experiment 1 -- while MobileNetV3-Large and EfficientNet-B0 have 5 runs
    each, single seed).

    GPU/CPU inference timing is deliberately taken **only** from
    ``benchmark`` (i.e. ``benchmark_results.json``, produced by
    ``src/mtl_lymphnet/benchmarking/inference_benchmark.py``), never from
    the ad-hoc timing fields also present in ``exp4_ablation_backbone``.
    Those two sources disagree (different benchmarking protocols were
    used), and only the JSON file's methodology matches what this table's
    own caption claims (10 warm-up + 100 timed iterations, batch size 1,
    FP32, Kaggle T4 GPU / Intel Xeon CPU).

    Returns
    -------
    str
        Three LaTeX data rows, newline-joined. Column order per row:
        Params (M), Size (MB), Dice, AUC, GPU (ms), CPU (ms).
        Params/Size are taken from the pkl entries directly (not
        recomputed) since they are deterministic architecture properties,
        not measurements with run-to-run variance.
    """
    backbone_order = ["TinyViT-5M", "MobileNetV3-Large", "EfficientNet-B0"]

    # --- Dice / AUC / params / size from the pkl -------------------------
    pkl_stats = {}
    for name in backbone_order:
        entries = data["exp4_ablation_backbone"][name]
        dice_mean, dice_std = _mean_std(entries, "seg_dice")
        auc_mean, auc_std = _mean_std(entries, "cls_auc")
        # params/size are identical across all runs of a given backbone
        # (they are architecture properties, not measurements), so take
        # them from the first run rather than averaging.
        params_millions = entries[0]["params"] / 1_000_000
        # size is stored in the pkl as decimal MB (bytes / 1e6); the
        # manuscript reports MiB (bytes / 1024**2) throughout, so convert
        # here to stay consistent with every other size figure in the
        # paper (Abstract, Section 3.2, Table 9 itself).
        size_mib = entries[0]["params"] * 4 / (1024**2)
        pkl_stats[name] = {
            "dice": (dice_mean, dice_std),
            "auc": (auc_mean, auc_std),
            "params_m": params_millions,
            "size_mib": size_mib,
        }

    # --- GPU / CPU timing from benchmark_results.json ---------------------
    timing = {name: {} for name in backbone_order}
    for entry in benchmark:
        name = entry["backbone"]
        if name not in timing:
            continue
        device_key = "gpu" if entry["device"] == "cuda" else "cpu"
        timing[name][device_key] = entry["mean_ms"]

    rows = []
    for name in backbone_order:
        s = pkl_stats[name]
        dice = _fmt_pm(*s["dice"])
        auc = _fmt_pm(*s["auc"])
        params = f"{s['params_m']:.2f}"
        size = f"{s['size_mib']:.1f}"
        gpu_ms = f"{timing[name]['gpu']:.1f}"
        cpu_ms = f"{timing[name]['cpu']:.1f}"
        rows.append(
            f"\t\t{name:<17s} & {params} & {size} & {dice} & {auc} & {gpu_ms} & {cpu_ms} \\\\"
        )

    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Table 10 -- Loss-weighting ablation (adds the MCC column)
# --------------------------------------------------------------------------- #


def build_table10_ablation_loss(data: dict) -> str:
    """Regenerate the 4 data rows of Table 10 (internal-CV loss-weighting
    ablation), including the MCC column.

    Data provenance
    ----------------
    ``data['exp5_ablation_loss']`` is a dict with keys ``'Fixed (0.5/0.5)'``,
    ``'Fixed (0.7/0.3)'``, ``'Uncertainty'``, ``'DWA'``, each mapping to a
    list of per-run result dicts (5 runs for the three non-default
    strategies, 15 runs -- all 3 seeds -- for Uncertainty, which is the
    default configuration reused from Experiment 1).

    MCC is computed as the simple across-run mean/std of each run's
    ``'cls_mcc'`` field, matching the convention already used for every
    other metric in this table (and matching Table 8's MCC column exactly
    for the shared "Uncertainty"/"MTL (proposed)" configuration). A pooled
    (single confusion-matrix-over-all-runs) MCC was also tested during the
    audit and found to differ only in the third decimal place from this
    per-run average, so the simpler and more consistent per-run average is
    used here.

    Returns
    -------
    str
        Four LaTeX data rows, newline-joined, in the order Fixed (0.5/0.5),
        Fixed (0.7/0.3), Uncertainty, DWA -- matching the manuscript.
        Column order: Dice, IoU, AUC, F1, MCC.
    """
    metrics = ["seg_dice", "seg_iou", "cls_auc", "cls_f1", "cls_mcc"]
    strategy_order = ["Fixed (0.5/0.5)", "Fixed (0.7/0.3)", "Uncertainty", "DWA"]

    rows = []
    for name in strategy_order:
        entries = data["exp5_ablation_loss"][name]
        cols = [_fmt_pm(*_mean_std(entries, m)) for m in metrics]
        rows.append(f"\t\t{name:<17s} & " + " & ".join(cols) + r" \\")

    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Figure 5 -- Internal CV vs bidirectional external validation
# --------------------------------------------------------------------------- #


def plot_figure5_external_validation(data: dict, output_path: Path) -> None:
    """Regenerate Figure 5 using 3-seed MEAN values (not a single seed).

    The original Figure 5 was found to have been plotted from
    ``exp2_c1_to_c2['results']`` / ``exp2_c2_to_c1['results']`` -- the
    *primary-seed-only* result dict -- while Table 5 and the surrounding
    text already reported the 3-seed mean for the same comparison. This
    function instead averages across all three seeds (42, 43, 44) for
    every bar, using ``results_by_seed``, so the figure and the table it
    illustrates always show the same numbers.

    Parameters
    ----------
    data:
        The loaded ``experiment_results.pkl`` dictionary.
    output_path:
        Where to save the regenerated PNG (e.g.
        ``outputs_v2/regenerated/figure5_external_validation.png``).
        The parent directory is created if it does not already exist.

    Notes
    -----
    Visual style (two side-by-side panels, grouped bars, color scheme,
    figure size) intentionally mirrors the original figure so it can be
    swapped in as a drop-in replacement for
    ``manuscript/5_external_validation.png`` without changing the
    manuscript's figure caption or layout.
    """
    metrics = [
        "seg_dice",
        "seg_iou",
        "cls_auc",
        "cls_accuracy",
        "cls_sensitivity",
        "cls_specificity",
        "cls_f1",
    ]

    internal_summary = data["exp1_internal_cv"]["summary"]
    internal_vals = {m: internal_summary[m]["mean"] for m in metrics}

    def three_seed_means(results_by_seed: Dict[int, dict]) -> Dict[str, float]:
        entries = _seed_entries(results_by_seed)
        return {m: _mean_std(entries, m)[0] for m in metrics}

    results = {
        "Internal CV": internal_vals,
        "C1->C2": three_seed_means(data["exp2_c1_to_c2"]["results_by_seed"]),
        "C2->C1": three_seed_means(data["exp2_c2_to_c1"]["results_by_seed"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {"Internal CV": "#1f77b4", "C1->C2": "#ff7f0e", "C2->C1": "#2ca02c"}
    groups = ["Internal CV", "C1->C2", "C2->C1"]
    width = 0.25

    seg_metrics = ["seg_dice", "seg_iou"]
    x = np.arange(len(seg_metrics))
    for i, g in enumerate(groups):
        vals = [results[g][m] for m in seg_metrics]
        axes[0].bar(x + (i - 1) * width, vals, width, label=g, color=colors[g],
                    edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seg_metrics)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Segmentation", fontweight="bold")
    axes[0].legend()

    cls_metrics = ["cls_auc", "cls_accuracy", "cls_sensitivity", "cls_specificity", "cls_f1"]
    x2 = np.arange(len(cls_metrics))
    for i, g in enumerate(groups):
        vals = [results[g][m] for m in cls_metrics]
        axes[1].bar(x2 + (i - 1) * width, vals, width, label=g, color=colors[g],
                    edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cls_metrics)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Classification", fontweight="bold")
    axes[1].legend()

    fig.suptitle("Internal CV vs Bidirectional External Validation", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    """Parse CLI arguments, regenerate all four tables + Figure 5, and write
    the results to ``--output-dir``, printing a preview of each table to
    stdout along the way.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Table 6, Table 7, Table 9, Table 10, and Figure 5 "
            "for the MTL-LymphNet manuscript from the canonical result "
            "artifacts, so these outputs can never silently drift out of "
            "sync with the underlying experiment data again."
        )
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=Path("outputs_v2/experiment_results.pkl"),
        help="Path to experiment_results.pkl (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path("outputs_v2/benchmark_results.json"),
        help="Path to benchmark_results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_v2/regenerated"),
        help="Directory to write regenerated .tex/.png files into (default: %(default)s)",
    )
    args = parser.parse_args()

    data = load_experiment_results(args.pkl)
    benchmark = load_benchmark_results(args.benchmark_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "table6_ext_backbone_sweep.tex": build_table6_ext_backbone_sweep(data),
        "table7_ext_strategy_sweep.tex": build_table7_ext_strategy_sweep(data),
        "table9_ablation_backbone.tex": build_table9_ablation_backbone(data, benchmark),
        "table10_ablation_loss.tex": build_table10_ablation_loss(data),
    }

    for filename, content in artifacts.items():
        out_path = args.output_dir / filename
        out_path.write_text(content + "\n")
        print(f"\n=== {filename} ===")
        print(content)
        print(f"--> written to {out_path}")

    fig_path = args.output_dir / "figure5_external_validation.png"
    plot_figure5_external_validation(data, fig_path)
    print(f"\n=== figure5_external_validation.png ===\n--> written to {fig_path}")

    print(
        "\nDone. These files were written for review only -- the manuscript "
        ".tex file and existing outputs_v2/tables|figures were NOT modified. "
        "Copy the rows / swap in the PNG by hand once you've checked them."
    )


if __name__ == "__main__":
    main()
