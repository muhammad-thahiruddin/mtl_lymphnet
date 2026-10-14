"""Command-line entrypoints for the MTL-LymphNet framework.

Usage
-----
    mtl-lymphnet --config configs/exp1_internal_cv.yaml run --experiment internal_cv
    mtl-lymphnet --config configs/base.yaml run --experiment all
    mtl-lymphnet smoke-test
    mtl-lymphnet export-tables --results outputs_v2/experiment_results.pkl
    mtl-lymphnet export-figures --results outputs_v2/experiment_results.pkl
    mtl-lymphnet stats --results outputs_v2/experiment_results.pkl
    mtl-lymphnet calibration --results outputs_v2/experiment_results.pkl
    mtl-lymphnet --config configs/exp1_internal_cv.yaml benchmark

Global options (--config/--data-dir/--output-dir/--mode/--seed) must be given
BEFORE the subcommand name, e.g. `mtl-lymphnet --mode debug run --experiment all`.

Bug B1 fix -- running a sweep of external-validation configurations:
    Use --backbone / --loss-strategy / --w-seg / --seeds to override a
    single base config (e.g. configs/exp2_external_validation.yaml) without
    editing YAML files or mutating source via switch_backbone.py. Each
    distinct combination gets its own checkpoint, TensorBoard run, and
    result-dict entry (see experiments/external_validation.py), so a sweep
    like:

        for bb in tu-tiny_vit_5m_224.dist_in22k_ft_in1k tu-mobilenetv3_large_100 tu-efficientnet_b0; do
            mtl-lymphnet --config configs/exp2_external_validation.yaml run \\
                --experiment external_validation --backbone $bb --seeds 42,43,44
        done

    will not overwrite earlier iterations. Omitting all three overrides
    reproduces the original single-configuration behavior exactly, storing
    results under the original `exp2_c1_to_c2` / `exp2_c2_to_c1` keys. Every
    swept configuration (accumulated under `results["exp2_variants"]`) is
    automatically picked up by `export-tables`, `stats`, and `calibration`
    below -- no extra flags needed at reporting time.

R1.11 fix -- `benchmark` now covers every backbone x both devices by
    default in a single invocation (see cmd_benchmark), instead of
    silently overwriting one `benchmark_result.json` per run.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .config import Config
from .env import print_environment_summary, setup_environment


def _load_config(args: argparse.Namespace) -> Config:
    base_cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg, env = setup_environment(
        mode=args.mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed if args.seed is not None else base_cfg.seed,
        base_config=base_cfg,
    )
    print_environment_summary(cfg, env)
    return cfg


def _load_raw_yaml(path: str | None) -> dict:
    """Read experiment-level keys (e.g. loss_strategy, w_seg) that live in the
    YAML file but are not part of `Config` itself."""
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}



def _load_master_df(cfg: Config) -> pd.DataFrame:
    from .data.inventory import build_dataset_inventory

    df = build_dataset_inventory(cfg.data_dir)
    if df.empty:
        raise RuntimeError(f"No samples found under {cfg.data_dir}. Check --data-dir.")
    return df


def _iter_external_variants(results: dict):
    """Yield (label, direction, ext_result) for every external-validation
    result present -- both the primary/default (`exp2_c1_to_c2` /
    `exp2_c2_to_c1`) and any backbone/loss-strategy sweep configurations
    accumulated under `results["exp2_variants"]` (Bug B1 fix). Reporting
    code (stats, tables, calibration) should iterate via this helper rather
    than hardcoding the two primary keys, so a sweep run is picked up
    automatically without further code changes.
    """
    for key, direction in (("exp2_c1_to_c2", "C1->C2"), ("exp2_c2_to_c1", "C2->C1")):
        if key in results:
            yield "default", direction, results[key]
    for variant_key, directions in results.get("exp2_variants", {}).items():
        for direction, ext_result in directions.items():
            yield variant_key, direction, ext_result


# --------------------------------------------------------------------------- #
# eda
# --------------------------------------------------------------------------- #
def cmd_eda(args: argparse.Namespace) -> None:
    from .data.inventory import validate_inventory

    cfg = _load_config(args)
    df = _load_master_df(cfg)
    report = validate_inventory(df, verify_readable=not args.skip_verify)

    print("\n" + "=" * 50)
    print("  Dataset Inventory")
    print("=" * 50)
    print(f"  Total samples   : {report['n_total']}")
    print(f"  Missing masks   : {report['n_missing_mask']}")
    print(f"  Missing lesions : {report['n_missing_lesion']}")
    print(f"  Corrupt images  : {len(report['corrupt_images'])}")
    dist = df.groupby(["center", "label_name"]).size().unstack(fill_value=0)
    print("\n" + dist.to_string())

    out_csv = os.path.join(cfg.output_dir, "dataset_inventory.csv")
    os.makedirs(cfg.output_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nInventory saved to {out_csv}")


def _build_json_summary(results: dict) -> dict:
    """Build the lightweight (means/stds only) JSON summary from the full
    results dict. Factored out of cmd_run so it can also be called directly
    -- e.g. to regenerate `experiment_results.json` from an existing
    `experiment_results.pkl` without re-running any training (see Bug B4).
    """
    from .experiments.base import summarize_folds

    json_summary: dict = {}
    for k, v in results.items():
        if k == "exp2_variants":
            # Bug B4 fix (found after first real Kaggle run): this is a
            # nested {variant_key: {direction: ext_result_dict}} structure
            # -- NOT a dict of fold-result lists -- so it needs its own
            # branch instead of falling through to the generic
            # `summarize_folds` branch below, which assumes every dict value
            # is a list of per-fold dicts and crashed with `KeyError: 0` on
            # the first real sweep.
            json_summary[k] = {
                variant_key: {
                    direction: (
                        ext_result["summary"] if ext_result.get("summary")
                        else {mk: mv for mk, mv in ext_result["results"].items() if not str(mk).startswith("_")}
                    )
                    for direction, ext_result in directions.items()
                }
                for variant_key, directions in v.items()
            }
        elif isinstance(v, dict) and "summary" in v:
            json_summary[k] = v["summary"]
        elif isinstance(v, dict) and "results" in v:
            json_summary[k] = {mk: mv for mk, mv in v["results"].items() if not str(mk).startswith("_")}
        elif isinstance(v, dict):
            json_summary[k] = {vk: summarize_folds(vv) for vk, vv in v.items()}
    return json_summary


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> None:
    from .experiments.ablation_backbone import run_ablation_backbone
    from .experiments.ablation_loss import run_ablation_loss
    from .experiments.ablation_mtl import run_ablation_mtl
    from .experiments.base import load_pickle, save_json, save_pickle, summarize_folds
    from .experiments.external_validation import run_external_validation
    from .experiments.internal_cv import run_internal_cv

    cfg = _load_config(args)
    df = _load_master_df(cfg)
    df_c1 = df[df["center"] == "Center 1"].reset_index(drop=True)
    df_c2 = df[df["center"] == "Center 2"].reset_index(drop=True)

    raw_cfg = _load_raw_yaml(args.config)
    loss_strategy = raw_cfg.get("loss_strategy", "uncertainty")
    w_seg = raw_cfg.get("w_seg", 0.5)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    exp1_path = os.path.join(cfg.output_dir, "exp1_results.pkl")

    # Load previously saved results so experiments ACCUMULATE across separate
    # CLI invocations (e.g. separate notebook cells) instead of overwriting.
    # This was the root cause of the "incomplete outputs" bug: each invocation
    # started with results={}, ran one experiment, then overwrote the pkl.
    out_pkl = os.path.join(cfg.output_dir, "experiment_results.pkl")
    if os.path.exists(out_pkl):
        results: dict = load_pickle(out_pkl)
        print(f"Loaded {len(results)} existing experiment(s) from {out_pkl}")
    else:
        results: dict = {}

    def ensure_exp1() -> dict:
        if os.path.exists(exp1_path) and not args.force_rerun_exp1:
            print(f"Loading cached Experiment 1 results from {exp1_path}")
            return load_pickle(exp1_path)
        exp1 = run_internal_cv(cfg, df, experiment_name="exp1_mtl_main", loss_strategy=loss_strategy, w_seg=w_seg, seeds=seeds)
        save_pickle(exp1, exp1_path)
        return exp1

    if args.experiment in ("internal_cv", "all"):
        results["exp1_internal_cv"] = ensure_exp1()

    if args.experiment in ("external_validation", "all"):
        # Bug B1 fix: --backbone / --loss-strategy / --w-seg let a single
        # base config be swept across configurations (e.g. all 3 backbones,
        # or all loss-weighting strategies) without editing YAML files.
        # When none of these overrides (nor multi-seed) are given, behavior
        # is UNCHANGED from before -- results land in the original
        # exp2_c1_to_c2 / exp2_c2_to_c1 keys. Otherwise, results are kept
        # separate under exp2_variants[<variant_key>] so they never collide
        # with the primary run or with each other.
        ext_backbone = args.backbone or None
        ext_strategy = args.loss_strategy or loss_strategy
        ext_w_seg = args.w_seg if args.w_seg is not None else w_seg
        # NOTE: a --seeds override alone does NOT count as a sweep config --
        # it upgrades the SAME (default backbone, default strategy) primary
        # result to multi-seed in place. Only an explicit --backbone /
        # --loss-strategy / --w-seg override represents a genuinely
        # different configuration that needs to be tracked separately under
        # exp2_variants (otherwise a plain multi-seed re-run of the default
        # config would show up as a confusing duplicate "variant" row
        # alongside the now-stale single-seed exp2_c1_to_c2/exp2_c2_to_c1).
        is_sweep_config = bool(args.backbone or args.loss_strategy or args.w_seg is not None)

        r_c1c2 = run_external_validation(
            cfg, df_c1, df_c2, "C1->C2",
            backbone=ext_backbone, loss_strategy=ext_strategy, w_seg=ext_w_seg, seeds=seeds,
        )
        r_c2c1 = run_external_validation(
            cfg, df_c2, df_c1, "C2->C1",
            backbone=ext_backbone, loss_strategy=ext_strategy, w_seg=ext_w_seg, seeds=seeds,
        )

        if is_sweep_config:
            variant_key = r_c1c2["variant_key"]
            results.setdefault("exp2_variants", {})[variant_key] = {"C1->C2": r_c1c2, "C2->C1": r_c2c1}
            print(f"\nStored under exp2_variants['{variant_key}'] "
                  f"(non-default config -- kept separate from exp2_c1_to_c2/exp2_c2_to_c1).")
        else:
            results["exp2_c1_to_c2"] = r_c1c2
            results["exp2_c2_to_c1"] = r_c2c1

    if args.experiment in ("ablation_mtl", "all"):
        results["exp3_ablation_mtl"] = run_ablation_mtl(cfg, df, results.get("exp1_internal_cv") or ensure_exp1())

    if args.experiment in ("ablation_backbone", "all"):
        results["exp4_ablation_backbone"] = run_ablation_backbone(cfg, df, results.get("exp1_internal_cv") or ensure_exp1())

    if args.experiment in ("ablation_loss", "all"):
        results["exp5_ablation_loss"] = run_ablation_loss(cfg, df, results.get("exp1_internal_cv") or ensure_exp1())


    save_pickle(results, out_pkl)
    print(f"\nFull results ({len(results)} experiment(s), incl. per-image arrays, weight trajectories) saved to {out_pkl}")


    json_summary = _build_json_summary(results)
    out_json = os.path.join(cfg.output_dir, "experiment_results.json")
    save_json(json_summary, out_json)
    print(f"Lightweight summary (means/stds) saved to {out_json}")


# --------------------------------------------------------------------------- #
# smoke-test
# --------------------------------------------------------------------------- #
def cmd_smoke_test(args: argparse.Namespace) -> None:
    """CPU-friendly end-to-end pipeline check: tiny subset, few epochs, 2 folds.

    Exercises the full leakage-safe path (nested split -> fit -> held-out test
    evaluation) without requiring a GPU.
    """
    from .experiments.internal_cv import run_internal_cv

    cfg = _load_config(args)
    cfg = cfg.replace(
        mode="debug", num_epochs=args.epochs, n_folds=2, batch_size=2,
        subset_fraction=1.0, use_amp=False, patience=5, num_workers=0,
        img_size=args.img_size, pretrained="imagenet" if args.pretrained else None,
    )

    df = _load_master_df(cfg)
    df_tiny = (
        df.groupby("label", group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(n=min(args.n_per_class, len(g)), random_state=cfg.seed))
        .reset_index(drop=True)
    )
    print(f"Smoke test: {len(df_tiny)} samples, {cfg.n_folds} folds, {cfg.num_epochs} epochs, device={cfg.device}")

    exp1 = run_internal_cv(cfg, df_tiny, experiment_name="smoke_test")

    for split in exp1["fold_results"]:
        assert "_n_train" in split and "_n_val" in split and "_n_test" in split
    print("\nSmoke test PASSED -- nested train/val/test split executed without leakage.")
    print(f"  Dice: {exp1['summary']['seg_dice']['mean']:.4f}")
    print(f"  AUC : {exp1['summary']['cls_auc']['mean']:.4f}")


# --------------------------------------------------------------------------- #
# export-tables / export-figures / stats / calibration
# --------------------------------------------------------------------------- #
def cmd_export_tables(args: argparse.Namespace) -> None:
    from .experiments.base import load_pickle
    from .reporting.tables import (
        build_backbone_ablation_table,
        build_external_validation_table,
        build_internal_cv_table,
        build_internal_cv_table_compact,
        build_internal_cv_table_detailed,
        build_loss_ablation_table,
        build_mtl_ablation_table,
        to_latex_booktabs,
    )

    results = load_pickle(args.results)
    tables_dir = args.output_dir or os.path.join(os.path.dirname(args.results), "tables")
    exp1 = results.get("exp1_internal_cv")

    if exp1:
        # Detailed: per-seed, per-fold + sub-means + pooled
        df_det, mr_det = build_internal_cv_table_detailed(exp1)
        to_latex_booktabs(
            df_det,
            "Internal cross-validation results -- per-seed breakdown (leakage-safe nested split).",
            "tab:internal_cv_detailed", "table2_internal_cv_detailed.tex",
            tables_dir, midrule_after=mr_det,
            note="Each seed uses 5-fold cross-validation. Pooled row aggregates all seed$\\times$fold runs.")
        print("Saved table2_internal_cv_detailed.tex")

        # Compact: per-seed sub-means + pooled only
        df_cmp, mr_cmp = build_internal_cv_table_compact(exp1)
        to_latex_booktabs(
            df_cmp,
            "Internal cross-validation results -- seed stability summary.",
            "tab:internal_cv_compact", "table2_internal_cv_compact.tex",
            tables_dir, midrule_after=mr_cmp)
        print("Saved table2_internal_cv_compact.tex")

        # Legacy flat table (backward compat)
        df_t2 = build_internal_cv_table(exp1["fold_results"])
        to_latex_booktabs(
            df_t2, "Internal cross-validation results (leakage-safe nested split).",
            "tab:internal_cv", "table2_internal_cv.tex", tables_dir)
        print("Saved table2_internal_cv.tex")

    if all(k in results for k in ("exp1_internal_cv", "exp2_c1_to_c2", "exp2_c2_to_c1")):
        # Bug B1 fix: fold in any backbone/loss-strategy sweep variants as
        # extra rows, so table3 covers the full R1.4/R1.5 comparison once a
        # sweep has been run -- no code change needed at that point.
        extra_variants = {
            f"{variant_key} [{direction}]": ext_result
            for variant_key, direction, ext_result in _iter_external_variants(results)
            if variant_key != "default"
        }
        df_t3 = build_external_validation_table(
            results["exp1_internal_cv"]["summary"],
            results["exp2_c1_to_c2"], results["exp2_c2_to_c1"],
            extra_variants=extra_variants or None)
        to_latex_booktabs(
            df_t3, "Bidirectional external validation results.",
            "tab:external_val", "table3_external_val.tex", tables_dir,
            note=("Additional rows are backbone/loss-strategy sweep configurations "
                  "(see Bug B1, exp2_variants)." if extra_variants else None))
        print(f"Saved table3_external_val.tex ({len(extra_variants)} extra variant row(s))"
              if extra_variants else "Saved table3_external_val.tex")

    if "exp3_ablation_mtl" in results:
        df_t4, mr4 = build_mtl_ablation_table(results["exp3_ablation_mtl"], exp1)
        to_latex_booktabs(
            df_t4, "Ablation study: multi-task learning vs.\\ single-task models.",
            "tab:ablation_mtl", "table4_ablation_mtl.tex", tables_dir,
            midrule_after=mr4,
            note="Proposed model reported with all seeds (n=15) and primary seed (n=5) for fair comparison.")
        print("Saved table4_ablation_mtl.tex")

    if "exp4_ablation_backbone" in results:
        df_t5, mr5 = build_backbone_ablation_table(results["exp4_ablation_backbone"], exp1)
        to_latex_booktabs(
            df_t5, "Ablation study: encoder backbone comparison.",
            "tab:ablation_backbone", "table5_ablation_backbone.tex", tables_dir,
            midrule_after=mr5,
            note="Proposed backbone reported with all seeds and primary seed for fair comparison.")
        print("Saved table5_ablation_backbone.tex")

    if "exp5_ablation_loss" in results:
        df_t6, mr6 = build_loss_ablation_table(results["exp5_ablation_loss"], exp1)
        to_latex_booktabs(
            df_t6, "Ablation study: MTL loss weighting strategies.",
            "tab:ablation_loss", "table6_ablation_loss.tex", tables_dir,
            midrule_after=mr6,
            note="Uncertainty (proposed) reported with all seeds and primary seed for fair comparison.")
        print("Saved table6_ablation_loss.tex")

    print(f"\nTables written to {tables_dir}")


def cmd_export_figures(args: argparse.Namespace) -> None:
    from .experiments.base import load_pickle
    from .reporting.figures import (
        plot_backbone_pareto,
        plot_external_validation_bars,
        plot_fold_auc_bar,
        plot_kfold_boxplot,
        plot_loss_ablation_bars,
        plot_training_curves,
        plot_weight_trajectory,
    )
    from .reporting.tables import build_weight_trajectory_table

    results = load_pickle(args.results)
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.results), "figures")

    if "exp1_internal_cv" in results:
        exp1 = results["exp1_internal_cv"]
        plot_kfold_boxplot(exp1["fold_results"], os.path.join(out_dir, "fig_kfold_boxplot.png"))
        plot_training_curves(exp1["fold_histories"], os.path.join(out_dir, "fig_training_curves.png"))
        plot_fold_auc_bar(exp1["fold_results"], os.path.join(out_dir, "fig_fold_auc.png"))
        wdf = build_weight_trajectory_table(exp1["fold_results"])
        if not wdf.empty:
            plot_weight_trajectory(wdf, os.path.join(out_dir, "fig_weight_trajectory.png"), strategy_name="Uncertainty")
        print("Saved Experiment 1 figures.")

    if all(k in results for k in ("exp1_internal_cv", "exp2_c1_to_c2", "exp2_c2_to_c1")):
        # NOTE: plot_external_validation_bars still only plots the primary
        # (default-config) C1->C2/C2->C1 pair -- its bar-width layout is
        # hardcoded for 3 scenarios. Extending it to N backbone/strategy
        # sweep variants is left for a later, deliberate figure redesign
        # once real sweep data exists (not done as part of Bug B1 / Step
        # 1.5), rather than generalized here.
        plot_external_validation_bars(
            results["exp1_internal_cv"]["summary"], results["exp2_c1_to_c2"], results["exp2_c2_to_c1"],
            os.path.join(out_dir, "fig_external_validation.png"),
        )
        print("Saved external validation figure (primary/default config only -- see note in source).")

    if "exp4_ablation_backbone" in results:
        plot_backbone_pareto(results["exp4_ablation_backbone"], os.path.join(out_dir, "fig_backbone_pareto.png"))
        print("Saved backbone pareto figure.")

    if "exp5_ablation_loss" in results:
        plot_loss_ablation_bars(results["exp5_ablation_loss"], os.path.join(out_dir, "fig_loss_ablation.png"))
        print("Saved loss ablation figure.")

    print(f"\nFigures written to {out_dir}")


def _pool_dice(fold_results: list[dict]) -> np.ndarray:
    """Pool per-image Dice scores across folds into a single array."""
    vals: list[float] = []
    for fr in fold_results:
        vals.extend(fr.get("_per_image_dice", []))
    return np.array(vals)


def _pool_cls(fold_results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pool classification probabilities and targets across folds."""
    probs, targets = [], []
    for fr in fold_results:
        probs.extend(fr.get("_cls_probs", []))
        targets.extend(fr.get("_cls_targets", []))
    return np.array(probs), np.array(targets)


def _get_primary_seed_folds(exp1_data: dict) -> list[dict]:

    """Extract fold results from the primary (first) seed only.

    When exp1 is run with multiple seeds (R1.6), `fold_results` contains
    N_seeds × N_folds entries.  Paired tests (Wilcoxon, DeLong) require
    equal-length arrays, and ablation experiments use only the primary seed.
    This function returns the primary seed's folds so comparisons are fair.
    """
    results_by_seed = exp1_data.get("results_by_seed", {})
    seeds = exp1_data.get("seeds", [])

    if results_by_seed and seeds:
        primary_seed = seeds[0]
        return results_by_seed[primary_seed]

    # Fallback: single-seed run, fold_results is already 5 entries
    return exp1_data["fold_results"]


def cmd_stats(args: argparse.Namespace) -> None:
    from .experiments.base import load_pickle, summarize_folds
    from .stats.bootstrap import bootstrap_auc_ci, bootstrap_ci, bootstrap_paired_diff_ci
    from .stats.significance import pairwise_delong, pairwise_wilcoxon

    results = load_pickle(args.results)
    out_dir = args.output_dir or os.path.dirname(args.results)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Multi-seed stability report (R1.6)
    # ------------------------------------------------------------------ #
    if "exp1_internal_cv" in results:
        exp1 = results["exp1_internal_cv"]
        seeds = exp1.get("seeds", [])
        results_by_seed = exp1.get("results_by_seed", {})

        if len(seeds) > 1:
            print(f"\n{'=' * 60}")
            print(f"  Multi-seed stability report ({len(seeds)} seeds: {seeds})")
            print(f"{'=' * 60}")
            for seed in seeds:
                seed_summary = summarize_folds(results_by_seed[seed])
                dice_m = seed_summary["seg_dice"]["mean"]
                dice_s = seed_summary["seg_dice"]["std"]
                auc_m = seed_summary["cls_auc"]["mean"]
                auc_s = seed_summary["cls_auc"]["std"]
                print(f"  Seed {seed}: Dice={dice_m:.4f}±{dice_s:.4f}  AUC={auc_m:.4f}±{auc_s:.4f}")

            all_summary = exp1["summary"]
            print(f"  {'─' * 50}")
            print(f"  Pooled:   Dice={all_summary['seg_dice']['mean']:.4f}±{all_summary['seg_dice']['std']:.4f}"
                  f"  AUC={all_summary['cls_auc']['mean']:.4f}±{all_summary['cls_auc']['std']:.4f}")
            print()

    # ------------------------------------------------------------------ #
    # Paired significance tests (use primary seed only for equal lengths)
    # ------------------------------------------------------------------ #
    dice_participants: dict[str, np.ndarray] = {}
    auc_probs: dict[str, np.ndarray] = {}
    auc_targets: dict[str, np.ndarray] = {}

    if "exp1_internal_cv" in results:
        # Use primary seed only for fair paired comparison with ablation runs
        primary_folds = _get_primary_seed_folds(results["exp1_internal_cv"])
        dice_participants["MTL Proposed"] = _pool_dice(primary_folds)
        p, t = _pool_cls(primary_folds)
        auc_probs["MTL Proposed"], auc_targets["MTL Proposed"] = p, t

    if "exp3_ablation_mtl" in results:
        dice_participants["Seg-Only"] = _pool_dice(results["exp3_ablation_mtl"]["seg_only"])
        p, t = _pool_cls(results["exp3_ablation_mtl"]["cls_only"])
        auc_probs["Cls-Only"], auc_targets["Cls-Only"] = p, t

    if "exp4_ablation_backbone" in results:
        from .models.backbones import variant_names
        for bb in variant_names():
            if bb in results["exp4_ablation_backbone"]:
                dice_participants[f"MTL {bb}"] = _pool_dice(results["exp4_ablation_backbone"][bb])
                p, t = _pool_cls(results["exp4_ablation_backbone"][bb])
                auc_probs[f"MTL {bb}"], auc_targets[f"MTL {bb}"] = p, t

    if "exp5_ablation_loss" in results:
        for strat in ("Fixed (0.5/0.5)", "Fixed (0.7/0.3)", "DWA"):
            if strat in results["exp5_ablation_loss"]:
                dice_participants[f"MTL {strat}"] = _pool_dice(results["exp5_ablation_loss"][strat])
                p, t = _pool_cls(results["exp5_ablation_loss"][strat])
                auc_probs[f"MTL {strat}"], auc_targets[f"MTL {strat}"] = p, t

    # Log array lengths for debugging
    print("\nPaired test array lengths:")
    for name, arr in dice_participants.items():
        print(f"  {name}: {len(arr)} per-image Dice values")

    if len(dice_participants) >= 2:
        df_dice = pairwise_wilcoxon(dice_participants)
        df_dice.to_csv(os.path.join(out_dir, "stats_dice_pairwise.csv"), index=False)
        print(f"\nFriedman chi2={df_dice.attrs.get('friedman_stat')}, p={df_dice.attrs.get('friedman_p')}")
        print(df_dice.to_string(index=False))

    if len(auc_probs) >= 2:
        df_auc = pairwise_delong(auc_probs, auc_targets)
        df_auc.to_csv(os.path.join(out_dir, "stats_auc_pairwise.csv"), index=False)
        print(df_auc.to_string(index=False))

    if "MTL Proposed" in dice_participants and "Seg-Only" in dice_participants:
        a, b = dice_participants["MTL Proposed"], dice_participants["Seg-Only"]
        if len(a) == len(b):
            ci = bootstrap_paired_diff_ci(a, b)
            print(f"\nDelta Dice (MTL Proposed - Seg-Only), bootstrap 95% CI: {ci}")
        else:
            print(f"\nSkipping paired Dice CI: mismatched lengths ({len(a)} vs {len(b)}).")

    # ------------------------------------------------------------------ #
    # External validation confidence intervals (R1.6; Step 1.5 / Bug B1
    # follow-up). Iterates BOTH the legacy exp2_c1_to_c2/exp2_c2_to_c1 keys
    # AND any exp2_variants[...] produced by a --backbone/--loss-strategy
    # sweep (see _iter_external_variants), so every external configuration
    # gets CI automatically -- no re-inference from checkpoints needed, since
    # `_cls_probs`/`_cls_targets`/`_per_image_dice` are already saved in each
    # result's `results` dict by engine/trainer.py::validate().
    # ------------------------------------------------------------------ #
    ci_rows = []
    for variant_label, direction, ext_result in _iter_external_variants(results):
        r = ext_result.get("results", {})
        probs = np.array(r.get("_cls_probs", []))
        targets = np.array(r.get("_cls_targets", []))
        dice_vals = np.array(r.get("_per_image_dice", []))

        row = {
            "variant": variant_label,
            "direction": direction,
            "backbone": ext_result.get("backbone", ""),
            "loss_strategy": ext_result.get("loss_strategy", ""),
            "seeds": ",".join(str(s) for s in ext_result.get("seeds", [])),
            "n_test": ext_result.get("n_test", len(targets)),
        }

        if len(probs) > 0 and len(np.unique(targets)) >= 2:
            auc_ci = bootstrap_auc_ci(probs, targets)
            row["auc_primary_seed"] = auc_ci["mean"]
            row["auc_ci_low"] = auc_ci["ci_low"]
            row["auc_ci_high"] = auc_ci["ci_high"]
        if len(dice_vals) > 0:
            dice_ci = bootstrap_ci(dice_vals, statistic=np.mean)
            row["dice_primary_seed"] = dice_ci["mean"]
            row["dice_ci_low"] = dice_ci["ci_low"]
            row["dice_ci_high"] = dice_ci["ci_high"]

        # Across-seed variability (training variance), when multi-seed.
        summary = ext_result.get("summary", {})
        if "cls_auc" in summary:
            row["auc_seed_mean"] = summary["cls_auc"]["mean"]
            row["auc_seed_std"] = summary["cls_auc"]["std"]
        if "seg_dice" in summary:
            row["dice_seed_mean"] = summary["seg_dice"]["mean"]
            row["dice_seed_std"] = summary["seg_dice"]["std"]

        ci_rows.append(row)

    if ci_rows:
        df_ext_ci = pd.DataFrame(ci_rows)
        df_ext_ci.to_csv(os.path.join(out_dir, "stats_external_ci.csv"), index=False)
        print(f"\nExternal validation CI (R1.6), {len(ci_rows)} configuration(s):")
        print(df_ext_ci.to_string(index=False))
        print("Saved stats_external_ci.csv")

    print(f"\nStatistics written to {out_dir}")



def cmd_calibration(args: argparse.Namespace) -> None:
    from .calibration.reliability import calibration_summary
    from .experiments.base import load_pickle
    from .reporting.figures import plot_reliability_grid

    results = load_pickle(args.results)
    out_dir = args.output_dir or os.path.dirname(args.results)

    scenarios: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "exp1_internal_cv" in results:
        # Use primary seed only to avoid counting the same images multiple times
        primary_folds = _get_primary_seed_folds(results["exp1_internal_cv"])
        scenarios["Internal CV"] = _pool_cls(primary_folds)

    # Bug B1 follow-up: covers the primary exp2_c1_to_c2/exp2_c2_to_c1 pair
    # AND every backbone/loss-strategy sweep variant under exp2_variants.
    # plot_reliability_grid already lays out an arbitrary number of panels
    # (min(3, n) columns), so no figure-layout change was needed here.
    for variant_label, direction, ext_result in _iter_external_variants(results):
        r = ext_result.get("results", {})
        probs, targets = np.array(r.get("_cls_probs", [])), np.array(r.get("_cls_targets", []))
        label = direction if variant_label == "default" else f"{variant_label} [{direction}]"
        scenarios[label] = (probs, targets)

    summary_rows = []
    for name, (probs, targets) in scenarios.items():
        if len(probs) == 0:
            continue
        summary = calibration_summary(probs, targets)
        summary["scenario"] = name
        summary_rows.append(summary)
        print(f"{name}: ECE={summary['ece']:.4f}, MCE={summary['mce']:.4f}, {summary['verdict']}")

    if summary_rows:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)

    valid_scenarios = {k: v for k, v in scenarios.items() if len(v[0]) > 0}
    if valid_scenarios:
        plot_reliability_grid(valid_scenarios, os.path.join(out_dir, "figures", "reliability_diagrams.png"))
        print(f"Saved reliability_diagrams.png ({len(valid_scenarios)} panel(s))")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Standardized inference benchmark (R1.11).

    Bug-B1-style fix: the original implementation benchmarked a single
    (backbone, device) pair -- whichever `cfg.backbone` and `cfg.device`
    happened to resolve to -- and always overwrote the same
    `benchmark_result.json`, so re-running for a different backbone or
    device silently discarded the previous result. By default this now
    benchmarks EVERY backbone in the registry on BOTH cpu and cuda (when
    available) in a single invocation, saving the full list to
    `benchmark_results.json` (plural) so R1.11's backbone/device comparison
    table can be built from one file with nothing overwritten.

    Use --backbone / --device to restrict to a single configuration (e.g.
    for a quick check) -- that single result is still appended to the same
    `benchmark_results.json` list rather than replacing it, unless
    --output-dir points elsewhere.
    """
    import json

    import torch

    from .benchmarking.inference_benchmark import benchmark_inference
    from .experiments.base import save_json
    from .models.backbones import BACKBONES
    from .models.mtl_unet import MTLUNet

    cfg = _load_config(args)

    if args.backbone:
        backbones = {args.backbone: BACKBONES.get(args.backbone, args.backbone)}
    else:
        backbones = dict(BACKBONES)

    if args.device:
        devices = [args.device]
    else:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    out_path = os.path.join(cfg.output_dir, "benchmark_results.json")
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        if not isinstance(all_results, list):
            all_results = []
    else:
        all_results = []

    new_results = []
    for short_name, timm_key in backbones.items():
        for device in devices:
            print(f"\nBenchmarking {short_name} ({timm_key}) on {device}...")
            model = MTLUNet(backbone=timm_key, pretrained=None)
            result = benchmark_inference(
                model, (3, cfg.img_size, cfg.img_size), device=device,
                batch_size=args.batch_size, precision=args.precision,
                n_warmup=args.warmup, n_reps=args.reps,
            )
            row = result.to_dict()
            row["backbone"] = short_name
            row["backbone_timm_key"] = timm_key
            print(f"  mean={row['mean_ms']:.2f}ms  p95={row['p95_ms']:.2f}ms  throughput={row['throughput_ips']:.1f} img/s")
            new_results.append(row)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Replace any existing entry for the same (backbone, device) pair instead
    # of blindly appending duplicates across repeated invocations.
    existing = {(r.get("backbone"), r.get("device")): r for r in all_results}
    for r in new_results:
        existing[(r["backbone"], r["device"])] = r
    all_results = list(existing.values())

    save_json(all_results, out_path)
    print(f"\nSaved {len(all_results)} benchmark configuration(s) total to {out_path}")


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtl-lymphnet", description="Modular MTL-LymphNet framework")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["debug", "full"])
    parser.add_argument("--seed", type=int, default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    p_eda = sub.add_parser("eda", help="Scan and validate the dataset inventory")
    p_eda.add_argument("--skip-verify", action="store_true", help="Skip per-image readability verification")
    p_eda.set_defaults(func=cmd_eda)

    p_run = sub.add_parser("run", help="Run one or all experiments (leakage-safe nested split)")
    p_run.add_argument(
        "--experiment",
        choices=["internal_cv", "external_validation", "ablation_mtl", "ablation_backbone", "ablation_loss", "all"],
        default="all",
    )
    p_run.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds for multi-seed runs (internal_cv and/or external_validation), e.g. 42,43,44")
    p_run.add_argument("--force-rerun-exp1", action="store_true", help="Ignore any cached exp1_results.pkl")
    p_run.add_argument("--backbone", type=str, default=None, help="[external_validation only, Bug B1 fix] Override backbone (short registry name e.g. MobileNetV3-Large, or a raw timm model string) for this run, without editing config files. Results are kept separate under exp2_variants[...] instead of overwriting exp2_c1_to_c2/exp2_c2_to_c1.")
    p_run.add_argument("--loss-strategy", type=str, default=None, help="[external_validation only, Bug B1 fix] Override loss-weighting strategy (e.g. uncertainty, fixed, dwa) for this run.")
    p_run.add_argument("--w-seg", type=float, default=None, help="[external_validation only, Bug B1 fix] Override fixed segmentation loss weight (used when --loss-strategy fixed).")
    p_run.set_defaults(func=cmd_run)

    p_smoke = sub.add_parser("smoke-test", help="CPU-friendly quick pipeline sanity check")
    p_smoke.add_argument("--epochs", type=int, default=2)
    p_smoke.add_argument("--n-per-class", type=int, default=12)
    p_smoke.add_argument("--img-size", type=int, default=128, help="Smaller than the default 256 for CPU speed")
    p_smoke.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False, help="Download ImageNet weights (default: off, for offline-friendly speed)")
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_tables = sub.add_parser("export-tables", help="Export LaTeX tables from saved results")
    p_tables.add_argument("--results", type=str, required=True, help="Path to experiment_results.pkl")
    p_tables.set_defaults(func=cmd_export_tables)

    p_figs = sub.add_parser("export-figures", help="Export figures from saved results")
    p_figs.add_argument("--results", type=str, required=True)
    p_figs.set_defaults(func=cmd_export_figures)

    p_stats = sub.add_parser("stats", help="Run statistical significance tests (Wilcoxon/Friedman/DeLong + bootstrap CI)")
    p_stats.add_argument("--results", type=str, required=True)
    p_stats.set_defaults(func=cmd_stats)

    p_calib = sub.add_parser("calibration", help="Run calibration analysis (ECE/MCE/reliability diagrams)")
    p_calib.add_argument("--results", type=str, required=True)
    p_calib.set_defaults(func=cmd_calibration)

    p_bench = sub.add_parser("benchmark", help="Standardized inference benchmark, all backbones x cpu/cuda by default (R1.11)")
    p_bench.add_argument("--batch-size", type=int, default=1)
    p_bench.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p_bench.add_argument("--warmup", type=int, default=10)
    p_bench.add_argument("--reps", type=int, default=100)
    p_bench.add_argument("--backbone", type=str, default=None, help="Restrict to a single backbone (short registry name, e.g. TinyViT-5M). Default: benchmark every registered backbone.")
    p_bench.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Restrict to a single device. Default: benchmark cpu, and cuda if available.")
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
