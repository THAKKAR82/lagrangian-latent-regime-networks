#!/usr/bin/env python
"""Leakage-safe model selection evaluation.

For each fold t, selects a model using only historical validation performance
(folds < t), then evaluates that model's test artifact. No test-fold labels
are used for selection decisions.

Candidate models must have prediction artifacts saved under predictions/<model>/.
Val artifacts (fold_XX_val.parquet) drive selection; test artifacts
(fold_XX_test.parquet) are evaluated.

Usage:
    python scripts/model_selector.py
    python scripts/model_selector.py --fold-start 20 --fold-end 40
    python scripts/model_selector.py --models xgb ensemble forced_scalar_damped_lagrangian
    python scripts/model_selector.py --rolling-k 3 --alpha 0.3
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import evaluate
from src.evaluation.model_selector import METHODS, ModelSelector, compute_val_metrics
from src.evaluation.predictions import PredictionArtifact

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_DEFAULT_MODELS = [
    "xgb",
    "ensemble",
    "lagrangian_conv1d",
    "forced_scalar_damped_lagrangian",
]

# Human-readable names written to walk_forward_summary_*.csv for compare_all.py
_METHOD_DISPLAY = {
    "previous_best_macro_f1": "model_selector_previous_best",
    "rolling_best_macro_f1": "model_selector_rolling_best",
    "previous_best_calibrated_score": "model_selector_calibrated",
    "rolling_best_calibrated_score": "model_selector_rolling_calibrated",
    "fallback_static": "model_selector_fallback_static",
}


def _audit_artifacts(pred_root: Path, models: list[str]) -> tuple[list[str], list[str]]:
    available, missing = [], []
    for m in models:
        (available if (pred_root / m).exists() else missing).append(m)
    return available, missing


def _discover_folds(pred_root: Path, models: list[str]) -> list[int]:
    fold_ids: set[int] = set()
    for model in models:
        for p in (pred_root / model).glob("fold_*_test.parquet"):
            try:
                fold_ids.add(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return sorted(fold_ids)


def _load_artifact(pred_root: Path, model: str, fold_id: int, split: str) -> PredictionArtifact | None:
    path = pred_root / model / f"fold_{fold_id:02d}_{split}.parquet"
    return PredictionArtifact.load(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe model selector evaluation.")
    parser.add_argument("--models", nargs="+", default=_DEFAULT_MODELS,
                        help="Candidate model names (must match predictions/<model>/ dirs)")
    parser.add_argument("--fold-start", type=int, default=None)
    parser.add_argument("--fold-end", type=int, default=None)
    parser.add_argument("--rolling-k", type=int, default=5,
                        help="Window size for rolling_* methods (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="ECE penalty weight for calibrated methods (default: 0.5)")
    parser.add_argument("--fallback", type=str, default="ensemble",
                        help="Model to use when no selection history exists (default: ensemble)")
    args = parser.parse_args()

    pred_root = _PROJECT_ROOT / "predictions"
    out_dir = _PROJECT_ROOT / "outputs" / "model_selector"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Artifact audit ---
    available_models, missing_models = _audit_artifacts(pred_root, args.models)
    if missing_models:
        for m in missing_models:
            log.warning(f"MISSING predictions/{m}/ — excluded from selection")
            if m == "lagrangian_conv1d":
                log.warning(
                    "  To generate: python -m src.training.train_lagrangian model=lagrangian_conv1d"
                )
            elif m == "xgb":
                log.warning(
                    "  To generate folds 0-70: python -m src.training.train_baseline"
                    " (then +fold_start/+fold_end for subsets)"
                )
            elif m == "ensemble":
                log.warning(
                    "  To generate folds 0-70: python -m src.training.train_ensemble"
                    " (then +fold_start/+fold_end for subsets)"
                )

    if not available_models:
        log.error("No models have any artifacts. Run training scripts first.")
        sys.exit(1)

    # Resolve fallback: if requested fallback is unavailable, use first available
    fallback = args.fallback if args.fallback in available_models else available_models[0]
    if fallback != args.fallback:
        log.warning(f"Requested fallback '{args.fallback}' not available; using '{fallback}'")

    # --- Fold discovery ---
    all_folds = _discover_folds(pred_root, available_models)
    folds = [
        f for f in all_folds
        if (args.fold_start is None or f >= args.fold_start)
        and (args.fold_end is None or f <= args.fold_end)
    ]
    if not folds:
        log.error(f"No folds found in [{args.fold_start}, {args.fold_end}]. Available: {all_folds}")
        sys.exit(1)

    log.info(f"Candidate models: {available_models}")
    log.info(f"Missing models (skipped): {missing_models}")
    log.info(f"Processing folds {folds[0]}–{folds[-1]} ({len(folds)} folds)")
    log.info(f"Methods: {list(METHODS)}")
    log.info(f"Fallback model: {fallback}")

    # --- Initialise one selector per method ---
    selectors: dict[str, ModelSelector] = {
        method: ModelSelector(
            models=available_models,
            method=method,
            k=args.rolling_k,
            alpha=args.alpha,
            fallback=fallback,
        )
        for method in METHODS
    }

    # Accumulate per-fold results per method
    fold_rows: dict[str, list[dict]] = defaultdict(list)
    # Per-fold baseline results (one row per model per fold)
    baseline_rows: list[dict] = []

    for fold_id in folds:
        log.info(f"--- Fold {fold_id} ---")

        # Step 1: Select model for this fold (uses only history from folds < fold_id)
        selections: dict[str, str] = {}
        selector_scores: dict[str, dict[str, float]] = {}
        for method, sel in selectors.items():
            selected, scores = sel.select()
            selections[method] = selected
            selector_scores[method] = scores

        # Step 2: Load test artifacts (cached per model to avoid re-reading)
        test_cache: dict[str, PredictionArtifact | None] = {}
        for model in available_models:
            test_cache[model] = _load_artifact(pred_root, model, fold_id, "test")

        # Step 3: Evaluate each method's selected model
        for method in METHODS:
            selected = selections[method]
            art = test_cache.get(selected)
            if art is None:
                log.warning(
                    f"  [{method}] selected={selected} — test artifact missing"
                    f" for fold {fold_id}, skipping"
                )
                continue

            y_pred = art.probs.argmax(axis=1)
            result = evaluate(art.true_labels, y_pred, art.probs)
            display = _METHOD_DISPLAY.get(method, method)

            fold_rows[method].append({
                "fold_id": fold_id,
                "method": method,
                "display_name": display,
                "selected_model": selected,
                "macro_f1": result.macro_f1,
                "balanced_accuracy": result.balanced_accuracy,
                "brier_score": result.brier_score,
                "ece": result.ece,
                "switch_frequency": result.switch_frequency,
                "mean_entropy": result.mean_entropy,
                "class_f1_0": float(result.class_f1[0]),
                "class_f1_1": float(result.class_f1[1]),
                "class_f1_2": float(result.class_f1[2]),
                "class_f1_3": float(result.class_f1[3]),
                **{
                    f"score_{m}": selector_scores[method].get(m, float("nan"))
                    for m in available_models
                },
            })
            log.info(
                f"  [{_METHOD_DISPLAY.get(method, method)}] selected={selected}"
                f" F1={result.macro_f1:.4f} Brier={result.brier_score:.4f} ECE={result.ece:.4f}"
            )

        # Step 4: Evaluate each baseline model (for comparison table)
        for model in available_models:
            art = test_cache.get(model)
            if art is None:
                continue
            y_pred = art.probs.argmax(axis=1)
            result = evaluate(art.true_labels, y_pred, art.probs)
            baseline_rows.append({
                "fold_id": fold_id,
                "model": model,
                "macro_f1": result.macro_f1,
                "balanced_accuracy": result.balanced_accuracy,
                "brier_score": result.brier_score,
                "ece": result.ece,
            })

        # Step 5: Record val metrics for ALL models (used in selection for fold+1 onwards)
        for model in available_models:
            art = _load_artifact(pred_root, model, fold_id, "val")
            if art is None:
                log.debug(f"  No val artifact for {model} at fold {fold_id}")
                continue
            val_metrics = compute_val_metrics(art.true_labels, art.probs)
            for sel in selectors.values():
                sel.record_val(model, val_metrics)

    # ------------------------------------------------------------------ outputs
    if not fold_rows and not baseline_rows:
        log.warning("No folds processed.")
        return

    # Fold-level results (all methods concatenated)
    all_fold_dfs = [pd.DataFrame(rows) for rows in fold_rows.values() if rows]
    if all_fold_dfs:
        fold_df = pd.concat(all_fold_dfs, ignore_index=True)
        p = out_dir / "selector_fold_results.csv"
        fold_df.to_csv(p, index=False)
        log.info(f"Saved fold results → {p}")

    # Summary table
    summary_rows: list[dict] = []

    for method, rows in fold_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        display = _METHOD_DISPLAY.get(method, method)
        summary_rows.append({
            "model": display,
            "method": method,
            "n_folds": len(df),
            "fold_min": int(df.fold_id.min()),
            "fold_max": int(df.fold_id.max()),
            "macro_f1": df.macro_f1.mean(),
            "macro_f1_std": df.macro_f1.std(),
            "brier_score": df.brier_score.mean(),
            "ece": df.ece.mean(),
            "balanced_accuracy": df.balanced_accuracy.mean(),
            "class_f1_0": df.class_f1_0.mean(),
            "class_f1_1": df.class_f1_1.mean(),
            "class_f1_2": df.class_f1_2.mean(),
            "class_f1_3": df.class_f1_3.mean(),
            "type": "selector",
        })

    if baseline_rows:
        bdf = pd.DataFrame(baseline_rows)
        for model, grp in bdf.groupby("model"):
            summary_rows.append({
                "model": model,
                "method": "baseline",
                "n_folds": len(grp),
                "fold_min": int(grp.fold_id.min()),
                "fold_max": int(grp.fold_id.max()),
                "macro_f1": grp.macro_f1.mean(),
                "macro_f1_std": grp.macro_f1.std(),
                "brier_score": grp.brier_score.mean(),
                "ece": grp.ece.mean(),
                "balanced_accuracy": grp.balanced_accuracy.mean(),
                "type": "baseline",
            })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    p = out_dir / "selector_summary.csv"
    summary_df.to_csv(p, index=False)
    log.info(f"Saved summary → {p}")

    # Model selection counts
    counts_rows: list[dict] = []
    for method, rows in fold_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        display = _METHOD_DISPLAY.get(method, method)
        counts = df.selected_model.value_counts().to_dict()
        counts_rows.append({"method": display, **{m: counts.get(m, 0) for m in available_models}})
    if counts_rows:
        counts_df = pd.DataFrame(counts_rows)
        p = out_dir / "selector_model_counts.csv"
        counts_df.to_csv(p, index=False)
        log.info(f"Saved model counts → {p}")

    # walk_forward_summary_*.csv files so compare_all.py picks them up automatically
    for method, rows in fold_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        display = _METHOD_DISPLAY.get(method, method)
        wf = df[["fold_id", "macro_f1", "balanced_accuracy", "brier_score", "ece"]].copy()
        wf.insert(0, "model", display)
        csv_path = _PROJECT_ROOT / f"walk_forward_summary_{display}.csv"
        wf.to_csv(csv_path, index=False)
        log.info(f"Saved walk-forward summary → {csv_path}")

    # Print results
    print("\n=== Model Selector Results ===")
    show_cols = [c for c in [
        "model", "n_folds", "fold_min", "fold_max",
        "macro_f1", "macro_f1_std", "brier_score", "ece", "balanced_accuracy", "type",
    ] if c in summary_df.columns]
    print(summary_df[show_cols].to_markdown(index=False, floatfmt=".4f"))

    if counts_rows:
        print("\n=== Selection Counts ===")
        print(counts_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
