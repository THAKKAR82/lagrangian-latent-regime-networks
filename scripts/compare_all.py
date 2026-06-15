#!/usr/bin/env python
"""Aggregate walk_forward_summary_*.csv files and produce a benchmark table.

Usage:
    python scripts/compare_all.py
    python scripts/compare_all.py --fold-start 20 --fold-end 40
    python scripts/compare_all.py --fold-start 20 --fold-end 40 --require-complete-fold-range
    python scripts/compare_all.py --output outputs/comparisons/bench.md

Discovery:
    Scans the project root for walk_forward_summary_*.csv files.
    Each row must have a fold_id column. Rows are filtered by fold range if requested.
    Models with partial coverage are shown unless --require-complete-fold-range is set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _load_all(root: Path) -> pd.DataFrame:
    """Load and concatenate all walk_forward_summary_*.csv files."""
    frames = []
    for csv_path in sorted(root.glob("walk_forward_summary_*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        # Back-compat: derive model name from filename if column missing
        if "model" not in df.columns:
            df["model"] = csv_path.stem.replace("walk_forward_summary_", "")
        # Source file for debugging
        df["_source"] = csv_path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_folds(df: pd.DataFrame, fold_start: int | None, fold_end: int | None) -> pd.DataFrame:
    if fold_start is not None:
        df = df[df["fold_id"] >= fold_start]
    if fold_end is not None:
        df = df[df["fold_id"] <= fold_end]
    return df


def _aggregate(df: pd.DataFrame, require_complete: bool, fold_start: int | None, fold_end: int | None) -> pd.DataFrame:
    """Group by model, compute summary stats, flag partial coverage."""
    expected_folds: set[int] | None = None
    if require_complete and fold_start is not None and fold_end is not None:
        expected_folds = set(range(fold_start, fold_end + 1))

    rows = []
    for model, grp in df.groupby("model", sort=False):
        fold_ids = sorted(grp["fold_id"].unique())
        n_folds = len(fold_ids)
        fold_min = int(min(fold_ids))
        fold_max = int(max(fold_ids))

        if expected_folds is not None:
            present = set(fold_ids)
            if not expected_folds.issubset(present):
                continue  # skip incomplete

        is_partial = False
        if fold_start is not None and fold_min > fold_start:
            is_partial = True
        if fold_end is not None and fold_max < fold_end:
            is_partial = True

        row: dict = {
            "model": model,
            "n_folds": n_folds,
            "fold_min": fold_min,
            "fold_max": fold_max,
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(),
            "brier_mean": grp["brier_score"].mean(),
            "ece_mean": grp["ece"].mean(),
        }
        if "balanced_accuracy" in grp.columns:
            row["balanced_accuracy_mean"] = grp["balanced_accuracy"].mean()
        row["partial"] = is_partial and not require_complete
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return out


def _print_table(summary: pd.DataFrame) -> None:
    display_cols = [c for c in [
        "model", "n_folds", "fold_min", "fold_max",
        "macro_f1_mean", "macro_f1_std", "brier_mean", "ece_mean",
        "balanced_accuracy_mean", "partial",
    ] if c in summary.columns]
    print(summary[display_cols].to_markdown(index=False, floatfmt=".4f"))


def _save_outputs(summary: pd.DataFrame, fold_start: int | None, fold_end: int | None, require_complete: bool, out_dir: Path, md_path: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if fold_start is not None and fold_end is not None:
        tag = f"folds_{fold_start}_{fold_end}"
        prefix = "benchmark_complete_" if require_complete else "benchmark_"
    else:
        tag = "all"
        prefix = "benchmark_"

    csv_out = out_dir / f"{prefix}{tag}.csv"
    summary.to_csv(csv_out, index=False)
    print(f"\nSaved CSV  → {csv_out}")

    display_cols = [c for c in [
        "model", "n_folds", "fold_min", "fold_max",
        "macro_f1_mean", "macro_f1_std", "brier_mean", "ece_mean",
        "balanced_accuracy_mean", "partial",
    ] if c in summary.columns]
    md_content = f"# Benchmark Results\n\n{summary[display_cols].to_markdown(index=False, floatfmt='.4f')}\n"

    md_out = out_dir / f"{prefix}{tag}.md"
    md_out.write_text(md_content)
    print(f"Saved MD   → {md_out}")

    if md_path:
        p = Path(md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md_content)
        print(f"Saved MD   → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare walk-forward benchmark results.")
    parser.add_argument("--fold-start", type=int, default=None, help="Minimum fold_id to include")
    parser.add_argument("--fold-end", type=int, default=None, help="Maximum fold_id to include")
    parser.add_argument("--require-complete-fold-range", action="store_true",
                        help="Only show models that have every fold in the requested range")
    parser.add_argument("--output", type=str, default=None, help="Additional markdown output path")
    args = parser.parse_args()

    root = Path(".")
    out_dir = root / "outputs" / "comparisons"

    all_rows = _load_all(root)
    if all_rows.empty:
        print("No walk_forward_summary_*.csv files found in current directory.")
        sys.exit(0)

    filtered = _filter_folds(all_rows, args.fold_start, args.fold_end)
    if filtered.empty:
        print(f"No rows matched fold range [{args.fold_start}, {args.fold_end}].")
        sys.exit(0)

    summary = _aggregate(filtered, args.require_complete_fold_range, args.fold_start, args.fold_end)
    if summary.empty:
        print("No models have complete coverage for the requested fold range.")
        sys.exit(0)

    # Header line
    if args.fold_start is not None or args.fold_end is not None:
        lo = args.fold_start if args.fold_start is not None else "start"
        hi = args.fold_end if args.fold_end is not None else "end"
        suffix = " (complete range only)" if args.require_complete_fold_range else ""
        print(f"\n=== Benchmark Table — folds {lo}–{hi}{suffix} ===")
    else:
        print("\n=== Benchmark Table — all folds ===")

    _print_table(summary)
    _save_outputs(summary, args.fold_start, args.fold_end, args.require_complete_fold_range, out_dir, args.output)


if __name__ == "__main__":
    main()
