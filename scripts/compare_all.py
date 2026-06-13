#!/usr/bin/env python
"""Aggregate all walk_forward_summary_*.csv files and produce a benchmark table.

Usage:
    python scripts/compare_all.py
    python scripts/compare_all.py --output reports/benchmark.md

Scans the current directory for files matching walk_forward_summary_*.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def collect_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(root.glob("walk_forward_summary_*.csv")):
        df = pd.read_csv(csv_path)
        model = csv_path.stem.replace("walk_forward_summary_", "")
        if "model" in df.columns:
            model = df["model"].iloc[0]
        n_folds = len(df)
        rows.append({
            "model": model,
            "n_folds": n_folds,
            "macro_f1_mean": df["macro_f1"].mean(),
            "macro_f1_std": df["macro_f1"].std(),
            "brier_mean": df["brier_score"].mean(),
            "ece_mean": df["ece"].mean(),
            "source": str(csv_path.name),
        })
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def main() -> None:
    root = Path(".")
    out_path = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            out_path = Path(args[i + 1])
        elif arg.startswith("--output="):
            out_path = Path(arg.split("=", 1)[1])

    df = collect_summaries(root)
    if df.empty:
        print("No walk_forward_summary_*.csv files found in current directory.")
        return

    table = df[["model", "n_folds", "macro_f1_mean", "macro_f1_std", "brier_mean", "ece_mean"]]
    md_table = table.to_markdown(index=False, floatfmt=".4f")
    print("\n=== Benchmark Table ===")
    print(md_table)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"# Benchmark Results\n\n{md_table}\n")
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
