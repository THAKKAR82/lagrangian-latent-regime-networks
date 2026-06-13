#!/usr/bin/env python
"""Ablation runner: run a list of train commands and collect walk_forward_summary CSVs.

Usage:
    python scripts/run_ablation.py [--subset]

  --subset: runs with +fold_start=20 +fold_end=40 for faster iteration.

Results are printed as a markdown table at the end.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ABLATIONS = [
    {
        "name": "conv1d_baseline",
        "cmd": "python -m src.training.train_lagrangian model=lagrangian_conv1d",
    },
    {
        "name": "v9_transition_head",
        "cmd": "python -m src.training.train_lagrangian model=lagrangian_v9",
    },
    {
        "name": "focal_loss",
        "cmd": "python -m src.training.train_lagrangian model=lagrangian_focal",
    },
    {
        "name": "ensemble_equal",
        "cmd": "python -m src.training.train_ensemble",
    },
    {
        "name": "ensemble_grid_search",
        "cmd": "python -m src.training.train_ensemble +ensemble_grid_search=true",
    },
]


def run_ablation(cmd: str, subset: bool, out_label: str) -> Path | None:
    full_cmd = cmd
    if subset:
        full_cmd += " +fold_start=20 +fold_end=40"
    print(f"\n=== Running: {out_label} ===")
    print(f"  {full_cmd}")
    result = subprocess.run(full_cmd, shell=True)
    if result.returncode != 0:
        print(f"  FAILED (returncode={result.returncode})")
        return None
    # Hydra writes to outputs/<date>/<time>/walk_forward_summary.csv
    matches = sorted(
        Path("outputs").glob("**/*walk_forward_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def main() -> None:
    subset = "--subset" in sys.argv
    rows = []
    for abl in ABLATIONS:
        csv_path = run_ablation(abl["cmd"], subset, abl["name"])
        if csv_path is None:
            rows.append({
                "name": abl["name"],
                "macro_f1_mean": float("nan"),
                "brier_mean": float("nan"),
                "ece_mean": float("nan"),
            })
            continue
        df = pd.read_csv(csv_path)
        rows.append({
            "name": abl["name"],
            "macro_f1_mean": df["macro_f1"].mean(),
            "brier_mean": df["brier_score"].mean(),
            "ece_mean": df["ece"].mean(),
        })

    result_df = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    print("\n\n=== Ablation Results ===")
    print(result_df.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
