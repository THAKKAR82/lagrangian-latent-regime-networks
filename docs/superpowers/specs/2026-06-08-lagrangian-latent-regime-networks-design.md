---
name: lagrangian-latent-regime-networks-design
description: Full architecture and implementation spec for the Lagrangian Latent Regime Networks project — a financial time-series regime forecasting system using learned Lagrangian dynamics in latent space, with XGBoost/LSTM/GRU/NODE baselines and walk-forward evaluation.
metadata:
  type: project
---

# Lagrangian Latent Regime Networks — Design Spec

**Date:** 2026-06-08  
**Project:** Lagrangian Latent Regime Networks (LLRN)  
**Goal:** Test whether a latent state evolved with a learned Lagrangian and discretized Euler–Lagrange dynamics produces smoother, more robust, and better-calibrated market regime forecasts than standard baselines (XGBoost, LSTM, GRU) and a Neural ODE baseline.

---

## 1. Project Framing

This is a **market-state / regime forecasting** project for risk-aware temporal representation learning. It is not a stock-price prediction project and not a trading bot.

**Target task:** Predict market regime labels over the next 5–10 trading days using rolling daily input windows of 30–60 trading days.

**Regime taxonomy:** 4 mutually exclusive classes defined by a 2×2 cross of directional return state × volatility state:

| Label | ID | Return axis | Volatility axis |
|---|---|---|---|
| Bull / Calm | 0 | Positive forward return | Low realized vol |
| Bull / Stress | 1 | Positive forward return | High realized vol |
| Bear / Calm | 2 | Negative forward return | Low realized vol |
| Bear / Stress | 3 | Negative forward return | High realized vol |

Thresholds are **adaptive quantile-based** (default: median split on both axes), fit on train data only.

---

## 2. Architecture Overview

**Approach:** Monolithic `src/` tree (Approach A) with a strict purity constraint:
- `src/data/`, `src/features/`, `src/labels/` — pure Python / pandas / numpy only, no PyTorch imports
- `src/models/`, `src/training/` — PyTorch + torchdiffeq
- `src/evaluation/`, `src/visualization/` — mixed (sklearn metrics, matplotlib/plotly)

**Config system:** Hydra with composable YAML config groups. Ablation sweeps use `--multirun`.

**Data source:** `yfinance` as primary (SPY, QQQ, TLT, GLD, ^VIX). Synthetic GBM+GARCH fixtures for unit tests (no network calls in CI).

**Walk-forward evaluation:** Expanding train window, fixed-size validation (252 days), fixed-size test (252 days), configurable step size (default 63 days).

---

## 3. Repository Structure

```
lagrange/
├── README.md
├── requirements.txt
├── configs/
│   ├── config.yaml                  # top-level Hydra defaults list
│   ├── data/default.yaml
│   ├── features/default.yaml
│   ├── labels/default.yaml
│   ├── splits/default.yaml
│   └── model/
│       ├── xgb.yaml
│       ├── lstm.yaml
│       ├── gru.yaml
│       ├── node.yaml
│       └── lagrangian.yaml
├── data/                            # gitignored
├── docs/superpowers/specs/
├── notebooks/
├── reports/figures/
├── src/
│   ├── data/
│   │   ├── manager.py               # DataManager: all disk I/O
│   │   └── download.py              # yfinance fetch + cache
│   ├── features/
│   │   └── engineer.py              # causal feature construction
│   ├── labels/
│   │   ├── base.py                  # BaseLabeler ABC stub
│   │   └── quantile_labeler.py      # 2x2 quadrant labeler
│   ├── models/
│   │   ├── baseline_xgb.py
│   │   ├── baseline_lstm.py
│   │   ├── baseline_gru.py
│   │   ├── baseline_node.py
│   │   └── lagrangian_regime_net.py
│   ├── training/
│   │   ├── train_baseline.py        # XGBoost entrypoint
│   │   ├── train_rnn.py             # LSTM/GRU shared trainer
│   │   ├── train_node.py
│   │   └── train_lagrangian.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── run_walkforward.py
│   ├── visualization/
│   │   └── plots.py
│   └── utils/
│       ├── reproducibility.py
│       └── dataset_builder.py
├── tests/
│   ├── conftest.py                  # synthetic OHLCV fixtures
│   ├── test_splits.py
│   ├── test_labels.py
│   ├── test_features.py
│   └── test_shapes.py
└── scripts/
    └── collect_results.py           # aggregate ablation outputs
```

---

## 4. Data Layer

### DataManager

```python
@dataclass
class DataManager:
    raw_dir: Path
    processed_dir: Path
    tickers: list[str]       # ["SPY", "QQQ", "TLT", "GLD", "^VIX"]
    start_date: str          # "2000-01-01"
    end_date: str            # "2024-12-31"
```

Responsibilities:
- Resolve all file paths (raw cache, processed features, labels, splits)
- Download raw OHLCV via `yfinance` if cache miss, load from parquet on hit
- Save/load processed DataFrames at each pipeline stage
- Save SHA-256 hash of raw files for reproducibility auditing
- Never transform data — pure I/O and path management

Raw data: one parquet per ticker at `data/raw/{ticker}.parquet`.  
Processed outputs: `data/processed/`.

### Download

- `yfinance.download(auto_adjust=True)` for adjusted OHLCV
- VIX fetched as `^VIX` (close only, no volume)
- All columns standardized to lowercase
- Hard failure (not silent) on partial/missing data

### Test Fixtures

`tests/conftest.py` generates synthetic OHLCV via geometric Brownian motion with GARCH-like vol clustering (~500 rows, 5 tickers, fixed seed). All unit tests use this fixture. Integration tests marked `@pytest.mark.integration` are skipped by default.

---

## 5. Feature Engineering

### Causality Contract

Every feature at time `t` uses only information available at close of day `t`. No `shift(-1)` on targets mixed into features. All rolling operations use explicit `min_periods` to produce NaN at warmup rather than silently using partial windows.

### Feature Groups

All configurable window sizes live in `configs/features/default.yaml`.

**Returns:**
- `log_return_1d` — log(close_t / close_{t-1})
- `log_return_5d`, `log_return_21d` — multi-horizon momentum

**Rolling statistics** (windows: 5, 10, 21, 63):
- `roll_mean_return_{w}` — rolling mean of daily log returns
- `roll_realized_vol_{w}` — rolling std × √252 (annualized)
- `roll_zscore_{w}` — (close - roll_mean) / roll_std

**Drawdown:**
- `roll_drawdown_{w}` — (close - rolling_max) / rolling_max

**Moving average distance:**
- `ma_dist_{w}` — (close - SMA_w) / SMA_w

**Volume:**
- `vol_change_1d` — log(volume_t / volume_{t-1})
- `roll_vol_ratio` — volume / rolling mean volume

**Volatility regime:**
- `vol_ratio_short_long` — realized_vol_5d / realized_vol_63d

**Cross-asset** (SPY vs QQQ, TLT, GLD, VIX):
- `roll_corr_{asset}_{w}` — rolling Pearson correlation of log returns

**VIX-specific:**
- `vix_level`, `vix_change_1d`, `vix_roll_zscore_21`

### Output Contract

```python
def build_features(
    prices: dict[str, pd.DataFrame],  # ticker -> OHLCV DataFrame
    cfg: FeaturesConfig,
) -> pd.DataFrame:                     # date-indexed, all features, NaNs at head
```

No in-place mutation. No scaler fit here — scaling is deferred to the dataset builder.

---

## 6. Regime Labeling

### QuantileLabeler

Inherits `BaseLabeler`. Fit on train data only — thresholds computed from train fold, applied unchanged to val/test. This is a hard leakage boundary.

```python
@dataclass
class LabelConfig:
    horizon: int = 5
    vol_window: int = 21
    return_quantile: float = 0.5
    vol_quantile: float = 0.5
    smoothing: bool = True
    smoothing_min_periods: int = 3
```

### Label Construction Steps

1. **Forward return:** `fwd_return_h = log(close_{t+h} / close_t)` on SPY. Last `h` rows dropped (no label).
2. **Realized vol:** rolling std of daily log returns over `vol_window`, annualized. Trailing window — fully causal.
3. **Fit thresholds on train:** `return_thresh` = `return_quantile` quantile of `fwd_return_h` over train dates. `vol_thresh` = `vol_quantile` quantile of `roll_vol` over train dates.
4. **Assign quadrant:** 2 binary flags → one of 4 integer labels (0–3).
5. **Persistence smoothing (optional):** Regime flip only accepted if new regime persists ≥ `smoothing_min_periods` consecutive days. Forward-pass scan only — no lookahead.

### Saved Artifacts

Label DataFrame columns: `fwd_return_h`, `roll_vol`, `return_thresh`, `vol_thresh`, `label_raw`, `label`, `regime_name`.

### BaseLabeler Stub

```python
# src/labels/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseLabeler(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseLabeler": ...
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)
```

Future HMM labeler implements the same interface.

---

## 7. Dataset Builder & Walk-Forward Splits

### Rolling Window Construction

Two output formats:

- **Sequence format** `(N, window_len, n_features)` — for LSTM/GRU/NODE/Lagrangian
- **Flat format** `(N, window_len * n_features)` — for XGBoost

Same index alignment logic for both. Labels are the regime at the end of the forward horizon. Rows where `label` is NaN (the last `h` rows of any split, due to the forward return window) are dropped before constructing windows.

### Walk-Forward Split Config

```python
@dataclass
class SplitConfig:
    train_start: str = "2000-01-01"
    val_size: int = 252
    test_size: int = 252
    step_size: int = 63
    min_train_size: int = 504
```

Generator pattern — yields `Fold` dataclasses, never materializes all folds at once.

### Scaler Fit Boundary

Inside each fold:
1. Fit `StandardScaler` on `train_idx` feature rows only
2. Transform train, val, test with fitted scaler
3. Attach fitted scaler to `Fold` object

Never fit scaler on val or test. Enforced by construction.

### Fold Dataclass

```python
@dataclass
class Fold:
    fold_id: int
    train_X: np.ndarray      # (N_train, window_len, n_features) or flat
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    scaler: StandardScaler
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    label_meta: pd.DataFrame
```

---

## 8. XGBoost Baseline

```python
@dataclass
class XGBConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"
    seed: int = 42
    n_jobs: int = -1
```

Input: flat features `(N, window_len * n_features)`. Uses XGBoost native `eval_set` for early stopping. One model trained per fold. Checkpoints + metrics JSON saved per fold.

`train_baseline.py` pipeline: load config → DataManager → build_features → QuantileLabeler → walk-forward folds → train → evaluate → aggregate.

---

## 9. Neural Models

### Shared Training Loop Structure

```
for epoch in range(max_epochs):
    train one epoch → log train loss
    evaluate on val → log val loss + macro F1
    early stopping check (patience on val macro F1)
    save checkpoint if val improves
```

Seed control: `set_global_seed(seed)` at top of every entrypoint.

### LSTM / GRU

```python
@dataclass
class RNNConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
```

Architecture: RNN encoder → last hidden state → LayerNorm → Linear(hidden_dim, 4).  
Cross-entropy loss with class-frequency inverse weighting.

### Neural ODE Baseline

```python
@dataclass
class NODEConfig:
    latent_dim: int = 32
    hidden_dim: int = 64
    rtol: float = 1e-3
    atol: float = 1e-4
    method: str = "dopri5"
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
```

MLP encoder → `z_0`. `torchdiffeq.odeint` with 2-layer MLP dynamics. Linear head → logits. Direct comparison point for Lagrangian model.

### Lagrangian Regime Network

```python
@dataclass
class LagrangianConfig:
    latent_dim: int = 8
    hidden_dim: int = 64
    n_steps: int = 4
    damping: float = 0.1
    use_forcing: bool = True
    dt: float = 1.0
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 15
```

**Encoder:** MLP → `(z_0, ż_0)` of shape `(batch, latent_dim)` each.

**Lagrangian:** `L_θ(z, ż) = T_θ(z, ż) - V_θ(z)`
- `T_θ = ½ żᵀ M(z) ż` — `M(z)` is diagonal positive-definite (softplus-activated MLP output)
- `V_θ(z)` — 2-layer MLP, scalar output

**Euler-Lagrange update:** Explicit finite-difference across `n_steps` latent steps. Damping term `γ ż` added. Optional exogenous forcing: linear projection of summary features added to acceleration.

**Decoder:** Linear(latent_dim → 4) on final `z_T`.

**Trainability safeguards:** gradient clipping (`max_norm=1.0`), softplus mass (never zero), potential MLP initialized near zero.

---

## 10. Evaluation Suite

### Metrics (`evaluation/metrics.py`)

`evaluate(y_true, y_pred, y_prob) -> EvalResult`:

| Metric | Method |
|---|---|
| Macro F1 | `sklearn f1_score(average='macro')` |
| Balanced accuracy | `sklearn balanced_accuracy_score` |
| Confusion matrix | `sklearn confusion_matrix` |
| Brier score | MSE between `y_prob` and one-hot `y_true` |
| Expected calibration error | 10-bin reliability diagram |
| Regime switch frequency | Fraction of consecutive prediction flips |
| Prediction entropy | Mean H(y_prob) |

`evaluate_noisy(X, model, σ=0.1)` — adds Gaussian noise, re-evaluates, reports F1 degradation.

### Walk-Forward Aggregation

Per-fold metrics collected into a `pd.DataFrame`. Summary: mean ± std across folds. Saved as CSV + printed markdown table.

### Plots (`visualization/plots.py`)

- Regime timeline (color-coded true vs predicted)
- Confusion matrix heatmap (per fold + aggregated)
- Calibration plot (reliability diagram per class, ECE annotation)
- Rolling macro F1 (63-day window over test timeline)
- XGBoost feature importance bar chart
- Fold summary bar chart (macro F1 per fold with error bars)

All plots → `reports/figures/` as PNG. Timeline plots also as interactive HTML (plotly).

---

## 11. Ablation Suite

Phase 7 sweeps via Hydra `--multirun`:

```bash
python -m src.training.train_lagrangian --multirun model.latent_dim=4,8,16,32
python -m src.training.train_lagrangian --multirun data.window_len=20,40,60
python -m src.training.train_lagrangian --multirun labels.horizon=5,10
python -m src.training.train_lagrangian --multirun model.damping=0.0,0.05,0.1,0.3
python -m src.training.train_lagrangian --multirun model.use_forcing=true,false
```

`scripts/collect_results.py` scans Hydra output dirs, loads metrics JSONs, produces benchmark table as CSV + markdown.

---

## 12. Reproducibility

- `set_global_seed(seed)` sets torch, numpy, random seeds + `cudnn.deterministic=True`
- All configs include `seed: 42` as top-level field
- Raw data SHA-256 hash logged at run start
- Hydra auto-saves full resolved config per run to `outputs/YYYY-MM-DD/HH-MM-SS/`

---

## 13. Testing

| File | Coverage |
|---|---|
| `test_splits.py` | No date overlap train/val/test; scaler fit only on train; correct window boundaries; fold temporal ordering |
| `test_labels.py` | 4 classes present; no NaN labels; thresholds train-only; smoothing reduces switch freq; label counts plausible |
| `test_features.py` | No future-leaking columns; output shape matches input index; no NaN outside warmup rows |
| `test_shapes.py` | Each model forward pass: input `(batch, window_len, n_features)` → output `(batch, 4)` |

---

## 14. CLI Entrypoints

```bash
python -m src.training.train_baseline          # XGBoost
python -m src.training.train_rnn model=lstm    # LSTM
python -m src.training.train_rnn model=gru     # GRU
python -m src.training.train_node              # Neural ODE
python -m src.training.train_lagrangian        # Lagrangian net
python -m src.evaluation.run_walkforward       # aggregate walk-forward eval
```

---

## 15. Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Repo scaffold, DataManager, download pipeline | MVP |
| 2 | Feature engineering | MVP |
| 3 | Regime labeling (QuantileLabeler) | MVP |
| 4 | Dataset builder + walk-forward splits | MVP |
| 5 | XGBoost baseline + evaluation suite | MVP |
| 6 | LSTM + GRU baselines | Post-MVP |
| 7 | Neural ODE baseline | Post-MVP |
| 8 | Lagrangian Regime Network | Post-MVP |
| 9 | Ablations + benchmark table | Post-MVP |
| 10 | Polish, README, dashboard | Post-MVP |

---

## 16. Nice-to-Haves (post-MVP)

- SHAP attribution for XGBoost
- Latent trajectory visualization (PCA/UMAP of z_t colored by regime)
- Illustrative risk overlay (regime → {100%, 50%, 0%, 25%} equity exposure, labeled "illustrative only")
- Streamlit dashboard (`scripts/dashboard.py`, optional dependency)

---

## 17. Non-Goals

- Stock price prediction
- Live trading or backtested trading strategies
- Profitability claims
- Real-time data feeds
