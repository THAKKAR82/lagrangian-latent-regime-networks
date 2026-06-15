# Lagrangian Latent Regime Networks

A research platform for benchmarking structured latent-dynamics models against standard baselines on financial market regime classification. The core model evolves latent state through a discrete symplectic Euler integrator inspired by Lagrangian mechanics, treating the latent space as a generalized coordinate system with learned potential and damping.

## What problem does this solve?

Most regime classifiers treat the hidden state as an unconstrained feature vector. This project asks: *does imposing energy-based dynamics on the latent space produce better-calibrated regime predictions?*

Answer so far: **the dynamics themselves don't help much — the encoder does.** Replacing a flat MLP encoder with a causal 1D-convolutional encoder improved macro-F1 from 0.3700 to 0.3976. The best result is a 50/50 ensemble of XGBoost and Lagrangian-conv1d, which beats both individual models on calibration.

## Regime taxonomy

Four mutually exclusive classes, 2×2 cross of return direction × volatility level:

| ID | Name | Return | Volatility |
|----|------|--------|------------|
| 0 | Bull/Calm | ↑ positive | low |
| 1 | Bull/Stress | ↑ positive | high |
| 2 | Bear/Calm | ↓ negative | low |
| 3 | Bear/Stress | ↓ negative | high |

Labels are defined on SPY using rolling quantile thresholds with smoothing.

## Benchmark results — 71 walk-forward folds (2004–2026)

| Model | Mean Macro F1 | Mean Brier | Mean ECE |
|-------|:-------------:|:----------:|:--------:|
| **Ensemble (XGBoost + conv1d)** | **0.4046** | **0.5899** | **0.1210** |
| XGBoost | 0.4208 | 0.6246 | 0.1678 |
| LSTM | 0.4062 | 0.6152 | 0.1477 |
| GRU | 0.4008 | 0.6071 | 0.1477 |
| Lagrangian conv1d | 0.3976 | 0.6214 | 0.1473 |
| NODE | 0.3850 | 0.6281 | 0.1450 |
| Lagrangian v3 (MLP) | 0.3700 | 0.6373 | 0.1258 |

Per-fold CSVs are in the repo root (`walk_forward_summary_*.csv`).

## Architecture

### LagrangianRegimeNet

```
Input (B, T=40, F)
  └─ Encoder (conv1d / tcn / mlp / hybrid_conv)
       └─ h (B, encoder_dim)
            ├─ z0_head    → z₀  (B, latent_dim)
            └─ zdot0_head → ż₀  (B, latent_dim)

Discrete symplectic Euler, n_steps iterations:
  q    = coord_net(z)                 [optional coordinate transform]
  V    = potential_net(q)
  dV   = autograd.grad(V, q)
  z̈    = -(dV/dq + γ·ż) / M_diag     [scalar / vector / no damping]
  ż   += dt · z̈                       [velocity first — symplectic]
  z   += dt · ż

Classifier: LayerNorm → Linear(latent_dim, 4)
```

**Encoder variants:**

| Encoder | Key params | Notes |
|---------|-----------|-------|
| `conv1d` | `conv_channels`, `conv_kernel_size` | 2 causal conv layers, last-step readout |
| `tcn` | `tcn_channels`, `tcn_dilations=[1,2,4,8]` | Dilated residual blocks |
| `mlp` | `hidden_dim` | Flatten → 2-layer MLP |
| `hybrid_conv` | `conv_kernel_size` | 3-layer causal conv |

All convolutions use left-padding only (causal, no future leakage).

**Damping modes** (mutually exclusive via config):
- `use_scalar_damping: true` — single learned scalar γ
- `use_vector_damping: true` — per-dimension γ from a small network
- both false — undamped Hamiltonian dynamics

**Optional extensions:** `use_forcing`, `use_coord_transform`, `use_transition_head`

## Walk-forward evaluation

- **Data:** SPY, QQQ, TLT, GLD, ^VIX via yfinance (2004-11-01 to 2026-06-08)
- **Folds:** 71 total, expanding train window, val = 252 days, test = 252 days, step = 63 days
- **Window:** 40 trading days per sample
- **Features:** 37 base (rolling returns, vol, momentum, correlations) + optional 29 econophysics features
- **Subset runs:** `+fold_start=20 +fold_end=40` for faster iteration before committing to a full run

## Quickstart

```bash
# 1. Create conda environment
conda create -n lagrange python=3.11 -y
conda activate lagrange
pip install -r requirements.txt

# 2. Run XGBoost baseline (downloads data automatically on first run)
python -m src.training.train_baseline

# 3. Run Lagrangian conv1d (subset first to verify)
python -m src.training.train_lagrangian model=lagrangian_conv1d +fold_start=20 +fold_end=40

# 4. Full 71-fold run
python -m src.training.train_lagrangian model=lagrangian_conv1d

# 5. Compare results
python scripts/compare_all.py --fold-start 20 --fold-end 40 --require-complete-fold-range
```

## Training all models

```bash
# Baselines
python -m src.training.train_baseline model=xgb
python -m src.training.train_rnn model=lstm
python -m src.training.train_rnn model=gru
python -m src.training.train_node model=node

# Lagrangian variants
python -m src.training.train_lagrangian model=lagrangian_conv1d
python -m src.training.train_lagrangian model=pure_lagrangian        # undamped (ablation)
python -m src.training.train_lagrangian model=scalar_damped_lagrangian
python -m src.training.train_lagrangian model=vector_damped_lagrangian
python -m src.training.train_lagrangian model=forced_scalar_damped_lagrangian

# Ensemble (XGBoost + conv1d, probability average)
python -m src.training.train_ensemble
```

Config keys not in the base config must be prefixed with `+`:
```bash
python -m src.training.train_lagrangian model=lagrangian_conv1d +fold_start=20 +fold_end=40
```

## Leakage-safe model selector

Dynamically selects which model to deploy each fold using only historical validation performance — no test-fold labels:

```bash
python scripts/model_selector.py \
  --fold-start 20 --fold-end 40 \
  --models xgb ensemble forced_scalar_damped_lagrangian

# Methods: previous_best_macro_f1 | rolling_best_macro_f1 | previous_best_calibrated_score | ...
```

Results appear automatically in `compare_all.py` via `walk_forward_summary_model_selector_*.csv`.

## Comparing experiments

```bash
python scripts/compare_all.py
python scripts/compare_all.py --fold-start 20 --fold-end 40 --require-complete-fold-range
```

## Repo structure

```
src/
  data/               download.py, manager.py
  features/           engineer.py (37 features), econophysics.py (29 features)
  labels/             quantile_labeler.py, multi_horizon_labeler.py
  models/             lagrangian_regime_net.py, baseline_*.py
  training/           train_lagrangian.py, train_rnn.py, train_baseline.py,
                      train_node.py, train_ensemble.py, train_stacked.py
  evaluation/         metrics.py, predictions.py, model_selector.py
  postprocessing/     temperature.py, ensemble.py, stacker.py, thresholds.py, adaptive.py
  utils/              dataset_builder.py, reproducibility.py
  visualization/      plots.py

configs/model/        one YAML per model variant
scripts/              compare_all.py, model_selector.py
reports/figures/      fold_summary.png per model
tests/test_shapes.py  147 tests
walk_forward_summary_*.csv  per-experiment benchmark results
```

## Key lessons from experiments

- **Encoder matters more than dynamics**: MLP encoder was the bottleneck; conv1d fixed it (+2.8pp over 71 folds)
- **Econophysics features hurt conv1d** (~-2pp consistently) — add noise the conv encoder can't filter
- **Multi-horizon loss hurts conv1d** — combining conv1d + MH supervision regressed vs standalone
- **XGBoost fails early, conv1d holds**: Pearson r = 0.29 between fold-level F1 of the two models
- **Ensemble beats both**: lower variance and complementary errors make 50/50 prob average the winner
- **Change one axis at a time**: encoder, features, or loss objective — not two at once

## Tests

```bash
python -m pytest tests/test_shapes.py -v        # 147 tests
python -m pytest tests/test_shapes.py -k lstm   # filter by pattern
```

## Research framing

This is regime forecasting for risk-aware temporal representation learning — not a trading bot, not a price prediction system, and no profitability claims are made.
