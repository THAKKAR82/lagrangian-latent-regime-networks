# CLAUDE.md

## Agent instructions

After completing any meaningful work in this repo — new experiment results, architecture changes, new configs, lessons learned — update this file to reflect the current state. Keep the benchmark table, experiment history, and lessons current.

Also update `PRD.md` to reflect completed work: mark phases done, update the benchmark table, and revise open questions if they've been answered. Do **not** add new requirements or scope to the PRD — only update what has already been built or measured.

## Project

Market regime classification research platform. The goal is to build and benchmark a Lagrangian-inspired latent dynamics model (`LagrangianRegimeNet`) against standard baselines (XGBoost, LSTM, GRU, Neural ODE) on a 4-class regime prediction task (BullCalm=0, BullStress=1, BearCalm=2, BearStress=3).

## Environment

```
Python: C:\Users\dhruv\.conda\envs\lagrange\python.exe  (3.11.15)
Run tests: python -m pytest tests/test_shapes.py -v
Run training: python -m src.training.train_<model> model=<config>
```

Always use the conda env above — not the system Python.

## Repo layout

```
src/
  data/           download.py, manager.py — yfinance fetch + caching
  features/       engineer.py (37 base features), econophysics.py (29 extra)
  labels/         quantile_labeler.py (4-class), multi_horizon_labeler.py
  models/         lagrangian_regime_net.py, lagrangian_regime_net_mh.py, baseline_*.py
  training/       train_lagrangian.py, train_lagrangian_mh.py, train_rnn.py, train_baseline.py, train_node.py
  evaluation/     metrics.py
  utils/          dataset_builder.py, multi_horizon_builder.py, reproducibility.py
  visualization/  plots.py
configs/model/    one yaml per model variant
tests/            test_shapes.py  (87 tests, must stay green)
```

## Walk-forward evaluation

- 71 folds, expanding train window, fixed val=252 / test=252 days, step=63 days
- Data: SPY, QQQ, TLT, GLD, ^VIX from 2004-11-01 to 2026-06-08
- Input windows: `(batch, window_len=40, n_features)`
- Always run a subset first (`+fold_start=20 +fold_end=40`, 21 folds) before full 71-fold
- Promotion threshold: +1pp mean macro F1 over best comparable subset baseline

## Benchmark results (71-fold, full run)

| Model | Mean Macro F1 | Mean Brier | Mean ECE |
|-------|--------------|------------|---------|
| **Ensemble (XGBoost + conv1d)** | **0.4046** | **0.5899** | **0.1210** |
| XGBoost | 0.4208 | 0.6246 | 0.1678 |
| LSTM | 0.4062 | 0.6152 | 0.1477 |
| GRU | 0.4008 | 0.6071 | 0.1477 |
| Lagrangian conv1d | 0.3976 | 0.6214 | 0.1473 |
| NODE | 0.3850 | 0.6281 | 0.1450 |
| Lagrangian v3 (MLP) | 0.3700 | 0.6373 | 0.1258 |

Saved CSVs: `walk_forward_summary_ensemble.csv`, `walk_forward_summary_lagrangian_conv1d.csv`, `walk_forward_summary_xgb.csv`, `walk_forward_summary_lagrangian_v3.csv`, etc. LSTM/GRU/NODE results are in Hydra output logs only (not saved as named CSVs).

## LagrangianRegimeNet architecture

**Core model** (`lagrangian_regime_net.py` — single horizon):
- Encoder (modular, `encoder_type`): MLP | conv1d | tcn | hybrid_conv → `h (batch, encoder_dim)`
- Two heads: `z0_head`, `z_dot0_head`: `h → (batch, latent_dim)`
- Discrete symplectic Euler integrator for `n_steps`:
  - `q = coord_net(z)` (optional coordinate transform)
  - `V = potential_net(q)` (DeepPotentialNet when vector damping on)
  - `dV_dq` via `autograd.grad(..., create_graph=True)` during training
  - Under `@no_grad`: use `torch.enable_grad()` context to compute gradient
  - `z_ddot = -(dV_dq + gamma * z_dot) / M_diag`
  - Symplectic Euler: velocity first, then position
- Classifier: `LayerNorm → Linear(latent_dim, 4)`

**MH model** (`lagrangian_regime_net_mh.py` — multi-horizon):
- Same architecture + 3 heads: `head_5`, `head_10`, `head_20`
- `forward(x)` → 5-day logits (backward compat)
- `forward_multi(x)` → `{logits_5, logits_10, logits_20}`
- Encoder is modular (same `_build_encoder` factory as single-horizon model)

**Key implementation constraints:**
- `autograd.grad` requires `create_graph=True` during training — removing it breaks backprop
- `last_trajectory` is diagnostic only — never use it for gradients (always `.detach()`)
- `_softplus_inverse(y)` requires `y > 0` — guard before calling
- Causal convolutions: left-pad by `(kernel-1) * dilation` only — no right padding

## Encoders

All in `lagrangian_regime_net.py`. All accept `(batch, T, F)` and return `(batch, encoder_dim)`.

| Encoder | Key params | Notes |
|---------|-----------|-------|
| `mlp` | `hidden_dim`, `encoder_dim` | Flatten → 2-layer MLP |
| `conv1d` | `conv_channels`, `conv_kernel_size`, `encoder_dim` | 2 causal conv layers, last-step readout |
| `tcn` | `tcn_channels`, `tcn_kernel_size`, `tcn_dilations`, `encoder_dim` | Residual blocks, dilations [1,2,4,8] |
| `hybrid_conv` | `conv_kernel_size`, `encoder_dim` | 3 layers F→64→128→64, last-step readout |

`CausalConv1d` left-pads by `(kernel-1)*dilation` to preserve sequence length — always causal.

## Config system (Hydra)

Base config: `configs/config.yaml`. Model configs in `configs/model/`.

```bash
# Single-horizon Lagrangian
python -m src.training.train_lagrangian model=lagrangian_conv1d

# Multi-horizon Lagrangian (v6/v7 style)
python -m src.training.train_lagrangian_mh model=lagrangian_v6

# Subset run (folds 20-40)
python -m src.training.train_lagrangian model=lagrangian_conv1d +fold_start=20 +fold_end=40

# Baselines
python -m src.training.train_baseline model=xgb
python -m src.training.train_rnn model=lstm
python -m src.training.train_node model=node
```

New keys not in the config struct must use `+key=value` prefix. Access optional keys in code with `getattr(cfg.model, 'key', default)`.

## Features

**Base (37 features)**: rolling returns, vol, momentum, correlations across 4 assets  
**Econophysics (29 features)**: vol-of-vol, kurtosis, skew, tail ratio, sign autocorr, sq/abs return means, avg cross-corr — all causal, enabled via `use_econophysics_features: true` in config

Feature expansion: 37 → 66 when econophysics enabled. `input_dim` in config is overridden at runtime by actual `n_features`.

## Labels

4-class composite: 2×2 quadrant of (return state) × (vol state), both thresholded at rolling medians. Smoothing applied by default. `label_asset: SPY`.

Multi-horizon: `MultiHorizonLabeler` wraps one `QuantileLabeler` per horizon. Last `h` rows NaN per horizon (correct, no leakage). All horizons must be valid for a sample to be included in multi-horizon training.

## Experiment history & lessons

| Variant | Mean F1 (subset 20-40) | Key change | Outcome |
|---------|----------------------|------------|---------|
| v3 (MLP, latent=16) | 0.358 | latent_dim=16, n_steps=8 | Best single-horizon MLP |
| v5 (vector damp + coord xform) | 0.350 | More expressive dynamics | Regressed — added instability |
| v6 (MLP + econo + MH) | 0.385 | Econophysics + multi-horizon | +2.7pp — best subset result |
| conv1d (37 feat, single-horizon) | 0.381 | New encoder | +2.3pp over MLP baseline |
| v7 (conv1d + econo + MH) | 0.354 | Combined everything | Regressed — MH loss hurt conv1d |
| v7b (conv1d + econo 66ch) | 0.357 | Drop MH, keep econo | Regressed — econo features add noise |
| TCN (no econo) | 0.367 | TCN encoder, 37 feat | -1.4pp vs conv1d — TCN not better here |
| v8 (conv1d + skip connection) | 0.379 | Last-row skip to latent | -0.2pp — skip redundant, conv already uses last step |
| **Ensemble (XGBoost + conv1d avg)** | **0.383 subset / 0.4046 full** | **Best Lagrangian-family result** | +0.7pp over conv1d on 71 folds |

**Key lessons:**
- Flatten+MLP encoder discards temporal order — conv1d encoder was a first-order fix (+2.8pp on 71 folds)
- Combining conv1d + multi-horizon loss degraded performance (v7 < v6 and < standalone conv1d)
- Econophysics features consistently hurt all conv variants by ~2pp — they add noise the model can't filter
- XGBoost dominates via realized vol (21d/63d) + cross-asset correlations (63d) + VIX level — top 3 groups = 59% importance
- XGBoost fails in early folds (2007-2008) due to insufficient training data; conv1d holds up better there (Pearson r = 0.29)
- Ensemble (prob avg) beats both individual models: lower variance, complementary errors across fold types
- Skip connection is redundant: conv1d's last-step readout already captures current-state features
- Change one axis at a time: encoder, features, or loss objective
- Multi-horizon supervision is an experimental extension, not a default

## Ensemble model

`train_ensemble.py` trains XGBoost (flat features) + Lagrangian conv1d (sequential) per fold, averages softmax probabilities at test time.

```bash
python -m src.training.train_ensemble
python -m src.training.train_ensemble +fold_start=20 +fold_end=40
```

No model config required — uses hardcoded `_LAG_CFG` and `_XGB_CFG` dicts matching the best known hyperparameters.

## Tests

121 tests in `tests/test_shapes.py`. Must stay green before any commit.

Key test groups: XGBoost, LSTM/GRU, NODE, Lagrangian base, v5 (vector damping), v6 MH model, econophysics features, multi-horizon labels, encoder ablation (4 variants × 7 tests + 3 standalone), skip connection (5 tests).

```bash
python -m pytest tests/test_shapes.py -v        # full suite
python -m pytest tests/test_shapes.py -k "skip_connection"  # skip connection tests only
```
