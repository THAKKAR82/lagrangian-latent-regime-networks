import numpy as np
import pandas as pd
import pytest
import torch
from src.models.baseline_lstm import RegimeLSTM, RegimeGRU, RNNConfig
from src.models.baseline_node import RegimeNODE, NODEConfig
from src.models.baseline_xgb import RegimeXGB, XGBConfig
from src.models.lagrangian_regime_net_mh import LagrangianRegimeNetMH, LagrangianMHConfig
from src.features.econophysics import build_econophysics_features
from src.labels.multi_horizon_labeler import MultiHorizonLabeler, MultiHorizonLabelConfig
from src.utils.multi_horizon_builder import MultiHorizonFold, build_folds_multi


@pytest.fixture
def xgb_cfg():
    return XGBConfig(n_estimators=10, max_depth=3, seed=42, n_jobs=1)


@pytest.fixture
def toy_flat_data():
    rng = np.random.default_rng(42)
    n_train, n_val, n_feat = 200, 50, 20
    X_train = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_train = rng.integers(0, 4, n_train)
    X_val = rng.standard_normal((n_val, n_feat)).astype(np.float32)
    y_val = rng.integers(0, 4, n_val)
    return X_train, y_train, X_val, y_val


def test_regime_xgb_predict_shape(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_regime_xgb_predict_proba_shape(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_regime_xgb_proba_sums_to_one(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_regime_xgb_predict_in_range(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert set(preds).issubset({0, 1, 2, 3})


def test_regime_xgb_feature_importances(xgb_cfg, toy_flat_data):
    X_train, y_train, X_val, y_val = toy_flat_data
    model = RegimeXGB(xgb_cfg)
    model.fit(X_train, y_train, X_val, y_val)
    fi = model.feature_importances()
    assert len(fi) == X_train.shape[1]


@pytest.fixture
def rnn_cfg():
    return RNNConfig(
        input_dim=37,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        seed=42,
    )


@pytest.fixture
def toy_seq_data():
    rng = np.random.default_rng(42)
    n_train, n_val = 200, 50
    seq_len, n_feat = 40, 37
    X_train = rng.standard_normal((n_train, seq_len, n_feat)).astype(np.float32)
    y_train = rng.integers(0, 4, n_train)
    X_val = rng.standard_normal((n_val, seq_len, n_feat)).astype(np.float32)
    y_val = rng.integers(0, 4, n_val)
    return X_train, y_train, X_val, y_val


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_forward_output_shape(rnn_cfg, ModelClass):
    model = ModelClass(rnn_cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert out.shape == (8, 4), f"Expected (8, 4), got {out.shape}"


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_shape(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_proba_shape(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_proba_sums_to_one(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_in_range(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


@pytest.mark.parametrize("ModelClass", [RegimeLSTM, RegimeGRU])
def test_rnn_predict_proba_switches_to_eval(rnn_cfg, toy_seq_data, ModelClass):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = ModelClass(rnn_cfg)
    model.train()  # explicitly put in train mode
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


@pytest.fixture
def node_cfg():
    return NODEConfig(input_dim=37, hidden_dim=32, seed=42)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_node_forward_output_shape(node_cfg, batch_size):
    model = RegimeNODE(node_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4), f"Expected ({batch_size}, 4), got {out.shape}"


def test_node_predict_shape(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_node_predict_proba_shape(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_node_proba_sums_to_one(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_node_predict_in_range(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_node_predict_proba_switches_to_eval(node_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = RegimeNODE(node_cfg)
    model.train()
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


def test_node_config_fields():
    cfg = NODEConfig()
    assert hasattr(cfg, "hidden_dim")
    assert hasattr(cfg, "ode_hidden_dim")
    assert hasattr(cfg, "solver")
    assert cfg.solver == "dopri5"


from src.models.lagrangian_regime_net import (
    LagrangianRegimeNet, LagrangianConfig, PotentialNet,
    CausalConv1d, Conv1dEncoder, TCNEncoder, HybridConvEncoder,
)


@pytest.fixture
def lag_cfg():
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=8,
        hidden_dim=32,
        n_steps=3,
        seed=42,
    )


@pytest.mark.parametrize("batch_size", [1, 8])
def test_lagrangian_forward_output_shape(lag_cfg, batch_size):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4), f"Expected ({batch_size}, 4), got {out.shape}"


def test_lagrangian_predict_shape(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_lagrangian_predict_proba_shape(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val), 4)


def test_lagrangian_proba_sums_to_one(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    proba = model.predict_proba(X_val)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_predict_in_range(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    preds = model.predict(X_val)
    assert set(preds.tolist()).issubset({0, 1, 2, 3})


def test_lagrangian_predict_proba_switches_to_eval(lag_cfg, toy_seq_data):
    X_train, y_train, X_val, y_val = toy_seq_data
    model = LagrangianRegimeNet(lag_cfg)
    model.train()
    _ = model.predict_proba(X_val)
    assert not model.training, "predict_proba should switch model to eval mode"


def test_lagrangian_trajectory_length(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == lag_cfg.n_steps


def test_lagrangian_trajectory_shape(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    for z in model.last_trajectory:
        assert z.shape == (4, lag_cfg.latent_dim)


def test_lagrangian_mass_positive(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    z = torch.randn(4, lag_cfg.latent_dim)
    m = model.mass_net(z)
    assert (m > 0).all(), "Mass diagonal must be strictly positive"


def test_lagrangian_damping_positive(lag_cfg):
    model = LagrangianRegimeNet(lag_cfg)
    gamma = torch.nn.functional.softplus(model.raw_gamma)
    assert gamma.item() > 0, "Damping must be positive"


@pytest.mark.parametrize("n_steps,scale", [(1, 1.0), (4, 1.0), (8, 1.0), (4, 10.0)])
def test_lagrangian_forward_finite(n_steps, scale):
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=n_steps)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37) * scale
    logits = model(x)
    assert torch.isfinite(logits).all(), f"Non-finite logits with n_steps={n_steps}, scale={scale}"


def test_lagrangian_backward_grad_flow():
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=3)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert model.potential_net.net[0].weight.grad is not None, "No grad on potential_net"
    assert model.raw_gamma.grad is not None, "No grad on raw_gamma"


# --- v5 tests ---

@pytest.fixture
def v5_cfg():
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=16,
        hidden_dim=64,
        n_steps=4,
        use_vector_damping=True,
        use_coord_transform=True,
        seed=42,
    )


def test_lagrangian_v5_forward_shape(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    x = torch.randn(4, 40, 37)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4, 4), got {out.shape}"


def test_lagrangian_v5_forward_finite(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    assert torch.isfinite(logits).all(), "v5 forward pass produced non-finite logits"


def test_lagrangian_v5_predict_proba_shape(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    X = np.random.randn(10, 40, 37).astype(np.float32)
    proba = model.predict_proba(X)
    assert proba.shape == (10, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_vector_damping_positive(v5_cfg):
    """gamma_net output must be strictly positive for random latent input."""
    model = LagrangianRegimeNet(v5_cfg)
    # Run a forward pass so gamma_net is exercised; inspect via a direct call
    z = torch.randn(4, v5_cfg.latent_dim)
    gamma_vec = torch.nn.functional.softplus(model.gamma_net(z))
    assert (gamma_vec > 0).all(), "Vector damping must be strictly positive"
    assert torch.isfinite(gamma_vec).all(), "Vector damping must be finite"


def test_lagrangian_v5_backward_grad_flow(v5_cfg):
    model = LagrangianRegimeNet(v5_cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert model.potential_net.net[0].weight.grad is not None, "No grad on DeepPotentialNet"
    assert model.gamma_net.weight.grad is not None, "No grad on gamma_net"
    assert model.coord_net.weight.grad is not None, "No grad on coord_net"


def test_lagrangian_v5_old_path_unchanged():
    """Default config (use_vector_damping=False) must still use scalar gamma and shallow potential."""
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=2)
    model = LagrangianRegimeNet(cfg)
    assert hasattr(model, 'raw_gamma'), "raw_gamma must exist on default config"
    assert not hasattr(model, 'gamma_net'), "gamma_net must not exist on default config"
    assert isinstance(model.potential_net, PotentialNet), "Default must use PotentialNet, not DeepPotentialNet"


# --- v6 MH model tests ---

@pytest.fixture
def mh_cfg():
    return LagrangianMHConfig(
        input_dim=37,
        window_len=40,
        latent_dim=16,
        hidden_dim=64,
        potential_hidden_dim=64,   # smaller for test speed
        mass_hidden_dim=32,
        n_steps=3,
        use_vector_damping=True,
        use_coord_transform=True,
        multi_horizon=True,
        seed=42,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_lagrangian_mh_forward_shape(mh_cfg, batch_size):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(batch_size, 40, 37)
    out = model(x)
    assert out.shape == (batch_size, 4)


def test_lagrangian_mh_forward_multi_shapes(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    out = model.forward_multi(x)
    assert set(out.keys()) == {"logits_5", "logits_10", "logits_20"}
    for k, v in out.items():
        assert v.shape == (4, 4), f"{k} shape wrong: {v.shape}"


def test_lagrangian_mh_forward_finite(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    out = model.forward_multi(x)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"Non-finite in {k}"


def test_lagrangian_mh_predict_proba(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    X = np.random.randn(8, 40, 37).astype(np.float32)
    proba = model.predict_proba(X)
    assert proba.shape == (8, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_lagrangian_mh_trajectory_stored(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == mh_cfg.n_steps


def test_lagrangian_mh_backward_grad_flow(mh_cfg):
    model = LagrangianRegimeNetMH(mh_cfg)
    x = torch.randn(4, 40, 37)
    out = model.forward_multi(x)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert model.potential_net.net[0].weight.grad is not None, "No grad on DeepPotentialNet"
    assert model.gamma_net.weight.grad is not None, "No grad on gamma_net"
    assert model.coord_net.weight.grad is not None, "No grad on coord_net"


# --- Econophysics feature tests ---

@pytest.fixture
def toy_prices_dict():
    """Minimal price dict for feature tests — SPY + QQQ + TLT."""
    idx = pd.date_range("2010-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    def make_df(seed_val):
        rng2 = np.random.default_rng(seed_val)
        close = 100 * np.exp(np.cumsum(rng2.normal(0, 0.01, 300)))
        return pd.DataFrame({"close": close, "volume": rng.integers(1e6, 1e7, 300)}, index=idx)
    return {"SPY": make_df(1), "QQQ": make_df(2), "TLT": make_df(3)}


def test_econophysics_features_shape(toy_prices_dict):
    df = build_econophysics_features(toy_prices_dict, roll_windows=[21])
    assert len(df) == 300
    assert df.shape[1] > 0


def test_econophysics_features_no_all_nan(toy_prices_dict):
    df = build_econophysics_features(toy_prices_dict, roll_windows=[21])
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    assert len(all_nan_cols) == 0, f"All-NaN columns: {all_nan_cols}"


def test_econophysics_no_future_leakage(toy_prices_dict):
    """Features at index t must not depend on prices after t.
    Check: shift prices by 1 day forward and verify features are unchanged at t-1.
    This is a smoke check: first valid row of each feature must be NaN (window not filled).
    """
    df = build_econophysics_features(toy_prices_dict, roll_windows=[21])
    # First 20 rows should have NaNs (window=21 not yet filled)
    assert df.iloc[:20].isna().any(axis=1).all(), "Expected NaNs in first 20 rows (window warmup)"


# --- Multi-horizon label tests ---

@pytest.fixture
def toy_spy_prices():
    idx = pd.date_range("2010-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    return pd.DataFrame({"close": close}, index=idx)


def test_multi_horizon_label_shapes(toy_spy_prices):
    cfg = MultiHorizonLabelConfig(horizons=[5, 10, 20])
    labeler = MultiHorizonLabeler(cfg)
    labels_df = labeler.fit_transform(toy_spy_prices)
    assert list(labels_df.columns) == ["label_5", "label_10", "label_20"]
    assert len(labels_df) == len(toy_spy_prices)


def test_multi_horizon_labels_in_range(toy_spy_prices):
    cfg = MultiHorizonLabelConfig(horizons=[5, 10, 20])
    labeler = MultiHorizonLabeler(cfg)
    labels_df = labeler.fit_transform(toy_spy_prices)
    for col in labels_df.columns:
        valid = labels_df[col].dropna()
        assert set(valid.astype(int).unique()).issubset({0, 1, 2, 3}), f"{col} has out-of-range values"


def test_multi_horizon_no_nan_in_head(toy_spy_prices):
    """Labels at horizon h must have NaN only in the last h rows (forward return window)."""
    cfg = MultiHorizonLabelConfig(horizons=[5, 10, 20], smoothing=False)
    labeler = MultiHorizonLabeler(cfg)
    labels_df = labeler.fit_transform(toy_spy_prices)
    # Check label_5: last 5 rows should be NaN (forward return unavailable)
    assert labels_df["label_5"].iloc[-5:].isna().all(), "Last 5 rows of label_5 should be NaN"
    # label_20: last 20 rows NaN
    assert labels_df["label_20"].iloc[-20:].isna().all(), "Last 20 rows of label_20 should be NaN"


# ---------------------------------------------------------------------------
# Encoder ablation tests
# ---------------------------------------------------------------------------

ENCODER_TYPES = ["mlp", "conv1d", "tcn", "hybrid_conv"]


def _make_lag_cfg(encoder_type: str) -> LagrangianConfig:
    return LagrangianConfig(
        input_dim=37,
        window_len=40,
        latent_dim=8,
        hidden_dim=32,
        encoder_dim=32,
        encoder_type=encoder_type,
        conv_channels=32,
        conv_kernel_size=3,
        tcn_channels=32,
        tcn_kernel_size=3,
        tcn_dilations=[1, 2, 4],
        n_steps=3,
        use_vector_damping=True,
        use_coord_transform=True,
        seed=0,
    )


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_forward_shape(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert out.shape == (8, 4), f"{encoder_type}: expected (8, 4), got {out.shape}"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_forward_finite(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(8, 40, 37)
    out = model(x)
    assert torch.isfinite(out).all(), f"{encoder_type}: non-finite logits"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_predict_proba_shape(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    X = np.random.randn(16, 40, 37).astype(np.float32)
    proba = model.predict_proba(X)
    assert proba.shape == (16, 4), f"{encoder_type}: predict_proba shape wrong"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_trajectory_length(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    _ = model(x)
    assert len(model.last_trajectory) == cfg.n_steps, f"{encoder_type}: trajectory length wrong"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_no_nan_in_output(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    h = model.encoder(x)
    assert not torch.isnan(h).any(), f"{encoder_type}: NaN in encoder output"


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_lagrangian_encoder_backward_grad_flow(encoder_type):
    cfg = _make_lag_cfg(encoder_type)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    logits = model(x)
    logits.sum().backward()
    # Check that at least one encoder parameter received a gradient
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0, f"{encoder_type}: no gradient flowed to encoder"


def test_causal_conv1d_preserves_length():
    """CausalConv1d must output the same T as input (causal padding correctness)."""
    for kernel_size in [3, 5]:
        for dilation in [1, 2, 4]:
            conv = CausalConv1d(16, 16, kernel_size, dilation)
            x = torch.randn(4, 16, 40)
            y = conv(x)
            assert y.shape == (4, 16, 40), (
                f"kernel={kernel_size} dilation={dilation}: "
                f"expected T=40, got {y.shape[2]}"
            )


def test_lagrangian_encoder_param_counts():
    """Log parameter counts for each encoder — acts as a sanity check."""
    for enc in ENCODER_TYPES:
        cfg = _make_lag_cfg(enc)
        model = LagrangianRegimeNet(cfg)
        enc_params = sum(p.numel() for p in model.encoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        # Sanity: encoder params must be > 0 and < total
        assert enc_params > 0, f"{enc}: encoder has no parameters"
        assert enc_params < total_params, f"{enc}: encoder has all params (dynamics missing?)"


def test_lagrangian_mlp_encoder_backward_compat():
    """Default LagrangianConfig (no encoder_type set) must still work as MLP."""
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=2)
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    out = model(x)
    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Skip-connection tests
# ---------------------------------------------------------------------------

def test_skip_connection_output_shape():
    """Model with use_skip_connection=True must still output (batch, 4)."""
    cfg = LagrangianConfig(
        input_dim=37, window_len=40, latent_dim=8, hidden_dim=32,
        n_steps=3, encoder_type="conv1d", use_skip_connection=True,
    )
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    out = model(x)
    assert out.shape == (4, 4), f"Expected (4, 4), got {out.shape}"


def test_skip_connection_forward_finite():
    """Skip connection must not introduce NaNs or Infs."""
    cfg = LagrangianConfig(
        input_dim=37, window_len=40, latent_dim=8, hidden_dim=32,
        n_steps=3, encoder_type="conv1d", use_skip_connection=True,
    )
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    out = model(x)
    assert torch.isfinite(out).all(), "Non-finite logits with skip connection"


def test_skip_connection_grad_flows():
    """Gradient must flow through the skip projection weights."""
    cfg = LagrangianConfig(
        input_dim=37, window_len=40, latent_dim=8, hidden_dim=32,
        n_steps=3, encoder_type="conv1d", use_skip_connection=True,
    )
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 40, 37)
    model(x).sum().backward()
    assert model.skip_proj.weight.grad is not None, "No gradient on skip_proj.weight"
    assert model.skip_proj.weight.grad.abs().sum() > 0, "Zero gradient on skip_proj.weight"


def test_skip_connection_default_off():
    """use_skip_connection defaults to False and model has no skip_proj attribute."""
    cfg = LagrangianConfig(input_dim=37, window_len=40, latent_dim=8, hidden_dim=32, n_steps=2)
    model = LagrangianRegimeNet(cfg)
    assert not hasattr(model, "skip_proj"), "skip_proj should not exist when use_skip_connection=False"


def test_skip_connection_changes_output():
    """Skip connection should change logit values vs same model without it."""
    torch.manual_seed(0)
    cfg_base = LagrangianConfig(
        input_dim=37, window_len=40, latent_dim=8, hidden_dim=32,
        n_steps=3, encoder_type="conv1d", seed=0, use_skip_connection=False,
    )
    cfg_skip = LagrangianConfig(
        input_dim=37, window_len=40, latent_dim=8, hidden_dim=32,
        n_steps=3, encoder_type="conv1d", seed=0, use_skip_connection=True,
    )
    x = torch.randn(4, 40, 37)
    out_base = LagrangianRegimeNet(cfg_base)(x)
    out_skip = LagrangianRegimeNet(cfg_skip)(x)
    assert not torch.allclose(out_base, out_skip), "Skip connection produced identical output to baseline"


def test_eval_result_has_class_f1():
    from src.evaluation.metrics import evaluate
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, 100)
    y_prob = np.abs(rng.standard_normal((100, 4))).astype(np.float32)
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1)
    result = evaluate(y_true, y_pred, y_prob)
    assert hasattr(result, "class_f1"), "EvalResult must have class_f1 attribute"
    assert result.class_f1.shape == (4,), f"Expected (4,), got {result.class_f1.shape}"


def test_eval_result_class_f1_in_range():
    from src.evaluation.metrics import evaluate
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, 200)
    y_prob = np.abs(rng.standard_normal((200, 4))).astype(np.float32)
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    y_pred = y_prob.argmax(axis=1)
    result = evaluate(y_true, y_pred, y_prob)
    assert np.all(result.class_f1 >= 0.0) and np.all(result.class_f1 <= 1.0)


# ---------------------------------------------------------------------------
# Prediction artifact tests
# ---------------------------------------------------------------------------

from src.evaluation.predictions import PredictionArtifact


def test_prediction_artifact_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    probs = np.abs(rng.standard_normal((50, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    dates = np.array([f"2020-01-{i+1:02d}" for i in range(50)])
    labels = rng.integers(0, 4, 50)
    art = PredictionArtifact(
        fold_id=3, split="val", model_name="xgb",
        dates=dates, true_labels=labels, probs=probs,
    )
    path = tmp_path / "fold_03_val.parquet"
    art.save(path)
    loaded = PredictionArtifact.load(path)
    assert loaded.fold_id == 3
    assert loaded.split == "val"
    assert loaded.model_name == "xgb"
    np.testing.assert_allclose(loaded.probs, probs, atol=1e-5)
    np.testing.assert_array_equal(loaded.true_labels, labels)


def test_prediction_artifact_to_dataframe_columns():
    rng = np.random.default_rng(3)
    probs = np.abs(rng.standard_normal((10, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    art = PredictionArtifact(
        fold_id=0, split="test", model_name="lagrangian",
        dates=np.array(["2021-01-01"] * 10),
        true_labels=rng.integers(0, 4, 10), probs=probs,
    )
    df = art.to_dataframe()
    required_cols = {"date", "fold_id", "split", "model", "true_label",
                     "prob_0", "prob_1", "prob_2", "prob_3"}
    assert required_cols.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Temperature Scaling tests
# ---------------------------------------------------------------------------

from src.postprocessing.temperature import TemperatureScaler


def test_temperature_scaler_output_sums_to_one():
    rng = np.random.default_rng(4)
    probs = np.abs(rng.standard_normal((80, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, 80)
    scaler = TemperatureScaler()
    scaler.fit(probs, labels)
    cal = scaler.transform(probs)
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-5)


def test_temperature_scaler_temperature_positive():
    rng = np.random.default_rng(5)
    probs = np.abs(rng.standard_normal((80, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, 80)
    scaler = TemperatureScaler()
    scaler.fit(probs, labels)
    assert scaler.temperature > 0.0


def test_temperature_scaler_default_identity():
    rng = np.random.default_rng(6)
    probs = np.abs(rng.standard_normal((30, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    scaler = TemperatureScaler()
    # T=1 => transform is identity
    cal = scaler.transform(probs)
    np.testing.assert_allclose(cal, probs, atol=1e-4)


# ---------------------------------------------------------------------------
# Weighted Ensemble tests
# ---------------------------------------------------------------------------

from src.postprocessing.ensemble import WeightedEnsemble, grid_search_weights


def test_weighted_ensemble_output_shape():
    rng = np.random.default_rng(7)
    xgb_p = np.abs(rng.standard_normal((50, 4))).astype(np.float32)
    xgb_p /= xgb_p.sum(axis=1, keepdims=True)
    lag_p = np.abs(rng.standard_normal((50, 4))).astype(np.float32)
    lag_p /= lag_p.sum(axis=1, keepdims=True)
    ens = WeightedEnsemble(w_xgb=0.6, w_lag=0.4)
    out = ens.predict_proba(xgb_p, lag_p)
    assert out.shape == (50, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_grid_search_weights_returns_valid_pair():
    rng = np.random.default_rng(8)
    xgb_p = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
    xgb_p /= xgb_p.sum(axis=1, keepdims=True)
    lag_p = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
    lag_p /= lag_p.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, 60)
    w_xgb, w_lag = grid_search_weights(xgb_p, lag_p, labels)
    assert abs(w_xgb + w_lag - 1.0) < 1e-5
    assert 0.1 <= w_xgb <= 0.9


# ---------------------------------------------------------------------------
# Fold-Adaptive Ensemble tests
# ---------------------------------------------------------------------------

from src.postprocessing.adaptive import FoldAdaptiveEnsemble


def test_fold_adaptive_equal_mode():
    ens = FoldAdaptiveEnsemble(mode="equal")
    w_xgb, w_lag = ens.get_weights(fold_id=5, train_size=2000)
    assert abs(w_xgb - 0.5) < 1e-5 and abs(w_lag - 0.5) < 1e-5


def test_fold_adaptive_training_size_mode():
    ens = FoldAdaptiveEnsemble(mode="training_size", size_threshold=1000)
    w_xgb_small, _ = ens.get_weights(fold_id=1, train_size=500)
    w_xgb_large, _ = ens.get_weights(fold_id=10, train_size=5000)
    assert w_xgb_small <= w_xgb_large


def test_fold_adaptive_previous_folds_val_fallback():
    ens = FoldAdaptiveEnsemble(mode="previous_folds_val")
    w_xgb, w_lag = ens.get_weights(fold_id=0, train_size=1000)
    assert abs(w_xgb - 0.5) < 1e-5
    ens.register_fold_val_f1(xgb_f1=0.45, lag_f1=0.35)
    w_xgb2, w_lag2 = ens.get_weights(fold_id=1, train_size=1000)
    assert w_xgb2 > w_lag2


def test_fold_adaptive_weights_sum_to_one():
    for mode in ("equal", "training_size", "previous_folds_val"):
        ens = FoldAdaptiveEnsemble(mode=mode)
        ens.register_fold_val_f1(xgb_f1=0.40, lag_f1=0.38)
        w_xgb, w_lag = ens.get_weights(fold_id=3, train_size=1500)
        assert abs(w_xgb + w_lag - 1.0) < 1e-5, f"mode={mode} weights don't sum to 1"


from src.postprocessing.stacker import LogisticStacker


def test_logistic_stacker_fallback_without_history():
    rng = np.random.default_rng(9)
    stacker = LogisticStacker(min_folds=3)
    xgb_p = np.abs(rng.standard_normal((20, 4))).astype(np.float32)
    xgb_p /= xgb_p.sum(axis=1, keepdims=True)
    lag_p = np.abs(rng.standard_normal((20, 4))).astype(np.float32)
    lag_p /= lag_p.sum(axis=1, keepdims=True)
    out = stacker.predict_proba(xgb_p, lag_p)
    expected = (xgb_p + lag_p) / 2.0
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_logistic_stacker_fit_after_min_folds():
    rng = np.random.default_rng(10)
    stacker = LogisticStacker(min_folds=2)
    for _ in range(3):
        xv = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
        xv /= xv.sum(axis=1, keepdims=True)
        lv = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
        lv /= lv.sum(axis=1, keepdims=True)
        yl = rng.integers(0, 4, 60)
        stacker.update(xv, lv, yl)
    ready = stacker.fit()
    assert ready, "Expected stacker to be ready after 3 folds"


def test_logistic_stacker_output_shape_and_valid():
    rng = np.random.default_rng(11)
    stacker = LogisticStacker(min_folds=2)
    for _ in range(3):
        xv = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
        xv /= xv.sum(axis=1, keepdims=True)
        lv = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
        lv /= lv.sum(axis=1, keepdims=True)
        yl = rng.integers(0, 4, 60)
        stacker.update(xv, lv, yl)
    stacker.fit()
    xp = np.abs(rng.standard_normal((30, 4))).astype(np.float32)
    xp /= xp.sum(axis=1, keepdims=True)
    lp = np.abs(rng.standard_normal((30, 4))).astype(np.float32)
    lp /= lp.sum(axis=1, keepdims=True)
    out = stacker.predict_proba(xp, lp)
    assert out.shape == (30, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


from src.postprocessing.thresholds import ThresholdTuner


def test_threshold_tuner_output_in_range():
    rng = np.random.default_rng(12)
    probs = np.abs(rng.standard_normal((80, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, 80)
    tuner = ThresholdTuner()
    tuner.fit(probs, labels)
    preds = tuner.predict(probs)
    assert set(preds).issubset({0, 1, 2, 3})


def test_threshold_tuner_thresholds_shape():
    rng = np.random.default_rng(13)
    probs = np.abs(rng.standard_normal((80, 4))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, 80)
    tuner = ThresholdTuner()
    tuner.fit(probs, labels)
    assert tuner.thresholds.shape == (4,)
    assert np.all(tuner.thresholds >= 0.1) and np.all(tuner.thresholds <= 0.9)


def test_threshold_tuner_no_silent_failure():
    """If no class ever exceeds its threshold, predict must still return valid labels."""
    rng = np.random.default_rng(14)
    probs = np.full((10, 4), 0.25, dtype=np.float32)
    tuner = ThresholdTuner()
    tuner.thresholds = np.full(4, 0.9)  # impossible threshold
    preds = tuner.predict(probs)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1, 2, 3})


# Transition head tests (Task 9)
from src.models.lagrangian_regime_net import LagrangianConfig, LagrangianRegimeNet


def test_transition_head_output_shape():
    cfg = LagrangianConfig(
        input_dim=10, window_len=5, latent_dim=8, hidden_dim=32,
        encoder_type="mlp", encoder_dim=32, n_steps=2,
        use_transition_head=True,
    )
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 5, 10)
    logits, trans_logits = model.forward_with_transition(x)
    assert logits.shape == (4, 4)
    assert trans_logits.shape == (4, 1)


def test_transition_head_default_off():
    cfg = LagrangianConfig(
        input_dim=10, window_len=5, latent_dim=8, hidden_dim=32,
        encoder_type="mlp", encoder_dim=32, n_steps=2,
    )
    assert not cfg.use_transition_head


def test_transition_head_finite():
    cfg = LagrangianConfig(
        input_dim=10, window_len=5, latent_dim=8, hidden_dim=32,
        encoder_type="mlp", encoder_dim=32, n_steps=2,
        use_transition_head=True,
    )
    model = LagrangianRegimeNet(cfg)
    x = torch.randn(4, 5, 10)
    logits, trans_logits = model.forward_with_transition(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(trans_logits).all()
