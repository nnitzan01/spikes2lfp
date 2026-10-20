import torch

import config
import models


def test_build_lstm_matches_config_and_runs():
    model = models.build_lstm(input_size=7, device='cpu', seqlength=20)
    assert model.model.num_hidden == config.LSTM_KWARGS['num_hidden']
    assert model.model.conv == config.LSTM_KWARGS['conv']
    y = model.evaluate(torch.rand(40, 7))      # 2 sequences of 20 bins, flattened
    assert y.shape == (40,)


def test_linear_regression_fits_all_batches():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from preprocess_data import get_data_loaders
    rng = np.random.default_rng(0)
    X = rng.random((40, 10, 3)).astype(np.float32)
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=(40, 10))).astype(np.float32)[..., None]
    train_dl, test_dl = get_data_loaders(X[:32], X[32:], y[:32], y[32:], batch_size=4)
    model = models.LinearRegressionModel(input_size=3)
    model.train(train_dl, test_dl)
    ref = LinearRegression().fit(X[:32].reshape(-1, 3), y[:32].reshape(-1))
    np.testing.assert_allclose(model.model.coef_, ref.coef_, rtol=1e-5)


def _future_perturbation_changes_past(net, t0=10, n_units=6, seq=24):
    """True if changing inputs at bins >= t0 changes any output before t0."""
    torch.manual_seed(1)
    x = torch.rand(2, seq, n_units)
    x2 = x.clone()
    x2[:, t0:] += torch.randn(2, seq - t0, n_units)
    net.eval()
    with torch.no_grad():
        y1, y2 = (o[0] if isinstance(o, tuple) else o for o in (net(x), net(x2)))
    return not torch.allclose(y1[:, :t0], y2[:, :t0]), not torch.allclose(y1[:, t0:], y2[:, t0:])


def test_cnn_lstm_is_causal():
    changes_past, changes_future = _future_perturbation_changes_past(
        models.LSTMnet(6, 8, 1, 24, conv=True, kernel_size=3, out_channels=4))
    assert not changes_past and changes_future


def test_causal_conv_sees_exactly_kernel_size_bins():
    conv = models.CausalConv1d(1, 1, kernel_size=3)
    x = torch.zeros(1, 10, 1)
    x[0, 4, 0] = 1.0                                  # impulse at bin 4
    with torch.no_grad():
        y = conv(x)[0, :, 0] - conv(torch.zeros(1, 10, 1))[0, :, 0]
    assert torch.nonzero(y).flatten().tolist() == [4, 5, 6]   # bins t with t-2 <= 4 <= t


def test_transformer_is_causal():
    net = models.SpikingTransformer(6, seqlength=24, embedding_dim=8, num_heads=2, num_layers=1, dropout=0.0,
                                    use_conv=True, conv_kernel_size=3, conv_out_channels=8)
    changes_past, changes_future = _future_perturbation_changes_past(net)
    assert not changes_past and changes_future
