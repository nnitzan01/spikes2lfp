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
