import numpy as np

from analysis.lagged_ridge import (RidgePath, fit_select_predict, lagged_features, r2_columns,
                                   raised_cosine_basis)


def test_basis_tiles_lags_and_is_normalised():
    basis = raised_cosine_basis(250, 8)
    assert basis.shape == (250, 8)
    np.testing.assert_allclose(basis.sum(axis=0), 1)
    assert (basis.sum(axis=1) > 0).all()                               # every lag is covered
    assert np.argmax(basis[:, 0]) == 0 and np.argmax(basis[:, -1]) == 249
    assert np.argmax(basis[:, 1]) < 250 / 8                            # short lags get narrow kernels


def test_features_are_causal():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 3))
    basis = raised_cosine_basis(40, 4)
    rows = np.arange(300)
    F = lagged_features(Z, basis, rows, columns_per_pass=2)
    Z2 = Z.copy()
    Z2[200:] = rng.normal(size=(100, 3))                              # change only the future of bin 199
    np.testing.assert_allclose(lagged_features(Z2, basis, rows)[:200], F[:200], atol=1e-6)
    # column order: input-major, basis-minor
    np.testing.assert_allclose(F[:, 1 * 4 + 2], np.convolve(Z[:, 1], basis[:, 2])[:300], atol=1e-6)


def test_ridge_recovers_a_lagged_linear_system():
    rng = np.random.default_rng(1)
    n, n_lags = 6000, 50
    basis = raised_cosine_basis(n_lags, 6)
    Z = rng.poisson(0.3, size=(n, 4)).astype(float)
    kernel = basis @ rng.normal(size=6)                                # a filter the basis can represent
    y = np.convolve(Z[:, 0], kernel)[:n] - 0.5 * np.convolve(Z[:, 2], kernel)[:n] + 0.01 * rng.normal(size=n)
    Y = np.stack([y, rng.normal(size=n)], axis=1)                      # second target: pure noise
    rows = np.arange(n)
    pred, alpha, val_r2 = fit_select_predict(Z, Y, basis, rows[:4000], rows[4000:5000], rows[5000:], [1e-6, 1e-3, 1])
    r2 = r2_columns(Y[5000:], pred)
    assert r2[0] > 0.98 and r2[1] < 0.05
    assert val_r2.shape == (3, 2) and alpha.shape == (2,)


def test_ridge_path_matches_closed_form():
    rng = np.random.default_rng(2)
    F, Y = rng.normal(size=(200, 5)).astype(np.float32), rng.normal(size=(200, 2))
    path = RidgePath(F, Y, [0.1], chunk=37)
    Fz = (F - F.mean(axis=0)) / F.std(axis=0)
    W = np.linalg.solve(Fz.T @ Fz + 0.1 * 200 * np.eye(5), Fz.T @ (Y - Y.mean(axis=0)))
    np.testing.assert_allclose(path.weights(0.1), W, atol=1e-6)
