import numpy as np

from analysis import coupling_measures as cm


def data(seed=0, n_bins=3000, n_units=6):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.1, size=(n_bins, n_units))
    counts[:, 5] = 0                                                   # a silent unit
    y = np.convolve(counts[:, 0], np.r_[np.zeros(3), np.ones(5)], mode='full')[:n_bins]   # unit 0 drives y 3-7 bins later
    Y = np.column_stack([y + 0.1 * rng.normal(size=n_bins), rng.normal(size=n_bins)])
    return counts, (Y - Y.mean(axis=0)) / Y.std(axis=0)


def test_lagged_products_match_brute_force():
    counts, Y = data()
    rows, lags = np.arange(100, 2900), np.array([0, 2, 5, 9])
    lp = cm.lagged_products(counts, Y, rows, lags)
    t = rows[np.isin(rows + 9, rows)]
    for i, l in enumerate(lags):
        sta = counts[t].T @ Y[t + l] / counts[t].sum(axis=0)[:, None]
        np.testing.assert_allclose(cm.spike_triggered_target(lp)[i, :5], sta[:5], rtol=1e-5)
        for u in range(5):
            for k in range(2):
                r = np.corrcoef(counts[t, u], Y[t + l, k])[0, 1]
                np.testing.assert_allclose(cm.spike_lfp_correlation(lp)[i, u, k], r, rtol=1e-4, atol=1e-6)
    assert np.isnan(cm.spike_triggered_target(lp)[:, 5]).all()
    stlfp = cm.max_abs_over_lags(cm.spike_triggered_target(lp))
    assert stlfp[5, 0] == 0 and np.argmax(stlfp[:, 0]) == 0               # the driving unit ranks first


def test_population_coupling_matches_brute_force():
    rng = np.random.default_rng(1)
    shared = rng.poisson(0.3, size=4000)
    counts = rng.poisson(0.1, size=(4000, 5)) + shared[:, None] * np.array([1, 1, 0, 0, 0])[None, :]
    rows = np.arange(4000)
    r = cm.population_coupling(counts, rows, coarse_bins=5)
    x = counts.reshape(800, 5, 5).sum(axis=1)
    for u in range(5):
        np.testing.assert_allclose(r[u], np.corrcoef(x[:, u], x.sum(axis=1) - x[:, u])[0, 1], rtol=1e-6)
    assert r[:2].min() > r[2:].max()


def test_firing_rate_and_rank_order():
    counts, _ = data()
    np.testing.assert_allclose(cm.firing_rate(counts, np.arange(3000), 0.004), counts.mean(axis=0) / 0.004)
    rng = np.random.default_rng(2)
    order = cm.rank_order(np.array([0.5, 0, 2.0, 0, np.nan, 1.0]), rng)
    assert list(order[:3]) == [2, 5, 0] and set(order[3:5]) == {1, 3} and order[5] == 4


def test_ridge_weight_norms():
    W = np.arange(12, dtype=float).reshape(6, 2)                          # 3 units x 2 basis functions, 2 targets
    norms = cm.ridge_weight_norms(W, 2)
    np.testing.assert_allclose(norms[1, 0], np.hypot(4, 6))
