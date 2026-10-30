import numpy as np
import pandas as pd

from analysis.stimulus_clock import (apply_template, bin_conditions, fit_template, per_presentation_r2,
                                     presentation_conditions, presentation_segments, r2, residual_r2,
                                     shuffle_presentations)


def test_conditions_and_lags():
    stim = pd.DataFrame({'image_name': ['a', 'b', 'omitted', 'a'], 'is_change': [False, True, False, False]})
    np.testing.assert_array_equal(presentation_conditions(stim), ['a', 'change', 'omitted', 'a'])
    cond, lag, names = bin_conditions(np.array([2, 5, 9]), np.array(['a', 'b', 'a']), n_bins=14, max_lag=4)
    assert names == ['a', 'b']
    np.testing.assert_array_equal(cond, [-1, -1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, -1])
    np.testing.assert_array_equal(lag, [-1, -1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, -1])


def test_template_recovers_evoked_response_from_training_bins_only():
    rng = np.random.default_rng(0)
    period, n_events = 10, 60
    onsets = np.arange(n_events) * period
    labels = np.where(np.arange(n_events) % 2 == 0, 'a', 'b')
    evoked = {'a': np.sin(np.arange(period)), 'b': np.cos(np.arange(period))}
    y = np.concatenate([evoked[label] for label in labels]) + 0.1 * rng.normal(size=n_events * period)
    cond, lag, names = bin_conditions(onsets, labels, len(y), period)
    train = np.arange(len(y) // 2)
    table, fallback = fit_template(y, cond, lag, train, len(names), period)
    np.testing.assert_allclose(table[0], evoked['a'], atol=0.06)
    T = apply_template(table, fallback, cond, lag)
    test = np.arange(len(y) // 2, len(y))
    assert r2(y[test], T[test]) > 0.95
    # the template explains the evoked part, so a perfect copy of the template leaves residual R2 = 0
    assert abs(residual_r2(y[test], T[test], T[test])) < 1e-12
    assert residual_r2(y[test], y[test], T[test]) == 1


def test_template_falls_back_to_mean_and_handles_multidimensional_targets():
    y = np.arange(12, dtype=float).reshape(6, 2)
    cond = np.array([0, 0, -1, 1, 1, 1])
    lag = np.array([0, 1, -1, 0, 1, 2])
    table, fallback = fit_template(y, cond, lag, np.arange(3), n_conditions=2, max_lag=3)
    T = apply_template(table, fallback, cond, lag)
    np.testing.assert_allclose(T[:2], y[:2])
    np.testing.assert_allclose(T[2:], np.broadcast_to(y[:3].mean(axis=0), (4, 2)))   # unseen / no condition


def test_shuffle_keeps_condition_and_breaks_pairing():
    rng = np.random.default_rng(1)
    onsets = np.arange(0, 100, 10)
    segments = presentation_segments(onsets, 100, min_len=5, max_len=10)
    conditions = np.array(['a', 'b'] * 5)
    X = np.repeat(np.arange(10), 10)[:, None].astype(float)          # each segment holds its own index
    Xs = shuffle_presentations(X, segments, conditions, rng)
    donors = Xs[segments[:, 0], 0].astype(int)
    np.testing.assert_array_equal(conditions[donors], conditions)  # same condition
    assert sorted(donors) == list(range(10)) and (donors != np.arange(10)).any()
    np.testing.assert_allclose(per_presentation_r2(X[:, 0] + np.arange(100) % 10, X[:, 0] + np.arange(100) % 10,
                                                   segments), 1)
