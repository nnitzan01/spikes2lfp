import numpy as np
import pandas as pd

from analysis import stats


def test_fdr_matches_benjamini_hochberg():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    q = stats.fdr(p)
    assert np.allclose(q[[0, 2, 1, 4]], [0.04, 0.0533333, 0.0533333, 0.5]) and np.isnan(q[3])


def test_mouse_level_tests_average_sessions_first():
    diff = np.array([1.0, 3.0, 2.0, -1.0, 5.0])
    mice = np.array([7, 7, 8, 9, 9])
    means, ids = stats.mouse_means(diff, mice)
    assert ids.tolist() == [7, 8, 9] and means.tolist() == [2.0, 2.0, 2.0]
    med, pos, n, p = stats.wilcoxon_mice(diff, mice)
    assert (med, pos, n) == (2.0, 3, 3)


def test_cluster_bootstrap_ci_covers_the_mean():
    rng = np.random.default_rng(1)
    mice = np.repeat(np.arange(30), 2)
    values = rng.normal(0.5, 1, 30)[mice] + rng.normal(0, 0.1, 60)       # mice differ, sessions within mice agree
    point, low, high = stats.cluster_bootstrap(values, mice, stat=np.mean, n_boot=2000, rng=rng)
    assert low < point < high and low < 0.5 < high
    _, low_h, high_h = stats.cluster_bootstrap(values, mice, stat=np.mean, n_boot=2000, rng=rng, hierarchical=True)
    assert low_h < point < high_h
    assert 'mice > 0' in stats.describe(values, mice, rng=rng)


def test_mixed_model_recovers_a_slope():
    rng = np.random.default_rng(2)
    rows = []
    for mouse in range(12):
        for session in range(2):
            offset = rng.normal(0, 1) + rng.normal(0, 0.3)
            x = rng.uniform(-1, 1, 40)
            rows.append(pd.DataFrame(dict(y=0.8 * x + offset + rng.normal(0, 0.2, 40), x=x, mouse=mouse,
                                          session=f'{mouse}_{session}')))
    fit = stats.mixed_model(pd.concat(rows), 'y ~ x')
    assert abs(fit.params['x'] - 0.8) < 0.1 and fit.pvalues['x'] < 1e-6
