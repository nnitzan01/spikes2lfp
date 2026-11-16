"""
Statistics with the mouse as the independent unit (reviewer 3, comment 9). Sessions are nested in mice (41 of
the 48 mice were recorded twice) and units in sessions, so:

  mouse_means(values, mice)                 average each mouse's sessions -> (values, mice) per mouse
  wilcoxon_mice(diff, mice)                 Wilcoxon signed-rank over mice of the per-mouse mean difference
  cluster_bootstrap(values, mice, ...)      CI of a statistic, resampling mice with replacement (with all their
                                            sessions); hierarchical=True also resamples sessions within each mouse
  fdr(p)                                    Benjamini-Hochberg q-values over a family of tests
  mixed_model(frame, formula)               linear mixed model with random intercepts for mice and for sessions
                                            within mice (unit-level data)
  describe(diff, mice)                      'median [95% CI], k/n mice > 0, p' for one paired difference
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

N_BOOT = 10_000


def mouse_means(values, mice):
    """values (sessions, ...) and their mice -> (per-mouse mean over sessions (mice, ...), mouse ids); NaN sessions ignored."""
    values, mice = np.asarray(values, float), np.asarray(mice)
    ids = np.unique(mice)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)              # mice whose sessions are all NaN
        means = np.stack([np.nanmean(values[mice == m], axis=0) for m in ids])
    return means, ids


def wilcoxon_mice(diff, mice):
    """(median of the per-mouse means, mice > 0, mice, two-sided Wilcoxon p over mice)."""
    m, _ = mouse_means(diff, mice)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return np.nan, 0, 0, np.nan
    p = wilcoxon(m).pvalue if np.any(m != 0) else 1.0
    return float(np.median(m)), int(np.sum(m > 0)), len(m), float(p)


def cluster_bootstrap(values, mice, stat=np.median, n_boot=N_BOOT, rng=None, hierarchical=False, ci=95):
    """
    Statistic of the per-mouse means and its percentile CI from resampling mice with replacement (and, if
    hierarchical, sessions within each resampled mouse). Returns (stat, low, high).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    values, mice = np.asarray(values, float), np.asarray(mice)
    ok = np.isfinite(values)
    values, mice = values[ok], mice[ok]
    ids = np.unique(mice)
    if len(ids) == 0:
        return np.nan, np.nan, np.nan
    per_mouse = [values[mice == m] for m in ids]
    point = stat(np.array([v.mean() for v in per_mouse]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(len(ids), size=len(ids))
        if hierarchical:
            boots[b] = stat(np.array([rng.choice(per_mouse[i], len(per_mouse[i])).mean() for i in draw]))
        else:
            boots[b] = stat(np.array([per_mouse[i].mean() for i in draw]))
    low, high = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(point), float(low), float(high)


def fdr(p):
    """Benjamini-Hochberg q-values (NaN p-values stay NaN and are not counted)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    ranked = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    q[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def mixed_model(frame, formula, mouse='mouse', session='session'):
    """
    statsmodels MixedLM of `formula` with a random intercept per mouse and a variance component for sessions
    within mice. Returns the fitted result (params, bse, pvalues, conf_int()).
    """
    import statsmodels.formula.api as smf
    frame = frame.dropna(subset=[c for c in frame.columns if c in formula or c in (mouse, session)]).reset_index(drop=True)
    model = smf.mixedlm(formula, frame, groups=mouse, re_formula='1',
                        vc_formula={session: f'0 + C({session})'})
    return model.fit(reml=True, method='lbfgs')


def describe(diff, mice, rng=None):
    """'median [95% CI] (k/n mice > 0, p = ...)' for a per-session paired difference, at the mouse level."""
    med, pos, n, p = wilcoxon_mice(diff, mice)
    _, low, high = cluster_bootstrap(diff, mice, rng=rng)
    return f'{med:+.3f} [{low:+.3f}, {high:+.3f}] ({pos}/{n} mice > 0, p = {p:.1e})'


def mouse_ids(sessions):
    """Mouse of each session id (tables/sessions.csv); a session missing from the table is its own mouse."""
    import paths
    meta = pd.read_csv(paths.REPO_DIR / 'tables' / 'sessions.csv').set_index('session_id')['mouse_id']
    return np.array([meta.get(s, s) for s in sessions])
