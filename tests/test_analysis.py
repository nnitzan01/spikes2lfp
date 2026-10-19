import numpy as np

from analysis.attribution_summaries import ipi_mean_attribution, mean_attribution, relative_mean_attribution
from analysis.peri_events import (event_start_indices, extract_snippets, proportional_influence,
                                  regional_dominance, snippet_time_axis)
from analysis.ranking import rank_active_neurons


def _sparse_attrs(seed=0, shape=(400, 9), p=0.2):
    rng = np.random.default_rng(seed)
    attrs = rng.normal(size=shape) * (rng.random(shape) < p)
    attrs[:, 4] = 0          # a unit that never fires
    attrs[10] = 0            # a bin with no activity
    return attrs


def test_summaries_match_previous_scripts():
    attrs = _sparse_attrs()
    # save_relative_attribution.py
    total = np.nansum(np.abs(attrs), axis=1, keepdims=True)
    old_relative = np.mean(attrs / (total + 1e-10), axis=0)
    # save_instantaneous_predictive_influence.py
    norm = np.divide(attrs, total, out=np.zeros_like(attrs), where=total > 1e-10)
    norm[np.abs(attrs) < 1e-10] = np.nan
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_ipi = np.nan_to_num(np.nanmean(norm, axis=0))
    np.testing.assert_allclose(mean_attribution(attrs), attrs.mean(axis=0))
    # the old script added 1e-10 to the denominator instead of masking empty bins
    np.testing.assert_allclose(relative_mean_attribution(attrs), old_relative, rtol=1e-6)
    np.testing.assert_allclose(ipi_mean_attribution(attrs), old_ipi)
    assert ipi_mean_attribution(attrs)[4] == 0


def test_regional_dominance_matches_previous_loop():
    attrs = _sparse_attrs(1, shape=(600, 12))
    ranks = rank_active_neurons(attrs)
    active = np.abs(attrs) > 1e-10
    labels = np.array(list('AABBBCCCCDDD'))
    starts = [0, 150, 300, 420]
    rank_snip, act_snip = extract_snippets(ranks, starts, 100), extract_snippets(active, starts, 100)

    dominance, regions = regional_dominance(rank_snip, act_snip, labels)

    # calculate_ipd.py loop
    for trial in range(len(starts)):
        with np.errstate(divide='ignore'):
            inv = np.where(act_snip[trial], 1.0 / rank_snip[trial].astype(np.float32), 0.0)
        for r, region in enumerate(regions):
            mask = labels == region
            num, den = inv[:, mask].sum(1), act_snip[trial][:, mask].sum(1)
            expected = np.where(den > 0, num / np.maximum(den, 1), 0.0)
            np.testing.assert_allclose(dominance[trial, r], expected, rtol=1e-6)
    share = proportional_influence(dominance.mean(axis=0))
    np.testing.assert_allclose(share.sum(axis=0)[share.sum(axis=0) > 0], 1, rtol=1e-6)


def test_event_windows():
    timestamps = np.arange(1000) * 0.004
    t = snippet_time_axis((-0.25, 0.5), 0.004)
    starts, kept = event_start_indices(timestamps, [0.1, 1.0, 3.9], (-0.25, 0.5), len(t))
    np.testing.assert_array_equal(kept, [1.0])            # first starts too early, last runs past the end
    assert starts[0] == 188 and len(t) == 188     # first bin edge at or after 0.75 s, as before
