"""
Per-unit summaries of a (time bins x units) attribution matrix.

With the zero IG baseline, attribution is exactly 0 in bins where a unit did not spike.
"""
import warnings

import numpy as np

from analysis.ranking import EPSILON


def relative_attribution(attrs, eps=EPSILON):
    """Each bin's attributions divided by that bin's summed |attribution| (signed; bins with no activity -> 0)."""
    total = np.nansum(np.abs(attrs), axis=1, keepdims=True)
    return np.divide(attrs, total, out=np.zeros_like(attrs, dtype=np.float64), where=total > eps)


def mean_attribution(attrs):
    """Mean signed attribution over all bins, (units,). Scales with how often a unit fires."""
    return np.mean(attrs, axis=0)


def relative_mean_attribution(attrs, eps=EPSILON):
    """Mean over all bins of the unit's share of the bin's total |attribution|, (units,)."""
    return np.mean(relative_attribution(attrs, eps), axis=0)


def ipi_mean_attribution(attrs, eps=EPSILON):
    """
    Instantaneous predictive influence: the unit's mean relative attribution over the bins in which
    it was active only (per-spike share, largely independent of firing rate), (units,).
    Units that never fire get 0.
    """
    relative = relative_attribution(attrs, eps)
    relative[np.abs(attrs) < eps] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)    # all-NaN columns: units that never fire
        result = np.nanmean(relative, axis=0)
    return np.nan_to_num(result, nan=0.0)
