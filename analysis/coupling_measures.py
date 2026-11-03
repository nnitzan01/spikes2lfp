"""
Simple per-unit measures of spike-LFP relationship, to compare with attribution (reviewer 2,
comment 1; reviewer 1, "simpler methods"). All are computed on the training block only, so a
ranking chosen from them can be scored on the test block.

counts: (bins x units) spike counts; Y: (bins x targets) z-scored model targets (one column per
channel x target); rows: the bins to use (sorted). Lags are in bins; a positive lag l pairs a spike
at t with the target at t + l (spikes before the LFP, the direction the causal model uses).

  firing_rate            spikes per second
  spike-triggered target sta_u(l) = sum_t c_u(t) y(t + l) / sum_t c_u(t), in target SDs;
                         stLFP = max over lags of |sta_u(l)|, the effect of one spike
  spike-LFP correlation  r_u(l) = Pearson correlation of c_u(t) with y(t + l);
                         max over lags of |r_u(l)|, which also grows with the number of spikes
                         (a broadband stand-in for spike-LFP coherence)
  population coupling    Pearson correlation of a unit's counts with the summed counts of all
                         other units, in coarser bins (Okun et al., 2015); one value per unit
  ridge weight           norm, over the lag basis, of a unit's standardised weights in the causal
                         lagged ridge (analysis/lagged_ridge.py) fitted to the target
"""
import numpy as np
from scipy import sparse

BLOCK = 100_000      # rows of counts copied at a time (a dense float copy of a whole block is a few GB)


def sparse_rows(counts, rows):
    """counts[rows] as a sparse CSR matrix (float32), built block by block."""
    rows = np.asarray(rows)
    return sparse.vstack([sparse.csr_matrix(np.asarray(counts[rows[a:a + BLOCK]], dtype=np.float32))
                          for a in range(0, len(rows), BLOCK)], format='csr')


def column_moments(C):
    """Mean and SD of each column of a sparse matrix."""
    n = C.shape[0]
    mean = np.asarray(C.sum(axis=0), dtype=np.float64).ravel() / n
    mean_sq = np.asarray(C.multiply(C).sum(axis=0), dtype=np.float64).ravel() / n
    return mean, np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))


def firing_rate(counts, rows, bin_size):
    return column_moments(sparse_rows(counts, rows))[0] / bin_size


def usable_rows(rows, max_lag):
    """Rows t whose t + max_lag is also in rows (so every lag stays inside the block)."""
    rows = np.asarray(rows)
    return rows[np.isin(rows + max_lag, rows)]


def lagged_products(counts, Y, rows, lags):
    """
    sum_t c_u(t) y_k(t + l) over t in usable_rows(rows, max(lags)), for every lag: (lags x units x
    targets), with the row set, spike sums and target moments needed by spike_triggered_target and
    spike_lfp_correlation.
    """
    lags = np.asarray(lags)
    t = usable_rows(rows, lags.max())
    C = sparse_rows(counts, t)
    CT = C.T.tocsr()
    products = np.stack([np.asarray(CT @ np.asarray(Y[t + l], dtype=np.float32), dtype=np.float64) for l in lags])
    c_mean, c_sd = column_moments(C)
    y_mean = np.stack([Y[t + l].mean(axis=0) for l in lags])                 # lags x targets
    y_sd = np.stack([Y[t + l].std(axis=0) for l in lags])
    return dict(products=products, n=len(t), spikes=c_mean * len(t), c_mean=c_mean, c_sd=c_sd,
                y_mean=y_mean, y_sd=y_sd)


def spike_triggered_target(lp):
    """Spike-triggered average of each target (lags x units x targets); NaN for silent units."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return lp['products'] / lp['spikes'][None, :, None]


def spike_lfp_correlation(lp):
    """Pearson correlation of each unit's counts with each target at each lag (lags x units x targets)."""
    cov = lp['products'] / lp['n'] - lp['c_mean'][None, :, None] * lp['y_mean'][:, None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        return cov / (lp['c_sd'][None, :, None] * lp['y_sd'][:, None, :])


def max_abs_over_lags(x):
    """max over the first axis of |x|, with NaN (silent units) as 0."""
    return np.nanmax(np.abs(np.nan_to_num(x, nan=0.0)), axis=0)


def population_coupling(counts, rows, coarse_bins=5):
    """Correlation of each unit with the sum of all other units, in bins of coarse_bins (rows in contiguous runs)."""
    rows = np.asarray(rows)
    rows = rows[:len(rows) // coarse_bins * coarse_bins]
    block = BLOCK // coarse_bins * coarse_bins
    x = np.concatenate([np.asarray(counts[rows[a:a + block]], dtype=np.float32).reshape(-1, coarse_bins, counts.shape[1])
                        .sum(axis=1) for a in range(0, len(rows), block)]).astype(np.float64)
    pop = x.sum(axis=1)
    x_c, pop_c = x - x.mean(axis=0), pop - pop.mean()
    cov_xp = x_c.T @ pop_c / len(x)                                   # cov(x_u, P)
    var_x, var_p = x.var(axis=0), pop.var()
    cov = cov_xp - var_x                                              # cov(x_u, P - x_u)
    var_rest = var_p - 2 * cov_xp + var_x                             # var(P - x_u)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = cov / np.sqrt(var_x * var_rest)
    return np.nan_to_num(r, nan=0.0)


def ridge_weight_norms(W, n_basis):
    """Per-unit norm of standardised lagged-ridge weights W (units * n_basis features x targets): units x targets."""
    return np.sqrt((W.reshape(-1, n_basis, W.shape[1]) ** 2).sum(axis=1))


def rank_order(scores, rng):
    """Unit indices from the highest score to the lowest, ties (e.g. silent units at 0) in random order."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    return np.lexsort((rng.random(len(scores)), -scores))
