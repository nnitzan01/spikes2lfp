"""
Targets, windowing and per-unit summaries for history-aware attribution (integrated_gradients.SequenceIG).

A *target* is a bin t of a continuous recording; its input window is the `history` bins ending at t
(bins t - history + 1 ... t), and the attributed quantity is the model's prediction at t, or the mean
over the last `target_bins` bins of the window. Attributions A[target, lag, unit] are reduced to:

    total[target, unit]   = sum over lags of A           (unit's contribution to that prediction)
    kernel[lag, unit]     = mean over targets of A       (how far back the unit's spikes matter)

Per-unit summaries over a set of targets (all reported per channel x model target):

    mean       = mean_t total[t, u]                                   (signed)
    mean_abs   = mean_t |total[t, u]|
    per_spike  = sum_t total[t, u] / sum_t spikes[t, u]               (spikes = spike count in the window)
    share      = mean_t |total[t, u]| / sum_u' |total[t, u']|         (fraction of all units' |attribution|)
"""
import numpy as np
import torch


def window_index(ends, history):
    """(n, history) bin indices of the windows ending at `ends`."""
    ends = np.asarray(ends)
    return ends[:, None] - history + 1 + np.arange(history)[None, :]


def window_spike_counts(counts, ends, history):
    """Spike count of every unit within each window: (n, units). counts: (bins x units) raw counts."""
    csum = np.concatenate([np.zeros((1, counts.shape[1])), np.cumsum(counts, axis=0)])
    ends = np.asarray(ends)
    return (csum[ends + 1] - csum[ends + 1 - history]).astype(np.int32)


def valid_ends(candidates, n_bins, history):
    """Target bins whose full window lies inside the recording."""
    candidates = np.asarray(candidates)
    return candidates[(candidates >= history - 1) & (candidates < n_bins)]


def attribute_targets(ig, X, ends, history, target_bins=1, baseline=0.0, batch_windows=256,
                      keep_kernel=False, keep_full=False):
    """
    Run SequenceIG for every target in `ends` over the continuous input X (bins x units tensor on the
    model's device), building windows batch by batch to bound memory.

    Returns a dict of numpy arrays: total (n, units), f_x, f_b, delta (n,), and if keep_kernel the
    sums over targets of A and |A| (history, units) as kernel_sum / kernel_abs_sum; if keep_full,
    A itself (n, history, units).
    """
    ends = np.asarray(ends)
    out = {k: [] for k in ('total', 'f_x', 'f_b', 'delta')}
    kernel_sum = kernel_abs_sum = None
    full = []
    for start in range(0, len(ends), batch_windows):
        idx = torch.as_tensor(window_index(ends[start:start + batch_windows], history), device=X.device)
        attr, f_x, f_b, delta = ig.attribute(X[idx], baseline=baseline, target_bins=target_bins)
        out['total'].append(attr.sum(dim=1).numpy())
        out['f_x'].append(f_x.numpy())
        out['f_b'].append(f_b.numpy())
        out['delta'].append(delta.numpy())
        if keep_kernel:
            s, a = attr.sum(dim=0).double().numpy(), attr.abs().sum(dim=0).double().numpy()
            kernel_sum = s if kernel_sum is None else kernel_sum + s
            kernel_abs_sum = a if kernel_abs_sum is None else kernel_abs_sum + a
        if keep_full:
            full.append(attr.numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    if keep_kernel:
        res['kernel_sum'], res['kernel_abs_sum'] = kernel_sum, kernel_abs_sum
    if keep_full:
        res['full'] = np.concatenate(full)
    return res


def coarsen_lags(kernel, factor):
    """Sum consecutive lags in groups of `factor` (oldest first): (history, ...) -> (history // factor, ...)."""
    n = kernel.shape[0] // factor
    trimmed = kernel[kernel.shape[0] - n * factor:]          # drop the oldest lags that don't fill a group
    return trimmed.reshape((n, factor) + kernel.shape[1:]).sum(axis=1)


def unit_summaries(totals, spikes):
    """Per-unit summaries over targets (see module docstring). totals, spikes: (targets, units)."""
    totals = np.asarray(totals, dtype=np.float64)
    abs_tot = np.abs(totals)
    row_sum = abs_tot.sum(axis=1, keepdims=True)
    share = np.divide(abs_tot, row_sum, out=np.zeros_like(abs_tot), where=row_sum > 0)
    n_spikes = np.asarray(spikes).sum(axis=0)
    return dict(mean=totals.mean(axis=0),
                mean_abs=abs_tot.mean(axis=0),
                per_spike=np.divide(totals.sum(axis=0), n_spikes, out=np.zeros(totals.shape[1]), where=n_spikes > 0),
                share=share.mean(axis=0))


def relative_completeness_error(f_x, f_b, delta):
    """
    Median |delta| over targets relative to the median |f(x) - f(b)|: the typical completeness error
    as a fraction of the typical change in prediction being explained. (A median of per-target
    ratios is dominated by targets where the prediction barely changes and is not used.)
    """
    scale = np.median(np.abs(np.asarray(f_x) - np.asarray(f_b)))
    return float(np.median(np.abs(np.asarray(delta))) / scale) if scale > 0 else 0.0
