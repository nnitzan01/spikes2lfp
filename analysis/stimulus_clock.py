"""
How much of the predicted LFP is the stimulus clock?

Spikes and LFP are both locked to image onsets, so a model could score well by reproducing the
mean evoked response. These functions separate the stimulus-locked part from single-trial
fluctuations.

Condition of a presentation: its image, or 'change' (a change presentation, any image), or
'omitted'. Every bin belongs to the most recent onset before it, at a lag of 0, 1, ... bins; bins
more than `max_lag` bins after an onset (or before the first one) have no condition.

Template T: the mean target at each (condition, lag), fitted on training bins only. It is the
best prediction possible from stimulus identity and time since onset alone. Bins without a
condition, or whose (condition, lag) never occurs in training, get the training mean.

R2 definitions (all over the given bins):
  r2(y, yhat)                    1 - SS(y - yhat) / SS(y - mean y)
  residual_r2(y, yhat, T)        1 - SS(y - yhat) / SS(y - T): the fraction of the variance the
                                 template leaves that the prediction explains (0 = no better than
                                 the template; negative = worse)
  r2 of residuals                r2(y - T_y, yhat - T_yhat), each with its own template: do the
                                 single-trial deviations of the prediction follow those of the data?
"""
import numpy as np


def presentation_conditions(presentations):
    """Condition label of each presentation (see module docstring)."""
    image = presentations.image_name.astype(str).to_numpy()
    change = presentations.is_change.fillna(False).to_numpy(bool)
    return np.where(image == 'omitted', 'omitted', np.where(change, 'change', image))


def onset_bins(onset_times, bin_starts, bin_size):
    """Bin index of each onset time (bins counted from bin_starts[0])."""
    return np.round((np.asarray(onset_times) - bin_starts[0]) / bin_size).astype(int)


def bin_conditions(onsets, conditions, n_bins, max_lag):
    """
    For each bin: the index of the condition (in the returned list of names) of the most recent
    onset and the lag in bins since it; -1 for both where there is none within max_lag bins.
    onsets: bin index of each presentation (sorted), conditions: its condition label.
    """
    names = sorted(set(conditions))
    code = np.searchsorted(names, conditions)
    bins = np.arange(n_bins)
    last = np.searchsorted(onsets, bins, side='right') - 1            # most recent onset at or before each bin
    lag = np.where(last >= 0, bins - onsets[np.maximum(last, 0)], -1)
    valid = (last >= 0) & (lag < max_lag)
    return np.where(valid, code[np.maximum(last, 0)], -1), np.where(valid, lag, -1), names


def fit_template(y, cond, lag, fit_bins, n_conditions, max_lag):
    """
    Mean of y (bins x ...) at each (condition, lag) over fit_bins:
    returns (table: conditions x lags x ..., fallback: mean over fit_bins). Unseen cells are NaN.
    """
    y = np.asarray(y, dtype=np.float64)
    fit_bins = np.asarray(fit_bins)
    fallback = y[fit_bins].mean(axis=0)
    keep = fit_bins[cond[fit_bins] >= 0]
    cell = cond[keep] * max_lag + lag[keep]
    flat = y[keep].reshape(len(keep), -1)
    sums = np.zeros((n_conditions * max_lag, flat.shape[1]))
    np.add.at(sums, cell, flat)
    counts = np.bincount(cell, minlength=n_conditions * max_lag)
    with np.errstate(invalid='ignore', divide='ignore'):
        table = sums / counts[:, None]
    table[counts == 0] = np.nan
    return table.reshape((n_conditions, max_lag) + y.shape[1:]), fallback


def apply_template(table, fallback, cond, lag):
    """Template value of every bin: table[cond, lag], or the fallback where there is none."""
    out = np.broadcast_to(fallback, cond.shape + np.shape(fallback)).copy()
    has = cond >= 0
    values = table[cond[has], lag[has]]
    seen = np.isfinite(values).reshape(len(values), -1).all(axis=1)
    idx = np.flatnonzero(has)[seen]
    out[idx] = values[seen]
    return out


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else np.nan


def residual_r2(y, yhat, template):
    """Fraction of the variance left by the template that the prediction explains (see module docstring)."""
    y, yhat, template = (np.asarray(a, float) for a in (y, yhat, template))
    ss = np.sum((y - template) ** 2)
    return 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else np.nan


def presentation_segments(onsets, n_bins, min_len, max_len):
    """(start, stop) bin ranges from each onset to the next (or to max_len bins), keeping those >= min_len bins."""
    stops = np.minimum(np.append(onsets[1:], n_bins), onsets + max_len)
    keep = (stops - onsets >= min_len) & (onsets >= 0)
    return np.stack([onsets[keep], stops[keep]], axis=1)


def per_presentation_r2(y, yhat, segments):
    """R2 within each presentation segment (start, stop); returns one value per segment."""
    return np.array([r2(y[a:b], yhat[a:b]) for a, b in segments])


def shuffle_presentations(X, segments, conditions, rng):
    """
    A copy of X (bins x units) in which the spikes of each presentation segment are replaced by
    those of another segment of the same condition and length, chosen by a random permutation within
    each (condition, length) group. Stimulus alignment is kept; single-trial pairing with the
    target is broken. Bins outside the segments are unchanged.
    """
    X_out = X.copy()
    lengths = segments[:, 1] - segments[:, 0]
    groups = {}
    for i, key in enumerate(zip(conditions, lengths)):
        groups.setdefault(key, []).append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        donors = rng.permutation(members)
        for target, donor in zip(members, donors):
            (a, b), (c, d) = segments[target], segments[donor]
            X_out[a:b] = X[c:d]
    return X_out
