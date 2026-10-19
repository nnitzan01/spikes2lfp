"""
Per-time-bin ranking of neurons by attribution, and the stability measures built on it.

All functions take attributions as a (time bins x neurons) array. A neuron is *active* in a bin
when |attribution| > EPSILON. With the zero integrated-gradients baseline used in this project,
attribution is exactly zero wherever the neuron did not spike, so "active" means "spiked".

Silent neurons are deliberately left unranked (rank 0). Giving them ranks would order them by
unit index (np.argsort breaks ties by position), and whenever fewer than K neurons are active
the top-K set would be padded with the same arbitrary silent neurons, inflating Jaccard
similarity.
"""
import numpy as np

EPSILON = 1e-10


def rank_active_neurons(attrs, eps=EPSILON, chunk_size=10_000):
    """
    Rank neurons in every time bin by |attribution|.

    Returns a uint16 array shaped like `attrs`: 1 = largest |attribution| in that bin, ranks run
    1..n_active, and silent neurons (|attribution| <= eps) get 0. Rows are processed in chunks
    to bound memory on full-session matrices.
    """
    n_bins, n_neurons = attrs.shape
    if n_neurons >= np.iinfo(np.uint16).max:
        raise ValueError(f"{n_neurons} neurons do not fit in uint16 ranks")
    ranks = np.zeros((n_bins, n_neurons), dtype=np.uint16)
    rank_values = np.arange(1, n_neurons + 1, dtype=np.uint16)[np.newaxis, :]
    for start in range(0, n_bins, chunk_size):
        abs_chunk = np.abs(attrs[start:start + chunk_size])
        # descending order; active neurons (> eps) always sort ahead of silent ones
        order = np.argsort(-abs_chunk, axis=1, kind='stable')
        chunk_ranks = np.empty(abs_chunk.shape, dtype=np.uint16)
        np.put_along_axis(chunk_ranks, order, rank_values, axis=1)
        chunk_ranks[abs_chunk <= eps] = 0
        ranks[start:start + chunk_size] = chunk_ranks
    return ranks


def inverse_rank(ranks):
    """1/rank for ranked (active) neurons and 0 for unranked (silent) ones, as float32."""
    inv = np.zeros(ranks.shape, dtype=np.float32)
    ranked = ranks > 0
    inv[ranked] = 1.0 / ranks[ranked]
    return inv


def top_k_mask(ranks, k):
    """Boolean (bins x neurons) membership of the top-k active neurons in each bin.

    Bins with fewer than k active neurons contain only the active ones.
    """
    return (ranks > 0) & (ranks <= k)


def region_presence(mask, region_labels):
    """
    Collapse a (bins x neurons) boolean mask to (bins x regions): True where any selected neuron
    belongs to the region. Use it to compute region-level Jaccard scores with the functions below.

    Returns (presence, regions), where regions is the sorted array of unique labels.
    """
    region_labels = np.asarray(region_labels)
    regions = np.unique(region_labels)
    presence = np.zeros((mask.shape[0], len(regions)), dtype=bool)
    for r, region in enumerate(regions):
        presence[:, r] = mask[:, region_labels == region].any(axis=1)
    return presence, regions


def jaccard_consecutive(mask, chunk_size=100_000):
    """
    Jaccard similarity between the selected sets of consecutive bins, J[t] = |S_t & S_t+1| / |S_t | S_t+1|.

    Returns a float32 array of length n_bins. J is NaN where both sets are empty (undefined),
    and the last bin is NaN because it has no successor; average with np.nanmean.
    """
    n_bins = mask.shape[0]
    jaccard = np.full(n_bins, np.nan, dtype=np.float32)
    for start in range(0, n_bins - 1, chunk_size):
        stop = min(start + chunk_size, n_bins - 1)
        current, following = mask[start:stop], mask[start + 1:stop + 1]
        intersection = np.count_nonzero(current & following, axis=1)
        union = np.count_nonzero(current | following, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            jaccard[start:stop] = np.where(union > 0, intersection / union, np.nan)
    return jaccard


def jaccard_to_reference(mask, reference):
    """
    Jaccard similarity between each bin's selected set and a fixed reference set.

    mask: (bins x items) boolean; reference: (items,) boolean.
    Returns float32 of length n_bins, NaN where both sets are empty.
    """
    reference = np.asarray(reference, dtype=bool)
    intersection = np.count_nonzero(mask & reference, axis=1)
    union = np.count_nonzero(mask | reference, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(union > 0, intersection / union, np.nan).astype(np.float32)


def baseline_reference_set(attrs, event_start_indices, baseline_bins, k, eps=EPSILON):
    """
    The k neurons with the largest mean relative attribution over the baseline bins of all events.

    attrs: (bins x neurons) attributions for the whole recording.
    event_start_indices: index of the first bin of each event snippet.
    baseline_bins: (start, stop) bin offsets of the baseline window within a snippet.

    Relative attribution is |attribution| divided by the summed |attribution| of all neurons in
    that bin, so a silent neuron contributes 0. (The earlier version used the mean *rank*, which
    included silent neurons' arbitrary ranks.) Neurons that never fire in the baseline are
    excluded, so the set can be smaller than k.

    Returns an (neurons,) boolean mask.
    """
    b_start, b_stop = baseline_bins
    rows = np.concatenate([np.arange(s + b_start, s + b_stop) for s in event_start_indices])
    abs_attrs = np.abs(attrs[rows])
    total = abs_attrs.sum(axis=1, keepdims=True)
    relative = np.divide(abs_attrs, total, out=np.zeros_like(abs_attrs, dtype=np.float64),
                         where=total > eps)
    score = relative.mean(axis=0)
    top = np.argsort(-score, kind='stable')[:k]
    reference = np.zeros(attrs.shape[1], dtype=bool)
    reference[top[score[top] > 0]] = True
    return reference
