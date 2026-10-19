"""Event-aligned snippets of binned data and the regional dominance (IPD) measure."""
import numpy as np

from analysis.ranking import inverse_rank


def event_start_indices(timestamps, event_times, window, n_bins_snippet):
    """
    Bin index at which each event's snippet starts, for events whose whole window lies inside the recording.

    timestamps: (bins,) left bin edges in seconds; window: (start, stop) s relative to the event;
    n_bins_snippet: snippet length in bins (len(snippet_time_axis(window, bin_size))).
    Returns (start_indices, kept_event_times).
    """
    event_times = np.asarray(event_times)
    starts = np.searchsorted(timestamps, event_times + window[0])
    valid = ((event_times + window[0]) >= timestamps[0]) & ((event_times + window[1]) < timestamps[-1]) \
        & (starts + n_bins_snippet <= len(timestamps))
    return starts[valid], event_times[valid]


def snippet_time_axis(window, bin_size):
    """Time of each snippet bin relative to the event."""
    return np.arange(window[0], window[1], bin_size)


def extract_snippets(arr, start_indices, n_bins):
    """Stack arr[s:s + n_bins] for every start index -> (events, n_bins, ...)."""
    return np.stack([arr[s:s + n_bins] for s in start_indices]) if len(start_indices) else \
        np.zeros((0, n_bins) + arr.shape[1:], dtype=arr.dtype)


def regional_dominance(rank_snippets, active_snippets, region_labels):
    """
    Instantaneous predictive dominance (IPD) of each region.

    For every event and bin: the mean of 1/rank over the region's active units (0 if none of
    them is active), where rank 1 is the most influential unit in that bin.

    rank_snippets, active_snippets: (events, bins, units); region_labels: (units,)
    Returns (dominance (events, regions, bins), region names).
    """
    inv = inverse_rank(np.where(active_snippets, rank_snippets, 0))
    regions = np.unique(region_labels)
    dominance = np.zeros((rank_snippets.shape[0], len(regions), rank_snippets.shape[1]), dtype=np.float32)
    for r, region in enumerate(regions):
        in_region = np.asarray(region_labels) == region
        numerator = inv[:, :, in_region].sum(axis=2)
        count = active_snippets[:, :, in_region].sum(axis=2)
        dominance[:, r, :] = np.divide(numerator, count, out=np.zeros_like(numerator), where=count > 0)
    return dominance, regions


def proportional_influence(average_dominance):
    """Each region's share of the summed dominance at every bin (regions x bins [x channels]); 0 where the sum is 0."""
    total = np.sum(average_dominance, axis=0, keepdims=True)
    return np.divide(average_dominance, total, out=np.zeros_like(average_dominance), where=total > 0)
