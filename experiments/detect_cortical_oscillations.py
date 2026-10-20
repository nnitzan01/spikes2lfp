"""
Detect high-voltage spindles (HVS; 2-6 Hz cortical oscillations) in the VISp LFP of one session.

Detection follows bz_FindRipples: band-pass, power envelope smoothed over WIN_SIZE_SEC, z-scored;
candidate events are runs above THRESHOLD_STD_LOW (merged across gaps < MAX_GAP_SEC) whose peak
exceeds THRESHOLD_STD_HIGH, that last at least MIN_DURATION_SEC and contain at least MIN_PEAKS
filtered-LFP peaks above THRESHOLD_STD_HIGH standard deviations. The channel with the most
2-6 Hz power is used. The search covers the active epoch through the end of the passive epoch.

usage (from the repo root):
    python -m experiments.detect_cortical_oscillations <session_dir>

Saves variables/<session_id>/detected_hvs_events.npy: (events x [start, peak, end]) in seconds,
read by the 'peri' stage of run_session.py, and validation plots of up to 10 random events
under variables/<session_id>/hvs_validation_plots/.
"""
import argparse
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, hilbert, welch

import config
import paths
from preprocess_data import pre_process_lfp
from run_session import SessionPipeline
from spectral_analysis import bandpass_filter

# --- detection parameters ---
F_RANGE = (2, 6)              # Hz
WIN_SIZE_SEC = 0.5            # power smoothing window, also the boundary search range
THRESHOLD_STD_LOW = 1.0       # z-scored power: event extent
THRESHOLD_STD_HIGH = 2.5      # z-scored power: required peak; also the LFP peak threshold (in SDs)
MIN_DURATION_SEC = 0.75
MAX_GAP_SEC = 0.15            # merge supra-threshold runs separated by less than this
MIN_PEAKS = 3                 # filtered-LFP peaks above threshold within the event
TIME_PADDING_SEC = 2.0        # context shown around events in validation plots


def get_best_channel(lfp_data, fs, f_range):
    """Index of the channel with the highest mean PSD in f_range."""
    band_power = []
    for channel_lfp in lfp_data.T:
        f, psd = welch(channel_lfp, fs=fs, nfft=2048, average='mean')
        band_power.append(np.mean(psd[(f >= f_range[0]) & (f <= f_range[1])]))
    return int(np.argmax(band_power))


def z_power_envelope(lfp_filtered, fs):
    """Z-scored band power, smoothed with a WIN_SIZE_SEC moving average."""
    win = int(WIN_SIZE_SEC * fs)
    power = np.convolve(np.abs(hilbert(lfp_filtered))**2, np.ones(win) / win, mode='same')
    return (power - np.nanmean(power)) / (np.nanstd(power) + 1e-10)


def refine_boundaries(start_idx, end_idx, z_power, threshold, fs, min_samples):
    """
    Move event edges outward to the nearest sample below `threshold` within WIN_SIZE_SEC.
    Returns (start, end), or (None, None) if the event is shorter than min_samples.

    Called with the same threshold that defined the event, which only extends each edge by one
    sample. A separate, lower boundary threshold may have been intended.
    """
    search = int(WIN_SIZE_SEC * fs)
    lo = max(0, start_idx - search)
    below = np.flatnonzero(z_power[lo:start_idx] < threshold)
    new_start = lo + below[-1] if len(below) else start_idx
    hi = min(len(z_power) - 1, end_idx + search)
    below = np.flatnonzero(z_power[end_idx:hi] < threshold)
    new_end = end_idx + below[0] if len(below) else end_idx
    return (new_start, new_end) if new_end - new_start >= min_samples else (None, None)


def detect_oscillations(lfp_data, fs, f_range=F_RANGE):
    """Detect events in a 1D LFP. Returns (events x [start, peak, end]) sample indices."""
    if lfp_data.ndim != 1:
        raise ValueError('detect_oscillations expects 1D LFP data')
    lfp_filtered = bandpass_filter(lfp_data, f_range, fs)
    z_power = z_power_envelope(lfp_filtered, fs)

    above = np.flatnonzero(z_power > THRESHOLD_STD_LOW)
    if len(above) == 0:
        return np.zeros((0, 3), dtype=int)
    # split supra-threshold samples into events wherever the gap exceeds MAX_GAP_SEC
    breaks = np.flatnonzero(np.diff(above) > int(MAX_GAP_SEC * fs))
    starts = np.insert(above[breaks + 1], 0, above[0])
    ends = np.append(above[breaks], above[-1])

    min_samples = int(MIN_DURATION_SEC * fs)
    peak_threshold = THRESHOLD_STD_HIGH * np.std(lfp_filtered)
    events = []
    for start, end in zip(starts, ends):
        if np.max(z_power[start:end]) < THRESHOLD_STD_HIGH:
            continue
        start, end = refine_boundaries(start, end, z_power, THRESHOLD_STD_LOW, fs, min_samples)
        if start is None:
            continue
        segment = lfp_filtered[start:end]
        peaks, _ = find_peaks(segment)
        if np.sum(segment[peaks] > peak_threshold) >= MIN_PEAKS:
            events.append([start, start + np.argmax(z_power[start:end]), end])
    return np.array(events, dtype=int).reshape(-1, 3)


def plot_random_events(lfp_1d, events, fs, output_dir, n_plots=10, seed=config.TORCH_SEED):
    """Save validation plots of up to n_plots random events (replaces any earlier plots in output_dir)."""
    if len(events) == 0:
        return
    z_power = z_power_envelope(bandpass_filter(lfp_1d, F_RANGE, fs), fs)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    pad = int(TIME_PADDING_SEC * fs)
    rng = np.random.default_rng(seed)
    for i, event_index in enumerate(rng.choice(len(events), min(len(events), n_plots), replace=False)):
        start, peak, end = events[event_index]
        lo, hi = max(0, start - pad), min(len(lfp_1d), end + pad)
        t = np.arange(lo, hi) / fs
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, lfp_1d[lo:hi], color='gray', linewidth=0.7, label='Raw LFP')
        ax2 = ax.twinx()
        ax2.plot(t, z_power[lo:hi], color='orange', linewidth=1.5, alpha=0.7, label='Power envelope')
        ax.axvspan(start / fs, end / fs, color='green', alpha=0.2, label='Detected Event')
        ax.axvline(peak / fs, color='purple', linestyle=':', linewidth=2, label='Peak Power')
        ax.set_title(f'Validation Plot {i+1} (Event {event_index})', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('LFP Amplitude (Raw)')
        ax2.set_ylabel('Z-scored power', color='orange')
        ax.grid(axis='x', alpha=0.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right')
        fig.savefig(output_dir / f'validation_event_{event_index}.png')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    pipeline = SessionPipeline(data_root, session_id, device='cpu')
    start, stop = pipeline.session.active_times[0], pipeline.session.passive_times[1]   # active through passive epoch
    lfp = pre_process_lfp(session_id, pipeline.session.channels, start, stop, data_root)
    fs = lfp.sampling_rate

    best = get_best_channel(lfp.data, fs, F_RANGE)
    print(f'Detecting on VISp channel {best}', flush=True)
    events = detect_oscillations(lfp.data[:, best], fs)
    print(f'Found {len(events)} events', flush=True)

    out = paths.hvs_events_file(pipeline.dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, lfp.timestamps[events] if len(events) else np.zeros((0, 3)))
    plot_random_events(lfp.data[:, best], events, fs, out.parent / 'hvs_validation_plots')
    print('Saved', out)


if __name__ == '__main__':
    main()
