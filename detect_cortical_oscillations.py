import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, hilbert
from typing import List, Tuple
import os
import random
import sys
import argparse
from pathlib import Path
import pandas as pd
from preprocess_data import *
from process_session import session as ps
from scipy.signal import welch


# --- 1. DETECTION PARAMETERS (Adjust based on final tuning) ---
FS = 1250.0                 # LFP Sampling Rate (Hz)
F_RANGE = [2, 6]            # Target frequency band (Hz)
WIN_SIZE_SEC = 0.5          # RMS smoothing window (seconds)
THRESHOLD_STD = 2.5         # Primary threshold (STD)
MIN_DURATION_SEC = 0.75     # Minimum event duration (seconds)
MAX_GAP_SEC = 0.15          # Max gap to merge events (seconds)
BOUNDARY_THRESHOLD_STD = 0.5 # Low-amplitude threshold for refinement
TIME_PADDING_SEC = 2.0      # Time window for plotting validation

# ====================================================================
# A. HELPER FUNCTIONS (CORE DETECTION LOGIC)
# ====================================================================

def get_best_channel(lfp_data: np.ndarray, fs: float, f_range: list) -> int:
    N_channels = lfp_data.shape[1]
    max_band_power = np.zeros(N_channels)

    for i in range(N_channels):
        channel_lfp = lfp_data[:, i]
        f, psd = welch(channel_lfp, fs=fs, nfft=2048, average='mean')
        f_mask = (f >= f_range[0]) & (f <= f_range[1])
        band_psd = psd[f_mask]
        max_band_power[i] = np.mean(band_psd)

    best_channel_index = np.argmax(max_band_power)
    return best_channel_index


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def refine_boundaries(start_idx: int, end_idx: int, rms_amplitude: np.ndarray, threshold_low: float, fs: float, min_samples: int) -> Tuple[int, int]:
    """Refines event start/end indices by looking for low-amplitude threshold crossing."""
    search_window_samples = int(WIN_SIZE_SEC * fs) 
    
    # Refine Start Boundary (Look backward)
    search_start = max(0, start_idx - search_window_samples)
    low_amp_indices = np.where(rms_amplitude[search_start:start_idx] < threshold_low)[0]
    if len(low_amp_indices) > 0:
        new_start_idx = search_start + low_amp_indices[-1]
    else:
        new_start_idx = start_idx
        
    # Refine End Boundary (Look forward)
    search_end = min(len(rms_amplitude) - 1, end_idx + search_window_samples)
    low_amp_indices = np.where(rms_amplitude[end_idx:search_end] < threshold_low)[0]
    if len(low_amp_indices) > 0:
        new_end_idx = end_idx + low_amp_indices[0]
    else:
        new_end_idx = end_idx
        
    if new_end_idx - new_start_idx >= min_samples:
        return new_start_idx, new_end_idx
    else:
        return None, None 

def detect_oscillations(lfp_data: np.ndarray, fs: float, filt_freqs: List[float]) -> np.ndarray:
    """Detects 3-5 Hz oscillation events based on LFP amplitude thresholding and refinement."""
    
    fs_lfp = fs
    
    if lfp_data.ndim > 1:
        raise ValueError("detect_oscillations expects 1D LFP data")
    
    # 1. Filtering, Power, and RMS Calculation
    lfp_filtered = bandpass_filter(lfp_data, filt_freqs[0], filt_freqs[1], fs_lfp)
    power = np.abs(hilbert(lfp_filtered))
    
    win_size = int(WIN_SIZE_SEC * fs_lfp)
    squared_amplitude = power**2
    kernel = np.ones(win_size) / win_size
    rms_amplitude = np.sqrt(np.convolve(squared_amplitude, kernel, mode='same'))
    
    # 2. Primary Thresholding
    mu = np.nanmean(rms_amplitude)
    sigma = np.nanstd(rms_amplitude)
    primary_threshold = mu + THRESHOLD_STD * sigma
    low_threshold = mu + BOUNDARY_THRESHOLD_STD * sigma
    
    over_threshold_indices = np.where(rms_amplitude > primary_threshold)[0]
    if len(over_threshold_indices) == 0:
        return np.array([])
    
    # 3. Merging Events
    gap_samples = int(MAX_GAP_SEC * fs_lfp)
    over_threshold_indices_diff = np.diff(over_threshold_indices)
    break_points = np.where(over_threshold_indices_diff > gap_samples)[0]
    
    event_starts_idx = np.insert(over_threshold_indices[break_points + 1], 0, over_threshold_indices[0])
    event_ends_idx = np.append(over_threshold_indices[break_points], over_threshold_indices[-1])
    
    # 4. Boundary Refinement, Min Duration, and Peak Calculation
    min_duration_samples = int(MIN_DURATION_SEC * fs_lfp)
    final_events = []
    
    for start_idx, end_idx in zip(event_starts_idx, event_ends_idx):
        new_start, new_end = refine_boundaries(start_idx, end_idx, rms_amplitude, low_threshold, fs_lfp, min_samples=min_duration_samples)
        
        if new_start is not None:
            # Find the peak power index within the refined boundaries
            event_slice = rms_amplitude[new_start:new_end]
            peak_relative_idx = np.argmax(event_slice)
            peak_abs_idx = new_start + peak_relative_idx

            # Append: [Start Sample, Peak Sample, End Sample]
            final_events.append([new_start, peak_abs_idx, new_end])
                    
    return np.array(final_events)

# ====================================================================
# B. VALIDATION PLOTTING FUNCTION
# ====================================================================

def plot_random_events(lfp_data: np.ndarray, detected_events: np.ndarray, fs: float, 
                       best_channel_index: int, output_dir: str, n_plots: int = 10):
    """Plots a set of randomly selected detected events for visual validation."""
    
    if len(detected_events) == 0:
        print("No events detected to plot.")
        return
        
    num_events = len(detected_events)
    # Select up to n_plots random indices
    plot_indices = random.sample(range(num_events), min(num_events, n_plots))
    
    # Pre-calculate the RMS amplitude envelope for plotting reference
    lfp_data_1d = lfp_data[:, best_channel_index]
    lfp_filtered = bandpass_filter(lfp_data_1d, F_RANGE[0], F_RANGE[1], fs)
    power = np.abs(hilbert(lfp_filtered))
    win_size = int(WIN_SIZE_SEC * fs)
    kernel = np.ones(win_size) / win_size
    rms_amplitude = np.sqrt(np.convolve(power**2, kernel, mode='same'))
    
    os.makedirs(output_dir, exist_ok=True)

    for i, event_index in enumerate(plot_indices):
        start, peak, end = detected_events[event_index].astype(int)
        
        # Determine the window boundaries for plotting (including padding)
        padding_samples = int(TIME_PADDING_SEC * fs)
        plot_start_sample = max(0, start - padding_samples)
        plot_end_sample = min(lfp_data.shape[0], end + padding_samples)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        time_axis = np.arange(plot_start_sample, plot_end_sample) / fs 
        
        # --- Plot LFP and RMS ---
        ax.plot(time_axis, lfp_data_1d[plot_start_sample:plot_end_sample], color='gray', linewidth=0.7, label='Raw LFP')
        
        ax2 = ax.twinx()
        ax2.plot(time_axis, rms_amplitude[plot_start_sample:plot_end_sample], color='orange', linewidth=1.5, alpha=0.7, label='RMS Envelope')
        
        # --- Mark Event Boundaries and Peak ---
        ax.axvspan(start / fs, end / fs, color='green', alpha=0.2, label='Detected Event')
        ax.axvline(peak / fs, color='purple', linestyle=':', linewidth=2, label='Peak Power')
        
        # Cosmetics
        ax.set_title(f'Validation Plot {i+1} (Event {event_index})', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('LFP Amplitude (Raw)')
        ax2.set_ylabel('RMS Amplitude (Env.)', color='orange')
        ax.grid(axis='x', alpha=0.5)
        
        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right')
        
        # Save the figure
        fig.savefig(os.path.join(output_dir, f'validation_event_{event_index}.png'))
        plt.close(fig) # Close figure to prevent memory buildup on cluster

# ====================================================================
# C. MAIN CLUSTER EXECUTION FUNCTION
# ====================================================================

def run_detection_analysis(output_dir: str, session_id: str ):
    """
    Main function to run oscillation detection for a single session and save results.
    """
    print(f"--- Starting Analysis for Session: {session_id} ---")
    df = pd.read_csv('tables/units_info.csv')

    session_obj = ps(session_id, df, output_dir)
    lfp_obj = pre_process_lfp(session_id, session_obj.channels, session_obj.active_times[0],
                                session_obj.passive_times[1], output_dir)
    FS = lfp_obj.sampling_rate
    F_RANGE = [2,6]
    
    lfp_data = lfp_obj.data  # Shape: (N_samples, N_channels)
    best_channel_index = get_best_channel(lfp_data, FS, F_RANGE)
    print(f"Best channel for detection: {best_channel_index}")
    
    lfp_channel_data = lfp_data[:, best_channel_index]
    
    # 1. Run Detection
    detected_events = detect_oscillations(
        lfp_data=lfp_channel_data,
        fs=FS,
        filt_freqs=F_RANGE
    )
    
    detected_events_timestamps = lfp_obj.timestamps[detected_events] if len(detected_events) > 0 else np.array([])
    
    if len(detected_events) == 0:
        print(f"No events found in session {session_id}.")
        # Still save empty array for consistency
        output_session_dir = os.path.join(output_dir, str(session_id))
        os.makedirs(output_session_dir, exist_ok=True)
        np.save(os.path.join(output_session_dir, 'detected_events.npy'), np.array([]))
        return

    print(f"Found {len(detected_events)} events.")

    # 2. Define Output Paths
    output_session_dir = os.path.join(output_dir, f'session_{str(session_id)}')
    # os.makedirs(output_session_dir, exist_ok=True)
    
    # 3. Save Results (Event Indices)
    np.save(os.path.join(output_session_dir, 'detected_hvs_events.npy'), detected_events_timestamps)
    print(f"Events saved to {output_session_dir}")

    # 4. Generate Validation Plots
    plot_output_dir = os.path.join(output_session_dir, 'validation_plots')
    plot_random_events(lfp_data, detected_events, FS, best_channel_index, plot_output_dir, n_plots=10)
    print(f"Validation plots saved to {plot_output_dir}")

# ====================================================================
# D. CLUSTER EXECUTION BLOCK (Conceptual)
# ====================================================================

def main(args):
    dir = Path(args.dir)
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        sys.exit(1)
    session_id = int(os.path.basename(dir))
    root_dir = dir.parent.parent.parent
    print(f"Root dir is set to: {root_dir}", flush=True)
    print(f"Session ID is set to: {session_id}", flush=True)    
    run_detection_analysis(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)