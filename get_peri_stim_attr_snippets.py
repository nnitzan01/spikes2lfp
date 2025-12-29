import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess_data import *
from process_session import session as ps
from scipy.sparse import save_npz, coo_matrix
import warnings 
import gc

def get_peri_stim_attr_snippets(output_dir, session_id):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Load units info for session mapping
    if os.path.exists('./tables/units_info.csv'):
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("Error: units_info.csv not found.")
        sys.exit(1)

    print(f"Obtaining session data for Session {session_id}", flush=True)
    
    # --- CONSTANTS ---
    seqlength = 750
    bin_size  = 0.004
    EPSILON = 1e-10
    time_win = [-.25, .5]
    
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs_entire_session' / str(session_id))
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    # os.makedirs(variables_dir, exist_ok=True)

    # --- 1. Determine Channels ---
    files = os.listdir(output_dir_attrs)
    chan_numbers = sorted(list(set([int(f.split("_")[4].replace("chan", "")) for f in files if f.endswith("_band0.npz")])))
    num_channels = len(chan_numbers)
    
    # --- 2. Load Spiking Data for Metadata and Timestamps ---
    session_obj = ps(session_id, df, output_dir)
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.passive_times[1])
    spikes_obj.truncate(seqlength)
    
    t = np.arange(time_win[0], time_win[1], bin_size)
    n_bins_snippet = len(t)
    timestamps_full = spikes_obj.timestamps
    n_timepoints_spikes = len(timestamps_full)
    
    # Filter valid stimulus trials
    stim_df = session_obj.stimulus_presentation
    stim_mask = (stim_df.active == True) & (stim_df.image_name != "omitted")
    stim_st = stim_df.start_time[stim_mask].values
    
    valid_trials_mask = ((stim_st + time_win[0]) >= timestamps_full[0]) & ((stim_st + time_win[1]) < timestamps_full[-1])
    stim_st = stim_st[valid_trials_mask]
    n_trials = len(stim_st)
    n_neurons = spikes_obj.spkMat.shape[1]

    # Pre-calculate start indices for all trials
    trial_start_indices = [np.searchsorted(timestamps_full, s + time_win[0]) for s in stim_st]
    
    del spikes_obj
    gc.collect()

    # --- 3. Initialize Averaged Container ---
    # Instead of (Trials x Bins x Neurons x Channels), we save (Bins x Neurons x Channels)
    # This reduces space by a factor of n_trials (e.g., ~5000x reduction)
    avg_attr_matrix = np.zeros((n_bins_snippet, n_neurons, num_channels), dtype=np.float32)
    
    # To correctly handle NaNs where neurons didn't spike, we need a counter per bin/neuron/chan
    activity_counter = np.zeros((n_bins_snippet, n_neurons, num_channels), dtype=np.uint32)

    # --- 4. Channel Loop ---
    print(f"Processing {num_channels} channels...", flush=True)
    for ch_idx, ch_val in enumerate(chan_numbers): 
        print(f"Channel {ch_val} ({ch_idx+1}/{num_channels})", flush=True)
        
        # Load the massive attribution file for the entire session
        sparse_attr = np.load(output_dir_attrs / f'attribution_scores_entire_session_chan{ch_val}_band{0}.npz') 
        attrs = coo_matrix((sparse_attr['data'], (sparse_attr['row'], sparse_attr['col'])), 
                          shape=sparse_attr['shape']).toarray().astype(np.float32)
        
        n_timepoints_limit = min(n_timepoints_spikes, attrs.shape[0])
        abs_attrs = np.abs(attrs)

        # Iterate through trials and accumulate
        for s_idx in trial_start_indices:
            e_idx = s_idx + n_bins_snippet
            if e_idx <= n_timepoints_limit:
                snippet = attrs[s_idx:e_idx, :]
                mask = abs_attrs[s_idx:e_idx, :] > EPSILON
                
                # Add attribution scores only where active
                avg_attr_matrix[:, :, ch_idx] += np.where(mask, snippet, 0.0)
                # Increment counter where active for later division (the NaN-logic replacement)
                activity_counter[:, :, ch_idx] += mask.astype(np.uint32)

        del attrs, abs_attrs, sparse_attr
        gc.collect() 

    # --- 5. Finalize Average (Numerator / Denominator) ---
    # Result is NaN if a neuron never fired in that time bin across any trial
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_attr_matrix = np.divide(avg_attr_matrix, activity_counter, 
                                   out=np.full_like(avg_attr_matrix, np.nan), 
                                   where=activity_counter > 0)

    # --- 6. Saving ---
    # This file will be ~100MB to 300MB per session instead of 40GB+
    np.save(variables_dir / 'mean_attribution_snippet.npy', avg_attr_matrix)
    
    # Save metadata needed for plotting
    # np.save(variables_dir / 't_vector.npy', t)
    
    print(f"Successfully saved averaged attribution matrix to: {variables_dir}")

def main(args):
    gc.enable()
    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f"Directory {dir_path} does not exist.")
        sys.exit(1)
    
    session_id = int(dir_path.name)
    # Reconstructing root based on common Allen SDK dir structure
    root_dir = dir_path.parent.parent.parent
    get_peri_stim_attr_snippets(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the session directory")
    args = parser.parse_args()
    main(args)