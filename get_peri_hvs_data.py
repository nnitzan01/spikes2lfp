import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess_data import *
from process_session import session as ps
from scipy.sparse import save_npz, coo_matrix
import warnings 
import gc

def get_peri_hvs_data(output_dir, session_id):
    # Suppress specific warnings to keep HPC logs clean
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Check for unit info file
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
        hvs = pd.read_csv('./tables/hvs_detection.csv')
        if hvs['hvs'][hvs['session']==session_id].values[0] == False:
            print(f"Session {session_id} marked as non-HVS. Exiting.")
            sys.exit(0)
    else:
        print("units_info.csv not found in the repo.")
        sys.exit(1)

    print(f"Obtaining session data for Session {session_id}", flush=True)
    
    # --- CONSTANTS ---
    bin_size  = 0.004
    K = 40 # Top neurons for Jaccard
    EPSILON = 1e-10
    time_win = [-1, 1]
    baseline_win = [-1, -0.75]
    
    # Path setup
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs_entire_session' / str(session_id))
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    os.makedirs(variables_dir, exist_ok=True)
    hvs_file = Path(output_dir / f'session_{session_id}' / 'detected_hvs_events.npy')

    # --- 1. Determine Channel Numbers ---
    files = os.listdir(output_dir_attrs)
    chan_numbers = []
    for f in files:
        if f.startswith("attribution_scores_entire_session_chan") and f.endswith("_band0.npz"):
            try:
                # Expected format: attribution_scores_entire_session_chanX_band0.npz
                # Parts: [attr, scores, entire, session, chanX, band0.npz] -> index 4
                chan_part = f.split("_")[4].replace("chan", "")
                chan_numbers.append(int(chan_part))
            except (IndexError, ValueError):
                continue
    
    chan_numbers = sorted(list(set(chan_numbers)))
    num_channels = len(chan_numbers)
    if num_channels == 0:
        print("Error: No attribution files found.")
        return
    print(f"Number of channels found: {num_channels}")
    
    # --- 2. Load Spiking Data ---
    session_obj = ps(session_id, df, output_dir)
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.passive_times[1])
    spikes_obj.truncate(750) 
    
    t = np.arange(time_win[0], time_win[1], bin_size)
    n_bins_snippet = len(t)
    timestamps = spikes_obj.timestamps
    n_timepoints_total = len(timestamps)
    
    # Filter stimulus times (Active & non-omitted)
    stim_st = np.load(hvs_file)[:,1]
    
    # Ensure trials stay within session bounds
    valid_trials_mask = ((stim_st + time_win[0]) >= timestamps[0]) & ((stim_st + time_win[1]) < timestamps[-1])
    stim_st = stim_st[valid_trials_mask]
    n_trials = len(stim_st)
    print(f"Valid trials for analysis: {n_trials}")

    # Calculate PSTH
    print("Calculating PSTH")
    n_neurons = spikes_obj.spkMat.shape[1]
    psth = np.zeros((n_neurons, n_bins_snippet), dtype=np.float32)
    for i in range(n_trials):
        start_idx = np.searchsorted(timestamps, stim_st[i] + time_win[0])
        if start_idx + n_bins_snippet <= n_timepoints_total:
            psth += spikes_obj.spkMat[start_idx : start_idx + n_bins_snippet, :].T
    psth /= (n_trials * bin_size)
    print("PSTH successfully computed")
    # Store trial indices for reuse in the channel loop
    trial_start_indices = [np.searchsorted(timestamps, s + time_win[0]) for s in stim_st]
    
    del spikes_obj
    gc.collect()

    # --- 3. Initialize Snippet Containers ---
    # Memory efficient types: uint16 for ranks, bool for mask
    rank_snippets = np.zeros((n_trials, n_bins_snippet, n_neurons, num_channels), dtype=np.uint16)
    active_mask_snippet = np.zeros((n_trials, n_bins_snippet, n_neurons, num_channels), dtype=bool) 
    jaccard_raw_snippets = np.zeros((n_trials, n_bins_snippet, num_channels), dtype=np.float32)
    jaccard_ref_snippets = np.zeros((n_trials, n_bins_snippet, num_channels), dtype=np.float32)

    # Pre-calculate baseline relative indices
    b_start_rel = np.searchsorted(t, baseline_win[0])
    b_end_rel = np.searchsorted(t, baseline_win[1])

    # --- 4. Channel-by-Channel Processing ---
    print("Starting Channel Loop...", flush=True)
    for ch_idx, ch_val in enumerate(chan_numbers): 
        print(f"Processing Channel {ch_val} ({ch_idx+1}/{num_channels})", flush=True)
        
        # Load Sparse Attribution
        sparse_file = output_dir_attrs / f'attribution_scores_entire_session_chan{ch_val}_band0.npz'
        sparse_attr = np.load(sparse_file, allow_pickle=True)
        attrs = coo_matrix((sparse_attr['data'], (sparse_attr['row'], sparse_attr['col'])), 
                          shape=sparse_attr['shape']).toarray().astype(np.float32)
        
        # Normalize Attribution (Relative Load)
        total_attr = np.nansum(np.abs(attrs), axis=1, keepdims=True)
        attrs_normalized = np.divide(np.abs(attrs), total_attr, out=np.zeros_like(attrs), where=total_attr > EPSILON)
        
        # Compute Rank for the entire session for this channel
        rank_matrix_channel = np.zeros((n_timepoints_total, n_neurons), dtype=np.int16)
        # Sequence for rank assignment [n, n-1, ..., 1]
        rank_values = np.arange(n_neurons, 0, -1, dtype=np.int16)
        for ti in range(n_timepoints_total):
            rank_matrix_channel[ti, np.argsort(attrs_normalized[ti, :])] = rank_values

        # --- Jaccard Ref: Calculate Mean Baseline Rank across trials ---
        # Using a summation approach to avoid large intermediate arrays
        sum_baseline_rank = np.zeros(n_neurons, dtype=np.float32)
        count_baseline_bins = n_trials * (b_end_rel - b_start_rel)
        for s_idx in trial_start_indices:
            sum_baseline_rank += np.sum(rank_matrix_channel[s_idx + b_start_rel : s_idx + b_end_rel, :], axis=0)
        
        avg_baseline_rank = sum_baseline_rank / count_baseline_bins
        baseline_top_K = set(np.argsort(avg_baseline_rank)[:K])
        
        # --- Jaccard: Stability Analysis (Session-wide) ---
        j_raw_full = np.zeros(n_timepoints_total, dtype=np.float32)
        j_ref_full = np.zeros(n_timepoints_total, dtype=np.float32)
        for ti in range(n_timepoints_total - 1):
            s_t = set(np.where(rank_matrix_channel[ti, :] <= K)[0])
            s_t1 = set(np.where(rank_matrix_channel[ti+1, :] <= K)[0])
            
            # t vs t+1
            u = len(s_t | s_t1)
            j_raw_full[ti] = len(s_t & s_t1) / u if u > 0 else 0
            
            # t vs Baseline
            u_ref = len(s_t | baseline_top_K)
            j_ref_full[ti] = len(s_t & baseline_top_K) / u_ref if u_ref > 0 else 0

        # --- Snippet Extraction for this channel ---
        for i, s_idx in enumerate(trial_start_indices):
            e_idx = s_idx + n_bins_snippet
            if e_idx <= n_timepoints_total:
                rank_snippets[i, :, :, ch_idx] = rank_matrix_channel[s_idx:e_idx, :]
                # Active mask based on raw (non-normalized) zero attribution
                active_mask_snippet[i, :, :, ch_idx] = (np.abs(attrs[s_idx:e_idx, :]) > EPSILON)
                jaccard_raw_snippets[i, :, ch_idx] = j_raw_full[s_idx:e_idx]
                jaccard_ref_snippets[i, :, ch_idx] = j_ref_full[s_idx:e_idx]

        # Cleanup
        del attrs, attrs_normalized, total_attr, rank_matrix_channel, j_raw_full, j_ref_full
        gc.collect() 

    # --- 5. Saving ---
    print("Saving aggregated snippet data...", flush=True)
    np.save(variables_dir / 'psth_hvs.npy', psth)
    np.save(variables_dir / 'rank_snippets_hvs.npy', rank_snippets)
    np.save(variables_dir / 'jaccard_ref_snippets_hvs.npy', jaccard_ref_snippets)
    np.save(variables_dir / 'jaccard_raw_snippets_hvs.npy', jaccard_raw_snippets)
    
    # Sparse mask saving
    orig_shape = active_mask_snippet.shape
    sparse_mask = coo_matrix(active_mask_snippet.reshape(-1, orig_shape[-1]))
    save_npz(variables_dir / 'active_mask_snippet_sparse_hvs.npz', sparse_mask)
    np.save(variables_dir / 'active_mask_shape_hvs.npy', orig_shape)

    print(f"Success. Variables saved to: {variables_dir}")

def main(args):
    gc.enable()
    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f"Directory {dir_path} does not exist.")
        sys.exit(1)
    
    session_id = int(dir_path.name)
    # Reconstructing root based on common Allen SDK dir structure
    root_dir = dir_path.parent.parent.parent
    get_peri_hvs_data(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the session directory")
    args = parser.parse_args()
    main(args)