import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess_data import *
from process_session import session as ps
from scipy.sparse import coo_matrix
import warnings # Required for RuntimeWarning handling

def get_peri_stim_data(output_dir, session_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for unit info file
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        exit(1)

    print("Obtaining session data and calculating PSTHs.", flush=True)
    
    # --- CONSTANTS ---
    seqlength = 750
    bin_size  = 0.004
    K = 40 # number of top neurons for rank comparison
    EPSILON = 1e-10
    
    # Path to the attrs folder
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs_entire_session' / str(session_id))

    # --- 1. Determine Channel Numbers ---
    files = os.listdir(output_dir_attrs)
    
    # We must find the actual list of channel numbers present in the directory
    # We rely on the naming convention 'chanX' where X is the 0-indexed number
    chan_numbers_0_indexed = []
    for f in files:
        if f.startswith("attribution_scores_entire_session_chan") and f.endswith("_band0.npz"):
            parts = f.split("_")
            # Filename is e.g., 'attribution_scores_entire_session_chan5_band0.npz'
            # The channel number is embedded in the 4th part (index 3)
            try:
                # Assuming the channel number is the 0-based index
                chan_part = parts[2].replace("chan", "")
                chan_numbers_0_indexed.append(int(chan_part))
            except (IndexError, ValueError):
                continue
    
    if not chan_numbers_0_indexed:
        print("Error: No attribution files found for band 0.")
        return
        
    # Get the unique, sorted list of channel indices (e.g., [0, 1, 2, ..., 31])
    chan_numbers_0_indexed = sorted(list(set(chan_numbers_0_indexed)))
    num_channels_storage = len(chan_numbers_0_indexed)
    print(f"Number of channels found: {num_channels_storage}")
    
    # --- 2. Load Spiking Data and Time Vectors ---
    session_obj = ps(session_id, df, output_dir)
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.passive_times[1])
    spikes_obj.truncate(seqlength)
    
    time_win = [-.25, .5]
    t = np.arange(time_win[0], time_win[1], bin_size)
    timestamps = spikes_obj.timestamps
    
    # Filter stimulus times to ensure full snippet fits within session data
    stim_st = session_obj.stimulus_presentation.start_time
    stim_st = stim_st[((stim_st + time_win[0]) > timestamps[0]) & ((stim_st + time_win[1]) < timestamps[-1])]
    stim_st = stim_st.values
    
    # Calculate PSTHs
    psth = np.zeros((spikes_obj.spkMat.shape[1], len(t)))
    for i in range(len(stim_st)):
        # Calculate start index in the full spike matrix (stim_st[i] + time_win[0])
        start = np.argmin(np.abs(timestamps - (stim_st[i] + time_win[0])))
        # Note: The PSTH needs the spike matrix snippet from start to start + len(t)
        psth += spikes_obj.spkMat[start:start+len(t),:].T
        
    # The duration of the snippet is len(t). Number of trials is len(stim_st).
    psth = psth / (len(stim_st) * bin_size) 

    # --- 3. Initialize Global Rank and Mask Matrices ---
    n_neurons = spikes_obj.spkMat.shape[1]
    n_timepoints = len(timestamps)
    
    # Rank and Mask are 3D: (Time x Neuron x Channel)
    rank_matrix = np.zeros((n_timepoints, n_neurons , num_channels_storage))
    non_zero_att_mask_global = np.zeros((n_timepoints, n_neurons , num_channels_storage), dtype=bool)
    
    # --- 4. Rank Calculation Loop (Per Channel) ---
    print("Calculating Ranks and Active Masks...")
    
    # Iterate over the 0-indexed channel numbers found in the files
    for ch_0_indexed in chan_numbers_0_indexed: 
        ch_storage_idx = chan_numbers_0_indexed.index(ch_0_indexed) # 0-indexed storage position (0 to num_channels_storage-1)
        
        # Load the attribution data for the current channel and broadband (band 0)
        sparse_attr = np.load(output_dir_attrs / f'attribution_scores_entire_session_chan{ch_0_indexed}_band{0}.npz') 
        attrs = coo_matrix((sparse_attr['data'], (sparse_attr['row'], sparse_attr['col'])), shape=sparse_attr['shape']).toarray()
        
        # Calculate Relative Attribution (Share of Network Load)
        total_attr = np.nansum(np.abs(attrs), axis=1, keepdims=True)
        
        # Use np.divide and np.nan_to_num to handle division by total_attr=0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Add total_attr>EPSILON check for division safety
            attrs_normalized = np.divide(attrs, total_attr, out=np.zeros_like(attrs), where=total_attr > EPSILON)
            
        abs_attrs = np.abs(attrs_normalized)
        
        # Store the non-zero active mask
        non_zero_att_mask_global[:, :, ch_storage_idx] = (np.abs(attrs) > EPSILON) 
        
        # Calculate Rank for all time points
        for ti in range(n_timepoints):
            abs_values_t = abs_attrs[ti, :]
            sorted_indices_ascending = np.argsort(abs_values_t)
            ranks_descending = np.arange(n_neurons, 0, -1)
            
            # Assign the rank to the correct neuron index and channel index
            rank_matrix[ti, sorted_indices_ascending, ch_storage_idx] = ranks_descending
            
        del attrs, sparse_attr, attrs_normalized # Memory cleanup

    # Calculate Inverse Rank Matrix (Time x Neuron x Channel)
    InverseRank_Matrix = 1/rank_matrix
    
    # --- 5. Jaccard Stability Analysis ---
    
    # Jaccard scores are 2D: (Time x Channel)
    jaccard_scores_raw = np.zeros((n_timepoints - 1, num_channels_storage))
    jaccard_scores_baseline_ref = np.zeros((n_timepoints - 1, num_channels_storage))
    
    print("Calculating Jaccard Scores...")
    
    # Compute baseline reference ranks (Time-averaged top K neurons)
    baseline_win = [-.25, 0] # Example baseline window
    baseline_start_idx = np.argmin(np.abs(t - baseline_win[0]))
    baseline_end_idx = np.argmin(np.abs(t - baseline_win[1]))
    
    # Compute peri-stimulus rank matrix (Trial x TimeBin x Neuron x Channel)
    peri_stim_rank = np.zeros((len(stim_st), len(t), n_neurons, num_channels_storage))
    
    for i in range(len(stim_st)):
        start_time_sec = stim_st[i] + time_win[0]
        start_idx = np.argmin(np.abs(timestamps - start_time_sec))
        
        # Extract snippet for all channels simultaneously
        peri_stim_rank[i,:,:,:] = rank_matrix[start_idx : start_idx + len(t), :, :]
        
    baseline_ranks = peri_stim_rank[:, baseline_start_idx:baseline_end_idx, :, :]
    # Average across time bins and trials (Axis 0=Trial, Axis 1=TimeBin) -> (Neuron x Channel)
    avg_rank_per_neuron = np.mean(np.mean(baseline_ranks, axis=0), axis=0) 
    
    for ch_storage_idx in range(num_channels_storage):
        # Top K neurons for baseline (lowest rank is best, so we sort avg_rank_per_neuron ascending)
        baseline_top_K_indices = np.argsort(avg_rank_per_neuron[:, ch_storage_idx])[:K]
        S_Baseline = set(baseline_top_K_indices)
        
        for ti in range(n_timepoints - 1):
            # Raw Jaccard (t vs t+1)
            indices_t = np.where(rank_matrix[ti, :, ch_storage_idx] <= K)[0]
            S_t = set(indices_t)
            indices_t_plus_1 = np.where(rank_matrix[ti+1, :, ch_storage_idx] <= K)[0]
            S_t_plus_1 = set(indices_t_plus_1)
            
            intersection = len(S_t.intersection(S_t_plus_1))
            union = len(S_t.union(S_t_plus_1))
            jaccard_scores_raw[ti,ch_storage_idx] = intersection / union if union > 0 else 0
                
            # Baseline Jaccard (t vs Baseline)
            intersection_baseline = len(S_t.intersection(S_Baseline))
            union_baseline = len(S_t.union(S_Baseline))
            jaccard_scores_baseline_ref[ti,ch_storage_idx] = intersection_baseline / union_baseline if union_baseline > 0 else 0
            
    # --- 6. Final Snippet Extraction and Saving ---
    
    # Final snippets are 3D: (Trial x TimeBin x Channel)
    jaccard_ref_snippets = np.zeros((len(stim_st), len(t), num_channels_storage))
    jaccard_raw_snippets = np.zeros((len(stim_st), len(t), num_channels_storage)) 
    
    # Inverse Rank and Active Mask are 4D: (Trial x TimeBin x Neuron x Channel)
    inverse_rank_snippets = np.zeros((len(stim_st), len(t), n_neurons, num_channels_storage))
    active_mask_snippet = np.zeros((len(stim_st), len(t), n_neurons, num_channels_storage), dtype=bool) 
    
    print("Extracting and saving final snippets...")
    
    for i in range(len(stim_st)):
        start_time_sec = stim_st[i] + time_win[0]
        start_idx = np.argmin(np.abs(timestamps - start_time_sec))
        
        # Check for potential out-of-bounds error near end of session
        snippet_end_idx = start_idx + len(t)
        if snippet_end_idx > n_timepoints:
             # Should not happen due to initial stim_st filtering, but safe guard.
             print(f"Warning: Trial {i} snippet truncated due to session end.")
             snippet_end_idx = n_timepoints

        # Extracting 4D arrays (Inverse Rank, Active Mask)
        inverse_rank_snippets[i, :, :, :] = InverseRank_Matrix[start_idx : snippet_end_idx, :, :]
        active_mask_snippet[i, :, :, :]   = non_zero_att_mask_global[start_idx : snippet_end_idx, :, :]
        
        # Extracting 3D arrays (Jaccard Scores)
        jaccard_ref_snippets[i, :, :] = jaccard_scores_baseline_ref[start_idx : snippet_end_idx, :]
        jaccard_raw_snippets[i, :, :] = jaccard_scores_raw[start_idx : snippet_end_idx, :]
             
    # --- Saving ---
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    os.makedirs(variables_dir, exist_ok=True)
    
    np.save(variables_dir / 'psth.npy', psth)
    np.save(variables_dir / 'inverse_rank_snippets.npy', inverse_rank_snippets)
    np.save(variables_dir / 'active_mask_snippet.npy', active_mask_snippet)
    np.save(variables_dir / 'jaccard_ref_snippets.npy', jaccard_ref_snippets)
    np.save(variables_dir / 'jaccard_raw_snippets.npy', jaccard_raw_snippets)

    print(f"Results successfully saved in: {variables_dir}")
    print("Session complete.", flush=True)

def main(args):
    dir = Path(args.dir)
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        sys.exit(1)
    session_id = int(os.path.basename(dir))
    root_dir = dir.parent.parent.parent
    print(f"Root dir is set to: {root_dir}", flush=True)
    print(f"Session ID is set to: {session_id}", flush=True)
    get_peri_stim_data(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)