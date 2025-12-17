import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from process_session import session as ps
from preprocess_data import *
from scipy.sparse import load_npz
import argparse

def calculate_ipd(output_dir, session_id):
    
    # Check for unit info file
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        sys.exit(1)
    
    bin_size  = 0.004
    time_win = [-.25, .5]
    t = np.arange(time_win[0], time_win[1], bin_size)
    
    # Path setup
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    
    # --- 1. Load Data ---
    print(f"Loading data for Session {session_id}...", flush=True)
    
    session_obj = ps(session_id, df, output_dir)
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times)
    locs = spikes_obj.units['structure_acronym'].values
    
    # Load the Rank Snippets (Trial x Time x Neuron x Channel) - uint16
    rank_snippets = np.load(variables_dir / 'rank_snippets.npy')
    
    # Load and reconstruct Sparse Active Mask
    sparse_mask = load_npz(variables_dir / 'active_mask_snippet_sparse.npz')
    original_shape = np.load(variables_dir / 'active_mask_shape.npy')
    active_mask_snippet = sparse_mask.toarray().reshape(original_shape)
    
    n_trials, n_bins, n_neurons, n_channels = rank_snippets.shape
    areas = np.unique(locs)
    n_regions = len(areas)
    
    print(f"Calculating IPD for {n_regions} regions across {n_trials} trials and {n_channels} channels.", flush=True)

    # --- 2. Dominance Calculation ---
    # Container: (Trial x Region x Bin x Channel)
    # We use a 4D array to store IPD per trial/channel before averaging
    dominance_scores_4D = np.zeros((n_trials, n_regions, n_bins, n_channels), dtype=np.float32)

    # Pre-create region masks
    region_masks = {region: (locs == region) for region in areas}
    region_names = list(areas)

    for ch_idx in range(n_channels):
        print(f"Processing Channel Index {ch_idx}...", flush=True)
        
        for trial_idx in range(n_trials):
            # Extract snippets for this specific trial and channel
            # Shape: (n_bins x n_neurons)
            trial_ranks = rank_snippets[trial_idx, :, :, ch_idx].astype(np.float32)
            trial_active = active_mask_snippet[trial_idx, :, :, ch_idx]
            
            # Inverse Rank calculation (1/Rank) - only for active neurons
            # Note: We use np.where to avoid division by zero for inactive cells (rank 0 or masked)
            with np.errstate(divide='ignore'):
                inverse_rank_snippet = np.where(trial_active, 1.0 / trial_ranks, 0.0)

            for r_idx, region in enumerate(region_names):
                mask = region_masks[region]
                
                # Filter to neurons in the current region
                # Shapes: (n_bins x n_neurons_in_region)
                region_inv_ranks = inverse_rank_snippet[:, mask]
                region_active_mask = trial_active[:, mask]
                
                # Numerator: Sum of Inverse Ranks of ACTIVE neurons in region
                numerator = np.sum(region_inv_ranks, axis=1)
                
                # Denominator: Count of ACTIVE neurons in region
                denominator = np.sum(region_active_mask, axis=1)
                
                # IPD Calculation (Sum of 1/Rank / Count of active contributors)
                ipd_time_series = np.where(denominator > 0, numerator / denominator, 0.0)
                
                dominance_scores_4D[trial_idx, r_idx, :, ch_idx] = ipd_time_series

    # --- 3. Final Averaging and Saving ---
    print("Averaging results and calculating Proportional Influence...", flush=True)
    
    # Average Dominance per Region/Channel across trials
    # Shape: (Region x Bin x Channel)
    average_dominance = np.mean(dominance_scores_4D, axis=0)
    
    # Calculate Proportional Influence
    # Sum dominance across all regions at each time bin/channel
    total_regional_influence = np.sum(average_dominance, axis=0, keepdims=True)
    
    # Proportional Influence = Regional IPD / Total Session IPD
    # Shape: (Region x Bin x Channel)
    proportional_influence = np.where(total_regional_influence > 0, 
                                     average_dominance / total_regional_influence, 
                                     0.0)

    # --- 4. Save Results ---
    output_path = variables_dir / 'regional_dominance_results.npz'
    np.savez_compressed(
        output_path,
        average_dominance=average_dominance,
        proportional_influence=proportional_influence,
        region_names=region_names,
        t=t
    )
    
    print(f"Results successfully saved to: {output_path}")
    
def main(args):
    dir_path = Path(args.dir)
    if not dir_path.exists():
        print(f"Directory {dir_path} does not exist.")
        sys.exit(1)
    
    session_id = int(dir_path.name)
    # Reconstructing root based on common Allen SDK dir structure
    root_dir = dir_path.parent.parent.parent
    calculate_ipd(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the session directory")
    args = parser.parse_args()
    main(args)