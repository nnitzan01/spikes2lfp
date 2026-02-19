import os
import sys
import torch
import argparse
import numpy as np
from plots import *
import pandas as pd
import models as models
from pathlib import Path
from preprocess_data import *
from sklearn.metrics import r2_score
from process_session import session as ps
from scipy.sparse import coo_matrix, save_npz
from IntegratedGradient_local import IntegratedGradient
import copy

def test_firing_rate_contribution(output_dir, session_id):
    """
    Test the contribution of firing rate to attribution by downsampling high-attribution neurons.
    
    Parameters:
    -----------
    output_dir : str or Path
        Output directory path
    session_id : int
        Session identifier
    Returns:
    ----- ---
        None (saves results to a .npy file in the output directory)
    """
    
    n_top_neurons  = 10 
    n_iterations   = 50 
    target_channel = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load units info
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        return None

    print("Obtaining session, spikes, and LFP data.", flush=True)
    session_obj = ps(session_id, df, output_dir)

    # Model hyperparameters (same as start.py but only for broadband)
    input_size = len(session_obj.units)
    hidden_size = 50 
    num_layers = 1
    seqlength = 750
    num_epochs = 15
    criterion = torch.nn.MSELoss()
    batch_size = 32
    bin_size = 0.004

    # Preprocess spikes and LFP
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.active_times[1])
    spikes_obj.truncate(seqlength)
    spikes_obj.minmax()

    lfp_obj = pre_process_lfp(session_id, session_obj.channels, session_obj.active_times[0],
                                session_obj.active_times[1], output_dir) 
    lfp_obj.filter_lfp(take_power=True)
    lfp_obj.downsample_lfp(5)
    lfp_obj.truncate(seqlength)
    lfp_obj.align_lfp(spikes_obj.spkMat.shape[0])

    print("Training broadband model for channel 5", flush=True)
    
    # Train only broadband model (bandi=0) for target channel
    bandi = 0  # broadband
    chani = target_channel
    
    X_train, X_test, y_train, y_test = chunk_and_reshape(spikes_obj.spkMat, lfp_obj.lfpMat[:,:,bandi], 
                                                    seqlength, test_size=0.2, random_state=42)
    
    train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)          
    model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers, seqlength), criterion, device)
    train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
    
    print("Training completed. Calculating initial attributions...", flush=True)

    # Calculate initial attributions for all neurons
    attr_dur = 720
    X_attr = torch.tensor(spikes_obj.spkMat[:int(attr_dur/bin_size),:]).float().to(device)
    num_trials = int(int(attr_dur/bin_size)/seqlength)
    X_attr = X_attr.reshape(num_trials, seqlength, X_attr.shape[1])
    
    ig = IntegratedGradient(model.model.train().to(device), method='last time point', seqlength=seqlength)        
    attrs = ig.run(X_attr, baselines=0, n_batch=10, n_steps=25).cpu()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    attrs = np.array(attrs)
    
    # Calculate mean absolute attribution for each neuron
    mean_abs_attrs = np.mean(np.abs(attrs), axis=0)  # Mean across time
    
    # Find top N neurons with highest attribution
    top_neuron_indices = np.argsort(mean_abs_attrs)[-n_top_neurons:][::-1]  # Descending order
    
    print(f"Top {n_top_neurons} neurons by mean absolute attribution:")
    for i, neuron_idx in enumerate(top_neuron_indices):
        print(f"  {i+1}. Neuron {neuron_idx}: {mean_abs_attrs[neuron_idx]:.6f}")
    
    # Get area information for these neurons
    neuron_areas = []
    neuron_unit_ids = []
    
    for neuron_idx in top_neuron_indices:
        # Get the actual unit_id from the index, not the entire row
        unit_id = session_obj.units.index[neuron_idx]
        neuron_unit_ids.append(unit_id)
        
        # Get area information directly from session object
        area = session_obj.units['structure_acronym'].iloc[neuron_idx]
        neuron_areas.append(area)
        print(f"  Neuron {neuron_idx} (unit_id {unit_id}) is in area: {area}")
    
    # Calculate firing rates for all neurons and area averages
    original_spike_mat = spikes_obj.spkMat.copy()
    total_bins = original_spike_mat.shape[0]
    
    # Calculate firing rates (proportion of non-zero bins)
    firing_rates = np.sum(original_spike_mat > 0, axis=0) / total_bins
    
    # Calculate average firing rates by area
    area_avg_firing_rates = {}
    unique_areas = np.unique(neuron_areas)
    
    for area in unique_areas:
        # Find all neurons in this area from session data
        area_neuron_indices = []
        for idx in range(len(session_obj.units)):
            if session_obj.units['structure_acronym'].iloc[idx] == area:
                area_neuron_indices.append(idx)
        
        if area_neuron_indices:
            area_avg_firing_rates[area] = np.mean(firing_rates[area_neuron_indices])
            print(f"Average firing rate in {area}: {area_avg_firing_rates[area]:.4f}")
    
    # Results storage
    results = {
        'top_neuron_indices': top_neuron_indices,
        'top_neuron_unit_ids': neuron_unit_ids,
        'top_neuron_areas': neuron_areas,
        'original_attributions': mean_abs_attrs[top_neuron_indices],
        'downsampled_attributions': [],
        'area_avg_firing_rates': area_avg_firing_rates,
        'original_firing_rates': firing_rates[top_neuron_indices]
    }
    
    print(f"Starting downsampling analysis with {n_iterations} iterations per neuron...", flush=True)
    
    # For each top neuron, perform downsampling
    for neuron_i, neuron_idx in enumerate(top_neuron_indices):
        area = neuron_areas[neuron_i]
        original_fr = firing_rates[neuron_idx]
        target_fr = area_avg_firing_rates[area]
        
        print(f"\nProcessing neuron {neuron_idx} (area: {area})")
        print(f"  Original firing rate: {original_fr:.4f}")
        print(f"  Target firing rate: {target_fr:.4f}")
        
        # Calculate how many bins to zero out
        current_active_bins = np.sum(original_spike_mat[:, neuron_idx] > 0)
        target_active_bins = int(target_fr * total_bins)
        bins_to_zero = current_active_bins - target_active_bins
        
        if bins_to_zero <= 0:
            print(f"  Neuron already at or below target firing rate. Skipping.")
            results['downsampled_attributions'].append([mean_abs_attrs[neuron_idx]] * n_iterations)
            continue
            
        print(f"  Need to zero out {bins_to_zero} bins")
        
        # Store attributions for this neuron across iterations
        neuron_downsampled_attrs = []
        
        for iteration in range(n_iterations):
            # Create a copy of the spike matrix
            modified_spike_mat = original_spike_mat.copy()
            
            # Find active bins for this neuron
            active_bin_indices = np.where(modified_spike_mat[:, neuron_idx] > 0)[0]
            
            if len(active_bin_indices) < bins_to_zero:
                # Not enough active bins to zero out
                neuron_downsampled_attrs.append(0.0)
                continue
            
            # Choose a random starting point for continuous stretch
            max_start_idx = len(active_bin_indices) - bins_to_zero
            if max_start_idx < 0:
                neuron_downsampled_attrs.append(0.0)
                continue
                
            start_idx = np.random.randint(0, max_start_idx + 1)
            bins_to_zero_indices = active_bin_indices[start_idx:start_idx + bins_to_zero]
            
            # Zero out the selected bins
            modified_spike_mat[bins_to_zero_indices, neuron_idx] = 0
            
            # Recalculate attribution with modified spike matrix
            X_attr_modified = torch.tensor(modified_spike_mat[:int(attr_dur/bin_size),:]).float().to(device)
            # Verify shape consistency - should match original X_attr shape
            assert X_attr_modified.shape == X_attr.reshape(-1, X_attr.shape[-1]).shape, \
                f"Shape mismatch: original {X_attr.reshape(-1, X_attr.shape[-1]).shape}, modified {X_attr_modified.shape}"
            num_trials_mod = int(int(attr_dur/bin_size)/seqlength)
            X_attr_modified = X_attr_modified.reshape(num_trials_mod, seqlength, X_attr_modified.shape[1])
            
            # Calculate attribution with reduced batch size for memory efficiency
            torch.cuda.empty_cache()  # Clear cache before attribution
            attrs_modified = ig.run(X_attr_modified, baselines=0, n_batch=10, n_steps=25).cpu()  # Reduced batch and steps
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Move to CPU immediately to free GPU memory
                X_attr_modified = X_attr_modified.cpu()
                del X_attr_modified
            
            attrs_modified = np.array(attrs_modified)
            mean_abs_attr_modified = np.mean(np.abs(attrs_modified[:, neuron_idx]))
            
            neuron_downsampled_attrs.append(mean_abs_attr_modified)
            
            if (iteration + 1) % 10 == 0:
                print(f"    Completed {iteration + 1}/{n_iterations} iterations")
        
        results['downsampled_attributions'].append(neuron_downsampled_attrs)
        print(f"  Average downsampled attribution: {np.mean(neuron_downsampled_attrs):.6f}")
        print(f"  Original attribution: {mean_abs_attrs[neuron_idx]:.6f}")
    
    # Calculate population attribution averages by area
    print("\nCalculating population attribution averages by area...")
    area_population_attrs = {}
    
    for area in unique_areas:
        # Find all neurons in this area from session data  
        area_neuron_indices = []
        for idx in range(len(session_obj.units)):
            if session_obj.units['structure_acronym'].iloc[idx] == area:
                area_neuron_indices.append(idx)
        
        if area_neuron_indices:
            area_attrs = mean_abs_attrs[area_neuron_indices]
            area_population_attrs[area] = np.mean(area_attrs)
            print(f"  {area}: {area_population_attrs[area]:.6f} (n={len(area_neuron_indices)} neurons)")
    
    results['area_population_attributions'] = area_population_attrs
    
    print("Analysis complete!", flush=True)
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs' / str(session_id))
    filename = Path(output_dir_attrs / 'attribution_firing_rate_resampling_results.npy')
    np.save(filename, results)

def main(args):
    dir = Path(args.dir)
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        sys.exit(1)
    session_id = int(os.path.basename(dir))
    root_dir = dir.parent.parent.parent
    print(f"Root dir is set to: {root_dir}", flush=True)
    print(f"Session ID is set to: {session_id}", flush=True)    
    
    test_firing_rate_contribution(root_dir, session_id)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)