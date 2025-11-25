import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import models as models
from pathlib import Path
from preprocess_data import *
from process_session import session as ps
from scipy.sparse import coo_matrix, save_npz
from IntegratedGradient_local import IntegratedGradient

def calc_attribution_all(output_dir, session_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        exit(1)

    print("Obtaining session, spikes, and LFP data for the entire session.", flush=True)
    session_obj = ps(session_id, df, output_dir)

    # model hyperparameters (should match the original training)
    input_size = len(session_obj.units)
    hidden_size = 50 
    num_layers = 1
    seqlength = 750
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
    criterion = torch.nn.MSELoss()
    bin_size = 0.004

    # Process spikes times
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.passive_times[1])
    spikes_obj.truncate(seqlength)
    spikes_obj.minmax()

    # Process LFP for spontaneous times
    lfp_obj = pre_process_lfp(session_id, session_obj.channels, session_obj.active_times[0],
                                session_obj.passive_times[1], output_dir) 
    lfp_obj.filter_lfp(take_power=True)
    lfp_obj.downsample_lfp(5)
    lfp_obj.truncate(seqlength)
    lfp_obj.align_lfp(spikes_obj.spkMat.shape[0])
    num_channels = len(lfp_obj.channels)

    # Load saved models instead of training
    print("Loading saved models from disk", flush=True)
    models_dir = Path(output_dir / 'spikes2lfp' / 'models' / str(session_id))
    if not models_dir.exists():
        print(f"Models directory {models_dir} does not exist. Please run training first.")
        sys.exit(1)

    all_models = {}
    for bandi in range(len(bands)+1):
        for chani in range(num_channels):
            model_file = models_dir / f'lstm_model_chan{chani}_band{bandi}.pt'
            if not model_file.exists():
                print(f"Model file {model_file} does not exist.")
                continue
            
            # Create model with same architecture as training
            model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)

            # Load the saved state dict
            model.model.load_state_dict(torch.load(model_file, map_location=device))
            model.model.eval()  # Set to evaluation mode
            all_models[chani, bandi] = model
            
    print(f"Loaded {len(all_models)} models successfully", flush=True)

    # Calculate attribution scores for the entire session
    print("Calculating and saving attribution scores for the entire session")

    # Reduce data size to avoid memory issues
    # max_samples = min(75000, spikes_obj.spkMat.shape[0])  # Limit to ~300s of data
    X_attr = torch.tensor(spikes_obj.spkMat).float().to(device)
    num_trials = int(spikes_obj.spkMat.shape[0] / seqlength)
    
    if num_trials == 0:
        print("Not enough data for even one trial. Exiting.")
        sys.exit(1)
    
    print(f"Processing {num_trials} trials")
    
    # Reshape for trials
    X_attr = X_attr[:num_trials * seqlength, :].reshape(num_trials, seqlength, X_attr.shape[1])
    # X_attr_flat = torch.tensor(spikes_obj.spkMat[:num_trials * seqlength, :]).float().to('cpu')
    
    mean_attribution = np.zeros((num_channels, len(bands)+1, spikes_obj.spkMat.shape[1]))
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs_entire_session' / str(session_id))
    os.makedirs(output_dir_attrs, exist_ok=True)

    bandi=0 # run only for broadband for now
    # for bandi in range(len(bands)+1):
    for chani in range(num_channels):
        if (chani, bandi) not in all_models:
            print(f"Model for channel {chani}, band {bandi} not found. Skipping.")
            continue
            
        # Clear GPU cache before each attribution calculation
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        model = all_models[chani, bandi]
        ig = IntegratedGradient(model.model.train().to(device), method='last time point', seqlength=seqlength)        
        # Reduce batch size  to avoid CUDA out of memory
        attrs = ig.run(X_attr, baselines=0, n_batch=10, n_steps=50).cpu()
        
        # Move attribution to CPU immediately and clear GPU cache
        attrs = np.array(attrs)
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # convert to relative attribution
        total_attr = np.nansum(np.abs(attrs), axis=1, keepdims=True)
        attrs = attrs / (total_attr + 1e-10)  # avoid division by zero
        filename = Path(output_dir_attrs / f'attribution_scores_entire_session_chan{chani}_band{bandi}.npz')
        attrs_sparse = coo_matrix(attrs)
        save_npz(filename, attrs_sparse, compressed=True)
        mean_attribution[chani, bandi, :] = np.mean(attrs, axis=0)
            
    filename = Path(output_dir_attrs / f'attribution_scores_entire_session_mean.npy')
    np.save(filename, mean_attribution)
    print("Session attribution scores are saved in: ", output_dir_attrs)
    print("Session attribution analysis complete.", flush=True)

def main(args):
    dir = Path(args.dir)
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        sys.exit(1)
    session_id = int(os.path.basename(dir))
    root_dir = dir.parent.parent.parent
    print(f"Root dir is set to: {root_dir}", flush=True)
    print(f"Session ID is set to: {session_id}", flush=True)    
    calc_attribution_all(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)