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
from spectral_analysis import bandpass_filter, calculate_band_r2


def start(output_dir, session_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        exit(1)

    print("Session obj.", flush=True)
    session_obj = ps(session_id, df, output_dir)
    output_dir = Path(output_dir)
    
    # model hyperparameters
    hidden_size = 50
    num_layers = 1
    seqlength = 750
    num_epochs = 15
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
    criterion  = torch.nn.MSELoss()
    batch_size = 32
    bin_size=0.004
    sr = 1/bin_size

    print("Obtaining spikes obj.", flush=True)
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.active_times[1])
    spikes_obj.truncate(seqlength)
    spikes_obj.minmax()

    print("Obtaining LFP obj.", flush=True)
    lfp_obj = pre_process_lfp(session_id, session_obj.channels, session_obj.active_times[0],
                                session_obj.active_times[1], output_dir) 
    lfp_obj.filter_lfp(take_power = True)
    lfp_obj.downsample_lfp(5)
    lfp_obj.truncate(seqlength)
    lfp_obj.align_lfp(spikes_obj.spkMat.shape[0])
    
    locs = spikes_obj.units['structure_acronym'].values
    areas = np.unique(locs)

    ctx = [area for area in areas if 'VIS' in area]
    ctx_no_visp = [area for area in areas if 'VIS' in area and 'VISp' not in area]
    hpc = [area for area in areas if 'CA' in area or 'DG' in area or 'SUB' in area
        or 'HPF' in area or 'ProS' in area or 'POST' in area]
    subcortical = [area for area in areas if area not in ctx and area not in hpc]
    
    mask_ctx = spikes_obj.units['structure_acronym'].isin(ctx)
    mask_ctx_no_visp = spikes_obj.units['structure_acronym'].isin(ctx_no_visp)
    mask_hpc = spikes_obj.units['structure_acronym'].isin(hpc)
    mask_subcortical = spikes_obj.units['structure_acronym'].isin(subcortical)
    
    sum_ctx = spikes_obj.units['structure_acronym'].isin(ctx).sum()
    sum_ctx_no_visp = spikes_obj.units['structure_acronym'].isin(ctx_no_visp).sum()
    sum_hpc = spikes_obj.units['structure_acronym'].isin(hpc).sum()
    sum_subcortical = spikes_obj.units['structure_acronym'].isin(subcortical).sum()
    
    rate_ctx = session_obj.units.loc[session_obj.units['structure_acronym'].isin(ctx), 'firing_rate'].mean()
    rate_ctx_no_visp = session_obj.units.loc[session_obj.units['structure_acronym'].isin(ctx_no_visp), 'firing_rate'].mean()
    rate_hpc = session_obj.units.loc[session_obj.units['structure_acronym'].isin(hpc), 'firing_rate'].mean()
    rate_subcortical = session_obj.units.loc[session_obj.units['structure_acronym'].isin(subcortical), 'firing_rate'].mean()

    res = pd.DataFrame(index=['cortex', 'cortex_no_visp', 'hippocampus', 'subcortical'],
                    columns=['num_units', 'mean_firing_rate'])
    res.loc['cortex'] = [sum_ctx, rate_ctx]
    res.loc['cortex_no_visp'] = [sum_ctx_no_visp, rate_ctx_no_visp]
    res.loc['hippocampus'] = [sum_hpc, rate_hpc]
    res.loc['subcortical'] = [sum_subcortical, rate_subcortical]
    
    bandi = 0
    
    print("Training full model.", flush=True)
    r2_series_list = []          
    X_train, X_test, y_train, y_test  = chunk_and_reshape(spikes_obj.spkMat, lfp_obj.lfpMat[:,:,bandi], 
                                                         seqlength, test_size=0.2, random_state=42)
    input_size = X_train.shape[2]
    for chani in range(len(lfp_obj.channels)):
        train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)
        model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
        train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
        yHat = model.evaluate(test_dataloader.dataset.tensors[0])
        r2_broadband = r2_score(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), 
                       yHat.cpu().numpy().reshape(-1,1))
        r2_bands = calculate_band_r2(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1),
                                              np.array(yHat.to('cpu')), bands[:-2], sr)
        r2_bands.loc['broadband'] = r2_broadband
        r2_series_list.append(r2_bands.rename(f'channel_{chani}'))
    r2_results_full_model = pd.concat(r2_series_list, axis=1)
    
    print("Training cortex only model.", flush=True)
    r2_series_list = []
    X_train, X_test, y_train, y_test = chunk_and_reshape(spikes_obj.spkMat[:, mask_ctx], lfp_obj.lfpMat[:, :, bandi],
                                                         seqlength, test_size=0.2, random_state=42)
    input_size = X_train.shape[2]
    for chani in range(len(lfp_obj.channels)):
        train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)
        model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
        train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
        yHat = model.evaluate(test_dataloader.dataset.tensors[0])
        r2_broadband = r2_score(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), 
                       yHat.cpu().numpy().reshape(-1,1))
        r2_bands = calculate_band_r2(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1),
                                              np.array(yHat.to('cpu')), bands[:-2], sr)
        r2_bands.loc['broadband'] = r2_broadband
        r2_series_list.append(r2_bands.rename(f'channel_{chani}'))
    r2_results_cortex = pd.concat(r2_series_list, axis=1)

    print("Training cortex (no visp) model.", flush=True)
    r2_series_list = []
    X_train, X_test, y_train, y_test = chunk_and_reshape(spikes_obj.spkMat[:, mask_ctx_no_visp], lfp_obj.lfpMat[:, :, bandi],
                                                         seqlength, test_size=0.2, random_state=42)
    input_size = X_train.shape[2]
    for chani in range(len(lfp_obj.channels)):
        train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)
        model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
        train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
        yHat = model.evaluate(test_dataloader.dataset.tensors[0])
        r2_broadband = r2_score(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), 
                       yHat.cpu().numpy().reshape(-1,1))
        r2_bands = calculate_band_r2(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1),
                                              np.array(yHat.to('cpu')), bands[:-2], sr)
        r2_bands.loc['broadband'] = r2_broadband
        r2_series_list.append(r2_bands.rename(f'channel_{chani}'))
    r2_results_cortex_no_visp = pd.concat(r2_series_list, axis=1)

    print("Training hpc model.", flush=True)
    r2_series_list = []
    X_train, X_test, y_train, y_test = chunk_and_reshape(spikes_obj.spkMat[:, mask_hpc], lfp_obj.lfpMat[:, :, bandi],
                                                         seqlength, test_size=0.2, random_state=42)
    input_size = X_train.shape[2]
    for chani in range(len(lfp_obj.channels)):
        train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)
        model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
        train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
        yHat = model.evaluate(test_dataloader.dataset.tensors[0])
        r2_broadband = r2_score(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), 
                       yHat.cpu().numpy().reshape(-1,1))
        r2_bands = calculate_band_r2(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1),
                                              np.array(yHat.to('cpu')), bands[:-2], sr)
        r2_bands.loc['broadband'] = r2_broadband
        r2_series_list.append(r2_bands.rename(f'channel_{chani}'))
    r2_results_hpc = pd.concat(r2_series_list, axis=1)

    print("Training subcortical model.", flush=True)
    r2_series_list = []
    X_train, X_test, y_train, y_test = chunk_and_reshape(spikes_obj.spkMat[:, mask_subcortical], lfp_obj.lfpMat[:, :, bandi],
                                                         seqlength, test_size=0.2, random_state=42)
    input_size = X_train.shape[2]
    for chani in range(len(lfp_obj.channels)):
        train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)
        model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
        train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
        yHat = model.evaluate(test_dataloader.dataset.tensors[0])
        r2_broadband = r2_score(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), 
                       yHat.cpu().numpy().reshape(-1,1))
        r2_bands = calculate_band_r2(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1),
                                              np.array(yHat.to('cpu')), bands[:-2], sr)
        r2_bands.loc['broadband'] = r2_broadband
        r2_series_list.append(r2_bands.rename(f'channel_{chani}'))
    r2_results_subcortical = pd.concat(r2_series_list, axis=1)

    # save all the R2 scores as csv files
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    os.makedirs(variables_dir, exist_ok=True)

    r2_results_subcortical.to_csv(Path(variables_dir / 'r2_results_subcortical.csv'), index=True)
    r2_results_hpc.to_csv(Path(variables_dir / 'r2_results_hpc.csv'), index=True)
    r2_results_cortex_no_visp.to_csv(Path(variables_dir / 'r2_results_cortex_no_visp.csv'), index=True)
    r2_results_cortex.to_csv(Path(variables_dir / 'r2_results_cortex.csv'), index=True)
    r2_results_full_model.to_csv(Path(variables_dir / 'r2_results_full_model.csv'), index=True)
    
    res.to_csv(Path(variables_dir / 'units_counts_and_rates.csv'), index=True)
    
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
    start(root_dir, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to the root dir: ")
    args = parser.parse_args()
    main(args)