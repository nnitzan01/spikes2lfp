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
# from IntegratedGradient import IntegratedGradient
from IntegratedGradient_local import IntegratedGradient

def start(output_dir, session_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists('./tables/units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./tables/units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        exit(1)

    print("Obtaining session, spikes, and LFP data.", flush=True)
    session_obj = ps(session_id, df, output_dir)

    # model hyperparameters
    input_size = len(session_obj.units)
    hidden_size = 50 
    num_layers = 1
    seqlength = 750
    num_epochs = 15
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
    criterion  = torch.nn.MSELoss()
    batch_size = 32
    bin_size=0.004

    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.active_times[1])
    spikes_obj.truncate(seqlength)
    spikes_obj.minmax()

    lfp_obj = pre_process_lfp(session_id, session_obj.channels, session_obj.active_times[0],
                                session_obj.active_times[1], output_dir) 
    lfp_obj.filter_lfp(take_power = True)
    lfp_obj.downsample_lfp(5)
    lfp_obj.truncate(seqlength)
    lfp_obj.align_lfp(spikes_obj.spkMat.shape[0])
    num_channels = len(lfp_obj.channels)

    print("Training", flush=True)
    lossesTrain, lossesTest = (np.zeros((len(lfp_obj.channels), num_epochs, len(bands)+1)) for _ in range(2))
    R2_train, R2_test       = (np.zeros((len(lfp_obj.channels), len(bands)+1)) for _ in range(2))
    all_models = {}
    for bandi in range(len(bands)+1):
        X_train, X_test, y_train, y_test  = chunk_and_reshape(spikes_obj.spkMat, lfp_obj.lfpMat[:,:,bandi], 
                                                        seqlength, test_size=0.2, random_state=42)
        for chani in range(len(lfp_obj.channels)):
            train_dataloader, test_dataloader = get_data_loaders(X_train, X_test, y_train[:,:,chani], y_test[:,:,chani], batch_size)          
            model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
            train_loss, test_loss = model.train(train_dataloader, test_dataloader, num_epochs)
            lossesTrain[chani,:, bandi] = train_loss
            lossesTest[chani,:, bandi] = test_loss
            yHat = model.evaluate(test_dataloader.dataset.tensors[0])
            R2_test[chani, bandi] = r2_score(test_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), np.array(yHat.to('cpu')))
            yHat = model.evaluate(train_dataloader.dataset.tensors[0])
            R2_train[chani, bandi] = r2_score(train_dataloader.dataset.tensors[1].cpu().numpy().reshape(-1,1), np.array(yHat.to('cpu')))
            all_models[chani, bandi] = model
            
    yHat_active = np.zeros((lfp_obj.lfpMat.shape[0], lfp_obj.lfpMat.shape[1], len(bands)+1))
    for bandi in range(len(bands)+1):
        for chani in range(len(lfp_obj.channels)):
            yHat_active[:,chani,bandi] = np.array(all_models[chani, bandi].evaluate(torch.tensor(spikes_obj.spkMat).float().to(device)).to('cpu'))
            
    print("Training completed", flush=True)

    print("Plotting results")
    output_dir_plots = Path(output_dir / 'spikes2lfp') # '/plots' is created in the plots.py
    os.makedirs(output_dir_plots, exist_ok=True)
    plot_r2(R2_test, lfp_obj.channels, bands, 
            show_plot=False, save_fig=True, output_dir=output_dir_plots, session_id=session_id, fig_name='r2_scores.png')
    for bandi in range(len(bands)+1):
        plot_all_channel_loss(lfp_obj.channels, bands, bandi, lossesTest, 
                            show_plot=False, save_fig=True, output_dir=output_dir_plots, session_id=session_id, fig_name=f'all_channel_loss_band{bandi}.png')
    for chani in range(len(lfp_obj.channels)):
        plot_lfp_prediction(lfp_obj.lfpMat, yHat_active, chani, bands, start_time = 20, end_time = 30,
                            show_plot=False, save_fig=True, output_dir=output_dir_plots, session_id=session_id, fig_name=f'lfp_prediction_band{bandi}.png')
        plot_abs_error_change(lfp_obj.lfpMat, yHat_active, session_obj, spikes_obj.timestamps, chani, bands,
                              show_plot=False, save_fig=True, output_dir=output_dir_plots, session_id=session_id, fig_name=f'abs_error_change_chan{chani}.png')
        plot_psd(lfp_obj.lfpMat, yHat_active, chani,
                 show_plot=False, save_fig=True, output_dir=output_dir_plots, session_id=session_id, fig_name=f'psd_chan{chani}.png')
        
    output_dir_models = Path(output_dir / 'spikes2lfp' / 'models' / str(session_id))
    os.makedirs(output_dir_models, exist_ok=True)    
    print("Models are saved in: ", output_dir_models, flush=True)
    for idx in range(len(all_models.keys())):
        chan = list(all_models.keys())[idx][0]
        band = list(all_models.keys())[idx][1]
        filename = Path(output_dir_models / f'lstm_model_chan{chan}_band{band}.pt')
        torch.save(all_models[chan, band].model.state_dict(), filename)
    
    print("Calculating and saving attribution scores")
    attr_dur = 720
    X_attr = torch.tensor(spikes_obj.spkMat[:int(attr_dur/bin_size),:]).float().to(device)
    num_trials = int(int(attr_dur/bin_size)/seqlength)
    X_attr = X_attr.reshape(num_trials, seqlength, X_attr.shape[1])
    X_attr_flat = torch.tensor(spikes_obj.spkMat[:int(attr_dur/bin_size),:]).float().to('cpu')
    mean_attribution = np.zeros((num_channels, len(bands)+1 , spikes_obj.spkMat.shape[1]))
    output_dir_attrs = Path(output_dir / 'spikes2lfp' / 'attrs' / str(session_id))
    os.makedirs(output_dir_attrs, exist_ok=True)

    for bandi in range(len(bands)+1):
        for chani in range(num_channels):
            model = all_models[chani, bandi]
            ig = IntegratedGradient(model.model.train().to(device), method='last time point', seqlength=seqlength)        
            attrs = ig.run(X_attr, baselines = 0, n_batch=40, n_steps = 50).cpu()
            if device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            # attrs = attrs / (X_attr_flat + 1e-10)
            attrs = np.array(attrs)
            filename = Path(output_dir_attrs / f'attribution_scores_chan{chani}_band{bandi}.npz')
            attrs_sparse = coo_matrix(attrs)
            save_npz(filename, attrs_sparse, compressed=True)
            mean_attribution[chani, bandi, :] = np.mean(attrs, axis=0)
            
    filename = Path(output_dir_attrs / f'attribution_scores_mean.npy')
    np.save(filename, mean_attribution)
    print("Attribution scores are saved in: ", output_dir_attrs)
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