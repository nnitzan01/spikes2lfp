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

def start(output_dir, session_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists('./units_info.csv'):
        print("Loading units_info.csv from the repo.")
        df = pd.read_csv('./units_info.csv')
    else:
        print("units_info.csv not found in the repo.")
        exit(1)
        
    print("Session obj.", flush=True)
    session_obj = ps(session_id, df, output_dir)
    output_dir = Path(output_dir)
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
    
    print("Spikes obj.", flush=True)
    spikes_obj = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.active_times[1])
    spikes_obj.truncate(seqlength)
    spikes_obj.minmax()

    print("LFP obj.", flush=True)
    lfp_obj = pre_process_lfp(session_id, session_obj.channels, session_obj.active_times[0],
                                session_obj.active_times[1], output_dir) 
    lfp_obj.filter_lfp(take_power = True)
    lfp_obj.downsample_lfp(5)
    lfp_obj.truncate(seqlength)
    lfp_obj.align_lfp(spikes_obj.spkMat.shape[0])
    
    bandi = 0  # broadband
    chani = 5  # first channel
    
    #load the saved mean attributions of the full model 
    path2mean_attr = Path(output_dir / 'spikes2lfp' / 'attrs' / str(session_id) / 'attribution_scores_mean.npy')
    
    mean_attr   = np.abs(np.load(path2mean_attr))
    firing_rate = session_obj.units['firing_rate'] 
    fr_idx_sorted = np.argsort(firing_rate)[::-1]  # indices of neurons sorted by firing rate
    attr_idx_sorted = np.argsort(mean_attr[chani,bandi,:])[::-1]    # indices of neurons sorted by attribution
    
    # define 3 permutation schemes: randomly excluding X neurons, excluding X neurons according to their firing rate, excluding 
    # neurons according to their attribution
    # we do this for 1 channel and the broadband model 
    
    steps = np.arange(5,100,5)
    nperm = 100
    
    R2_fr   = np.zeros((len(steps), nperm))
    R2_attr = np.zeros((len(steps), nperm))
    R2_rand = np.zeros((len(steps), nperm))
    
    
    for permi in range(nperm):
        print(f"Permutation {permi+1} / {nperm}", flush=True)
        for step in steps:
            num_neurons = int((step/100) * len(session_obj.units)) # number of neurons to exclude
            
            # train a model excluding the top num_neurons neurons according to firing rate
            fr_excluded = fr_idx_sorted[num_neurons:]
            X_train_fr, X_test_fr, y_train_fr, y_test_fr  = chunk_and_reshape(spikes_obj.spkMat[:,fr_excluded], lfp_obj.lfpMat[:,chani,bandi], 
                                                      seqlength, prediction_lag=0, test_size=0.2, random_state=42)
            train_dataloader_fr , test_dataloader_fr = get_data_loaders(X_train_fr, X_test_fr, y_train_fr, y_test_fr, batch_size)
            lstm_model_fr = models.process_model(models.LSTMnet(len(session_obj.units) - num_neurons, hidden_size, num_layers,seqlength, 
                                                 conv=True,kernel_size=5, out_channels=32 ,batchNorm=False), criterion, device)
            train_loss, test_loss = lstm_model_fr.train(train_dataloader_fr,
                                        test_dataloader_fr , num_epochs)
            yHat_fr = lstm_model_fr.evaluate(test_dataloader_fr.dataset.tensors[0]).cpu().numpy()
            R2_fr[np.where(steps==step)[0][0], permi] = r2_score(test_dataloader_fr.dataset.tensors[1].cpu().numpy().reshape(-1,1), yHat_fr)


            # do the same for attribution-based exclusion
            attr_excluded = attr_idx_sorted[num_neurons:]
            X_train_attr, X_test_attr, y_train_attr, y_test_attr  = chunk_and_reshape(spikes_obj.spkMat[:,attr_excluded], lfp_obj.lfpMat[:,chani,bandi], 
                                                      seqlength, prediction_lag=0, test_size=0.2, random_state=42)
            train_dataloader_attr, test_dataloader_attr = get_data_loaders(X_train_attr, X_test_attr, y_train_attr, y_test_attr, batch_size)
            lstm_model_attr = models.process_model(models.LSTMnet(len(session_obj.units) - num_neurons, hidden_size, num_layers,seqlength, 
                                                 conv=True,kernel_size=5, out_channels=32 ,batchNorm=False), criterion, device)
            train_loss, test_loss = lstm_model_attr.train(train_dataloader_attr,
                                        test_dataloader_attr , num_epochs)
            yHat_attr = lstm_model_attr.evaluate(test_dataloader_attr.dataset.tensors[0]).cpu().numpy()
            R2_attr[np.where(steps==step)[0][0], permi] = r2_score(test_dataloader_attr.dataset.tensors[1].cpu().numpy().reshape(-1,1), yHat_attr)

            # do the same for random exclusion
            rand_excluded = np.random.choice(len(session_obj.units), num_neurons, replace=False)
            rand_excluded = np.setdiff1d(np.arange(len(session_obj.units)), rand_excluded)
            X_train_rand, X_test_rand, y_train_rand, y_test_rand  = chunk_and_reshape(spikes_obj.spkMat[:,rand_excluded], lfp_obj.lfpMat[:,chani,bandi], 
                                                      seqlength, prediction_lag=0, test_size=0.2, random_state=42)
            train_dataloader_rand, test_dataloader_rand = get_data_loaders(X_train_rand, X_test_rand, y_train_rand, y_test_rand, batch_size)
            lstm_model_rand = models.process_model(models.LSTMnet(len(session_obj.units) - num_neurons, hidden_size, num_layers,seqlength, 
                                                 conv=True,kernel_size=5, out_channels=32 ,batchNorm=False), criterion, device)
            train_loss, test_loss = lstm_model_rand.train(train_dataloader_rand,
                                        test_dataloader_rand , num_epochs)
            yHat_rand = lstm_model_rand.evaluate(test_dataloader_rand.dataset.tensors[0]).cpu().numpy()
            R2_rand[np.where(steps==step)[0][0], permi] = r2_score(test_dataloader_rand.dataset.tensors[1].cpu().numpy().reshape(-1,1), yHat_rand)

    # save the results
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    r2_fr_path = Path(variables_dir / 'r2_scores_fr.npy')
    np.save(r2_fr_path, R2_fr)
    r2_attr_path = Path(variables_dir / 'r2_scores_attr.npy')
    np.save(r2_attr_path, R2_attr)
    r2_rand_path = Path(variables_dir / 'r2_scores_rand.npy')
    np.save(r2_rand_path, R2_rand)
    print("R2 scores saved to:", r2_fr_path, r2_attr_path, r2_rand_path)
        
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
            