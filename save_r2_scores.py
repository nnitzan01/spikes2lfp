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
    spikes_obj_spont = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj_spont.getSpkMat(session_obj.spontaneous_times[0], session_obj.spontaneous_times[1])
    spikes_obj_spont.truncate(seqlength)
    spikes_obj_spont.minmax()
    
    spikes_obj_passive = pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=bin_size, sigma=3)
    spikes_obj_passive.getSpkMat(session_obj.passive_times[0], session_obj.passive_times[1])
    spikes_obj_passive.truncate(seqlength)
    spikes_obj_passive.minmax()

    print("LFP obj.", flush=True)
    lfp_obj_spont = pre_process_lfp(session_id, session_obj.channels, session_obj.spontaneous_times[0],
                                session_obj.spontaneous_times[1], output_dir) 
    lfp_obj_spont.filter_lfp(take_power = True)
    lfp_obj_spont.downsample_lfp(5)
    lfp_obj_spont.truncate(seqlength)
    lfp_obj_spont.align_lfp(spikes_obj_spont.spkMat.shape[0])

    lfp_obj_passive = pre_process_lfp(session_id, session_obj.channels, session_obj.passive_times[0],
                                session_obj.passive_times[1], output_dir) 
    lfp_obj_passive.filter_lfp(take_power = True)
    lfp_obj_passive.downsample_lfp(5)
    lfp_obj_passive.truncate(seqlength)
    lfp_obj_passive.align_lfp(spikes_obj_passive.spkMat.shape[0])

    print("Obtain model weights", flush=True)
    R2_test_spont   = np.zeros((len(lfp_obj_spont.channels), len(bands)+1))
    R2_test_passive = np.zeros((len(lfp_obj_passive.channels), len(bands)+1))
    all_models = {}
    for bandi in range(len(bands)+1):
        X_train_spont, X_test_spont, y_train_spont, y_test_spont = chunk_and_reshape(spikes_obj_spont.spkMat, lfp_obj_spont.lfpMat[:,:,bandi], 
                                                        seqlength, test_size=0.2, random_state=42)
        X_train_passive, X_test_passive, y_train_passive, y_test_passive = chunk_and_reshape(spikes_obj_passive.spkMat, lfp_obj_passive.lfpMat[:,:,bandi],
                                                        seqlength, test_size=0.2, random_state=42)
        
        for chani in range(len(lfp_obj_spont.channels)):
            _, test_dataloader_spont    = get_data_loaders(X_train_spont, X_test_spont, y_train_spont[:,:,chani], y_test_spont[:,:,chani], batch_size)
            _ , test_dataloader_passive = get_data_loaders(X_train_passive, X_test_passive, y_train_passive[:,:,chani], y_test_passive[:,:,chani], batch_size)
            model = models.process_model(models.LSTMnet(input_size, hidden_size, num_layers,seqlength), criterion, device)
            model_weights_dir = Path(output_dir / 'spikes2lfp' / 'models' / str(session_id) / f'lstm_model_chan{chani}_band{bandi}.pt')
            model.model.load_state_dict(torch.load(model_weights_dir))
            model.model.to(device)
            yHat = model.evaluate(test_dataloader_spont.dataset.tensors[0])
            R2_test_spont[chani, bandi] = r2_score(test_dataloader_spont.dataset.tensors[1].cpu().numpy().reshape(-1,1), np.array(yHat.to('cpu')))
            yHat = model.evaluate(test_dataloader_passive.dataset.tensors[0])
            R2_test_passive[chani, bandi] = r2_score(test_dataloader_passive.dataset.tensors[1].cpu().numpy().reshape(-1,1), np.array(yHat.to('cpu')))
            all_models[chani, bandi] = model

    # save the R2 scores
    variables_dir = Path(output_dir / 'spikes2lfp' / 'variables' / str(session_id))
    os.makedirs(variables_dir, exist_ok=True)

    r2_scores_spont_path = Path(variables_dir / 'r2_scores_spont.npy')
    np.save(r2_scores_spont_path, R2_test_spont)

    r2_scores_passive_path = Path(variables_dir / 'r2_scores_passive.npy')
    np.save(r2_scores_passive_path, R2_test_passive)
    print("R2 scores saved to:", r2_scores_spont_path, r2_scores_passive_path)
    
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