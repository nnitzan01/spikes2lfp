import os
import torch
import models as models
import numpy as np
from tqdm import tqdm

def save_models(models, output_dir, session_id):
    """
    Save the LSTM model weights to the output directory.

    input:
    models: dict, LSTM models, keys are (channel, band) pairs
    output_dir: str, parent folder to save the models
    session_id: int, session_id

    output:
    None

    file structure:
    output_dir/models/session_id/lstm_model_chan{channel}_band{band}.pt
    """
    for i, model in enumerate(models):
        os.makedirs(output_dir / 'models' / str(session_id), exist_ok=True)
        for idx in range(len(models.keys())):
            chan = list(models.keys())[idx][0]
            band = list(models.keys())[idx][1]
            filename = output_dir / 'models' / str(session_id) / f'lstm_model_chan{chan}_band{band}.pt'
            torch.save(models[chan, band].model.state_dict(), filename)

def load_models(output_dir, session_id, probe_obj, args):
    """
    Load .pt files from the output directory and return the LSTM models.

    input:
    output_dir: str, parent folder to load the models
    session_id: int, session_id
    probe_obj: object, contains the channels and bands for the analysis
    args: list, [input_size, hidden_size, num_layers, seqlength, device], used to reconstruct the models

    output:
    models: dict, LSTM models, keys are (channel, band) pairs
    """
    models = {}
    num_channels, bands = probe_obj.chans.shape[0], probe_obj.bands
    input_size, hidden_size, num_layers, seqlength, device = args
    criterion = torch.nn.MSELoss()
    for chani in range(num_channels):
        for bandi in range(len(bands)+1):
            model = lstm.process_model(lstm.LSTMnet(input_size, hidden_size, num_layers), seqlength, criterion, device)
            filename = output_dir / 'models' / str(session_id) / f'lstm_model_chan{chani}_band{bandi}.pt'
            model.model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
            models[chani, bandi] = model
    return models

def train_models(probe_obj, input_size, hidden_size, num_layers, seqlength, device, num_epochs, X_train, y_train):
    """
    Utility function to train LSTM models for each channel and band.
    
    input:
    probe_obj: object, contains the channels and bands for the analysis
    input_size: int, number of units
    hidden_size: int, number of hidden units
    num_layers: int, number of hidden layers
    seqlength: int, sequence length for each sample
    device: str, 'cuda' or 'cpu'
    num_epochs: int, number of epochs
    X_train: torch.tensor, (#trials*seqlength, #units)
    y_train: torch.tensor, (#trials*seqlength, #channels, #bands)

    output:
    models: dict, trained LSTM models, keys are (channel, band) pairs
    lossesAll: np.array, (#channels, num_epochs, #bands), losses for each combination
    """
    models = {}
    criterion = torch.nn.MSELoss()
    lossesAll = np.zeros((y_train.shape[1], num_epochs, len(probe_obj.bands)+1))
    for chani in tqdm(range(len(probe_obj.chans))):
        for bandi in range(len(probe_obj.bands)+1):
            model = lstm.process_model(lstm.LSTMnet(input_size, hidden_size, num_layers), seqlength, criterion, device)
            models[chani, bandi] = model
            model.model.to(device)
            # Here, if X_train are y_train are not flat, change y_train[:, chani, bandi] to y_train[:, :, chani, bandi]
            losses = model.train(X_train.to(device), y_train[:, chani, bandi].to(device), num_epochs)
            lossesAll[chani,:, bandi] = losses
    return models, lossesAll