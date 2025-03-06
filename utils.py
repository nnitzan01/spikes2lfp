import os
import torch
import lstm_module as lstm

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

def load_models(output_dir, session_id, bands, chans, args):
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
    num_channels = len(chans)
    input_size, hidden_size, num_layers, seqlength, device = args
    criterion = torch.nn.MSELoss()
    for chani in range(num_channels):
        for bandi in range(len(bands)+1):
            model = lstm.process_model(lstm.LSTMnet(input_size, hidden_size, num_layers), seqlength, criterion, device)
            filename = output_dir / 'models' / str(session_id) / f'lstm_model_chan{chani}_band{bandi}.pt'
            model.model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
            models[chani, bandi] = model
    return models