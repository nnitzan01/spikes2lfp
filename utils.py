import blosc2
import os
import torch
import lstm_module as lstm

def bl2_save(array, path):
    # current ver. assumes parent directory exists
    blosc2.save_tensor(array, path, mode='w')

def bl2_load(path):
    if not os.path.exists(path):
        print('File does not exist')
        return None
    else:
        return blosc2.load_tensor(path, mode='r')
    

def save_models(models, output_dir, session_id):
    for i, model in enumerate(models):
        os.makedirs(output_dir / 'models' / str(session_id), exist_ok=True)
        for idx in range(len(models.keys())):
            chan = list(models.keys())[idx][0]
            band = list(models.keys())[idx][1]
            filename = output_dir / 'models' / str(session_id) / f'lstm_model_chan{chan}_band{band}.pt'
            torch.save(models[chan, band].model.state_dict(), filename)

def load_models(output_dir, session_id, num_channels, args):
    models = {}
    bands, input_size, hidden_size, num_layers, seqlength, criterion, device = args
    for chani in range(num_channels):
        for bandi in range(len(bands)+1):
            model = lstm.process_model(lstm.LSTMnet(input_size, hidden_size, num_layers), seqlength, criterion, device)
            filename = output_dir / 'models' / str(session_id) / f'lstm_model_chan{chani}_band{bandi}.pt'
            # print(filename)
            # print(os.path.exists(filename))
            model.model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
            models[chani, bandi] = model
    return models