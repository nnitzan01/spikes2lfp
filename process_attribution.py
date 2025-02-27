import torch
from captum.attr import IntegratedGradients
import multiprocessing
from utils import bl2_save
from tqdm import tqdm
"""
# Structure of multiprocessing here:
# 1. divide_task_for_attr
#    For each channel, divide the task into 8 processes, one for each band
#    Saves the data returned from multi
# 2. multi
#    Organize the data for each band, and run the attribution algorithm
#    Returns the results
# 3. calculate_attr
#    Calsulates the attribution scores with the data provided
#    return [name, results]
# 4. Store the attribution scores
"""
import numpy as np
def multi_save(data, filenames):
    for datum, filename in zip(data, filenames):
        np.save(filename, datum)
        # bl2_save(datum, filename)

def set_single_process():
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

def calculate_attr(X_attr, integrated_gradients):
    set_single_process()
    attributions = []
    for i in tqdm(range(0, X_attr.shape[0], 100)):
        attributions.append(integrated_gradients.attribute(X_attr[i:i+100,:].contiguous(),target=0,n_steps=50))
    attributions = torch.cat(attributions).to(torch.float32)
    attributions = attributions.cpu().detach()
    return attributions


def multi(models_chani, X_attr, bands, filename):
    data = []
    filenames = []
    for bandi in range(len(bands)-2):
        mdl = models_chani[bandi].model.train().to('cpu')
        integrated_gradients = IntegratedGradients(mdl)
        filename_bandi = f'{filename}_band{bandi}.npy' 
        data.append([X_attr, integrated_gradients])
        filenames.append(filename_bandi)
    with multiprocessing.Pool(processes=6) as pool:
        results = pool.starmap(calculate_attr, data)
    return [results, filenames]

def divide_task_for_attr(models, session_id, output_dir, spikes_obj, bin_size, probe_obj, dur = 720):
    num_channels, bands = probe_obj.chans.shape[0], probe_obj.bands
    X_attr = spikes_obj.spkMat[:int(dur/bin_size), :] # first 720 seconds of data
    X_attr = torch.tensor(X_attr).float()
    bands_len = len(bands)+1
    path_name = output_dir / 'attrs' / str(session_id)
    if not path_name.exists():
        path_name.mkdir(parents=True)
    for chani in range(num_channels):
        models_chani = [models[chani, i] for i in range(bands_len)]
        filename = output_dir / 'attrs' / str(session_id) / f'attribution_scores_chan{chani}'
        results = multi(models_chani, X_attr, bands, filename)
        assert len(results) == 2
        multi_save(results[0], results[1])
        break

