import os
import torch
import blosc2
import numpy as np
from tqdm import tqdm
import multiprocessing
from captum.attr import IntegratedGradients

"""
# Structure of multiprocessing:
# 1. divide_task_for_attr()
#    For each channel, divide the task into 9 processes, one for each band
#    Saves the data returned from multi
# 2. multi()
#    For each channel, gather the data for each band and calculate the attribution scores in parallel 
#    Returns the results and the corresponding filenames in a list
# 3. calculate_attr()
#    For each channel & band, calculate the attribution scores
#    return [filename, result]
"""

def bl2_save(array, path):
    """
    save the attribution scores as .bl2 files
    
    input:
    array: torch.tensor, attribution scores
    path: str, complete path to the file to be saved
    
    note: current ver. assumes the parent dir exists
    """
    blosc2.save_tensor(array, path, mode='w')

def bl2_load(path):
    """
    save the attribution scores as .bl2 files

    input:
    path: str, complete path to the file to be loaded
    """
    if not os.path.exists(path):
        print('File does not exist')
        return None
    else:
        return blosc2.load_tensor(path)

def multi_save(data, filenames):
    """
    save multiple files
    input:
    data: list of data to be saved
    filenames: list of filenames
    """
    for datum, filename in zip(data, filenames):
        bl2_save(datum, filename)
        # to use this, change the data extension from .bl2 to .npy in multi()
        # np.save(filename, datum) 

def set_single_process():
    """
    experimental function to set the number of threads to 1
    attempt to improve performance for each vCPU
    currently the interop threads are not used
    """
    import torch
    torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

def calculate_attr(X_test, integrated_gradients):
    """
    calculate the attribution scores, used for each process in multi

    input:
    X_attr: torch.tensor, spikes data
    integrated_gradients: IntegratedGradients object, already initialized with the corresponding model

    output:
    attributions: torch.tensor, attribution scores
    """
    set_single_process()
    attributions = []
    for i in tqdm(range(0, X_test.shape[0])):
        trial = X_test[i].unsqueeze(0)
        # trial = trial.to('cpu')
        attributions.append(integrated_gradients.attribute(trial,target=0,n_steps=50).unsqueeze(0))
    attributions = torch.cat(attributions).to(torch.float32)
    attributions = attributions.cpu().detach()
    return attributions

def multi(models_chani, X_test, bands, filename):
    """
    middle function to handle the multiprocessing

    input:
    models_chani: list, list of models for each channel
    X_attr: torch.tensor, spikes data
    bands: list, frequency bands
    filename: str, filename to save the attribution scores

    output:
    results: list, list of attribution scores
    filenames: list, list of filenames
    """
    # should look like [[X_attr, integrated_gradients], [X_attr, integrated_gradients], ...]
    # the list contains 9 sublists, one for each process
    # each list will be passed to calculate_attr() automatically
    data = []
    filenames = []
    for bandi in range(len(bands)+1):
        mdl = models_chani[bandi].model.train().to('cpu')
        integrated_gradients = IntegratedGradients(mdl)
        filename_bandi = f'{filename}_band{bandi}.bl2' 
        data.append([X_test, integrated_gradients])
        filenames.append(filename_bandi)
    # starting the multiprocessing, 9 processes, one for each band
    # this might be changed because having too many processes slows down the process significantly
    with multiprocessing.Pool(processes=9) as pool:
        results = pool.starmap(calculate_attr, data)
    return [results, filenames]

def divide_task_for_attr(models, session_id, output_dir, X_test , bands):
    """
    highest level function to prepare the data for multiprocessing

    input:
    models: dict, LSTM models trained for each channel and band
    session_id: int, session id
    output_dir: str, parent dir to save the attribution scores
                will be later stuctured as output_dir/attrs/session_id/file_chanx_bandy.bl2
    spikes_obj: object, preprocessed spike data
    bin_size: float, bin size -> 4ms
    probe_obj: object, contains the channels and bands for the analysis
    dur: int, length of data to be analyzed

    output:
    none, saves the attribution scores as .bl2 files
    """
    num_channels = int(len(models.keys()) / (len(bands) + 1))
    bands_len = len(bands)+1 # including broadband
    path_name = output_dir / 'attrs' 
    if not path_name.exists():
        path_name.mkdir(parents=True)
    # for each channel, divide the task into 9 processes, one for each band
    for chani in range(num_channels):
        models_chani = [models[chani, i] for i in range(bands_len)]
        filename = output_dir / 'attrs' / f'attribution_scores_chan{chani}'
        results = multi(models_chani, X_test, bands, filename)
        assert len(results) == 2
        multi_save(results[0], results[1])
        
      