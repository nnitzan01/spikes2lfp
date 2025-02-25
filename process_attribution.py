import torch
from captum.attr import IntegratedGradients
import multiprocessing
from utils import bl2_save

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

def multi_save(data, filenames):
    for datum, filename in zip(data, filenames):
        bl2_save(datum, filename)

def calculate_attr(X_attr, integrated_gradients):
    # lowest level function, do the calculations and return values
    attributions = []
    for i in range(0, X_attr.shape[0],100):
        attributions.append(integrated_gradients.attribute(X_attr[i:i+100,:].contiguous(),  target=0,  n_steps=50))
    attributions = torch.cat(attributions).to(torch.float32)
    attributions = attributions.cpu().detach()
    return attributions

def multi(models_chani, X_attr, bands, filename):
    # organize X_attr for each band, and return the results
    # args needed: integrated_gradients, X_attr, metadata
    # organize the data into the following format:
    # [[integrated_gradients, X_attr, metadata], [integrated_gradients, X_attr, metadata], ...]
    data = []
    filenames = []
    for bandi in range(len(bands)+1):
        mdl = models_chani[bandi].model.train().to('cpu')
        integrated_gradients = IntegratedGradients(mdl)
        filename_bandi = f'{filename}_band{bandi}.bl2' 
        data.append([X_attr, integrated_gradients])
        filenames.append(filename_bandi)
    print(len(data))
    return None
    # with multiprocessing.Pool() as pool:
    #     results = pool.starmap(calculate_attr, data)
    # return [results, filenames]

def divide_task_for_attr(models, session_id, output_dir, spikes_obj, bin_size, num_channels, bands, dur = 720):
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
        break
        # assert len(results) == 2
        # multi_save(results[0], results[1])
        # break