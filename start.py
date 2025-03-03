
import torch
import utils
from pathlib import Path
import preprocess_data as ppd
from process_session import session as ss
import process_probe as pp
import process_attribution as pa
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Session obj contains stimulus, channel, and unit information
session_id = 1044385384
session_obj = ss(session_id)
# Spikes obj contains Spike matrix
spikes_obj = ppd.pre_process_spikes(session_obj.units, session_obj.spike_times, bin_size=0.004, sigma=3)
spikes_obj.getSpkMat(session_obj.active_times[0], session_obj.active_times[1])
spikes_obj.convolve_with_gaussian()
spikes_obj.zscore()
# Probe obj contains LFP data
# Getting LFP just for active trials
probe_obj = pp.probe(session_obj, session_obj.active_times[0], session_obj.active_times[1])
lfp_obj = ppd.pre_process_lfp(probe_obj.lfp, 1250)
lfp_obj.filter_lfp(probe_obj.bands)
# Training hyperparameters
input_size = spikes_obj.spkMat.shape[1]
hidden_size = 50
num_layers = 1
seqlength = 750
num_epochs = 15
X_train, y_train, X_test, y_test = ppd.generate_training_data(lfp_obj, spikes_obj, seqlength)
# np.savez_compressed('F:/temp_compare/data_modular.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
models, lossesAll = utils.train_models(probe_obj, input_size, hidden_size, num_layers, seqlength, device, num_epochs, X_train, y_train)
output_dir = Path('E:/vbn_s3_cache')
# Saving model weights
utils.save_models(models, output_dir, session_id)
# Loading model weights
output_dir = Path(r"Z:\Buzsakilabspace\LabShare\NoamNitzan\Open_Access\Allen_2022")
args = [input_size, hidden_size, num_layers, seqlength, device]
models = utils.load_models(output_dir, session_id, probe_obj, args)
# Calculating attribution scores
output_dir = Path(r'F:\vbn_s3_cache')
dur = 720
bin_size = 0.004
pa.divide_task_for_attr(models, session_id, output_dir, spikes_obj, bin_size, probe_obj, dur)