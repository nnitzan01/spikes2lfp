import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import tqdm
import torch

class pre_process_spikes:
    def __init__(self, units, spike_times, bin_size=0.004, sigma=1):
        self.units = units
        self.spike_times = spike_times
        self.bin_size = bin_size
        self.sigma  = sigma
        self.spkMat = []
        self.timestamps = []
        
    def getSpkMat(self, start, stop):
        dur = stop - start
        bin_count = int(np.ceil(dur/self.bin_size))
        bins = np.linspace(start,stop,num=bin_count+1)
        self.timestamps = np.linspace(start, stop, num=bin_count)
        self.spkMat = np.zeros((bin_count, len(self.units)))
        for i, unit in enumerate(tqdm.tqdm(self.units.index)):
            self.spkMat[:, i] = np.histogram(self.spike_times[unit], bins=bins)[0].tolist()

    def convolve_with_gaussian(self):
        self.spkMat = gaussian_filter1d(self.spkMat, self.sigma, axis=0)/self.bin_size

    def zscore(self):
        self.spkMat = (self.spkMat - self.spkMat.mean(axis=0))/self.spkMat.std(axis=0)
        

class pre_process_lfp:
    def __init__(self, data, sampling_rate=1250):
        self.data = data
        self.sampling_rate = sampling_rate
        self.lfpMat = []
        
    def filter_lfp(self, bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)]):
        self.lfpMat = np.zeros((self.data.shape[0], self.data.shape[1], len(bands)+1))
        # first entry is the raw signal
        self.lfpMat[:, :, 0] = zscore(self.data, axis=0)
        
        for i, band in enumerate(tqdm.tqdm(bands)):
            low  = bands[i][0] / (self.sampling_rate / 2)
            high = bands[i][1] / (self.sampling_rate / 2)
            b, a = butter(3, [low, high], btype='bandpass')
            # Apply the filter
            filt  = filtfilt(b, a, self.data.astype(np.float64), axis=0)
            power = np.abs(hilbert(filt, axis=0))**2 
            # z-score the power
            self.lfpMat[:, :, i+1] = zscore(power, axis=0)
    
    def downsample_lfp(self, factor):
        self.lfpMat = self.lfpMat[::factor,:,:]
    
    def align_lfp(self, length):
        if self.lfpMat.shape[0] > length:
            self.lfpMat = self.lfpMat[:length,:,:]

def generate_training_data(lfp_obj, spikes_obj, seqlength, test_size=0.2, make_val=False, val_size=0.5):
    lfp = lfp_obj.lfpMat[::5,:,:]
    if lfp.shape[0] > spikes_obj.spkMat.shape[0]:
        lfp = lfp[:spikes_obj.spkMat.shape[0],:,:]
    num_trials = int(lfp.shape[0] / seqlength)
    X = spikes_obj.spkMat[:num_trials * seqlength, :]
    X = X.reshape(num_trials, seqlength, X.shape[1])
    y = lfp[:num_trials * seqlength, :,:]
    y = y.reshape(num_trials, seqlength, y.shape[1], y.shape[2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = torch.Tensor(X_train).float()
    y_train = torch.Tensor(y_train).float()

    # X_train = torch.reshape(X_train, (X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
    # X_test = torch.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2]))
    # y_train = torch.reshape(y_train, (y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3]))
    # y_test = torch.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3]))
    
    if make_val:
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=42)
        X_test = torch.Tensor(X_test).float()
        y_test = torch.Tensor(y_test).float()    
        X_val = torch.Tensor(X_val).float()
        y_val = torch.Tensor(y_val).float()
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        X_test = torch.Tensor(X_test).float()
        y_test = torch.Tensor(y_test).float()
        return X_train, y_train, X_test, y_test