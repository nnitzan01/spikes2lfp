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

class pre_process_training_data:
    def __init__(self, lfp_obj, spikes_obj, seqlength):
        self.lfp_obj = lfp_obj
        self.spikes_obj = spikes_obj
        self.seqlength = seqlength
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def create_training_data(self, test_size=0.2):
        lfp = self.lfp_obj.lfpMat[::5,:,:] # downsample by 5
        if lfp.shape[0] > self.spikes_obj.spkMat.shape[0]:
            lfp = lfp[:self.spikes_obj.spkMat.shape[0],:,:]
        num_trials = int(lfp.shape[0] / self.seqlength)
        X = self.spikes_obj.spkMat[:num_trials * self.seqlength, :]
        X = X.reshape(num_trials, self.seqlength, X.shape[1])

        y = lfp[:num_trials * self.seqlength, :,:]
        y = y.reshape(num_trials, self.seqlength, y.shape[1], y.shape[2])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])
        y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2], y_train.shape[3])
        y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2], y_test.shape[3]) 

        X_trainT = torch.tensor(X_train).float()
        y_trainT = torch.tensor(y_train).float()
        X_testT = torch.tensor(X_test).float()
        y_testT = torch.tensor(y_test).float()

        self.X_train = X_trainT
        self.y_train = y_trainT
        self.X_test = X_testT
        self.y_test = y_testT

    def get_training_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

def generate_training_data(lfp_obj, spikes_obj, seqlength, test_size=0.2):
    lfp = lfp_obj.lfpMat[::5,:,:] # downsample by 5
    if lfp.shape[0] > spikes_obj.spkMat.shape[0]:
        lfp = lfp[:spikes_obj.spkMat.shape[0],:,:]
    num_trials = int(lfp.shape[0] / seqlength)
    X = spikes_obj.spkMat[:num_trials * seqlength, :]
    X = X.reshape(num_trials, seqlength, X.shape[1])

    y = lfp[:num_trials * seqlength, :,:]
    y = y.reshape(num_trials, seqlength, y.shape[1], y.shape[2])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train = torch.tensor(X_train.reshape(X_train.shape[0] * X_train.shape[1], X_train.shape[2])).float()
    X_test = torch.tensor(X_test.reshape(X_test.shape[0] * X_test.shape[1], X_test.shape[2])).float()
    y_train = torch.tensor(y_train.reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2], y_train.shape[3])).float()
    y_test = torch.tensor(y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2], y_test.shape[3])).float()

    return X_train, y_train, X_test, y_test

