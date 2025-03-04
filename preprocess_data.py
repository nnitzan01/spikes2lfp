import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import process_probe as pp
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
    def __init__(self, session_id, channels, start_time, stop_time, output_dir):
        chans, lfp = pp.load_lfp(output_dir, session_id, channels, start_time, stop_time)
        self.channels = chans
        self.data = lfp
        self.lfpMat = []
        self.sampling_rate = 1250

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
        del self.data
    
    def downsample_lfp(self, factor):
        self.lfpMat = self.lfpMat[::factor,:,:]
    
    def align_lfp(self, length):
        if self.lfpMat.shape[0] > length:
            self.lfpMat = self.lfpMat[:length,:,:]