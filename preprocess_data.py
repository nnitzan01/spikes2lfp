import torch
import numpy as np
import process_probe as pp
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config


class pre_process_spikes:
    """Binned spike counts (time bins x units) for a time window."""
    def __init__(self, units, spike_times, bin_size=config.BIN_SIZE, sigma=1):
        self.units = units
        self.spike_times = spike_times
        self.bin_size = bin_size
        self.sigma = sigma          # Gaussian width in bins, used only by convolve_with_gaussian()
        self.spkMat = []
        self.timestamps = []

    def getSpkMat(self, start, stop):
        """Count spikes in bins of bin_size from start to stop (s); timestamps are the left bin edges."""
        bin_count = int(np.round((stop - start) / self.bin_size))
        bins = start + np.arange(bin_count + 1) * self.bin_size
        self.timestamps = bins[:-1]
        self.spkMat = np.zeros((bin_count, len(self.units)), dtype=np.float32)
        for i, unit in enumerate(self.units.index):
            self.spkMat[:, i] = np.histogram(self.spike_times[unit], bins=bins)[0]

    def truncate(self, seqlength):
        """Drop trailing bins so the length is a whole number of sequences."""
        num_trials = int(self.spkMat.shape[0] / seqlength)
        self.spkMat = self.spkMat[:num_trials * seqlength, :]
        self.timestamps = self.timestamps[:num_trials * seqlength]

    def convolve_with_gaussian(self):
        self.spkMat = gaussian_filter1d(self.spkMat, self.sigma, axis=0)/self.bin_size

    def zscore(self):
        self.spkMat = (self.spkMat - self.spkMat.mean(axis=0))/self.spkMat.std(axis=0)

    def minmax(self):
        """
        Min-max scale each time bin across units (note the transposes: it is not per unit).

        With 4 ms bins nearly every count is 0 or 1 and the minimum across units is 0, so in
        practice each bin is divided by its largest count: the input is close to a binary spike
        indicator, and multi-spike bursts are capped. This does not remove firing-rate effects
        from attributions (a unit that fires more still has more non-zero bins); use per-spike
        attribution for rate-independent comparisons.
        """
        scaler = MinMaxScaler()
        self.spkMat = scaler.fit_transform(self.spkMat.T).T


class pre_process_lfp:
    """VISp LFP for a time window, band-filtered into model targets (time x channels x targets)."""
    def __init__(self, session_id, channels, start_time, stop_time, output_dir):
        chans, lfp, timestamps = pp.load_lfp(output_dir, session_id, channels, start_time, stop_time)
        self.channels = chans
        self.data = lfp
        self.timestamps = timestamps
        self.lfpMat = []
        self.sampling_rate = config.LFP_FS

    def filter_lfp(self, bands=config.BANDS, take_power=False):
        """
        Build lfpMat[:, :, 0] = z-scored broadband LFP and lfpMat[:, :, i] = z-scored band-passed
        LFP (or its Hilbert power envelope if take_power) for bands[i-1]. Frees self.data.
        """
        self.lfpMat = np.zeros((self.data.shape[0], self.data.shape[1], len(bands)+1))
        self.lfpMat[:, :, 0] = zscore(self.data, axis=0)
        nyquist = self.sampling_rate / 2
        for i, (low_hz, high_hz) in enumerate(bands):
            b, a = butter(3, [low_hz / nyquist, high_hz / nyquist], btype='bandpass')
            filt = filtfilt(b, a, self.data.astype(np.float64), axis=0)
            if take_power:
                power = np.abs(hilbert(filt, axis=0))**2
                self.lfpMat[:, :, i+1] = zscore(power, axis=0)
            else:
                self.lfpMat[:, :, i+1] = zscore(filt, axis=0)
        del self.data

    def downsample_lfp(self, factor):
        """
        Keep every factor-th sample (1250 Hz / 5 = 250 Hz, one sample per spike bin).

        No anti-alias filter is applied: broadband content above the new Nyquist (125 Hz) folds
        back into the target. Kept as-is so existing models stay comparable.
        """
        self.lfpMat = self.lfpMat[::factor,:,:]

    def truncate(self, seqlength):
        num_trials = int(self.lfpMat.shape[0] / seqlength)
        self.lfpMat = self.lfpMat[:num_trials * seqlength, :, :]

    def align_lfp(self, length):
        """Trim the LFP to `length` samples (the spike matrix length)."""
        if self.lfpMat.shape[0] < length:
            raise ValueError(f'LFP has {self.lfpMat.shape[0]} samples but spikes have {length} bins')
        self.lfpMat = self.lfpMat[:length,:,:]


def chunk_and_reshape(spikes, lfp, seqlength, test_size=config.TEST_SIZE, prediction_lag=0,
                      random_state=config.SPLIT_SEED, return_indices=False):
    """
    Chunks the spike and LFP data into equal segments, reshapes them,
    and splits them into training and testing sets at the chunk ("trial") level.

    Args:
        spikes: NumPy array of shape (num_timepoints, num_neurons) representing spiking data.
        lfp: NumPy array of shape (num_timepoints, num_lfp_channels) representing LFP data.
             If LFP is single channel, it may be 1D.
        seqlength: The length of each chunk (window size).
        test_size: The proportion of data to use for the test set.
        prediction_lag: predict the LFP this many bins after the spikes (0 = same bin).
        random_state: The random state for the train_test_split function.
        return_indices: also return the chunk indices of the training and test sets.

    Returns:
        X_train, X_test, y_train, y_test (, train_idx, test_idx)
        X: (chunks, seqlength, neurons), y: (chunks, seqlength, channels). Chunk i covers
        bins i*seqlength ... (i+1)*seqlength - 1 of the (lag-shifted) input.
    """
    if len(lfp.shape) == 1:
        lfp = lfp[:, np.newaxis]

    # pair spikes at t with the LFP at t + prediction_lag; the last lag bins have no target
    if prediction_lag > 0:
        lfp = lfp[prediction_lag:, :]
        spikes = spikes[:-prediction_lag, :]

    # truncate both to a common whole number of chunks
    length = min(spikes.shape[0], lfp.shape[0])
    num_trials = length // seqlength
    spikes = spikes[:num_trials * seqlength]
    lfp = lfp[:num_trials * seqlength]

    X_reshaped = np.reshape(spikes, (num_trials, seqlength, spikes.shape[1]))
    lfp_reshaped = np.reshape(lfp, (num_trials, seqlength, lfp.shape[1]))

    # splitting the chunk indices gives the same split as splitting the arrays themselves
    train_idx, test_idx = train_test_split(np.arange(num_trials), test_size=test_size,
                                           random_state=random_state)
    split = (X_reshaped[train_idx], X_reshaped[test_idx], lfp_reshaped[train_idx], lfp_reshaped[test_idx])
    return split + (train_idx, test_idx) if return_indices else split


def get_data_loaders(X_train, X_test, y_train, y_test, batch_size=config.BATCH_SIZE):
    """Wrap (chunks x seqlength x features) arrays in DataLoaders; only training data is shuffled."""
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
