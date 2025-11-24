import torch
import numpy as np
import process_probe as pp
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class pre_process_spikes:
    def __init__(self, units, spike_times, seqlength = 750, bin_size=0.004, sigma=1):
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
        for i, unit in enumerate(self.units.index):
            self.spkMat[:, i] = np.histogram(self.spike_times[unit], bins=bins)[0].tolist()

    def truncate(self, seqlength):
        num_trials = int(self.spkMat.shape[0] / seqlength)
        self.spkMat = self.spkMat[:num_trials * seqlength, :]
        self.timestamps = self.timestamps[:num_trials * seqlength]

    def convolve_with_gaussian(self):
        self.spkMat = gaussian_filter1d(self.spkMat, self.sigma, axis=0)/self.bin_size

    def zscore(self):
        self.spkMat = (self.spkMat - self.spkMat.mean(axis=0))/self.spkMat.std(axis=0)
        
    def minmax(self):
        scaler = MinMaxScaler()
        self.spkMat = scaler.fit_transform(self.spkMat.T).T
        
class pre_process_lfp:
    def __init__(self, session_id, channels, start_time, stop_time, output_dir):
        chans, lfp, timestamps = pp.load_lfp(output_dir, session_id, channels, start_time, stop_time)
        self.channels = chans
        self.data = lfp
        self.timestamps = timestamps
        self.lfpMat = []
        self.sampling_rate = 1250

    def filter_lfp(self, bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 400)], take_power = False):
        self.lfpMat = np.zeros((self.data.shape[0], self.data.shape[1], len(bands)+1))
        # first entry is the raw signal
        self.lfpMat[:, :, 0] = zscore(self.data, axis=0)
        for i, band in enumerate(bands):
            low  = bands[i][0] / (self.sampling_rate / 2)
            high = bands[i][1] / (self.sampling_rate / 2)
            b, a = butter(3, [low, high], btype='bandpass')
            # Apply the filter
            filt  = filtfilt(b, a, self.data.astype(np.float64), axis=0)
            if take_power:
                power = np.abs(hilbert(filt, axis=0))**2
                self.lfpMat[:, :, i+1] = zscore(power, axis=0)
            else:
                self.lfpMat[:, :, i+1] = zscore(filt, axis=0)
        del self.data
    
    def downsample_lfp(self, factor):
        self.lfpMat = self.lfpMat[::factor,:,:]   
        
    def truncate(self, seqlength):
        num_trials = int(self.lfpMat.shape[0] / seqlength)
        self.lfpMat = self.lfpMat[:num_trials * seqlength, :, :]
    
    def align_lfp(self, length):
        if self.lfpMat.shape[0] > length:
            self.lfpMat = self.lfpMat[:length,:,:]
            
def chunk_and_reshape(spikes, lfp, seqlength, test_size=0.2, prediction_lag = 0, random_state=42):
    """
    Chunks the spike and LFP data into equal segments, reshapes them,
    and splits them into training and testing sets.

    Args:
        spikes: NumPy array of shape (num_timepoints, num_neurons) representing spiking data.
        lfp: NumPy array of shape (num_timepoints, num_lfp_channels) representing LFP data.
             If LFP is single channel, should be (num_timepoints, 1).
        seqlength: The length of each chunk (window size).
        test_size: The proportion of data to use for the test set.
        random_state: The random state for the train_test_split function.

    Returns:
        X_train, X_test, y_train, y_test: NumPy arrays representing the training and testing sets
                                         for the spikes (X) and LFP (y) data.
    """

    if len(lfp.shape) == 1:
        lfp = lfp[:, np.newaxis]
    
    # shift the LFP if prediction_lag is not 0
    if prediction_lag > 0:
        lfp_shifted = np.roll(lfp, -prediction_lag, axis=0)
        lfp_shifted = lfp_shifted[:-prediction_lag, :]
        spikes = spikes[:-prediction_lag, :]
    else:
        lfp_shifted = lfp
    # The last time points can not be predicted and should be removed

    num_trials = int(lfp_shifted.shape[0] / seqlength)

    # Truncate spikes and LFP data to be multiples of seqlength
    if spikes.shape[0] % seqlength != 0:
        spikes = spikes[:-(spikes.shape[0] % seqlength), :]
        lfp_shifted = lfp_shifted[:-(lfp_shifted.shape[0] % seqlength), :]

    # Reshape the data into trials
    X_reshaped = np.reshape(spikes, (num_trials, seqlength, spikes.shape[1]))
    lfp_reshaped = np.reshape(lfp_shifted, (num_trials, seqlength, lfp_shifted.shape[1]))

    # Split into training and testing sets at the trial level
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, lfp_reshaped, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def get_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    
    # # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

# def chunk_and_reshape(spikes, lfp, seqlength, test_size=0.2, random_state=42):
#     """
#     Chunks the spike and LFP data into equal segments, reshapes them,
#     and splits them into training and testing sets.

#     Args:
#         spikes: NumPy array of shape (num_timepoints, num_neurons) representing spiking data.
#         lfp: NumPy array of shape (num_timepoints, num_lfp_channels) representing LFP data.
#              If LFP is single channel, should be (num_timepoints, 1).
#         seqlength: The length of each chunk (window size).
#         test_size: The proportion of data to use for the test set.
#         random_state: The random state for the train_test_split function.

#     Returns:
#         X_train, X_test, y_train, y_test: NumPy arrays representing the training and testing sets
#                                          for the spikes (X) and LFP (y) data.
#     """

#     if len(lfp.shape) == 1:
#         lfp = lfp[:, np.newaxis]
    
#     num_trials = int(lfp.shape[0] / seqlength)

#     # Truncate spikes and LFP data to be multiples of seqlength
#     if spikes.shape[0] % seqlength != 0:
#         spikes = spikes[:-(spikes.shape[0] % seqlength), :]
#         lfp = lfp[:-(lfp.shape[0] % seqlength), :]

#     # Reshape the data into trials
#     X_reshaped = np.reshape(spikes, (num_trials, seqlength, spikes.shape[1]))
#     lfp_reshaped = np.reshape(lfp, (num_trials, seqlength, lfp.shape[1]))

#     # Split into training and testing sets at the trial level
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_reshaped, lfp_reshaped, test_size=test_size, random_state=random_state
#     )

#     return X_train, X_test, y_train, y_test


# def chunk_and_reshape_sliding_window(spikes, lfp, seqlength, overlap_factor=0.5, test_size=0.2, random_state=42):
#     """
#     Chunks the spike and LFP data into overlapping segments for training,
#     reshapes them, and splits them into training and testing sets.
#     Testing data is not expanded.

#     Args:
#         spikes: NumPy array of shape (num_timepoints, num_neurons) representing spiking data.
#         lfp: NumPy array of shape (num_timepoints, num_lfp_channels) representing LFP data.
#              If LFP is single channel, should be (num_timepoints, 1).
#         seqlength: The length of each chunk (window size).
#         overlap_factor: The proportion of overlap between consecutive chunks (0 to <1).
#         test_size: The proportion of data to use for the test set.
#         random_state: The random state for the train_test_split function.

#     Returns:
#         X_train, X_test, y_train, y_test: NumPy arrays representing the training and testing sets
#                                          for the spikes (X) and LFP (y) data.
#     """
#     if len(lfp.shape) == 1:
#         lfp = lfp[:, np.newaxis]

#     # 1. Initial Reshape into Trials
#     num_trials = int(lfp.shape[0] / seqlength)
#     # Truncate spikes and LFP data to be multiples of seqlength
#     if spikes.shape[0] % seqlength != 0:
#         spikes = spikes[:-(spikes.shape[0] % seqlength), :]
#         lfp = lfp[:-(lfp.shape[0] % seqlength), :]
    
#     X_reshaped = np.reshape(spikes, (num_trials, seqlength, spikes.shape[1]))
#     lfp_reshaped = np.reshape(lfp, (num_trials, seqlength, lfp.shape[1]))

#     # 2. Train/Test Split
#     X_train_trials, X_test, y_train_trials, y_test = train_test_split(
#         X_reshaped, lfp_reshaped, test_size=test_size, random_state=random_state
#     )

#     # 3. Reshape Training Data for Expansion
#     X_train = X_train_trials.reshape(-1, X_train_trials.shape[-1])
#     y_train = y_train_trials.reshape(-1, y_train_trials.shape[-1])

#     # 4. Sliding Window Expansion (Training Data Only)
#     hop_length = int(seqlength * (1 - overlap_factor))
#     X_expanded = []
#     y_expanded = []
    
#     # Get the number of timepoints
#     num_timepoints = y_train.shape[0]
#     # Iterate through the number of time points, using a hop_length
#     for j in range(0, num_timepoints - seqlength + 1, hop_length):
#         start_idx = j 
#         end_idx = j + seqlength
#         X_expanded.append(X_train[start_idx:end_idx, :])
#         y_expanded.append(y_train[start_idx:end_idx, :])

#     X_train = np.array(X_expanded)
#     y_train = np.array(y_expanded)

#     # 6. Reshape Expanded Training Data into Trials
#     num_trials_expanded = X_train.shape[0]
#     X_train = np.reshape(X_train, (num_trials_expanded, seqlength, X_train.shape[2]))
#     y_train = np.reshape(y_train, (num_trials_expanded, seqlength, y_train.shape[2]))
    
#     # 7. Return
#     return X_train, X_test, y_train, y_test