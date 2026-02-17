import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import r2_score
from typing import Union, List, Tuple, Dict
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

def bandpass_filter(data: np.ndarray, band: tuple, fs: float, order: int = 3) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter (zero-phase via filtfilt).
    """
    if band[0] >= band[1]:
        raise ValueError("lowcut must be less than highcut.")
        
    nyquist = 0.5 * fs
    low = band[0] / nyquist
    high = band[1] / nyquist
    
    # Ensure band limits are below Nyquist frequency
    if high >= 1.0:
        high = 0.99 
        
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- Main Spectral R2 Function ---

def calculate_band_r2(
    actual_lfp: np.ndarray, 
    predicted_lfp: np.ndarray, 
    bands: Union[List[Tuple[float, float]], Tuple[float, float]],
    fs: float
) -> pd.Series:
    """
    Filters actual and predicted LFP signals within specified frequency bands 
    and calculates the R2 score for each band.

    Args:
        actual_lfp: 1D NumPy array of the true LFP signal.
        predicted_lfp: 1D NumPy array of the LFP predicted by the model.
        bands: A single tuple (low, high) or a list of tuples defining the 
               frequency bands (e.g., [(4, 8), (8, 12)]).
        fs: The sampling rate of the LFP data (Hz).

    Returns:
        A pandas Series where the index is the band string (e.g., '4-8 Hz') 
        and values are the R2 scores.
    """
    if actual_lfp.shape != predicted_lfp.shape:
        raise ValueError("Actual and predicted LFP signals must have the same shape.")

    results = {}
    
    # Ensure bands is iterable, even if a single tuple is passed
    if isinstance(bands, tuple) and len(bands) == 2 and isinstance(bands[0], (int, float)):
        bands_list = [bands]
    else:
        bands_list = bands

    for low, high in bands_list:
        band_name = f'{low}-{high} Hz'
        
        try:
            # 1. Filter the actual LFP
            lfp_actual_filtered = bandpass_filter(actual_lfp, (low, high), fs)
            
            # 2. Filter the predicted LFP (CRITICAL: Use the broadband model's output as is)
            lfp_predicted_filtered = bandpass_filter(predicted_lfp, (low, high), fs)
            
            # 3. Calculate R2 score between the two filtered signals
            # This R2 measures how much of the variance in the actual rhythm is explained 
            # by the predicted rhythm's component in that same frequency range.
            r2 = r2_score(lfp_actual_filtered, lfp_predicted_filtered)
            
            results[band_name] = r2
            
        except Exception as e:
            print(f"Error processing band {band_name}: {e}")
            results[band_name] = np.nan
            
    return pd.Series(results)


def _process_frequency(fi, frex, s, wavtime, half_wave, n_conv, dataX, data_shape, num_trials, return_tf_all=True):
    """Helper function for parallel frequency processing"""
    # Create wavelet
    wavelet = np.exp(2j * np.pi * frex[fi] * wavtime) * np.exp(-wavtime**2 / (2 * s[fi]**2))
    waveletX = fft(wavelet, n_conv)
    waveletX = waveletX / np.max(waveletX)
    
    # Convolution
    as_signal = ifft(waveletX * dataX)
    
    # Flatten to 1D for proper slicing
    as_signal = as_signal.flatten()
    
    # Proper trimming to match MATLAB behavior
    # In MATLAB: as = as(half_wave+1:end-half_wave);
    as_signal = as_signal[half_wave:len(as_signal)-half_wave]
    
    # The result should have length = data_shape[0] * num_trials
    expected_length = data_shape[0] * num_trials
    if len(as_signal) != expected_length:
        # Trim or pad to match expected length
        if len(as_signal) > expected_length:
            as_signal = as_signal[:expected_length]
        else:
            # This shouldn't happen if the math is right, but just in case
            raise ValueError(f"Convolution result too short: {len(as_signal)} vs expected {expected_length}")
    
    # Reshape back to time × trials
    as_signal = as_signal.reshape(data_shape[0], num_trials)
    
    # Power
    pow_signal = np.abs(as_signal)**2
    tf_freq = np.mean(pow_signal, axis=1)
    
    # ITPC
    itpc_freq = np.abs(np.mean(np.exp(1j * np.angle(as_signal)), axis=1))
    
    # Return pow_signal only if needed
    if return_tf_all:
        return fi, tf_freq, pow_signal, itpc_freq
    else:
        return fi, tf_freq, None, itpc_freq

def wavelet_power(data, timestamps, freqs=(0.5, 220), fs=1250, num_frex=200,
                  range_cycles=(4, 12), normalization='none',
                  baseline=None, scaling='lin', mkplt=True, n_jobs=None,
                  dtype='float32', return_tf_all=True):
    """
    Computes wavelet spectrogram and ITPC for single- or multi-channel data.
    
    Parameters:
    -----------
    data : array_like
        Time × trials × channels (channels optional)
    timestamps : array_like
        Time vector
    freqs : tuple, optional
        Frequency range (default: (0.5, 220))
    fs : float, optional
        Sampling frequency (default: 1250)
    num_frex : int, optional
        Number of frequencies (default: 200)
    range_cycles : tuple, optional
        Range of cycles (default: (4, 12))
    normalization : str, optional
        Normalization method: 'none', 'z-score', 'decibel', 'maxpower' (default: 'none')
    baseline : tuple, optional
        Baseline window (default: [min(timestamps), max(timestamps)])
    scaling : str, optional
        Frequency scaling: 'lin' or 'log' (default: 'lin')
    mkplt : bool, optional
        Make plot (default: True)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores, None uses single core (default: None)
    dtype : str, optional
        Data type for arrays: 'float32' (memory efficient) or 'float64' (default: 'float32')
    return_tf_all : bool, optional
        Whether to return tf_all array. Set False to save memory (default: True)
    
    Returns:
    --------
    tf : ndarray
        freq × time × channels
    tf_all : ndarray or None
        freq × time × trials × channels (None if return_tf_all=False)
    itpc : ndarray
        freq × time × channels
    frex : ndarray
        frequencies used
    """
    
    # Handle default baseline
    if baseline is None:
        baseline = (np.min(timestamps), np.max(timestamps))
    
    # Detect channels - reshape if 2D
    if data.ndim == 2:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    
    num_ch = data.shape[2]
    num_trials = data.shape[1]
    
    # Frequency parameters
    min_freq = freqs[0]
    max_freq = freqs[1]
    
    if scaling.lower() == 'log':
        frex = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
    else:
        frex = np.linspace(min_freq, max_freq, num_frex)
    
    # Wavelet parameters
    s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[-1]), num_frex) / (2 * np.pi * frex)
    wavtime = np.arange(-2, 2 + 1/np.round(fs), 1/np.round(fs))
    half_wave = (len(wavtime) - 1) // 2
    
    # FFT parameters
    n_wave = len(wavtime)
    n_data = data.shape[0] * num_trials
    n_conv = n_wave + n_data - 1
    
    # Set numpy dtype
    np_dtype = np.float32 if dtype == 'float32' else np.float64
    
    # Initialize outputs with channel dimension
    tf = np.zeros((num_frex, data.shape[0], num_ch), dtype=np_dtype)
    tf_all = np.zeros((num_frex, data.shape[0], num_trials, num_ch), dtype=np_dtype) if return_tf_all else None
    itpc = np.zeros((num_frex, data.shape[0], num_ch), dtype=np_dtype)
    
    # Handle parallel processing
    if n_jobs is None:
        use_parallel = False
    else:
        use_parallel = True
        if n_jobs == -1:
            n_jobs = cpu_count()
    
    # Multichannel loop
    for ch in range(num_ch):
        # Concatenate trials for this channel
        alldata = data[:, :, ch].reshape(1, -1)
        dataX = fft(alldata, n_conv)
        
        if use_parallel:
            # Parallel processing
            process_func = partial(_process_frequency, 
                                 frex=frex, s=s, wavtime=wavtime, half_wave=half_wave,
                                 n_conv=n_conv, dataX=dataX, data_shape=data.shape, 
                                 num_trials=num_trials, return_tf_all=return_tf_all)
            
            with Pool(n_jobs) as pool:
                results = pool.map(process_func, range(num_frex))
            
            # Collect results
            for fi, tf_freq, pow_signal, itpc_freq in results:
                tf[fi, :, ch] = tf_freq
                if return_tf_all and pow_signal is not None:
                    tf_all[fi, :, :, ch] = pow_signal
                itpc[fi, :, ch] = itpc_freq
        else:
            # Sequential processing (original)
            for fi in range(num_frex):
                # Create wavelet
                wavelet = np.exp(2j * np.pi * frex[fi] * wavtime) * np.exp(-wavtime**2 / (2 * s[fi]**2))
                waveletX = fft(wavelet, n_conv)
                waveletX = waveletX / np.max(waveletX)
                
                # Convolution
                as_signal = ifft(waveletX * dataX)
                
                # Flatten to 1D for proper slicing
                as_signal = as_signal.flatten()
                
                # Proper trimming to match MATLAB behavior
                as_signal = as_signal[half_wave:len(as_signal)-half_wave]
                
                # Ensure correct length
                expected_length = data.shape[0] * num_trials
                if len(as_signal) > expected_length:
                    as_signal = as_signal[:expected_length]
                
                # Reshape back to time × trials
                as_signal = as_signal.reshape(data.shape[0], num_trials)
                
                # Power
                pow_signal = np.abs(as_signal)**2
                tf[fi, :, ch] = np.mean(pow_signal, axis=1)
                if return_tf_all:
                    tf_all[fi, :, :, ch] = pow_signal
                
                # ITPC
                itpc[fi, :, ch] = np.abs(np.mean(np.exp(1j * np.angle(as_signal)), axis=1))
    
    # Normalization (applied channel-wise)
    baseline_window = [baseline[0], baseline[1]]
    baseidx = [np.argmin(np.abs(timestamps - baseline_window[0])), 
               np.argmin(np.abs(timestamps - baseline_window[1]))]
    
    tf_db = np.zeros_like(tf)
    norm_tf = np.zeros_like(tf)
    
    for ch in range(num_ch):
        basepow = tf[:, baseidx[0]:baseidx[1]+1, ch]
        
        tf_db[:, :, ch] = 10 * np.log10(tf[:, :, ch] / np.mean(basepow, axis=1, keepdims=True))
        
        mu = np.mean(basepow, axis=1, keepdims=True)
        sd = np.std(basepow, axis=1, keepdims=True)
        norm_tf[:, :, ch] = (tf[:, :, ch] - mu) / sd
    
    max_p = np.max(tf)
    
    # Plotting
    if mkplt:
        ch = 0  # default channel for plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if normalization == 'none':
            im = ax.contourf(timestamps, frex, tf[:, :, ch], levels=50, cmap='jet')
            cbar = plt.colorbar(im)
            cbar.set_label('Power [µV²]')
        elif normalization == 'z-score':
            im = ax.contourf(timestamps, frex, norm_tf[:, :, ch], levels=50, cmap='jet')
            cbar = plt.colorbar(im)
            cbar.set_label('Power [z-score]')
        elif normalization == 'decibel':
            im = ax.contourf(timestamps, frex, tf_db[:, :, ch], levels=50, cmap='jet')
            cbar = plt.colorbar(im)
            cbar.set_label('Power [dB]')
        elif normalization == 'maxpower':
            im = ax.contourf(timestamps, frex, tf[:, :, ch]/max_p, levels=50, cmap='jet')
            cbar = plt.colorbar(im)
            cbar.set_label('Power [a.u.]')
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        
        if scaling.lower() == 'log':
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    return tf, tf_all, itpc, frex


# standard bandpass filetering + hilbert transform
from scipy.signal import butter, filtfilt, hilbert

def bandpass_hilbert_power(data, bands=None, fs=1250):
    """
    Bandpass filtering + Hilbert transform to compute band power.
    """
    if bands is None:
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 25), (25, 50), (50, 100)]
        
    power_data = np.zeros((len(bands), data.shape[0], data.shape[1], data.shape[2]))
    
    for band_idx, band in enumerate(bands):
        filtered = butter_bandpass(data, band, fs)
        analytic_signal = hilbert(filtered, axis=0)
        power = np.abs(analytic_signal)**2
        power_data[band_idx,:, :, :] = power
    
    return power_data

    
# Bandpass filtering function
def butter_bandpass(data, band, fs, order=3):
    """
    Designs a Butterworth bandpass filter and applies it to the data.
    Parameters:
    data : array_like
        Input signal (time × trials × channels) or (time × trials)
    band : tuple
        Frequency band (low, high)
    fs : float
        Sampling frequency
    order : int, optional
        Filter order (default: 3)
    Returns:
    filtered_data : array_like
        Filtered signal
    
    """

    filtered_data = np.zeros_like(data)
    
    nyq = 0.5 * fs
    low = band[0] / nyq
    high = band[1] / nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = min(3 * max(len(a), len(b)) , data.shape[0]-1)  # Limit padlen to avoid warning
    if data.ndim == 2:
        filtered_data = filtfilt(b, a, data, padlen=padlen, axis=0)
        return filtered_data
    elif data.ndim == 3:
        for ch in range(data.shape[2]):
            filtered_data[:, :, ch] = filtfilt(b, a, data[:, :, ch], padlen=padlen, axis=0)
        return filtered_data
    else:
        raise ValueError("Data must be 2D or 3D array.")