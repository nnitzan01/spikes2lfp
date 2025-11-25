import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import r2_score
from typing import Union, List, Tuple, Dict
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


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


def wavelet_power(data, timestamps, freqs=[0.5, 220], Fs=1250, num_frex=200, range_cycles=[4, 12], normalization='none', baseline=None, scaling='lin', mkplt=1):
    """
    Computes the wavelet spectrogram and ITPC of a timeseries, translated from MATLAB.

    Args:
        data (np.ndarray): Input data, should be 2D (time x trials).
        timestamps (np.ndarray): Time stamps of the data in seconds. Must have same length as time dimension of data.
        freqs (list, optional): Min and max frequency to use. Defaults to [0.5, 220].
        Fs (int, optional): Sampling frequency. Defaults to 1250.
        num_frex (int, optional): Number of frequencies to use. Defaults to 200.
        range_cycles (list, optional): Number of cycles for wavelet for min and max frequency. Defaults to [4, 12].
        normalization (str, optional): Normalization method: 'z-score', 'decibel', 'maxpower', or 'none'. Defaults to 'none'.
        baseline (list, optional): Baseline for normalization in seconds. Defaults to the full time period.
        scaling (str, optional): Frequency scaling, 'lin' or 'log'. Defaults to 'lin'.
        mkplt (int, optional): Plotting option: 0 for none; 1 for tf; 2 for itpc; 3 for both. Defaults to 1.

    Returns:
        tuple: (tf, tf_all, itpc, frex)
            tf (np.ndarray): Raw power data (frex x time).
            tf_all (np.ndarray): Raw power data for all trials (frex x time x trials).
            itpc (np.ndarray): Inter-trial phase clustering (frex x time).
            frex (np.ndarray): Frequencies vector.
    """
    if mkplt not in [0, 1, 2, 3]:
        raise ValueError('variable mkplt must be 0, 1, 2 or 3')

    # Ensure data is 2D (time x trials)
    data = np.atleast_2d(data)
    n_time, n_trials = data.shape
    
    time = np.array(timestamps)
    if len(time) != n_time:
        raise ValueError(f"The length of timestamps ({len(time)}) must be equal to the number of time points in data ({n_time}).")

    # Frequency vector
    min_freq, max_freq = freqs[0], freqs[1]
    if scaling == 'log':
        frex = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
    else:
        frex = np.linspace(min_freq, max_freq, num_frex)

    # Wavelet parameters
    s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[-1]), num_frex) / (2 * np.pi * frex)
    step = 1 / round(Fs)
    wavtime = np.arange(-2, 2 + step, step)
    half_wave = (len(wavtime) - 1) // 2

    # FFT parameters
    nWave = len(wavtime)
    nData = n_time * n_trials
    nConv = nWave + nData - 1

    # Output arrays
    tf = np.zeros((num_frex, n_time))
    tf_all = np.zeros((num_frex, n_time, n_trials))
    itpc = np.zeros((num_frex, n_time))

    # FFT of all data concatenated
    alldata = data.flatten(order='F')  # MATLAB column-major
    dataX = fft(alldata, nConv)

    for fi, f in enumerate(frex):
        # Create wavelet
        wavelet = np.exp(2 * 1j * np.pi * f * wavtime) * np.exp(-wavtime ** 2 / (2 * s[fi] ** 2))
        waveletX = fft(wavelet, nConv)
        # Normalize wavelet to have a peak amplitude of 1 in the frequency domain.
        waveletX = waveletX / np.max(np.abs(waveletX))
        
        # Convolution
        asig = ifft(waveletX * dataX)
        asig = asig[half_wave:-half_wave]
        asig = asig.reshape((n_time, n_trials), order='F')
                
        # Power
        tf[fi, :] = np.mean(np.abs(asig) ** 2, axis=1)
        tf_all[fi, :, :] = np.abs(asig) ** 2
        
        # ITPC
        itpc[fi, :] = np.abs(np.mean(np.exp(1j * np.angle(asig)), axis=1))

    # Normalization
    if baseline is None:
        baseline = [time[0], time[-1]]
        
    baseidx = [np.argmin(np.abs(time - baseline[0])), np.argmin(np.abs(time - baseline[1]))]
    baseline_power = tf[:, baseidx[0]:baseidx[1]]
    
    mean_baseline_power = np.mean(baseline_power, axis=1, keepdims=True)
    std_baseline_power = np.std(baseline_power, axis=1, keepdims=True)
    
    # Add a small epsilon to avoid division by zero or log of zero
    tf_db = 10 * np.log10(tf / (mean_baseline_power + 1e-15))
    norm_tf = (tf - mean_baseline_power) / (std_baseline_power + 1e-15)
    maxP = np.max(tf)

    # Plotting
    if mkplt == 1 or mkplt == 3:
        plt.figure()
        if normalization == 'none':
            plot_data, cbar_label = tf, 'Power [uV^2]'
        elif normalization == 'z-score':
            plot_data, cbar_label = norm_tf, 'Power [z-score]'
        elif normalization == 'decibel':
            plot_data, cbar_label = tf_db, 'Power [dB]'
        elif normalization == 'maxpower':
            plot_data, cbar_label = tf / maxP, 'Power [a.u.]'
        else:
            raise ValueError('Please choose a valid normalization method')

        plt.pcolormesh(time, frex, plot_data, shading='auto', cmap='jet')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        c = plt.colorbar()
        c.set_label(cbar_label)

        if scaling == 'log':
            plt.yscale('log')
        plt.ylim([min_freq, max_freq])
        plt.title('Time-Frequency Power')
        plt.tight_layout()
        plt.show()
        
    if mkplt == 2 or mkplt == 3:
        plt.figure()
        plt.pcolormesh(time, frex, itpc, shading='auto', cmap='jet')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        c = plt.colorbar()
        c.set_label('ITPC')
        if scaling == 'log':
            plt.yscale('log')
        plt.ylim([min_freq, max_freq])
        plt.title('ITPC')
        plt.tight_layout()
        plt.show()

    return tf, tf_all, itpc, frex