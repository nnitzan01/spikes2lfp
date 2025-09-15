import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

def wavelet_power(data, timestamps, freqs=[0.5, 220], Fs=1250, num_frex=200, range_cycles=[4, 12], normalization='none', baseline=None, scaling='lin', mkplt=1):
    # Ensure data is 2D (time x trials)
    data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = data.T
    n_time, n_trials = data.shape
    time = np.array(timestamps)

    # Frequency vector
    min_freq, max_freq = freqs[0], freqs[1]
    if scaling == 'log':
        frex = np.logspace(np.log10(min_freq), np.log10(max_freq), num_frex)
    else:
        frex = np.linspace(min_freq, max_freq, num_frex)

    # Wavelet parameters
    s = np.logspace(np.log10(range_cycles[0]), np.log10(range_cycles[-1]), num_frex) / (2 * np.pi * frex)
    wavtime = np.arange(-2, 2, 1/round(Fs))
    half_wave = int((len(wavtime) - 1) / 2)

    # FFT parameters
    nWave = len(wavtime)
    nData = n_time * n_trials
    nConv = nWave + nData - 1

    # Output arrays
    tf = np.zeros((len(frex), n_time))
    tf_all = np.zeros((len(frex), n_time, n_trials))
    itpc = np.zeros((len(frex), n_time))

    # FFT of all data concatenated
    alldata = data.flatten(order='F')  # MATLAB column-major
    dataX = fft(alldata, nConv)

    for fi, f in enumerate(frex):
        # Create wavelet
        wavelet = np.exp(2 * 1j * np.pi * f * wavtime) * np.exp(-wavtime ** 2 / (2 * s[fi] ** 2))
        waveletX = fft(wavelet, nConv)
        waveletX = waveletX / np.max(np.abs(waveletX))
        # Convolution
        asig = ifft(waveletX * dataX)
        asig = asig[half_wave:-(half_wave)] if half_wave > 0 else asig
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
    tf_db = 10 * np.log10(tf / np.mean(baseline_power, axis=1, keepdims=True))
    norm_tf = (tf - np.mean(baseline_power, axis=1, keepdims=True)) / np.std(baseline_power, axis=1, keepdims=True)
    maxP = np.max(tf)

    # Plotting
    if mkplt == 1 or mkplt == 3:
        plt.figure()
        if normalization == 'none':
            plt.pcolormesh(time, frex, tf, shading='auto', cmap='jet')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            c = plt.colorbar()
            c.set_label('Power [uV^2]')
        elif normalization == 'z-score':
            plt.pcolormesh(time, frex, norm_tf, shading='auto', cmap='jet')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            c = plt.colorbar()
            c.set_label('Power [z-score]')
        elif normalization == 'decibel':
            plt.pcolormesh(time, frex, tf_db, shading='auto', cmap='jet')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            c = plt.colorbar()
            c.set_label('Power [dB]')
        elif normalization == 'maxpower':
            plt.pcolormesh(time, frex, tf / maxP, shading='auto', cmap='jet')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            c = plt.colorbar()
            c.set_label('Power [a.u.]')
        else:
            raise ValueError('Please choose a valid normalization method')
        if scaling == 'log':
            plt.yscale('log')
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
        plt.title('ITPC')
        plt.tight_layout()
        plt.show()

    return tf, tf_all, itpc, frex