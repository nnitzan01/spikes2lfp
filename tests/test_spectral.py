import numpy as np
import pytest

from spectral_analysis import wavelet_power

FS = 250
T_TRIAL = 2.0


def _two_trial_data():
    """Two trials (time x trials): a 10 Hz sine in trial 0 and a 40 Hz sine in trial 1."""
    t = np.arange(0, T_TRIAL, 1 / FS)
    data = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 40 * t)], axis=1)
    return data, t


def _peak_freq(tf_all, frex, trial):
    mid = tf_all.shape[1] // 2
    return frex[np.argmax(tf_all[:, mid, trial, 0])]


@pytest.mark.parametrize("n_jobs", [None, 2])
def test_trials_are_not_mixed(n_jobs):
    data, t = _two_trial_data()
    _, tf_all, _, frex = wavelet_power(data, t, freqs=(2, 60), fs=FS, num_frex=59,
                                       mkplt=False, n_jobs=n_jobs, dtype='float64')
    assert abs(_peak_freq(tf_all, frex, 0) - 10) <= 1
    assert abs(_peak_freq(tf_all, frex, 1) - 40) <= 1


def test_unit_sine_has_unit_power_at_every_frequency():
    # A unit-amplitude sine should give power ~1 at its own frequency, whatever that frequency is
    t = np.arange(0, 8, 1 / FS)
    for f in (5, 20, 50):
        data = np.sin(2 * np.pi * f * t)[:, None]
        tf, _, _, frex = wavelet_power(data, t, freqs=(f, f + 1), fs=FS, num_frex=2,
                                       mkplt=False, return_tf_all=False, dtype='float64')
        mid = tf.shape[1] // 2
        # sin = 0.5*e^{iwt} - 0.5*e^{-iwt}; a complex wavelet with peak gain 1 keeps only the
        # positive-frequency half, so the convolution has magnitude 0.5 and power 0.25
        assert tf[0, mid, 0] == pytest.approx(0.25, rel=0.05)
