import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess_data import chunk_and_reshape, pre_process_spikes


def test_split_matches_previous_array_based_split():
    rng = np.random.default_rng(0)
    spikes, lfp = rng.random((7500, 5)), rng.random((7500, 3))
    X_train, X_test, y_train, y_test, train_idx, test_idx = chunk_and_reshape(
        spikes, lfp, 750, return_indices=True)
    old = train_test_split(spikes.reshape(10, 750, 5), lfp.reshape(10, 750, 3), test_size=0.2, random_state=42)
    for new, ref in zip((X_train, X_test, y_train, y_test), old):
        np.testing.assert_array_equal(new, ref)
    np.testing.assert_array_equal(X_test[0], spikes[test_idx[0] * 750:(test_idx[0] + 1) * 750])


def test_truncates_to_common_length_and_applies_lag():
    spikes = np.arange(1600)[:, None].astype(float)
    lfp = np.arange(1700)[:, None].astype(float)          # longer than spikes
    X_tr, X_te, y_tr, y_te = chunk_and_reshape(spikes, lfp, 750, prediction_lag=3, test_size=0.5)
    X, y = np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te])
    assert X.shape == (2, 750, 1)
    np.testing.assert_array_equal(y - X, 3)               # spikes at t paired with LFP at t + 3


def test_spike_timestamps_are_left_bin_edges():
    units = pd.DataFrame(index=[1, 2])
    sp = pre_process_spikes(units, {1: np.array([0.001, 0.0055]), 2: np.array([3.9999])}, bin_size=0.004)
    sp.getSpkMat(0, 4)
    assert sp.spkMat.shape == (1000, 2)
    np.testing.assert_allclose(sp.timestamps[:3], [0, 0.004, 0.008])
    assert sp.spkMat[0, 0] == 1 and sp.spkMat[1, 0] == 1 and sp.spkMat[-1, 1] == 1


def test_scalers_use_training_statistics_only():
    from preprocess_data import SpikeScaler, Standardizer
    train = np.array([[0, 2, 0], [1, 4, 0], [2, 0, 0]], dtype=float)
    test = np.array([[4, 2, 1]], dtype=float)
    scaler = SpikeScaler().fit(train)
    np.testing.assert_allclose(scaler.transform(test), [[2.0, 0.5, 0.0]])   # not clipped; silent unit -> 0
    std = Standardizer().fit(np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(std.transform(np.array([[5.0]])), [[3.0]])
    restored = SpikeScaler.from_state(scaler.state())
    np.testing.assert_allclose(restored.transform(test), scaler.transform(test))


def test_decimation_removes_aliasing_and_keeps_alignment():
    from preprocess_data import _decimate
    fs, q = 1250, 5
    t = np.arange(0, 4, 1 / fs)
    slow, fast = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 240 * t)   # 240 Hz aliases to 10 Hz at 250 Hz
    y = _decimate((slow + fast)[:, None], q)[:, 0]
    assert len(y) == len(t[::q])
    core = slice(50, -50)                                                    # ignore filter edges
    np.testing.assert_allclose(y[core], slow[::q][core], atol=0.02)          # fast part removed, slow kept in place
    assert np.abs((slow + fast)[::q][core] - slow[::q][core]).max() > 0.5    # plain slicing aliases


def test_take_chunks():
    from preprocess_data import take_chunks
    arr = np.arange(10)[:, None]
    np.testing.assert_array_equal(take_chunks(arr, [2, 0], seqlength=3)[:, :, 0], [[6, 7, 8], [0, 1, 2]])
