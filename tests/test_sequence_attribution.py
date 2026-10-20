import pytest
import numpy as np
import torch

from analysis.sequence_attribution import (attribute_targets, coarsen_lags, unit_summaries, valid_ends,
                                           window_index, window_spike_counts)
from integrated_gradients import SequenceIG
from models import LSTMnet


def test_windows_and_spike_counts():
    np.testing.assert_array_equal(window_index([4, 6], 3), [[2, 3, 4], [4, 5, 6]])
    counts = np.arange(20).reshape(10, 2) % 3
    ends = np.array([2, 9])
    expected = np.stack([counts[e - 2:e + 1].sum(axis=0) for e in ends])
    np.testing.assert_array_equal(window_spike_counts(counts, ends, 3), expected)
    np.testing.assert_array_equal(valid_ends([0, 1, 2, 9, 10], 10, 3), [2, 9])


def test_attribute_targets_matches_direct_call_and_batching():
    torch.manual_seed(0)
    net = LSTMnet(4, 6, 1, 20, conv=True, kernel_size=3, out_channels=3)
    X = (torch.rand(200, 4) < 0.3).float()
    ends = np.array([30, 77, 120, 199, 45])
    ig = SequenceIG(net, n_steps=20)
    res = attribute_targets(ig, X, ends, history=20, batch_windows=2, keep_kernel=True, keep_full=True)
    direct, *_ = ig.attribute(X[torch.as_tensor(window_index(ends, 20))])
    np.testing.assert_allclose(res['full'], direct.numpy(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(res['total'], direct.sum(dim=1).numpy(), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(res['kernel_sum'], direct.sum(dim=0).numpy(), rtol=1e-5, atol=1e-6)
    assert res['total'].shape == (5, 4) and res['delta'].shape == (5,)


def test_unit_summaries_and_coarsening():
    totals = np.array([[1.0, -1.0, 0.0], [3.0, 1.0, 0.0]])
    spikes = np.array([[1, 2, 0], [1, 0, 0]])
    s = unit_summaries(totals, spikes)
    np.testing.assert_allclose(s['mean'], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(s['mean_abs'], [2.0, 1.0, 0.0])
    np.testing.assert_allclose(s['per_spike'], [2.0, 0.0, 0.0])           # (1+3)/2, (-1+1)/2, no spikes
    np.testing.assert_allclose(s['share'], [(0.5 + 0.75) / 2, (0.5 + 0.25) / 2, 0.0])
    k = np.arange(7)[:, None].astype(float)                                  # lags 0..6, oldest first
    np.testing.assert_allclose(coarsen_lags(k, 3)[:, 0], [1 + 2 + 3, 4 + 5 + 6])   # oldest lag dropped


def test_relative_completeness_error_is_robust_to_near_zero_changes():
    from analysis.sequence_attribution import relative_completeness_error
    f_b = np.zeros(5)
    f_x = np.array([1.0, 1.0, 1.0, 1e-6, 1e-6])     # two targets where the prediction barely changes
    delta = np.full(5, 1e-4)
    assert relative_completeness_error(f_x, f_b, delta) == pytest.approx(1e-4)   # not ~100 from the tiny ones
