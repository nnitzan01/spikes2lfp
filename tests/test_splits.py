import numpy as np
import pytest

from splits import blocked_folds, chunk_bins, describe, holdout_split


def _min_gap(a, b):
    return np.min(np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :]))


def test_holdout_is_past_train_future_test_with_buffers():
    split = holdout_split(100, test_fraction=0.2, val_fraction=0.1, buffer_chunks=2)
    assert split['test'].tolist() == list(range(80, 100))
    assert split['val'].tolist() == list(range(68, 78))
    assert split['train'].tolist() == list(range(0, 66))
    assert _min_gap(split['train'], split['val']) > 2 and _min_gap(split['val'], split['test']) > 2
    assert sorted(np.concatenate([split[k] for k in ('train', 'val', 'test', 'buffer')]).tolist()) == list(range(100))


@pytest.mark.parametrize('n_folds', [3, 5])
def test_blocked_folds_cover_each_chunk_once_as_test_and_keep_buffers(n_folds):
    folds = blocked_folds(103, n_folds=n_folds, val_fraction=0.1, buffer_chunks=2)
    tests = np.concatenate([f['test'] for f in folds])
    assert sorted(tests.tolist()) == list(range(103))
    for f in folds:
        assert np.all(np.diff(f['test']) == 1) and np.all(np.diff(f['val']) == 1)   # contiguous blocks
        assert _min_gap(f['train'], f['test']) > 2 and _min_gap(f['train'], f['val']) > 2
        assert len(np.intersect1d(f['val'], f['test'])) == 0


def test_chunk_bins_and_describe():
    np.testing.assert_array_equal(chunk_bins([0, 2], seqlength=3), [0, 1, 2, 6, 7, 8])
    d = describe(holdout_split(100), seqlength=750, bin_size=0.004)
    assert d['test']['n_chunks'] == 20 and d['test']['minutes'] == 1.0
