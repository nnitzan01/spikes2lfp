"""
Contiguous (blocked) train / validation / test splits of a recording cut into 3 s chunks.

Neighbouring chunks are strongly correlated, so random chunk-level splits overstate held-out
performance. Here every set is a contiguous block of chunks, and `buffer_chunks` chunks between
sets are left out entirely, so no train/test pair is closer than the buffer (and filter edge
effects from the continuous preprocessing stay inside the buffer).
"""
import numpy as np

import config


def holdout_split(n_chunks, test_fraction=config.TEST_FRACTION, val_fraction=config.VAL_FRACTION,
                  buffer_chunks=config.BUFFER_CHUNKS):
    """
    Train on the past, test on the future. In time order:

        train | buffer | validation | buffer | test

    Returns a dict of chunk-index arrays: 'train', 'val', 'test' (and 'buffer', the dropped chunks).
    """
    n_test = int(round(n_chunks * test_fraction))
    n_val = int(round(n_chunks * val_fraction))
    test = np.arange(n_chunks - n_test, n_chunks)
    val_stop = n_chunks - n_test - buffer_chunks
    val = np.arange(val_stop - n_val, val_stop)
    train = np.arange(0, val_stop - n_val - buffer_chunks)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f'{n_chunks} chunks are too few for this split')
    used = np.concatenate([train, val, test])
    return dict(train=train, val=val, test=test, buffer=np.setdiff1d(np.arange(n_chunks), used))


def blocked_folds(n_chunks, n_folds=5, val_fraction=config.VAL_FRACTION, buffer_chunks=config.BUFFER_CHUNKS):
    """
    Blocked cross-validation: fold k tests on the k-th contiguous block. Validation is the block of
    chunks just before the test block (just after it for the first fold), and training is every
    remaining chunk not within `buffer_chunks` of either.

    Returns a list of dicts like holdout_split().
    """
    edges = np.linspace(0, n_chunks, n_folds + 1).round().astype(int)
    n_val = max(1, int(round(n_chunks * val_fraction)))
    folds = []
    for k in range(n_folds):
        test = np.arange(edges[k], edges[k + 1])
        if k > 0:
            val = np.arange(max(0, test[0] - buffer_chunks - n_val), test[0] - buffer_chunks)
        else:
            val = np.arange(test[-1] + 1 + buffer_chunks, test[-1] + 1 + buffer_chunks + n_val)
        excluded = np.zeros(n_chunks, dtype=bool)
        for block in (test, val):
            excluded[max(0, block[0] - buffer_chunks):min(n_chunks, block[-1] + 1 + buffer_chunks)] = True
        train = np.flatnonzero(~excluded)
        used = np.concatenate([train, val, test])
        folds.append(dict(train=train, val=val, test=test, buffer=np.setdiff1d(np.arange(n_chunks), used)))
    return folds


def chunk_bins(chunks, seqlength=config.SEQLENGTH):
    """Bin indices covered by the given chunks."""
    chunks = np.asarray(chunks)
    return (chunks[:, None] * seqlength + np.arange(seqlength)).ravel()


def describe(split, seqlength=config.SEQLENGTH, bin_size=config.BIN_SIZE):
    """Chunk counts, durations and time ranges of each set, for the split schematic."""
    out = {}
    for name in ('train', 'val', 'test', 'buffer'):
        chunks = np.asarray(split[name])
        out[name] = dict(n_chunks=int(len(chunks)),
                         minutes=round(len(chunks) * seqlength * bin_size / 60, 2),
                         first_chunk=int(chunks.min()) if len(chunks) else None,
                         last_chunk=int(chunks.max()) if len(chunks) else None)
    return out
