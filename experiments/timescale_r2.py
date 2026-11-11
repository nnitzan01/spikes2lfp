"""
How well is each band predicted, and on what timescale? (reviewer 3, comment 6)

The band targets are power envelopes (|Hilbert|^2): heavy-tailed, so test R2 is dominated by brief
transients that the MSE-trained model can only predict on average, while it can track the slower
modulation of the band's power. For every channel x target of the seed-0 models, on the test block:

  r2_timescale        windows x ...: R2 after smoothing target and prediction with the same centred
                      boxcar of WINDOWS_MS (4 ms = no smoothing)
  r2_log, r2_amplitude   R2 of log power and of amplitude (sqrt power), the band targets only;
                      power is put back in raw units with the training scaler, clipped at
                      POWER_FLOOR x the target's median before the log / square root
  coherence           freqs x ...: magnitude-squared coherence of target and prediction (Welch,
                      COHERENCE_SECONDS segments); freqs saved as coherence_freqs (Hz)
  auc_transient       AUC of the prediction for transient bins (target > TRANSIENT_SD SD above its
                      test mean) vs the other bins; n_transients
  eta_target, eta_prediction   ... x lags: mean target and prediction around each transient (z-units of
                      the test block), lags ETA_MS before to after (eta_lags_ms)
and each of these with the spikes shifted 30-120 s back in time (as in stimulus_clock; the null), as
<name>_shift with a leading shifts axis (no ETAs).

usage (from the repo root):
    python -m experiments.timescale_r2 <session_dir> [--n-shifts 3]

Needs the 'train' stage. Saves variables/<session_id>/timescale_r2.npz (... = channels x targets;
NaN for the broadband target where a measure is band-only).
"""
import argparse

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import coherence
from sklearn.metrics import roc_auc_score

import config
import paths
from analysis.stimulus_clock import r2
from experiments.stimulus_clock import SHIFT_RANGE, contiguous, predict
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

WINDOWS_MS = [4, 20, 50, 100, 250]
COHERENCE_SECONDS = 2.0
TRANSIENT_SD = 3.0
ETA_MS = 200
POWER_FLOOR = 1e-3


def smooth(y, window_ms):
    n = max(1, int(round(window_ms / 1e3 / config.BIN_SIZE)))
    return uniform_filter1d(y, n, mode='nearest') if n > 1 else y


def band_scores(y, yhat, mean, std):
    """R2 of log power and of amplitude, from z-scored power y, yhat and the scaler's mean and std."""
    p, phat = y * std + mean, yhat * std + mean
    floor = POWER_FLOOR * np.median(p)
    p, phat = np.maximum(p, floor), np.maximum(phat, floor)
    return r2(np.log(p), np.log(phat)), r2(np.sqrt(p), np.sqrt(phat))


def transient_auc(y, yhat):
    events = (y - y.mean()) > TRANSIENT_SD * y.std()
    return (roc_auc_score(events, yhat) if 0 < events.sum() < len(y) else np.nan), events


def event_triggered(x, events, half):
    idx = np.flatnonzero(events)
    idx = idx[(idx >= half) & (idx < len(x) - half)]
    if not len(idx):
        return np.full(2 * half + 1, np.nan)
    return x[idx[:, None] + np.arange(-half, half + 1)].mean(axis=0)


def scores(y, yhat, mean, std, band):
    """All measures of one target / prediction pair."""
    fs = 1 / config.BIN_SIZE
    out = dict(r2_timescale=np.array([r2(smooth(y, w), smooth(yhat, w)) for w in WINDOWS_MS]),
               coherence=coherence(y, yhat, fs=fs, nperseg=int(COHERENCE_SECONDS * fs))[1])
    out['r2_log'], out['r2_amplitude'] = band_scores(y, yhat, mean, std) if band > 0 else (np.nan, np.nan)
    out['auc_transient'], events = transient_auc(y, yhat) if band > 0 else (np.nan, None)
    return out, events


def run(pipeline, n_shifts=3, seed=config.TORCH_SEED):
    rng = np.random.default_rng(seed)
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_bins, n_ch = X.shape[0], Y.shape[1]
    test_bins = contiguous(chunk_bins(pipeline.split['test']))
    n_chunks = len(pipeline.split['test'])
    scaler = pipeline.target_scaler
    shifts = rng.integers(*(np.array(SHIFT_RANGE) / config.BIN_SIZE).astype(int), size=n_shifts)
    shifted = [X[(test_bins - s) % n_bins] for s in shifts]
    fs = 1 / config.BIN_SIZE
    freqs = coherence(np.zeros(int(COHERENCE_SECONDS * fs)), np.zeros(int(COHERENCE_SECONDS * fs)),
                      fs=fs, nperseg=int(COHERENCE_SECONDS * fs))[0]
    half = int(round(ETA_MS / 1e3 / config.BIN_SIZE))

    shape = (n_ch, config.N_TARGETS)
    sizes = dict(r2_timescale=(len(WINDOWS_MS),), coherence=(len(freqs),), r2_log=(), r2_amplitude=(), auc_transient=())
    out = {}
    for k, extra in sizes.items():
        out[k] = np.full(extra + shape, np.nan)
        out[f'{k}_shift'] = np.full((n_shifts,) + extra + shape, np.nan)
    out['n_transients'] = np.zeros(shape, int)
    out['eta_target'], out['eta_prediction'] = (np.full(shape + (2 * half + 1,), np.nan) for _ in range(2))

    for (chan, band), model in sorted(pipeline.models.items()):
        y = Y[test_bins, chan, band].astype(float)
        mean, std = scaler.mean_[chan, band], scaler.std_[chan, band]
        yhat = predict(model, X, pipeline.split['test']).astype(float)
        res, events = scores(y, yhat, mean, std, band)
        for k, v in res.items():
            out[k][(Ellipsis, chan, band)] = v
        if events is not None:
            out['n_transients'][chan, band] = events.sum()
            out['eta_target'][chan, band] = event_triggered((y - y.mean()) / y.std(), events, half)
            out['eta_prediction'][chan, band] = event_triggered((yhat - y.mean()) / y.std(), events, half)
        for i, Xs in enumerate(shifted):
            null, _ = scores(y, predict(model, Xs, np.arange(n_chunks)).astype(float), mean, std, band)
            for k, v in null.items():
                out[f'{k}_shift'][(i, Ellipsis, chan, band)] = v
        log(f'chan {chan} band {band}: R2 by window ' + ' '.join(f'{v:.2f}' for v in out['r2_timescale'][:, chan, band])
            + f'; shifted {np.nanmean(out["r2_timescale_shift"][:, 0, chan, band]):.2f}')

    out.update(windows_ms=np.array(WINDOWS_MS), coherence_freqs=freqs, eta_lags_ms=np.arange(-half, half + 1) * config.BIN_SIZE * 1e3,
               shifts_s=shifts * config.BIN_SIZE)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--n-shifts', type=int, default=3)
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.n_shifts)
    f = pipeline.dirs.variables / 'timescale_r2.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
