"""
Why is the spontaneous-epoch R2 near 0 (v1: about 0.2)? The transfer stage scores the active-epoch
models on the passive and spontaneous epochs with inputs and targets scaled by the active training
scalers, and sklearn's R2 counts any offset or gain mismatch between prediction and target as error.
v1 z-scored the targets within each epoch. This scores the seed-0 models on each epoch four ways:

  r2_active        as the transfer stage: active training scalers for spikes and targets
  r2_epoch         spikes min-max scaled and targets z-scored on the epoch itself (as v1)
  r2_recalibrated  active scalers, then the prediction mapped by a linear fit (a + b * prediction)
                   on the first half of the epoch; R2 on the second half
  r2_corr          squared Pearson correlation of prediction and target (active scalers), the
                   ceiling of any linear recalibration
and the diagnostics, in active-scaled target units:
  target_mean, target_std, prediction_mean, prediction_std
  input_rate_ratio  per epoch, median over units of the epoch's mean spike count / the active
                    training mean (units that fire in training)

All arrays are epochs x channels x targets (input_rate_ratio: epochs), epochs = EPOCHS.

usage (from the repo root):
    python -m experiments.transfer_scaling <session_dir> [--targets 0,1,2,3,4]

Needs the 'train' stage. Saves variables/<session_id>/transfer_scaling.npz.
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.stimulus_clock import r2
from preprocess_data import SpikeScaler, Standardizer
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

EPOCHS = ['active_test', 'passive', 'spontaneous']
KEYS = ['r2_active', 'r2_epoch', 'r2_recalibrated', 'r2_corr', 'target_mean', 'target_std', 'prediction_mean',
        'prediction_std']


def recalibrated_r2(y, yhat):
    """R2 on the second half after fitting y = a + b * yhat on the first half."""
    half = len(y) // 2
    b, a = np.polyfit(yhat[:half], y[:half], 1)
    return r2(y[half:], a + b * yhat[half:])


def epoch_data(pipeline, epoch):
    """Spike counts (bins x units) and raw targets (bins x channels x targets) of an epoch."""
    if epoch == 'active_test':
        bins = chunk_bins(pipeline.split['test'])
        counts = pipeline.active_counts.spkMat
        raw = pipeline.lfp_targets(*pipeline.session.active_times, counts.shape[0]).lfpMat
        return counts[bins], raw[bins]
    start, stop = {'passive': pipeline.session.passive_times, 'spontaneous': pipeline.session.spontaneous_times}[epoch]
    counts = pipeline.spike_counts(start, stop).spkMat
    return counts, pipeline.lfp_targets(start, stop, counts.shape[0]).lfpMat


def run(pipeline, targets=(0, 1, 2, 3, 4)):
    n_ch = pipeline.n_channels
    out = {k: np.full((len(EPOCHS), n_ch, config.N_TARGETS), np.nan) for k in KEYS}
    out['input_rate_ratio'] = np.full(len(EPOCHS), np.nan)
    train_counts = pipeline.active_counts.spkMat[chunk_bins(pipeline.split['train'])]
    train_rate = train_counts.mean(axis=0)
    for e, epoch in enumerate(EPOCHS):
        counts, raw = epoch_data(pipeline, epoch)
        fires = train_rate > 0
        out['input_rate_ratio'][e] = np.median(counts.mean(axis=0)[fires] / train_rate[fires])
        inputs = {'active': pipeline.spike_scaler.transform(counts), 'epoch': SpikeScaler().fit(counts).transform(counts)}
        y_targets = {'active': pipeline.target_scaler.transform(raw), 'epoch': Standardizer().fit(raw).transform(raw)}
        for (chan, band), model in pipeline.models.items():
            if band not in targets:
                continue
            for scaling in ('active', 'epoch'):
                yhat = model.evaluate(inputs[scaling]).numpy()
                y = y_targets[scaling][:len(yhat), chan, band]
                out[f'r2_{scaling}'][e, chan, band] = r2(y, yhat)
                if scaling == 'active':
                    out['r2_recalibrated'][e, chan, band] = recalibrated_r2(y, yhat)
                    out['r2_corr'][e, chan, band] = np.corrcoef(y, yhat)[0, 1] ** 2
                    out['target_mean'][e, chan, band], out['target_std'][e, chan, band] = y.mean(), y.std()
                    out['prediction_mean'][e, chan, band], out['prediction_std'][e, chan, band] = yhat.mean(), yhat.std()
        log(f'{epoch}: median over channels (broadband) R2 active {np.nanmedian(out["r2_active"][e, :, 0]):.3f}, '
            f'epoch {np.nanmedian(out["r2_epoch"][e, :, 0]):.3f}, recalibrated {np.nanmedian(out["r2_recalibrated"][e, :, 0]):.3f}, '
            f'r^2 {np.nanmedian(out["r2_corr"][e, :, 0]):.3f}; input rate ratio {out["input_rate_ratio"][e]:.2f}')
    out['epochs'] = np.array(EPOCHS)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1,2,3,4')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, [int(t) for t in args.targets.split(',')])
    f = pipeline.dirs.variables / 'transfer_scaling.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
