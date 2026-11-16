"""
An example stretch of the test block for the overview figure: the recorded targets and the seed-0 predictions on every
channel (broadband, 0.5-4 Hz, 100-200 Hz), the firing rate of each region group, and the stimulus and lick times.

usage (from the repo root):
    python -m experiments.example_trace <session_dir> [--seconds 12] [--start-chunk 2]

Needs the 'train' stage. Saves variables/<session_id>/example_trace.npz:
  t (s from the start of the stretch), target / prediction (targets x bins x channels; scaled targets),
  targets, rate (bins x groups, spikes/s per unit, 20 ms bins), rate_t, groups, image_times, change_times,
  omission_times, lick_times (s from the start), layer4_channel,
  ridge_lagged / ridge_instant (targets x bins): the lagged unit ridge (1 s history) and the current-bin ridge of
  experiments/baselines.py, fitted on the training block for the layer 4 channel, on the same stretch
"""
import argparse

import numpy as np
import torch

import config
import paths
from experiments.explore_task import auto_rewarded_changes, layer4_channel, task_events
from experiments.region_ablation import region_masks
from experiments.region_events import GROUPS
from analysis.lagged_ridge import fit_select_predict, raised_cosine_basis
from preprocess_data import take_chunks
from run_session import SessionPipeline, save_npz
from splits import chunk_bins

TARGETS = list(range(9))              # broadband and the eight band envelopes
RIDGE_TARGETS = [0, 1]                # targets (of TARGETS) the ridge baselines are fitted for
RATE_BIN = 5                          # spike bins per rate bin (20 ms)


def ridge_predictions(pipeline, chan, bins):
    """Predictions of the lagged and the current-bin unit ridge (baselines.py) for RIDGE_TARGETS of one channel at bins."""
    Z, split = pipeline.active_spikes.spkMat, pipeline.split
    Y = pipeline.active_lfp.lfpMat[:, chan, RIDGE_TARGETS]
    train, val = (chunk_bins(split[p])[::config.RIDGE_TRAIN_STEP] for p in ('train', 'val'))
    basis = raised_cosine_basis(int(round(config.RIDGE_HISTORY / config.BIN_SIZE)), config.RIDGE_N_BASIS)
    out = {}
    for name, b in (('ridge_lagged', basis), ('ridge_instant', np.ones((1, 1)))):
        out[name] = np.full((len(TARGETS), len(bins)), np.nan)
        out[name][[TARGETS.index(t) for t in RIDGE_TARGETS]] = fit_select_predict(Z, Y, b, train, val, bins, config.RIDGE_ALPHAS)[0].T
    return out


def run(pipeline, seconds=12.0, start_chunk=2, ridge=True):
    n = int(round(seconds / config.BIN_SIZE))
    n_chunks = int(np.ceil(n / config.SEQLENGTH))
    test = pipeline.split['test']
    chunks = test[start_chunk:start_chunk + n_chunks]
    X = torch.as_tensor(take_chunks(pipeline.active_spikes.spkMat, chunks), dtype=torch.float32)
    Y = take_chunks(pipeline.active_lfp.lfpMat, chunks).reshape(-1, pipeline.n_channels, config.N_TARGETS)[:n]
    counts = take_chunks(pipeline.active_counts.spkMat, chunks).reshape(-1, pipeline.active_counts.spkMat.shape[1])[:n]
    t0 = pipeline.active_spikes.timestamps[chunks[0] * config.SEQLENGTH]
    targets = [t for t in TARGETS if t < config.N_TARGETS]
    target = np.stack([Y[:, :, t] for t in targets])
    prediction = np.full_like(target, np.nan)
    for j, band in enumerate(targets):
        for chan in range(pipeline.n_channels):
            if (chan, band) in pipeline.models:
                prediction[j, :, chan] = pipeline.models[chan, band].evaluate(X).numpy()[:n]
    masks = region_masks(np.asarray(pipeline.session.units.structure_acronym))
    m = n // RATE_BIN * RATE_BIN
    binned = counts[:m].reshape(-1, RATE_BIN, counts.shape[1]).sum(1) / (RATE_BIN * config.BIN_SIZE)
    rate = np.stack([binned[:, masks[g]].mean(1) if masks[g].any() else np.full(len(binned), np.nan) for g in GROUPS], 1)
    l4 = layer4_channel(pipeline.session_id, pipeline.n_channels)[0]
    extra = ridge_predictions(pipeline, l4, chunk_bins(chunks)[:n]) if ridge else {}
    s = pipeline.session
    events = task_events(s.active, s.licks.timestamps, auto_rewarded_changes(s))[0]
    in_window = lambda x: np.sort(np.asarray(x, float)[(np.asarray(x) >= t0) & (np.asarray(x) < t0 + n * config.BIN_SIZE)] - t0)
    changes = np.concatenate([events[k] for k in ('change_hit', 'change_miss') if k in events])
    return dict(t=np.arange(n) * config.BIN_SIZE, target=target, prediction=prediction, targets=np.array(targets),
                rate=rate, rate_t=(np.arange(len(rate)) + 0.5) * RATE_BIN * config.BIN_SIZE, groups=np.array(GROUPS),
                image_times=in_window(np.concatenate([events['image'], events['post_omission']])),
                change_times=in_window(changes), omission_times=in_window(events['omission']),
                lick_times=in_window(s.licks.timestamps),
                layer4_channel=np.array(l4), **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--seconds', type=float, default=12.0)
    parser.add_argument('--start-chunk', type=int, default=2, help='test-block chunk (3 s) to start from')
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.seconds, args.start_chunk)
    f = pipeline.dirs.variables / 'example_trace.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
