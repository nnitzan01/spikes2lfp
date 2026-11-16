"""
An example stretch of the test block for the figures: the recorded targets and the seed-0 predictions on every
channel (broadband and the eight band envelopes), the firing rate of each region group, and the stimulus and lick times.

The stretch is chosen on the shown channel's models of --select-targets (default broadband; mean R2): every run of consecutive test-block chunks of the
requested length is scored by its R2, and the run at the --quantile of those scores is taken (a typical good fit,
not the best one); --start-chunk sets it by hand instead.

usage (from the repo root):
    python -m experiments.example_trace <session_dir> [--seconds 12] [--channel 5] [--quantile 0.9] [--start-chunk N]
        [--select-targets 0] [--output example_trace]

Needs the 'train' stage. Saves variables/<session_id>/example_trace.npz:
  t (s from the start of the stretch), target / prediction (targets x bins x channels; scaled targets),
  targets, rate (bins x groups, spikes/s per unit, 20 ms bins), rate_t, groups, image_times, change_times,
  omission_times, lick_times (s from the start), layer4_channel, channel (the shown channel; default layer 4),
  start_chunk, window_r2 (broadband R2 of every candidate stretch, in test-chunk order), quantile,
  ridge_lagged / ridge_instant (targets x bins): the lagged unit ridge (1 s history) and the current-bin ridge of
  experiments/baselines.py, fitted on the training block for the shown channel, on the same stretch
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


def window_scores(pipeline, chan, n_chunks, targets=(0,)):
    """Mean R2 over `targets` of the shown channel's models on every run of n_chunks consecutive test chunks."""
    test = pipeline.split['test']
    X = torch.as_tensor(take_chunks(pipeline.active_spikes.spkMat, test), dtype=torch.float32)
    scores = np.zeros(len(test) - n_chunks + 1)
    for band in targets:
        y = take_chunks(pipeline.active_lfp.lfpMat[:, chan, band], test)                # chunks x bins
        p = pipeline.models[chan, band].evaluate(X).numpy().reshape(y.shape)
        for i in range(len(scores)):
            a, b = y[i:i + n_chunks].ravel(), p[i:i + n_chunks].ravel()
            scores[i] += (1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)) / len(targets)
    return scores


def run(pipeline, seconds=12.0, start_chunk=None, ridge=True, channel=None, quantile=0.9, select_targets=(0,)):
    n = int(round(seconds / config.BIN_SIZE))
    n_chunks = int(np.ceil(n / config.SEQLENGTH))
    test = pipeline.split['test']
    l4 = layer4_channel(pipeline.session_id, pipeline.n_channels)[0]
    channel = l4 if channel is None else channel
    scores = window_scores(pipeline, channel, n_chunks, select_targets)
    if start_chunk is None:
        start_chunk = int(np.argmin(np.abs(scores - np.quantile(scores, quantile))))
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
    extra = ridge_predictions(pipeline, channel, chunk_bins(chunks)[:n]) if ridge else {}
    s = pipeline.session
    events = task_events(s.active, s.licks.timestamps, auto_rewarded_changes(s))[0]
    in_window = lambda x: np.sort(np.asarray(x, float)[(np.asarray(x) >= t0) & (np.asarray(x) < t0 + n * config.BIN_SIZE)] - t0)
    changes = np.concatenate([events[k] for k in ('change_hit', 'change_miss') if k in events])
    return dict(t=np.arange(n) * config.BIN_SIZE, target=target, prediction=prediction, targets=np.array(targets),
                rate=rate, rate_t=(np.arange(len(rate)) + 0.5) * RATE_BIN * config.BIN_SIZE, groups=np.array(GROUPS),
                image_times=in_window(np.concatenate([events['image'], events['post_omission']])),
                change_times=in_window(changes), omission_times=in_window(events['omission']),
                lick_times=in_window(s.licks.timestamps),
                layer4_channel=np.array(l4), channel=np.array(channel), start_chunk=np.array(start_chunk),
                window_r2=scores, quantile=np.array(quantile), select_targets=np.array(select_targets), **extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--seconds', type=float, default=12.0)
    parser.add_argument('--channel', type=int, default=None, help='channel shown (default: the layer 4 channel)')
    parser.add_argument('--quantile', type=float, default=0.9, help='fit quantile of the chosen stretch among all stretches')
    parser.add_argument('--start-chunk', type=int, default=None, help='test-block chunk (3 s) to start from (overrides --quantile)')
    parser.add_argument('--select-targets', default='0', help='targets whose mean R2 chooses the stretch')
    parser.add_argument('--output', default='example_trace', help='file name (without .npz) in variables/<session_id>/')
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.seconds, args.start_chunk, channel=args.channel, quantile=args.quantile,
              select_targets=[int(t) for t in args.select_targets.split(',')])
    f = pipeline.dirs.variables / f'{args.output}.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
