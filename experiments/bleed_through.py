"""
Spike bleed-through (reviewer 3, comments 6 and 11): is the predicted high-frequency power only the
spike waveforms of units recorded near the LFP contact, which the model sees as inputs?

For every VISp LFP channel, units are excluded by distance from the contact (same probe, Euclidean
distance between the unit's peak channel and the contact, from probe_vertical_position and
probe_horizontal_position; units on other probes are never near):
  near_<r>   units within r um, for r in RADII_UM
  visp       every VISp unit (the local population)
and the prediction is scored with the timescale measures of experiments/timescale_r2.py
(R2 after smoothing, coherence, transient AUC) on the test block:

  inference  the seed-0 models with the excluded units replaced by their training mean
             (inf_<measure>: groups x ... x channels x targets; 'all' = no unit excluded)
  retrained  models trained without the excluded units (fit_and_score), for --retrain-groups and
             --retrain-targets, plus a model retrained on all units with the same seed for comparison
             (ret_<measure>: groups x ..., with 'all' first)

Also saves n_excluded (groups x channels), nearest_um (channels: distance of the nearest unit),
groups, retrain_groups, radii_um.

usage (from the repo root):
    python -m experiments.bleed_through <session_dir> [--retrain-groups near_100 | none]
        [--retrain-targets 1,5,6,7,8]

Needs the 'train' stage. Saves variables/<session_id>/bleed_through.npz.
"""
import argparse

import numpy as np
import torch

import config
import paths
from experiments.stimulus_clock import contiguous, predict
from experiments.timescale_r2 import WINDOWS_MS, coherence_freqs, scores
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

RADII_UM = [50, 100, 200]
GROUPS = [f'near_{r}' for r in RADII_UM] + ['visp']
MEASURES = {'r2_timescale': (len(WINDOWS_MS),), 'coherence': (len(coherence_freqs()),), 'auc_transient': ()}


def distances(units, channels):
    """Distance (um) of each unit's peak channel from each LFP channel: units x channels (inf on other probes)."""
    for table, name in ((units, 'units'), (channels, 'LFP channels')):
        missing = {'probe_id', 'probe_vertical_position'} - set(table.columns)
        if missing:
            raise KeyError(f'{name} table has no {sorted(missing)} (needs the session channel table)')

    def position(table):
        h = table['probe_horizontal_position'] if 'probe_horizontal_position' in table else 0
        return np.stack([np.asarray(table['probe_vertical_position'], float), np.broadcast_to(np.asarray(h, float), len(table))], 1)
    d = np.linalg.norm(position(units)[:, None] - position(channels)[None], axis=-1)
    same = np.asarray(units['probe_id'])[:, None] == np.asarray(channels['probe_id'])[None]
    return np.where(same, d, np.inf)


def exclusion_masks(units, channels):
    """{group: units x channels boolean, True = excluded from that channel's inputs}."""
    d = distances(units, channels)
    masks = {f'near_{r}': d <= r for r in RADII_UM}
    masks['visp'] = np.repeat((np.asarray(units['structure_acronym']) == 'VISp')[:, None], len(channels), 1)
    return masks


def empty(n_groups, n_ch, prefix):
    return {f'{prefix}_{k}': np.full((n_groups,) + extra + (n_ch, config.N_TARGETS), np.nan) for k, extra in MEASURES.items()}


def store(out, prefix, g, chan, band, res):
    for k in MEASURES:
        out[f'{prefix}_{k}'][(g, Ellipsis, chan, band)] = res[k]


def run(pipeline, retrain_groups=('near_100',), retrain_targets=(1, 5, 6, 7, 8)):
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch = Y.shape[1]
    test_bins = contiguous(chunk_bins(pipeline.split['test']))
    scaler = pipeline.target_scaler
    masks = exclusion_masks(pipeline.session.units, pipeline.active_lfp.channels)
    train_mean = X[chunk_bins(pipeline.split['train'])].mean(axis=0)
    groups = ['all'] + GROUPS
    out = {**empty(len(groups), n_ch, 'inf'), **empty(1 + len(retrain_groups), n_ch, 'ret')}

    def evaluate(model, inputs, chan, band):
        y = Y[test_bins, chan, band].astype(float)
        yhat = predict(model, inputs, pipeline.split['test']).astype(float)
        return scores(y, yhat, scaler.mean_[chan, band], scaler.std_[chan, band], band)[0]

    for (chan, band), model in sorted(pipeline.models.items()):
        for g, group in enumerate(groups):
            if group == 'all':
                inputs = X
            else:
                excluded = masks[group][:, chan]
                inputs = X.copy()
                inputs[:, excluded] = train_mean[excluded]
            store(out, 'inf', g, chan, band, evaluate(model, inputs, chan, band))
        log(f'chan {chan} band {band}: R2 at 250 ms, inference, ' + ', '.join(
            f'{group} {out["inf_r2_timescale"][g, -1, chan, band]:.3f}' for g, group in enumerate(groups)))

    for band in retrain_targets:
        for chan in range(n_ch):
            y = Y[:, chan, band]
            for g, group in enumerate(['all'] + list(retrain_groups)):
                keep = np.ones(X.shape[1], bool) if group == 'all' else ~masks[group][:, chan]
                if not keep.any():
                    continue
                model, _ = pipeline.fit_and_score(X[:, keep], y)
                store(out, 'ret', g, chan, band, evaluate(model, X[:, keep], chan, band))
            log(f'retrained chan {chan} band {band}: R2 at 250 ms ' + ', '.join(
                f'{group} {out["ret_r2_timescale"][g, -1, chan, band]:.3f}' for g, group in enumerate(['all'] + list(retrain_groups))))

    d = distances(pipeline.session.units, pipeline.active_lfp.channels)
    out.update(groups=np.array(groups), retrain_groups=np.array(['all'] + list(retrain_groups)), radii_um=np.array(RADII_UM),
               n_excluded=np.stack([masks[g].sum(axis=0) for g in GROUPS]), nearest_um=d.min(axis=0),
               windows_ms=np.array(WINDOWS_MS), coherence_freqs=coherence_freqs())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--retrain-groups', default='near_100', help=f"comma-separated from {GROUPS}, or 'none'")
    parser.add_argument('--retrain-targets', default='1,5,6,7,8')
    args = parser.parse_args()

    groups = [] if args.retrain_groups == 'none' else args.retrain_groups.split(',')
    unknown = set(groups) - set(GROUPS)
    if unknown:
        parser.error(f'unknown groups {sorted(unknown)}; choose from {GROUPS}')
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, groups, [int(t) for t in args.retrain_targets.split(',')] if groups else [])
    f = pipeline.dirs.variables / 'bleed_through.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
