"""
Does a unit's high attribution just reflect its high firing rate? (reviewer 3, comment 8)

For each of --bands on the layer 4 channel (alignment table; --channel to override), the --n-top units with the
largest mean |attribution| on the *training-block* targets are selected (the attr stage's zero-baseline
attribution, unit_attribution_train.npz) and evaluated on the *test-block* targets with the same model:

  matched control  for every top unit, the unit of the same area (any area if none) with the closest firing rate
                   among the units outside the top --exclude-fraction (default: only the top units themselves; high
                   rates and high attribution go together, so excluding more leaves no rate-matched units): if
                   attribution only restated firing rate, a top unit and its control would have the same attribution.
                   rate_control / rate_top records how close the match is
  downsampled      random spikes of the top unit removed until it fires at its area's mean rate (--n-iter
                   draws) and its attribution recomputed; compared with the area's mean attribution

Inference over sessions and mice is done by `summarize firing_rate_control` (paired, per session).

usage (from the repo root):
    python -m experiments.firing_rate_contribution <session_dir> [--bands 0,1] [--channel N] [--n-top 10]
        [--n-iter 5] [--exclude-fraction 0]

Needs the 'train', 'attr' and 'summaries' stages. Saves variables/<session_id>/firing_rate_control.npz
(bands x top units, except where stated; attribution = mean |attribution| and attribution per spike over the
test-block targets):
  bands, channel, top, control (unit indices), rate_top, rate_control, rate_area (spikes/s),
  attr_top, attr_control, attr_area (area mean), per_spike_top, per_spike_control,
  attr_downsampled, per_spike_downsampled (bands x top x iterations), same_area (control from the same area)
"""
import argparse

import numpy as np
import pandas as pd
import torch

import config
import paths
from analysis.sequence_attribution import attribute_targets, unit_summaries, window_spike_counts
from experiments.explore_task import layer4_channel
from integrated_gradients import SequenceIG
from run_session import SessionPipeline, save_npz


def test_summaries(pipeline, ig, counts):
    """Per-unit summaries over the test-block targets for a (possibly modified) spike-count matrix."""
    t = pipeline.attr_targets
    X = torch.as_tensor(pipeline.spike_scaler.transform(counts), device=pipeline.device)
    res = attribute_targets(ig, X, t['sampled_test'], history=config.ATTR_HISTORY,
                            baseline=pipeline.ig_baseline(X.shape[1]), batch_windows=pipeline.ig_batch)
    return unit_summaries(res['total'], window_spike_counts(counts, t['sampled_test'], config.ATTR_HISTORY))


def matched_controls(top, rate, areas, excluded):
    """For each top unit, the non-excluded unit of the same area (else any area) with the closest firing rate."""
    controls, same = [], []
    for u in top:
        pool = np.flatnonzero(~excluded & (areas == areas[u]))
        same.append(len(pool) > 0)
        pool = pool if len(pool) else np.flatnonzero(~excluded)
        controls.append(pool[np.argmin(np.abs(rate[pool] - rate[u]))])
    return np.array(controls), np.array(same)


def run_band(pipeline, chan, band, n_top, n_iter, exclude_fraction, rng):
    counts = pipeline.active_counts.spkMat
    areas = pipeline.session.units['structure_acronym'].to_numpy().astype(str)
    ig = SequenceIG(pipeline.models[chan, band].model.to(pipeline.device), n_steps=config.IG_N_STEPS)
    selection = np.load(pipeline.dirs.attrs / 'unit_attribution_train.npz')['mean_abs'][chan, band]
    order = np.argsort(-selection, kind='stable')                          # chosen on training targets
    top = order[:n_top]
    excluded = np.zeros(len(order), bool)
    excluded[order[:max(n_top, int(exclude_fraction * len(order)))]] = True
    rate = counts.sum(axis=0) / (counts.shape[0] * config.BIN_SIZE)      # spikes/s over the active epoch
    control, same = matched_controls(top, rate, areas, excluded)
    original = test_summaries(pipeline, ig, counts)                       # evaluated on test targets
    area_rate = pd.Series(rate).groupby(areas).mean()
    area_attr = pd.Series(original['mean_abs']).groupby(areas).mean()

    out = dict(top=top, control=control, same_area=same, rate_top=rate[top], rate_control=rate[control],
               rate_area=area_rate[areas[top]].to_numpy(), attr_top=original['mean_abs'][top],
               attr_control=original['mean_abs'][control], attr_area=area_attr[areas[top]].to_numpy(),
               per_spike_top=original['per_spike'][top], per_spike_control=original['per_spike'][control],
               attr_downsampled=np.full((n_top, n_iter), np.nan), per_spike_downsampled=np.full((n_top, n_iter), np.nan))
    for j, unit in enumerate(top):
        keep_fraction = min(1.0, area_rate[areas[unit]] / rate[unit]) if rate[unit] > 0 else 1.0
        spike_bins = np.repeat(np.arange(counts.shape[0]), counts[:, unit].astype(int))   # one entry per spike
        print(f'band {band} unit {unit} ({areas[unit]}): {rate[unit]:.2f} -> {area_rate[areas[unit]]:.2f} spikes/s; '
              f'control {control[j]} at {rate[control[j]]:.2f}', flush=True)
        for i in range(n_iter):
            if keep_fraction >= 1:
                out['attr_downsampled'][j, i], out['per_spike_downsampled'][j, i] = out['attr_top'][j], out['per_spike_top'][j]
                continue
            kept = rng.random(len(spike_bins)) < keep_fraction                # remove spikes at random
            modified = counts.copy()
            modified[:, unit] = np.bincount(spike_bins[kept], minlength=counts.shape[0])
            s = test_summaries(pipeline, ig, modified)
            out['attr_downsampled'][j, i], out['per_spike_downsampled'][j, i] = s['mean_abs'][unit], s['per_spike'][unit]
    return out


def run(pipeline, bands=(0, 1), chan=None, n_top=10, n_iter=5, exclude_fraction=0.0):
    rng = np.random.default_rng(config.TORCH_SEED)
    chan = layer4_channel(pipeline.session_id, pipeline.n_channels)[0] if chan is None else chan
    per_band = [run_band(pipeline, chan, b, n_top, n_iter, exclude_fraction, rng) for b in bands]
    out = {k: np.stack([r[k] for r in per_band]) for k in per_band[0]}
    return dict(out, bands=np.array(bands), channel=np.array(chan))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--bands', default='0,1', help='model targets')
    parser.add_argument('--channel', type=int, default=None, help='default: the layer 4 channel')
    parser.add_argument('--n-top', type=int, default=10, help='number of top-attribution units to test')
    parser.add_argument('--n-iter', type=int, default=5, help='random spike removals per unit')
    parser.add_argument('--exclude-fraction', type=float, default=0.0,
                        help='controls come from outside this top fraction of units')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, [int(b) for b in args.bands.split(',')], args.channel, args.n_top, args.n_iter, args.exclude_fraction)
    f = pipeline.dirs.variables / 'firing_rate_control.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
