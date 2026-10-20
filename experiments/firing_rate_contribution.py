"""
Does a unit's high attribution just reflect its high firing rate?

The units with the largest mean |attribution| on training-block targets (one channel, broadband
model) are selected. For each, random spikes are removed until it fires at the mean rate of its area,
and its history-aware attribution is recomputed with the same model on the *test-block* targets.
Repeated n_iter times with different random removals; the downsampled values are compared with the
unit's original test-block value and with its area's mean.

usage (from the repo root):
    python -m experiments.firing_rate_contribution <session_dir> [--channel 5] [--n-top 10] [--n-iter 10]

Needs the 'train', 'attr' and 'summaries' stages of run_session.py.
Saves attrs/<session_id>/attribution_firing_rate_resampling_results.npy, a pickled dict
(load with np.load(..., allow_pickle=True).item()); attribution values are mean |attribution| and
attribution per spike over the test-block targets.
"""
import argparse

import numpy as np
import pandas as pd
import torch

import config
import paths
from analysis.sequence_attribution import attribute_targets, unit_summaries, window_spike_counts
from integrated_gradients import SequenceIG
from run_session import SessionPipeline

BAND = 0    # broadband model


def test_summaries(pipeline, ig, counts):
    """Per-unit summaries over the test-block targets for a (possibly modified) spike-count matrix."""
    t = pipeline.attr_targets
    X = torch.as_tensor(pipeline.spike_scaler.transform(counts), device=pipeline.device)
    res = attribute_targets(ig, X, t['sampled_test'], history=config.ATTR_HISTORY,
                            baseline=pipeline.ig_baseline(X.shape[1]), batch_windows=pipeline.ig_batch)
    return unit_summaries(res['total'], window_spike_counts(counts, t['sampled_test'], config.ATTR_HISTORY))


def run(pipeline, chan, n_top, n_iter):
    rng = np.random.default_rng(config.TORCH_SEED)
    counts = pipeline.active_counts.spkMat
    areas = pipeline.session.units['structure_acronym'].to_numpy()
    ig = SequenceIG(pipeline.models[chan, BAND].model.to(pipeline.device), n_steps=config.IG_N_STEPS)

    selection = np.load(pipeline.dirs.attrs / 'unit_attribution_train.npz')['mean_abs'][chan, BAND]
    top = np.argsort(-selection, kind='stable')[:n_top]                 # chosen on training targets
    original = test_summaries(pipeline, ig, counts)                      # evaluated on test targets

    rate = counts.sum(axis=0) / (counts.shape[0] * config.BIN_SIZE)      # spikes/s over the active epoch
    area_rate = pd.Series(rate).groupby(areas).mean()
    area_attr = pd.Series(original['mean_abs']).groupby(areas).mean()

    results = {
        'top_neuron_indices': top,
        'top_neuron_unit_ids': pipeline.session.units.index[top].to_list(),
        'top_neuron_areas': areas[top].tolist(),
        'selection_mean_abs': selection[top],
        'original_attributions': original['mean_abs'][top],
        'original_per_spike': original['per_spike'][top],
        'original_firing_rates': rate[top],
        'area_avg_firing_rates': area_rate[np.unique(areas[top])].to_dict(),
        'area_population_attributions': area_attr[np.unique(areas[top])].to_dict(),
        'downsampled_attributions': [],
        'downsampled_per_spike': [],
    }
    for unit in top:
        keep_fraction = min(1.0, area_rate[areas[unit]] / rate[unit]) if rate[unit] > 0 else 1.0
        spike_bins = np.repeat(np.arange(counts.shape[0]), counts[:, unit].astype(int))   # one entry per spike
        print(f'unit {unit} ({areas[unit]}): {rate[unit]:.2f} -> {area_rate[areas[unit]]:.2f} spikes/s', flush=True)
        mean_abs, per_spike = [], []
        for _ in range(n_iter if keep_fraction < 1 else 0):
            kept = rng.random(len(spike_bins)) < keep_fraction                # remove spikes at random
            modified = counts.copy()
            modified[:, unit] = np.bincount(spike_bins[kept], minlength=counts.shape[0])
            s = test_summaries(pipeline, ig, modified)
            mean_abs.append(s['mean_abs'][unit])
            per_spike.append(s['per_spike'][unit])
        results['downsampled_attributions'].append(mean_abs or [original['mean_abs'][unit]] * n_iter)
        results['downsampled_per_spike'].append(per_spike or [original['per_spike'][unit]] * n_iter)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--channel', type=int, default=5)
    parser.add_argument('--n-top', type=int, default=10, help='number of top-attribution units to test')
    parser.add_argument('--n-iter', type=int, default=10, help='random spike removals per unit')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    results = run(pipeline, args.channel, args.n_top, args.n_iter)

    out = pipeline.dirs.attrs / 'attribution_firing_rate_resampling_results.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, results)
    print('Saved', out)


if __name__ == '__main__':
    main()
