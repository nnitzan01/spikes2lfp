"""
Does a unit's high attribution just reflect its high firing rate?

For the units with the largest mean |attribution| (one channel, broadband model), remove spikes
until the unit fires at its area's mean rate, recompute the unit's attribution with the same
model, and compare. Repeated with different random choices of the removed spikes.

usage (from the repo root):
    python -m experiments.firing_rate_contribution <session_dir> [--channel 5] [--n-top 10] [--n-iter 50]

Uses the trained model and the 'active' attribution window of run_session.py. Rates and spike
removal are restricted to that window, which is where attribution is recomputed.
Saves attrs/<session_id>/attribution_firing_rate_resampling_results.npy, a pickled dict
(load with np.load(..., allow_pickle=True).item()).
"""
import argparse

import numpy as np
import pandas as pd
import torch

import config
import paths
from integrated_gradients import IntegratedGradient
from run_session import SessionPipeline, load_sparse

BAND = 0    # broadband model


def unit_attribution(net, X, device):
    """Mean |attribution| of every unit over X (bins x units)."""
    ig = IntegratedGradient(net, method='last time point', seqlength=config.SEQLENGTH)
    attrs = ig.run(torch.tensor(X).float().to(device), baselines=config.IG_BASELINE,
                   n_batch=config.IG_N_BATCH, n_steps=config.IG_N_STEPS)
    return np.mean(np.abs(attrs.numpy()), axis=0)


def run(pipeline, chan, n_top, n_iter):
    rng = np.random.default_rng(config.TORCH_SEED)
    X = pipeline.spikes(*pipeline.window_times('active')).spkMat            # model input for the window
    areas = pipeline.session.units['structure_acronym'].to_numpy()
    net = pipeline.models[chan, BAND].model.train().to(pipeline.device)    # train(): cuDNN RNN backward

    attr_file = paths.attribution_file(pipeline.dirs, 'active', chan, BAND)
    mean_abs_attr = (np.mean(np.abs(load_sparse(attr_file)), axis=0) if attr_file.exists()
                     else unit_attribution(net, X, pipeline.device))
    top = np.argsort(-mean_abs_attr, kind='stable')[:n_top]

    # firing rate as the fraction of bins with a spike, within the window
    rate = np.mean(X > 0, axis=0)
    area_rate = pd.Series(rate).groupby(areas).mean()

    results = {
        'top_neuron_indices': top,
        'top_neuron_unit_ids': pipeline.session.units.index[top].to_list(),
        'top_neuron_areas': areas[top].tolist(),
        'original_attributions': mean_abs_attr[top],
        'original_firing_rates': rate[top],
        'area_avg_firing_rates': area_rate[np.unique(areas[top])].to_dict(),
        'area_population_attributions': pd.Series(mean_abs_attr).groupby(areas).mean()[np.unique(areas[top])].to_dict(),
        'downsampled_attributions': [],
    }

    for unit in top:
        active_bins = np.flatnonzero(X[:, unit] > 0)
        n_remove = len(active_bins) - int(area_rate[areas[unit]] * X.shape[0])
        print(f'unit {unit} ({areas[unit]}): rate {rate[unit]:.4f} -> {area_rate[areas[unit]]:.4f}, '
              f'removing {max(n_remove, 0)} of {len(active_bins)} spike bins', flush=True)
        if n_remove <= 0:      # already at or below its area's mean rate
            results['downsampled_attributions'].append([mean_abs_attr[unit]] * n_iter)
            continue
        downsampled = []
        for _ in range(n_iter):
            # remove a run of consecutive spike bins (consecutive in the list of the unit's spike
            # bins, so spread over time) starting at a random spike
            first = rng.integers(0, len(active_bins) - n_remove + 1)
            X_mod = X.copy()
            X_mod[active_bins[first:first + n_remove], unit] = 0
            downsampled.append(unit_attribution(net, X_mod, pipeline.device)[unit])
        results['downsampled_attributions'].append(downsampled)
        print(f'  attribution {mean_abs_attr[unit]:.6f} -> {np.mean(downsampled):.6f} (mean of {n_iter})')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--channel', type=int, default=5)
    parser.add_argument('--n-top', type=int, default=10, help='number of top-attribution units to test')
    parser.add_argument('--n-iter', type=int, default=50, help='random spike removals per unit')
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
