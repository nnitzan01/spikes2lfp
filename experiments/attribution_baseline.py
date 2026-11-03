"""
Does the IG baseline decide how well attribution finds the units the model depends on? (reviewer 2,
comment 1; reviewer 3, alternative IG baselines)

The main attribution integrates from a zero baseline (no spikes), so it also credits each unit's
average drive. The ranked ablation removes a unit by replacing it with its training mean, which only
takes away its fluctuations. Here the training-block targets are attributed again with the
training-mean baseline (as config.IG_BASELINE = 'mean' would in the attr stage), and the ranked
ablation (experiments/ranked_ablation.py) is rerun for MEASURES:

  attribution_mean_baseline            mean |attribution| with the mean baseline
  attribution_mean_baseline_per_spike  |attribution per spike| with the mean baseline
  and, for comparison on the same random orderings, the zero-baseline attribution, ridge weight and
  spike-LFP correlation from ranked_ablation_measures.npz

usage (from the repo root):
    python -m experiments.attribution_baseline <session_dir> [--targets 0,1,2] [--n-random 5]
        [--retrain-targets 0 | none] [--retrain-fractions 0.05,0.2]

Needs the 'train' stage and experiments/ranked_ablation.py (for the other measures' scores).
Saves variables/<session_id>/attribution_baseline.npz, with the keys of ranked_ablation.npz for
these MEASURES, plus
  mean_abs, per_spike          channels x targets x units, mean-baseline summaries (training block)
  completeness, n_steps        channels x targets
  spearman_zero_baseline, spearman_firing_rate   channels x targets, rank correlation of
                               mean |attribution| (mean baseline) with the zero-baseline one / with firing rate
"""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

import config
import paths
from analysis.sequence_attribution import attribute_targets, relative_completeness_error, unit_summaries
from experiments import ranked_ablation as ra
from integrated_gradients import SequenceIG
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

MEASURES = ['attribution_mean_baseline', 'attribution_mean_baseline_per_spike', 'attribution',
            'attribution_per_spike', 'ridge_weight', 'spike_lfp_correlation']


def mean_baseline_attribution(pipeline, targets):
    """Mean-baseline IG of the training-block targets: per-unit summaries (channels x targets x units) and completeness."""
    t = pipeline.attr_targets
    X = torch.as_tensor(pipeline.active_spikes.spkMat, device=pipeline.device)
    n_ch, n_units = pipeline.n_channels, X.shape[1]
    baseline = torch.as_tensor(pipeline.active_spikes.spkMat[chunk_bins(pipeline.split['train'])].mean(axis=0),
                               dtype=torch.float32, device=pipeline.device)
    shape = (n_ch, config.N_TARGETS)
    out = {k: np.full(shape + (n_units,), np.nan) for k in ('mean_abs', 'per_spike')}
    out['completeness'], out['n_steps'] = np.full(shape, np.nan), np.zeros(shape, int)
    for band in targets:
        for chan in range(n_ch):
            steps = config.IG_N_STEPS
            while True:
                ig = SequenceIG(pipeline.models[chan, band].model.to(pipeline.device), n_steps=steps,
                                max_sequences=config.IG_MAX_SEQUENCES)
                res = attribute_targets(ig, X, t['sampled_train'], config.ATTR_HISTORY, baseline=baseline,
                                        batch_windows=pipeline.ig_batch)
                err = relative_completeness_error(res['f_x'], res['f_b'], res['delta'])
                if err <= config.IG_MAX_ERROR or steps >= config.IG_MAX_STEPS:
                    break
                steps = min(4 * steps, config.IG_MAX_STEPS)
            summary = unit_summaries(res['total'], t['sampled_train_spikes'])
            out['mean_abs'][chan, band], out['per_spike'][chan, band] = summary['mean_abs'], summary['per_spike']
            out['completeness'][chan, band], out['n_steps'][chan, band] = err, steps
            log(f'mean-baseline attribution chan {chan} band {band}: {steps} steps, completeness error {err:.1e}')
    return out


def rank_correlation(a, b):
    """Spearman of two (units x channels x targets) arrays per channel x target."""
    out = np.full(a.shape[1:], np.nan)
    for idx in np.ndindex(*a.shape[1:]):
        x, y = a[(slice(None),) + idx], b[(slice(None),) + idx]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
            out[idx] = spearmanr(x[ok], y[ok])[0]
    return out


def run(pipeline, targets=(0, 1, 2), n_random=5, retrain_targets=(0,), retrain_fractions=(0.05, 0.2)):
    targets, retrain_targets = list(targets), list(retrain_targets)
    attr = mean_baseline_attribution(pipeline, sorted(set(targets) | set(retrain_targets)))
    scores = dict(np.load(pipeline.dirs.variables / 'ranked_ablation_measures.npz'))
    scores['attribution_mean_baseline'] = np.moveaxis(attr['mean_abs'], 2, 0)
    scores['attribution_mean_baseline_per_spike'] = np.abs(np.moveaxis(attr['per_spike'], 2, 0))

    rng = np.random.default_rng(config.TORCH_SEED)
    r2_full, r2_ablated, r2_random = ra.inference_ablation(pipeline, scores, targets, n_random, rng, MEASURES)
    out = dict(r2_full=r2_full, r2_ablated=r2_ablated, r2_random=r2_random, measures=np.array(MEASURES),
               fills=np.array(ra.FILLS), fractions=np.array(ra.FRACTIONS),
               retrain_fractions=np.array(retrain_fractions, dtype=float), **attr,
               spearman_zero_baseline=rank_correlation(scores['attribution_mean_baseline'], scores['attribution']),
               spearman_firing_rate=rank_correlation(scores['attribution_mean_baseline'], scores['firing_rate']))
    if retrain_targets:
        r2, r2_rand, r2_full_re = ra.retrained_ablation(pipeline, scores, retrain_targets, list(retrain_fractions), 1,
                                                        max(1, n_random // 2), rng, MEASURES)
        out.update(r2_retrained=r2, r2_retrained_random=r2_rand, r2_retrained_full=r2_full_re)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1,2')
    parser.add_argument('--n-random', type=int, default=5)
    parser.add_argument('--retrain-targets', default='0')
    parser.add_argument('--retrain-fractions', default='0.05,0.2')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, ra.parse_list(args.targets), args.n_random, ra.parse_list(args.retrain_targets),
              ra.parse_list(args.retrain_fractions, float))
    f = pipeline.dirs.variables / 'attribution_baseline.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
