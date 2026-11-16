"""
Robustness of the model and its attribution (reviewer 3, comments 3, 5, 7 and 8), on the pooled channels of
the session (tables/layer_4_borders_aligned.csv; every --channel-step-th) for --targets:

  seeds        the standard model retrained with seeds 1..--seeds (fit_and_score) and attributed with the
               mean baseline on the training-block targets (as experiments/attribution_baseline.py). Seed 0 is
               the session's saved model (r2_scores.npy, attribution_baseline.npz), so rank stability across
               initialisations is the agreement of seed s with seed 0. Also the test R2 spread over seeds.
  subsamples   --subsamples draws of the same number of units from every region group of GROUPS (--per-region,
               or the smallest group's size if smaller; groups with fewer than MIN_UNITS units left out), the
               model retrained on them (seed 0) and attributed: region comparisons without the difference in
               unit counts between regions.
  ablation     retrained ablation with seeds 0..--seeds (ranked_ablation.retrained_ablation) for
               --ablation-targets on every --ablation-channel-step-th pooled channel: the top --fractions of
               units by each of ABLATION_MEASURES (chosen on the training block) and --n-random random
               orderings removed and the model retrained; R2 on the test block.

usage (from the repo root):
    python -m experiments.robustness <session_dir> [--targets 0,1,2] [--seeds 2] [--subsamples 3]
        [--per-region 30] [--channel-step 1] [--ablation-targets 0,1] [--ablation-channel-step 2]
        [--fractions 0.05,0.2] [--n-random 2] [--skip seeds,subsamples,ablation]

Needs the 'train' stage, attribution_baseline.npz and ranked_ablation_measures.npz. Saves
variables/<session_id>/robustness.npz (channels and targets not run are NaN):
  channels, ablation_channels         channel indices run
  r2_seeds                            (seeds + 1) x channels x targets, test R2 (0 = saved model)
  attr_seeds                          (seeds + 1) x channels x targets x units, mean |attribution| (mean baseline)
  unit_groups                         units, index into GROUPS (-1: none of them)
  sub_units                           subsamples x units (bool); sub_n_per_group
  r2_sub, attr_sub                    subsamples x channels x targets (x units, NaN outside the draw)
  r2_ablation                         measures x fractions x (seeds + 1) x channels x targets
  r2_ablation_random                  orderings x fractions x (seeds + 1) x channels x targets
  r2_ablation_full                    (seeds + 1) x channels x targets (retrained on every unit)
  ablation_measures, fractions, groups, completeness_seeds, completeness_sub
"""
import argparse

import numpy as np
import pandas as pd
import torch

import config
import paths
from analysis.sequence_attribution import unit_summaries
from experiments import ranked_ablation as ra
from experiments.attribution_baseline import attribute_with_retries
from experiments.pool_sessions import ALIGNMENT_TABLE
from experiments.region_ablation import region_masks
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

GROUPS = ['VISp', 'visual_cortex_other', 'hippocampus', 'thalamus', 'midbrain']
ABLATION_MEASURES = ['attribution_mean_baseline', 'ridge_weight', 'spike_lfp_correlation', 'firing_rate']
MIN_UNITS = 10


def pooled_channels(session_id, n_ch):
    """The session's pooled channels (alignment table), or every channel if it is not in the table (or none fit)."""
    try:
        row = pd.read_csv(ALIGNMENT_TABLE).set_index('session').loc[session_id]
        channels = list(range(int(row['first_channel_index']), min(int(row['last_channel_index']) + 1, n_ch)))
    except (KeyError, FileNotFoundError):
        channels = []
    return channels or list(range(n_ch))


def attribute(pipeline, model, X):
    """Mean-baseline IG of the training-block targets: attribution totals (targets x units) and the completeness error."""
    t = pipeline.attr_targets
    baseline = torch.as_tensor(X[chunk_bins(pipeline.split['train'])].mean(axis=0), dtype=torch.float32, device=pipeline.device)
    res, err, _ = attribute_with_retries(model, torch.as_tensor(X, dtype=torch.float32, device=pipeline.device),
                                         t['sampled_train'], baseline, pipeline.ig_batch)
    return res['total'], err


def seeds(pipeline, channels, targets, n_seeds):
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch, n_tg, n_units = Y.shape[1], Y.shape[2], X.shape[1]
    spikes = pipeline.attr_targets['sampled_train_spikes']
    r2 = np.full((n_seeds + 1, n_ch, n_tg), np.nan)
    attr = np.full((n_seeds + 1, n_ch, n_tg, n_units), np.nan)
    completeness = np.full((n_seeds + 1, n_ch, n_tg), np.nan)
    r2[0] = np.load(pipeline.dirs.variables / 'r2_scores.npy')
    saved = np.load(pipeline.dirs.variables / 'attribution_baseline.npz')
    attr[0], completeness[0] = saved['mean_abs'], saved['completeness']
    for band in targets:
        for chan in channels:
            for s in range(1, n_seeds + 1):
                model, r2[s, chan, band] = pipeline.fit_and_score(X, Y[:, chan, band], seed=s)
                total, completeness[s, chan, band] = attribute(pipeline, model.model, X)
                attr[s, chan, band] = unit_summaries(total, spikes)['mean_abs']
            log(f'seeds chan {chan} band {band}: test R2 {np.round(r2[:, chan, band], 3)}')
    return dict(r2_seeds=r2, attr_seeds=attr, completeness_seeds=completeness)


def subsamples(pipeline, channels, targets, n_draws, per_region, rng):
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch, n_tg, n_units = Y.shape[1], Y.shape[2], X.shape[1]
    masks = region_masks(pipeline.session.units['structure_acronym'].to_numpy().astype(str))
    unit_groups = np.full(n_units, -1)
    for i, g in enumerate(GROUPS):
        unit_groups[masks[g]] = i
    present = [i for i in range(len(GROUPS)) if np.sum(unit_groups == i) >= MIN_UNITS]
    n = min([per_region] + [int(np.sum(unit_groups == i)) for i in present])
    if len(present) < 2:                                    # nothing to compare
        n_draws = 0
    sub_units = np.zeros((n_draws, n_units), bool)
    for d in range(n_draws):
        for i in present:
            sub_units[d, rng.choice(np.flatnonzero(unit_groups == i), n, replace=False)] = True
    spikes = pipeline.attr_targets['sampled_train_spikes']
    r2 = np.full((n_draws, n_ch, n_tg), np.nan)
    attr = np.full((n_draws, n_ch, n_tg, n_units), np.nan)
    completeness = np.full((n_draws, n_ch, n_tg), np.nan)
    for d in range(n_draws):
        keep = np.flatnonzero(sub_units[d])
        for band in targets:
            for chan in channels:
                model, r2[d, chan, band] = pipeline.fit_and_score(X[:, keep], Y[:, chan, band])
                total, completeness[d, chan, band] = attribute(pipeline, model.model, X[:, keep])
                attr[d, chan, band, keep] = unit_summaries(total, spikes[:, keep])['mean_abs']
        log(f'subsample {d}: {n} units from each of {[GROUPS[i] for i in present]}, '
            f'median test R2 by target {np.round(np.nanmedian(r2[d], axis=0)[targets], 3)}')
    return dict(unit_groups=unit_groups, sub_units=sub_units, sub_n_per_group=np.array(n), r2_sub=r2, attr_sub=attr,
                completeness_sub=completeness)


def ablation(pipeline, channels, targets, fractions, n_seeds, n_random, rng):
    v = pipeline.dirs.variables
    scores = dict(np.load(v / 'ranked_ablation_measures.npz'))
    scores['attribution_mean_baseline'] = np.moveaxis(np.load(v / 'attribution_baseline.npz')['mean_abs'], 2, 0)
    r2, r2_random, r2_full = ra.retrained_ablation(pipeline, scores, targets, fractions, n_seeds + 1, n_random, rng,
                                                   ABLATION_MEASURES, channels=channels)
    return dict(r2_ablation=r2, r2_ablation_random=r2_random, r2_ablation_full=r2_full,
                ablation_measures=np.array(ABLATION_MEASURES), fractions=np.array(fractions, dtype=float))


def run(pipeline, targets=(0, 1, 2), n_seeds=2, n_subsamples=3, per_region=30, channel_step=1,
        ablation_targets=(0, 1), ablation_channel_step=2, fractions=(0.05, 0.2), n_random=2, skip=()):
    n_ch = pipeline.active_lfp.lfpMat.shape[1]
    channels = pooled_channels(pipeline.session_id, n_ch)
    rng = np.random.default_rng(config.TORCH_SEED)
    out = dict(channels=np.array(channels[::channel_step], int), ablation_channels=np.array(channels[::ablation_channel_step], int),
               groups=np.array(GROUPS))
    if 'seeds' not in skip:
        out.update(seeds(pipeline, channels[::channel_step], list(targets), n_seeds))
    if 'subsamples' not in skip:
        out.update(subsamples(pipeline, channels[::channel_step], list(targets), n_subsamples, per_region, rng))
    if 'ablation' not in skip:
        out.update(ablation(pipeline, channels[::ablation_channel_step], list(ablation_targets), list(fractions),
                            n_seeds, n_random, rng))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1,2')
    parser.add_argument('--seeds', type=int, default=2, help='seeds besides the saved seed-0 model')
    parser.add_argument('--subsamples', type=int, default=3)
    parser.add_argument('--per-region', type=int, default=30)
    parser.add_argument('--channel-step', type=int, default=1)
    parser.add_argument('--ablation-targets', default='0,1')
    parser.add_argument('--ablation-channel-step', type=int, default=2)
    parser.add_argument('--fractions', default='0.05,0.2')
    parser.add_argument('--n-random', type=int, default=2)
    parser.add_argument('--skip', default='', help='comma-separated parts to leave out: seeds, subsamples, ablation')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, ra.parse_list(args.targets), args.seeds, args.subsamples, args.per_region, args.channel_step,
              ra.parse_list(args.ablation_targets), args.ablation_channel_step, ra.parse_list(args.fractions, float),
              args.n_random, [s for s in args.skip.split(',') if s])
    f = pipeline.dirs.variables / 'robustness.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
