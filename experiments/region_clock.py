"""
Evoked vs induced by region (reviewer 1; REVISION_STATUS section E): which regions carry the
trial-specific part of the prediction, and which only the stimulus-locked part?

On the test block, with the session's seed-0 models (every channel x --targets) and the stimulus
template of experiments/stimulus_clock.py (residual = data minus template):

  r2_model, r2_residual                  all units as recorded (residual R2: fraction of the variance
                                         left by the template that the prediction explains)
  r2_removed, r2_residual_removed        the region's units replaced by their training mean
  r2_swap, r2_residual_swap              only the region's spikes swapped between test presentations of
                                         the same condition (shuffles x ...): its stimulus-locked content
                                         is kept, its single-trial pairing with the target broken. The
                                         R2 lost is the region's trial-specific contribution.
  r2_swap_all, r2_residual_swap_all      every unit swapped (as stimulus_clock's r2_shuffle)

Region shares of the per-unit measures (summed over the region's units / over all units; channels x
targets), where the files exist:
  share_attribution_mean_baseline        mean |attribution|, mean baseline (attribution_baseline.npz)
  share_<measure>                        spike_lfp_correlation, stlfp, ridge_weight
                                         (ranked_ablation_measures.npz)

Regions as in experiments/region_ablation.py (REGIONS; NaN where the session has no units).

usage (from the repo root):
    python -m experiments.region_clock <session_dir> [--targets 0,1,2,3,4] [--n-shuffles 3]

Needs the 'train' stage. Saves variables/<session_id>/region_clock.npz: the R2s above as
(regions x) channels x targets, swaps with a leading shuffles axis; regions, region_n_units.
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.stimulus_clock import r2, residual_r2, shuffle_presentations
from experiments.region_ablation import REGIONS, region_masks
from experiments.stimulus_clock import clock, predict
from run_session import SessionPipeline, log, save_npz

SHARE_MEASURES = ['spike_lfp_correlation', 'stlfp', 'ridge_weight']


def region_shares(scores, masks):
    """Share of each region in a per-unit score (units x channels x targets) -> regions x channels x targets."""
    total = np.nansum(np.abs(scores), axis=0)
    return np.stack([np.nansum(np.abs(scores[masks[r]]), axis=0) / total if masks[r].any()
                     else np.full(total.shape, np.nan) for r in REGIONS])


def shares(variables, masks):
    out = {}
    f = variables / 'attribution_baseline.npz'
    if f.exists():
        out['share_attribution_mean_baseline'] = region_shares(np.moveaxis(np.load(f)['mean_abs'], -1, 0), masks)
    f = variables / 'ranked_ablation_measures.npz'
    if f.exists():
        d = np.load(f)
        out.update({f'share_{m}': region_shares(d[m], masks) for m in SHARE_MEASURES if m in d})
    return out


def run(pipeline, targets=(0, 1, 2, 3, 4), n_shuffles=3, seed=config.TORCH_SEED):
    rng = np.random.default_rng(seed)
    c = clock(pipeline)
    X, Y, T, test_bins, segments, seg_cond = c['X'], c['Y'], c['T'], c['test_bins'], c['segments'], c['seg_cond']
    masks = region_masks(pipeline.session.units['structure_acronym'].to_numpy().astype(str))
    present = [i for i, r in enumerate(REGIONS) if masks[r].any()]
    n_ch, n_chunks = Y.shape[1], len(pipeline.split['test'])
    Xt, Yt, Tt = X[test_bins], Y[test_bins], T[test_bins]
    train_mean = X[c['train_bins']].mean(axis=0)

    shape = (n_ch, config.N_TARGETS)
    out = {k: np.full(shape, np.nan) for k in ('r2_model', 'r2_residual')}
    out.update({k: np.full((len(REGIONS),) + shape, np.nan) for k in ('r2_removed', 'r2_residual_removed')})
    out.update({k: np.full((n_shuffles, len(REGIONS)) + shape, np.nan) for k in ('r2_swap', 'r2_residual_swap')})
    out.update({k: np.full((n_shuffles,) + shape, np.nan) for k in ('r2_swap_all', 'r2_residual_swap_all')})

    def score(key, index, Xs, chan, band, model):
        yhat = predict(model, Xs, np.arange(n_chunks))
        y, tmpl = Yt[:, chan, band], Tt[:, chan, band]
        out[f'r2_{key}'][index + (chan, band)] = r2(y, yhat)
        out[f'r2_residual_{key}'][index + (chan, band)] = residual_r2(y, yhat, tmpl)

    swapped = [shuffle_presentations(Xt, segments, seg_cond, rng) for _ in range(n_shuffles)]
    for chan, band in sorted(k for k in pipeline.models if k[1] in targets):
        model = pipeline.models[chan, band]
        y, tmpl = Yt[:, chan, band], Tt[:, chan, band]
        yhat = predict(model, Xt, np.arange(n_chunks))
        out['r2_model'][chan, band], out['r2_residual'][chan, band] = r2(y, yhat), residual_r2(y, yhat, tmpl)
        for i, Xs in enumerate(swapped):
            score('swap_all', (i,), Xs, chan, band, model)
        for r in present:
            mask = masks[REGIONS[r]]
            Xr = Xt.copy()
            Xr[:, mask] = train_mean[mask]
            score('removed', (r,), Xr, chan, band, model)
            for i, Xs in enumerate(swapped):
                Xr = Xt.copy()
                Xr[:, mask] = Xs[:, mask]
                score('swap', (i, r), Xr, chan, band, model)
        lost = out['r2_residual'][chan, band] - np.nanmean(out['r2_residual_swap'][:, :, chan, band], 0)
        log(f'chan {chan} band {band}: R2 {out["r2_model"][chan, band]:.3f}, residual R2 {out["r2_residual"][chan, band]:.3f}; '
            'residual R2 lost by swapping ' + ', '.join(f'{REGIONS[r]} {lost[r]:.3f}' for r in present[:6]))

    out.update(shares(pipeline.dirs.variables, masks))
    out.update(regions=np.array(REGIONS), region_n_units=np.array([masks[r].sum() for r in REGIONS]),
               n_test_presentations=np.array(len(segments)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1,2,3,4')
    parser.add_argument('--n-shuffles', type=int, default=3)
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, [int(t) for t in args.targets.split(',')], args.n_shuffles)
    f = pipeline.dirs.variables / 'region_clock.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
