"""
Does attribution identify the units the model depends on better than simpler measures do?
(reviewer 2, comment 1; reviewer 1, "simpler methods"; reviewer 3, comment 8)

Units are ranked separately for every channel x model target by each measure in MEASURES, all
computed on the training block (analysis/coupling_measures.py; attribution from the 'summaries'
stage's training-block targets):

  attribution            mean |attribution| (total contribution)
  attribution_per_spike  |attribution per spike|
  stlfp                  max |spike-triggered target| over lags 0-200 ms
  spike_lfp_correlation  max |correlation of counts with the target| over lags 0-200 ms
  population_coupling    correlation with the summed activity of the other units
  firing_rate
  ridge_weight           norm of the unit's weights in the causal lagged ridge (units variant)

and the top FRACTIONS of units are removed on the test block. A measure that ranks the units the
predictions depend on first makes test R2 fall fastest.

Inference-time ablation (the session's seed-0 models, every channel x --targets): the removed units'
inputs are replaced by their training mean ('mean'; removes their timing, keeps their average drive)
or by 0 ('zero'; the IG baseline). Random orderings give the chance curve.
Retrained ablation (--retrain-targets, default broadband): the standard model is retrained without
the removed units (fit_and_score, --retrain-seeds seeds), for --retrain-fractions.

usage (from the repo root):
    python -m experiments.ranked_ablation <session_dir> [--targets 0,1,2,3,4] [--n-random 5]
        [--retrain-targets 0 | none] [--retrain-fractions 0.05,0.2] [--retrain-seeds 1]

Needs the 'train' and 'summaries' stages. Saves variables/<session_id>/ranked_ablation.npz
(axes in the order of MEASURES, FILLS and FRACTIONS; targets not run are NaN):
  r2_full                    channels x targets, test R2 of the unablated model
  r2_ablated                 fills x measures x fractions x channels x targets
  r2_random                  fills x random orderings x fractions x channels x targets
  spearman_with_attribution  measures x channels x targets, rank correlation of each measure with attribution
  r2_retrained               measures x retrain fractions x seeds x channels x targets
  r2_retrained_random        random orderings x retrain fractions x seeds x channels x targets
  r2_retrained_full          seeds x channels x targets (retrained on all units)
  measures, fills, fractions, retrain_fractions
and the per-unit scores to variables/<session_id>/ranked_ablation_measures.npz (units x channels x targets each).
"""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import config
import paths
from analysis import coupling_measures as cm
from analysis.lagged_ridge import fit_select, raised_cosine_basis
from preprocess_data import take_chunks
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

MEASURES = ['attribution', 'attribution_per_spike', 'stlfp', 'spike_lfp_correlation', 'population_coupling',
            'firing_rate', 'ridge_weight']
FILLS = ['mean', 'zero']
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
STA_LAGS = np.arange(0, 51, 2)          # bins (0-200 ms in 8 ms steps)


def n_removed(fraction, n_units):
    return max(1, int(round(fraction * n_units)))


def unit_scores(pipeline, targets):
    """Every measure's per-unit score: {measure: units x channels x targets}, NaN for targets not in `targets`."""
    counts, X = pipeline.active_counts.spkMat, pipeline.active_spikes.spkMat
    Y3 = pipeline.active_lfp.lfpMat
    n_bins, n_ch, n_tg = Y3.shape
    n_units = counts.shape[1]
    train = chunk_bins(pipeline.split['train'])
    Y = Y3[:, :, targets].reshape(n_bins, n_ch * len(targets))

    def per_model(x):                    # (units x n_ch * len(targets)) -> units x channels x all targets
        out = np.full((n_units, n_ch, n_tg), np.nan)
        out[:, :, targets] = x.reshape(n_units, n_ch, len(targets))
        return out

    def per_unit(x):
        return per_model(np.repeat(np.asarray(x)[:, None], n_ch * len(targets), axis=1))

    scores = {}
    attr = np.load(pipeline.dirs.attrs / 'unit_attribution_train.npz')      # channels x targets x units
    scores['attribution'] = np.moveaxis(attr['mean_abs'], 2, 0)
    scores['attribution_per_spike'] = np.abs(np.moveaxis(attr['per_spike'], 2, 0))
    log('spike-triggered targets and correlations')
    lp = cm.lagged_products(counts, Y, train, STA_LAGS)
    scores['stlfp'] = per_model(cm.max_abs_over_lags(cm.spike_triggered_target(lp)))
    scores['spike_lfp_correlation'] = per_model(cm.max_abs_over_lags(cm.spike_lfp_correlation(lp)))
    scores['population_coupling'] = per_unit(cm.population_coupling(counts, train))
    scores['firing_rate'] = per_unit(cm.firing_rate(counts, train, config.BIN_SIZE))
    log('lagged ridge')
    n_lags = int(round(config.RIDGE_HISTORY / config.BIN_SIZE))
    basis = raised_cosine_basis(n_lags, config.RIDGE_N_BASIS)
    step = config.RIDGE_TRAIN_STEP
    val = chunk_bins(pipeline.split['val'])
    _, W, _, _ = fit_select(X, Y, basis, train[::step], val[::step], config.RIDGE_ALPHAS)
    scores['ridge_weight'] = per_model(cm.ridge_weight_norms(W, basis.shape[1]))
    return scores


def orders(scores, chan, band, n_random, rng, measures=MEASURES):
    """Unit orderings for one model: measures (most important first), then n_random random ones."""
    ranked = [cm.rank_order(scores[m][:, chan, band], rng) for m in measures]
    n_units = len(ranked[0])
    return ranked, [rng.permutation(n_units) for _ in range(n_random)]


def inference_ablation(pipeline, scores, targets, n_random, rng, measures=MEASURES):
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch, n_tg, n_units = Y.shape[1], Y.shape[2], X.shape[1]
    split = pipeline.split
    fill_values = {'mean': torch.as_tensor(X[chunk_bins(split['train'])].mean(axis=0), dtype=torch.float32),
                   'zero': torch.zeros(n_units)}
    X_test = torch.as_tensor(take_chunks(X, split['test']), dtype=torch.float32)
    y_test = take_chunks(Y, split['test']).reshape(-1, n_ch, n_tg)
    r2_full = np.full((n_ch, n_tg), np.nan)
    r2_ablated = np.full((len(FILLS), len(measures), len(FRACTIONS), n_ch, n_tg), np.nan)
    r2_random = np.full((len(FILLS), n_random, len(FRACTIONS), n_ch, n_tg), np.nan)

    for band in targets:
        for chan in range(n_ch):
            model, y = pipeline.models[chan, band], y_test[:, chan, band]
            r2_full[chan, band] = r2_score(y, model.evaluate(X_test).numpy())
            ranked, random_orders = orders(scores, chan, band, n_random, rng, measures)
            for f, fill in enumerate(FILLS):
                for out, ords in ((r2_ablated, ranked), (r2_random, random_orders)):
                    for i, order in enumerate(ords):
                        for k, fraction in enumerate(FRACTIONS):
                            removed = torch.as_tensor(order[:n_removed(fraction, n_units)])
                            kept = X_test[..., removed].clone()       # replaced in place, then restored
                            X_test[..., removed] = fill_values[fill][removed]
                            out[f, i, k, chan, band] = r2_score(y, model.evaluate(X_test).numpy())
                            X_test[..., removed] = kept
            k = len(FRACTIONS) // 2
            log(f'chan {chan} band {band}: full {r2_full[chan, band]:.3f}, top {FRACTIONS[k]:.0%} removed (mean fill) '
                + ', '.join(f'{m} {r2_ablated[0, i, k, chan, band]:.3f}' for i, m in enumerate(measures)))
    return r2_full, r2_ablated, r2_random


def retrained_ablation(pipeline, scores, targets, fractions, n_seeds, n_random, rng, measures=MEASURES):
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch, n_tg, n_units = Y.shape[1], Y.shape[2], X.shape[1]
    r2 = np.full((len(measures), len(fractions), n_seeds, n_ch, n_tg), np.nan)
    r2_random = np.full((n_random, len(fractions), n_seeds, n_ch, n_tg), np.nan)
    r2_full = np.full((n_seeds, n_ch, n_tg), np.nan)
    for band in targets:
        for chan in range(n_ch):
            y = Y[:, chan, band]
            ranked, random_orders = orders(scores, chan, band, n_random, rng, measures)
            for seed in range(n_seeds):
                r2_full[seed, chan, band] = pipeline.fit_and_score(X, y, seed)[1]
                for out, ords in ((r2, ranked), (r2_random, random_orders)):
                    for i, order in enumerate(ords):
                        for k, fraction in enumerate(fractions):
                            keep = np.sort(order[n_removed(fraction, n_units):])
                            out[i, k, seed, chan, band] = pipeline.fit_and_score(X[:, keep], y, seed)[1]
            log(f'retrained chan {chan} band {band}: full {r2_full[:, chan, band].mean():.3f}')
    return r2, r2_random, r2_full


def spearman_with_attribution(scores):
    n_units, n_ch, n_tg = scores['attribution'].shape
    out = np.full((len(MEASURES), n_ch, n_tg), np.nan)
    for i, m in enumerate(MEASURES):
        for chan in range(n_ch):
            for band in range(n_tg):
                a, b = scores['attribution'][:, chan, band], scores[m][:, chan, band]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 2 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
                    out[i, chan, band] = spearmanr(a[ok], b[ok])[0]
    return out


def run(pipeline, targets=(0, 1, 2, 3, 4), n_random=5, retrain_targets=(0,), retrain_fractions=(0.05, 0.2),
        retrain_seeds=1):
    targets, retrain_targets = list(targets), list(retrain_targets)
    rng = np.random.default_rng(config.TORCH_SEED)
    scores = unit_scores(pipeline, sorted(set(targets) | set(retrain_targets)))
    r2_full, r2_ablated, r2_random = inference_ablation(pipeline, scores, targets, n_random, rng)
    out = dict(r2_full=r2_full, r2_ablated=r2_ablated, r2_random=r2_random,
               spearman_with_attribution=spearman_with_attribution(scores),
               measures=np.array(MEASURES), fills=np.array(FILLS), fractions=np.array(FRACTIONS),
               retrain_fractions=np.array(retrain_fractions, dtype=float))
    if retrain_targets:
        r2, r2_rand, r2_full_re = retrained_ablation(pipeline, scores, retrain_targets, list(retrain_fractions),
                                                     retrain_seeds, max(1, n_random // 2), rng)
        out.update(r2_retrained=r2, r2_retrained_random=r2_rand, r2_retrained_full=r2_full_re)
    return out, scores


def parse_list(text, kind=int):
    return [] if text == 'none' else [kind(x) for x in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1,2,3,4', help='model targets for inference-time ablation')
    parser.add_argument('--n-random', type=int, default=5, help='random orderings (half as many when retraining)')
    parser.add_argument('--retrain-targets', default='0', help="model targets for retrained ablation, or 'none'")
    parser.add_argument('--retrain-fractions', default='0.05,0.2')
    parser.add_argument('--retrain-seeds', type=int, default=1)
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out, scores = run(pipeline, parse_list(args.targets), args.n_random, parse_list(args.retrain_targets),
                      parse_list(args.retrain_fractions, float), args.retrain_seeds)
    v = pipeline.dirs.variables
    save_npz(v / 'ranked_ablation.npz', **out)
    save_npz(v / 'ranked_ablation_measures.npz', compressed=True, **{m: s.astype(np.float32) for m, s in scores.items()})
    print('Saved', v / 'ranked_ablation.npz')


if __name__ == '__main__':
    main()
