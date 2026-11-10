"""
Region removal (reviewer 3, comment 5): how much held-out performance is lost when each region's
units are removed, and does it survive when stimulus timing, behaviour and population-wide activity
are given to the model?

Regions are a fixed list (REGIONS: the notebook's global regions plus single areas of interest),
so sessions can be pooled; a region with no units in the session is NaN.

Inference-time (the session's seed-0 models, every channel x --targets, test block): the region's
units are replaced by their training mean ('mean') or by 0 ('zero'):
  r2_removed   the region removed
  r2_only      every unit outside the region removed (what the region predicts on its own,
               through a model trained on everyone)
Retrained (--retrain-targets, default broadband; --retrain-seeds): the standard model is trained
without the region (fit_and_score):
  r2_retrained_full, r2_retrained_removed
and again with COVARIATES as extra inputs (stimulus onsets per image / change / omission, running,
pupil, licks, rewards, and the summed activity of all units):
  r2_cov_full, r2_cov_removed
A region whose removal still costs R2 with the covariates present carries information beyond
stimulus timing, arousal and global activity. --retrain-regions sets which regions are retrained
(default: the global regions).

usage (from the repo root):
    python -m experiments.region_ablation <session_dir> [--targets 0,1,2,3,4] [--retrain-targets 0 | none]
        [--retrain-regions groups | all | VISp,thalamus,...] [--retrain-seeds 1] [--no-covariates]
        [--output region_ablation]

Needs the 'train' stage (the covariates are computed on first use). Saves
variables/<session_id>/<--output>.npz (region axes in the order of REGIONS, NaN where not run):
  r2_full (channels x targets), r2_removed / r2_only (fills x regions x channels x targets),
  r2_retrained_full / r2_cov_full (seeds x channels x targets),
  r2_retrained_removed / r2_cov_removed (regions x seeds x channels x targets),
  regions, region_n_units, fills, covariate_names
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import r2_score

import config
import paths
from experiments.baselines import behaviour_inputs
from preprocess_data import SpikeScaler, take_chunks
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

GLOBAL_REGIONS = {
    'VISp': ['VISp'],
    'visual_cortex_other': ['VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'VISli', 'VISpl', 'VISpor', 'VISmma', 'VISmmp'],
    'hippocampus': ['CA1', 'CA3', 'DG', 'ProS', 'SUB', 'POST'],
    'thalamus': ['LG', 'LP', 'LT', 'MG', 'POL', 'PIL', 'PoT'],
    'midbrain': ['APN', 'MB', 'NOT', 'NB', 'MRN', 'SC', 'SGN', 'ZI'],
}
AREAS = ['VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'LG', 'LP', 'CA1', 'CA3', 'DG', 'SUB', 'POST',
         'APN', 'MRN', 'SC', 'NOT']
REGIONS = list(GLOBAL_REGIONS) + ['other'] + AREAS
FILLS = ['mean', 'zero']


def area_name(acronym):
    """Session acronym as used in REGIONS (SC subdivisions pooled; LG and MG are pooled by process_session)."""
    return 'SC' if acronym.startswith('SC') else acronym


def region_masks(acronyms):
    """{region: boolean unit mask} for every region in REGIONS."""
    areas = np.array([area_name(a) for a in acronyms])
    masks = {name: np.isin(areas, members) for name, members in GLOBAL_REGIONS.items()}
    masks['other'] = ~np.any(list(masks.values()), axis=0)
    masks.update({a: areas == a for a in AREAS})
    return masks


def covariate_inputs(pipeline):
    """Covariates on the active epoch's bins (bins x covariates) and their names."""
    counts = pipeline.active_counts.spkMat
    n_bins = counts.shape[0]
    train = chunk_bins(pipeline.split['train'])
    saved = pipeline.load_covariates('active')
    stimulus, behaviour = saved['stimulus'], behaviour_inputs(pipeline, train)
    names = [*stimulus.columns, *saved['behaviour'].columns[saved['behaviour'].notna().any()], 'population']
    population = counts.sum(axis=1, keepdims=True).astype(np.float32)
    population = SpikeScaler().fit(population[train]).transform(population)      # scaled like a unit's input
    Z = np.hstack([stimulus.to_numpy(np.float32)[:n_bins], behaviour[:n_bins].astype(np.float32), population])
    return Z, names


def inference_ablation(pipeline, masks, targets):
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch, n_tg, n_units = Y.shape[1], Y.shape[2], X.shape[1]
    split = pipeline.split
    fill_values = {'mean': torch.as_tensor(X[chunk_bins(split['train'])].mean(axis=0), dtype=torch.float32),
                   'zero': torch.zeros(n_units)}
    X_test = torch.as_tensor(take_chunks(X, split['test']), dtype=torch.float32)
    y_test = take_chunks(Y, split['test']).reshape(-1, n_ch, n_tg)
    r2_full = np.full((n_ch, n_tg), np.nan)
    r2_removed = np.full((len(FILLS), len(REGIONS), n_ch, n_tg), np.nan)
    r2_only = np.full_like(r2_removed, np.nan)

    def score(model, y, removed, fill):
        removed = torch.as_tensor(np.flatnonzero(removed))
        kept = X_test[..., removed].clone()                 # replaced in place, then restored
        X_test[..., removed] = fill_values[fill][removed]
        r2 = r2_score(y, model.evaluate(X_test).numpy())
        X_test[..., removed] = kept
        return r2

    for band in targets:
        for chan in range(n_ch):
            model, y = pipeline.models[chan, band], y_test[:, chan, band]
            r2_full[chan, band] = r2_score(y, model.evaluate(X_test).numpy())
            for f, fill in enumerate(FILLS):
                for r, region in enumerate(REGIONS):
                    mask = masks[region]
                    if mask.any():
                        r2_removed[f, r, chan, band] = score(model, y, mask, fill)
                        r2_only[f, r, chan, band] = score(model, y, ~mask, fill)
            loss = r2_full[chan, band] - r2_removed[0, :, chan, band]
            log(f'chan {chan} band {band}: full {r2_full[chan, band]:.3f}; R2 lost without '
                + ', '.join(f'{g} {loss[REGIONS.index(g)]:.3f}' for g in GLOBAL_REGIONS))
    return r2_full, r2_removed, r2_only


def retrained_ablation(pipeline, masks, targets, regions, n_seeds, Z=None):
    """Retrained R2 on all units and without each region in `regions`, with covariates Z appended if given."""
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat
    n_ch, n_tg = Y.shape[1], Y.shape[2]
    full = np.full((n_seeds, n_ch, n_tg), np.nan)
    removed = np.full((len(REGIONS), n_seeds, n_ch, n_tg), np.nan)

    def inputs(keep):
        return X[:, keep] if Z is None else np.hstack([X[:, keep], Z[:len(X)]])

    for band in targets:
        for chan in range(n_ch):
            y = Y[:, chan, band]
            for seed in range(n_seeds):
                full[seed, chan, band] = pipeline.fit_and_score(inputs(np.ones(X.shape[1], bool)), y, seed)[1]
                for region in regions:
                    mask = masks[region]
                    if mask.any() and not mask.all():
                        removed[REGIONS.index(region), seed, chan, band] = pipeline.fit_and_score(inputs(~mask), y, seed)[1]
            log(f'retrained{" with covariates" if Z is not None else ""} chan {chan} band {band}: '
                f'full {full[:, chan, band].mean():.3f}')
    return full, removed


def run(pipeline, targets=(0, 1, 2, 3, 4), retrain_targets=(0,), retrain_regions=None, retrain_seeds=1,
        covariates=True):
    masks = region_masks(pipeline.session.units['structure_acronym'].to_numpy().astype(str))
    retrain_regions = list(GLOBAL_REGIONS) + ['other'] if retrain_regions is None else list(retrain_regions)
    r2_full, r2_removed, r2_only = inference_ablation(pipeline, masks, list(targets))
    out = dict(r2_full=r2_full, r2_removed=r2_removed, r2_only=r2_only, regions=np.array(REGIONS),
               region_n_units=np.array([masks[r].sum() for r in REGIONS]), fills=np.array(FILLS))
    if retrain_targets:
        out['r2_retrained_full'], out['r2_retrained_removed'] = retrained_ablation(
            pipeline, masks, list(retrain_targets), retrain_regions, retrain_seeds)
        if covariates:
            Z, names = covariate_inputs(pipeline)
            out['r2_cov_full'], out['r2_cov_removed'] = retrained_ablation(
                pipeline, masks, list(retrain_targets), retrain_regions, retrain_seeds, Z)
            out['covariate_names'] = np.array(names)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1,2,3,4', help='model targets for inference-time removal')
    parser.add_argument('--retrain-targets', default='0', help="model targets for retrained removal, or 'none'")
    parser.add_argument('--retrain-regions', default='groups', help="'groups' (global regions), 'all' or a list")
    parser.add_argument('--retrain-seeds', type=int, default=1)
    parser.add_argument('--no-covariates', action='store_true', help='skip the retraining with covariates')
    parser.add_argument('--output', default='region_ablation', help='saved as variables/<session_id>/<output>.npz')
    args = parser.parse_args()

    regions = {'groups': None, 'all': REGIONS}.get(args.retrain_regions, args.retrain_regions.split(','))
    unknown = set(regions or []) - set(REGIONS)
    if unknown:
        parser.error(f'unknown regions {sorted(unknown)}; choose from {REGIONS}')
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    targets = [int(t) for t in args.targets.split(',')]
    retrain = [] if args.retrain_targets == 'none' else [int(t) for t in args.retrain_targets.split(',')]
    out = run(pipeline, targets, retrain, regions, args.retrain_seeds, not args.no_covariates)
    f = pipeline.dirs.variables / f'{args.output}.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
