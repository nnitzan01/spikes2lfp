"""
Baselines for the CNN-LSTM (reviewers 1 and 3, comment 3): how well do simpler models predict the
same targets, on the same blocked split?

Lagged ridge (analysis/lagged_ridge.py; RIDGE_HISTORY s of history on RIDGE_N_BASIS raised-cosine
lags, penalty chosen on the validation block) on these inputs:
  units_instant              the model's input (scaled spikes of every unit), current bin only:
                             the first version's linear model, fitted properly
  units                      the same with the lag basis
  regions                    population spike count of each area
  stimulus                   onset impulses per image, change and omission (covariates.py): a
                             linear stimulus-clock model
  behaviour                  running speed, pupil area, licks, rewards (z-scored)
  stimulus_behaviour, regions_stimulus_behaviour, units_stimulus_behaviour   combinations
LSTM (the standard CNN-LSTM, fit_and_score) on:
  lstm_regions               population counts per area, min-max scaled with training statistics
  lstm_regions_behaviour     the same plus the behavioural covariates

usage (from the repo root):
    python -m experiments.baselines <session_dir> [--ridge all|units,regions,...] [--lstm regions,... | none]

Needs the 'train' and 'covariates' stages. Saves variables/<session_id>/baselines.npz:
r2_<variant> (channels x targets, test R2), alpha_<variant> (chosen penalty, ridge only),
n_inputs_<variant>, and model_r2 (the session's CNN-LSTM, for reference).
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.lagged_ridge import fit_select_predict, r2_columns, raised_cosine_basis
from preprocess_data import SpikeScaler
from run_session import SessionPipeline, save_npz
from splits import chunk_bins

RIDGE_VARIANTS = ['units_instant', 'units', 'regions', 'stimulus', 'behaviour', 'stimulus_behaviour',
                  'regions_stimulus_behaviour', 'units_stimulus_behaviour']
LSTM_VARIANTS = ['regions', 'regions_behaviour']


def region_counts(pipeline):
    """Population spike count per area (bins x areas) and the area names."""
    counts = pipeline.active_counts.spkMat
    areas = pipeline.session.units['structure_acronym'].to_numpy()
    names = np.unique(areas)
    return np.stack([counts[:, areas == a].sum(axis=1) for a in names], axis=1), names


def behaviour_inputs(pipeline, train_bins):
    """Behavioural covariates z-scored with training statistics; columns that are all NaN (no eye tracking) dropped."""
    b = pipeline.load_covariates('active')['behaviour']
    b = b.loc[:, b.notna().any()].fillna(0).to_numpy(np.float64)
    sd = b[train_bins].std(axis=0)
    return (b - b[train_bins].mean(axis=0)) / np.where(sd > 0, sd, 1)


def inputs(pipeline, variant, train_bins):
    parts = {'units': lambda: pipeline.active_spikes.spkMat,
             'regions': lambda: region_counts(pipeline)[0],
             'stimulus': lambda: pipeline.load_covariates('active')['stimulus'].to_numpy(np.float64),
             'behaviour': lambda: behaviour_inputs(pipeline, train_bins)}
    names = variant.replace('_instant', '').split('_')
    n_bins = pipeline.active_spikes.spkMat.shape[0]
    return np.hstack([np.asarray(parts[n]()[:n_bins], dtype=np.float32) for n in names])


def run_ridge(pipeline, variants=RIDGE_VARIANTS):
    split, Y3 = pipeline.split, pipeline.active_lfp.lfpMat
    n_bins, n_ch, n_tg = Y3.shape
    Y = Y3.reshape(n_bins, n_ch * n_tg)
    step = config.RIDGE_TRAIN_STEP
    train, val, test = (chunk_bins(split[p]) for p in ('train', 'val', 'test'))
    n_lags = int(round(config.RIDGE_HISTORY / config.BIN_SIZE))
    out = {}
    for variant in variants:
        Z = inputs(pipeline, variant, train)
        basis = np.ones((1, 1)) if variant.endswith('_instant') else raised_cosine_basis(n_lags, config.RIDGE_N_BASIS)
        prediction, alpha, _ = fit_select_predict(Z, Y, basis, train[::step], val[::step], test, config.RIDGE_ALPHAS)
        out[f'r2_{variant}'] = r2_columns(Y[test], prediction).reshape(n_ch, n_tg)
        out[f'alpha_{variant}'] = alpha.reshape(n_ch, n_tg)
        out[f'n_inputs_{variant}'] = np.array(Z.shape[1])
        print(f'ridge {variant}: {Z.shape[1]} inputs, median test R2 by target '
              f'{np.round(np.nanmedian(out[f"r2_{variant}"], axis=0), 3)}', flush=True)
    return out


def run_lstm(pipeline, variants=LSTM_VARIANTS):
    Y = pipeline.active_lfp.lfpMat
    train = chunk_bins(pipeline.split['train'])
    out = {}
    for variant in variants:
        Z = inputs(pipeline, variant, train).astype(np.float32)
        if variant.startswith('regions'):                       # scale counts like the unit inputs
            n_regions = len(region_counts(pipeline)[1])
            Z[:, :n_regions] = SpikeScaler().fit(Z[train, :n_regions]).transform(Z[:, :n_regions])
        r2 = np.full(Y.shape[1:], np.nan)
        for band in range(Y.shape[2]):
            for chan in range(Y.shape[1]):
                r2[chan, band] = pipeline.fit_and_score(Z, Y[:, chan, band])[1]
        out[f'r2_lstm_{variant}'] = r2
        out[f'n_inputs_lstm_{variant}'] = np.array(Z.shape[1])
        print(f'lstm {variant}: median test R2 by target {np.round(np.nanmedian(r2, axis=0), 3)}', flush=True)
    return out


def parse(text, allowed):
    if text == 'none':
        return []
    chosen = allowed if text == 'all' else text.split(',')
    unknown = set(chosen) - set(allowed)
    if unknown:
        raise ValueError(f'unknown variants {sorted(unknown)}; choose from {allowed}')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--ridge', default='all', help=f"'all', 'none' or a comma-separated subset of {RIDGE_VARIANTS}")
    parser.add_argument('--lstm', default='all', help=f"'all', 'none' or a comma-separated subset of {LSTM_VARIANTS}")
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run_ridge(pipeline, parse(args.ridge, RIDGE_VARIANTS))
    out.update(run_lstm(pipeline, parse(args.lstm, LSTM_VARIANTS)))
    model_r2 = pipeline.dirs.variables / 'r2_scores.npy'
    if model_r2.exists():
        out['model_r2'] = np.load(model_r2)
    f = pipeline.dirs.variables / 'baselines.npz'
    if f.exists():                                  # keep variants computed by earlier runs
        out = {**dict(np.load(f)), **out}
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
