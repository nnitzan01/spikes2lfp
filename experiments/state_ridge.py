"""
Model-independence checks of the brain-state results of region_events (REVISION_STATUS 2026-11-24: the midbrain's
share of the 0.5-4 Hz prediction falls with running and pupil in the CNN and the CNN-LSTM). CPU only.

At the layer 4 channel, for --targets (default broadband and 0.5-4 Hz), over the active epoch and the passive replay:

  ridge     the causal lagged ridge on all units (analysis/lagged_ridge.py, as baselines.py 'units'), split exactly
            into region-group contributions: for a linear model with the training feature mean as baseline,
            c_g(t) = sum over the group's lag features of the standardised feature times its weight (the Shapley
            value of the group), and the prediction is y_mean + sum_g c_g
  rate      model free: each group's population rate (sum of its scaled units, causal RECTANGLE_S boxcar)

Both are reduced to the same per-state sums as region_events.state_stats (region_events.accumulate; levels are
deciles of running and pupil over both epochs, engaged 0 / 1): for 'ridge' with the ridge prediction as p,
for 'rate' with the target as p (so Cov(c, p) is the covariance of the group rate with the target).

usage (from the repo root):
    python -m experiments.state_ridge <session_dir> [--targets 0,1]

Needs the 'train' stage only for the split and scalers. Saves variables/<session_id>/state_ridge.npz.
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.lagged_ridge import fit_select, lagged_features, r2_columns, raised_cosine_basis
from experiments.explore_task import clean_pupil, layer4_channel
from experiments.region_events import (GROUP_STATS, GROUPS, N_LEVELS, STATE_EPOCHS, STATE_VARS, STATS, accumulate,
                                       behaviour_on_bins, epoch_data, state_levels)
from experiments.region_ablation import region_masks
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

RECTANGLE_S = 0.25                 # causal smoothing of the group rates (the 0.5-4 Hz envelope's timescale)
COLUMNS_PER_PASS = 16              # units per block when building lag features over a whole epoch


def group_contributions(path, W, Z, basis, unit_groups):
    """Exact ridge contributions (groups x bins x targets) over every bin of Z."""
    n_basis, rows = basis.shape[1], np.arange(len(Z))
    out = np.zeros((len(unit_groups), len(Z), W.shape[1]), np.float32)
    owner = np.full(Z.shape[1], -1)
    for g, m in enumerate(unit_groups):
        owner[m] = g
    for a in range(0, Z.shape[1], COLUMNS_PER_PASS):
        units = np.arange(a, min(Z.shape[1], a + COLUMNS_PER_PASS))
        F = lagged_features(Z[:, units], basis, rows)
        for k, u in enumerate(units):
            if owner[u] < 0:
                continue
            cols = u * n_basis + np.arange(n_basis)
            out[owner[u]] += ((F[:, k * n_basis:(k + 1) * n_basis] - path.mean[cols]) / path.scale[cols]) @ W[cols]
    return out


def causal_rate(Z, unit_groups):
    """Population rate per group (groups x bins), causal boxcar of RECTANGLE_S."""
    n = max(1, int(round(RECTANGLE_S / config.BIN_SIZE)))
    rates = np.stack([Z[:, m].sum(1) for m in unit_groups]).astype(np.float64)
    c = np.cumsum(rates, axis=1)
    c[:, n:] -= c[:, :-n].copy()                                          # window ends at the current bin
    return c / n


def run(pipeline, targets=(0, 1)):
    s = pipeline.session
    targets = list(targets)
    masks = region_masks(np.asarray(s.units.structure_acronym))
    present = [i for i, g in enumerate(GROUPS) if masks[g].any()]
    unit_groups = [masks[GROUPS[i]] for i in present]
    n_ch = pipeline.n_channels
    l4, _ = layer4_channel(pipeline.session_id, n_ch)

    split = pipeline.split
    train, val, test = (chunk_bins(split[p]) for p in ('train', 'val', 'test'))
    data = {e: epoch_data(pipeline, e) for e in ('active', 'passive')}
    Z, Y3, _ = data['active']
    Y = np.asarray(Y3[:, l4][:, targets], np.float64)
    basis = raised_cosine_basis(int(round(config.RIDGE_HISTORY / config.BIN_SIZE)), config.RIDGE_N_BASIS)
    step = config.RIDGE_TRAIN_STEP
    path, W, alpha, _ = fit_select(Z, Y, basis, train[::step], val[::step], config.RIDGE_ALPHAS)
    log(f'ridge on {Z.shape[1]} units, layer 4 channel {l4}, alpha {alpha}')

    eye = s.eye_tracking
    pupil = None
    if eye is not None and len(eye):
        t = eye.timestamps.to_numpy(float)
        pupil = (t, clean_pupil(t, eye.pupil_area.to_numpy(float), eye.likely_blink if 'likely_blink' in eye else None)['clean'])
    behaviour = {e: behaviour_on_bins(s, ts + config.BIN_SIZE / 2, pupil) for e, (_, _, ts) in data.items()}
    edges = {v: np.nanquantile(np.concatenate([behaviour[e][v] for e in data]), np.linspace(0, 1, N_LEVELS + 1)[:-1])
             for v in ('running', 'pupil')}
    edges = {v: (e if np.isfinite(e).all() else np.zeros(N_LEVELS)) for v, e in edges.items()}
    levels = {e: state_levels(behaviour[e], edges) for e in data}
    levels['passive']['engaged'][:] = -1

    n_stats = len(STATS) + len(GROUP_STATS) * len(present)
    shape = (len(STATE_EPOCHS), len(STATE_VARS), N_LEVELS, n_stats, len(targets))
    stats = {k: np.zeros(shape) for k in ('ridge', 'rate')}
    r2_test = None
    for e, (X, Yall, _) in data.items():
        y = np.asarray(Yall[:, l4][:, targets], np.float64)
        n = min(len(X), len(y))
        contrib = group_contributions(path, W, X[:n], basis, unit_groups)               # groups x bins x targets
        p = path.y_mean + contrib.sum(0)
        rates = causal_rate(X[:n], unit_groups)
        if e == 'active':
            r2_test = r2_columns(y[test], p[test])
        for j in range(len(targets)):
            lev = {v: levels[e][v][:n] for v in STATE_VARS}
            for kind, pp, c in (('ridge', p[:, j], contrib[:, :, j]), ('rate', y[:n, j], rates)):
                accumulate(stats[kind][STATE_EPOCHS.index(e), ..., j], lev, y[:n, j], pp, c)
                if e == 'active':
                    in_test = (np.arange(n) >= test[0]) & (np.arange(n) <= test[-1])
                    accumulate(stats[kind][STATE_EPOCHS.index('active_test'), ..., j],
                               {v: np.where(in_test, lev[v], -1) for v in STATE_VARS}, y[:n, j], pp, c)
        log(f'{e}: ridge test R2 {np.round(r2_test, 3)}; contributions of {len(present)} groups over {n} bins')

    out = {}
    for kind, st in stats.items():       # group sums at the full GROUPS positions (NaN for absent groups)
        full = np.full(st.shape[:3] + (len(STATS) + len(GROUP_STATS) * len(GROUPS),) + st.shape[4:], np.nan)
        full[:, :, :, :len(STATS)] = st[:, :, :, :len(STATS)]
        for j, i in enumerate(present):
            for k in range(len(GROUP_STATS)):
                full[:, :, :, len(STATS) + i * len(GROUP_STATS) + k] = st[:, :, :, len(STATS) + j * len(GROUP_STATS) + k]
        out[f'{kind}_stats'] = full
    out.update(groups=np.array(GROUPS), group_n_units=np.array([masks[g].sum() for g in GROUPS]), targets=np.array(targets),
               layer4_channel=l4, ridge_alpha=alpha, ridge_r2_test=r2_test, state_epochs=np.array(STATE_EPOCHS),
               state_vars=np.array(STATE_VARS), stat_names=np.array(STATS + [f'{g}_{k}' for g in GROUPS for k in GROUP_STATS]),
               state_edges=np.stack([edges['running'], edges['pupil']]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1')
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    pipeline = SessionPipeline(data_root, session_id, torch.device('cpu'))
    out = run(pipeline, [int(t) for t in args.targets.split(',')])
    f = pipeline.dirs.variables / 'state_ridge.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
