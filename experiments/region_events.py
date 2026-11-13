"""
Region contributions to the task-evoked LFP (reviewer 1; REVISION_STATUS Phase 2 F, and the data for G and H).

The seed-0 prediction of every channel x --targets is split into the contributions of the region groups
(GROUPS: GLOBAL_REGIONS of experiments/region_ablation.py and 'other') by exact Shapley values over the
groups the session has: the model is run with every subset (coalition) of groups, the absent groups'
units replaced by their training mean, and

    phi_r(t) = sum over coalitions S without r of |S|! (n - |S| - 1)! / n! * (f_{S+r}(t) - f_S(t))

so that sum_r phi_r(t) = f_all(t) - f_none(t) at every bin (f_none: every unit at its mean). This is done
over the whole active epoch (training blocks included: in-sample) and the passive replay, and reduced to

  eta_*           event-locked averages (event types x lags x channels x targets), of the target (eta_target),
                  the prediction (eta_prediction), f_none (eta_base) and each group's contribution
                  (eta_contribution: types x lags x groups x channels x targets). Event types as in
                  experiments/explore_task.py (active and passive_); eta_*_test: the active types on the
                  test block only (n_events_test)
  state_stats     sums for brain-state analyses (H): epochs (STATE_EPOCHS) x state variables (STATE_VARS)
                  x levels x STATS (+ per group) x channels x targets; levels are deciles of the variable over
                  both epochs (state_edges); 'engaged' has levels 0 / 1, 'all' one level. From the sums:
                  Var(prediction), Cov(phi_r, prediction) (Shapley share = Cov / Var, summing to 1 over
                  groups), and R2, per state level

and an IG cross-check on the same events (--ig-events per type, 0 = off): at the layer 4 channel, for each
event type in IG_TYPES and offset IG_OFFSETS_MS, the prediction at that bin from its ATTR_HISTORY-bin window,
attributed by SequenceIG with the training-mean baseline (summed over each group's units and all lags) and
by exact Shapley values over the same window (ig_contribution, ig_shapley: types x offsets x groups x
targets; ig_completeness).

usage (from the repo root):
    python -m experiments.region_events <session_dir> [--targets 0,1] [--ig-events 100]

Needs the 'train' stage. Saves variables/<session_id>/region_events.npz.
"""
import argparse
import itertools
from math import factorial

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

import config
import paths
from analysis.sequence_attribution import window_index
from analysis.stimulus_clock import onset_bins
from experiments.attribution_baseline import attribute_with_retries
from experiments.explore_task import (ACTIVE_TYPES, ENGAGED_REWARDS_PER_MIN, ENGAGED_WINDOW_S, EVENT_WINDOW,
                                      auto_rewarded_changes, clean_pupil, event_average, event_rate,
                                      layer4_channel, on_grid, passive_presentations, task_events)
from experiments.region_ablation import GLOBAL_REGIONS, region_masks
from integrated_gradients import SequenceIG
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

GROUPS = list(GLOBAL_REGIONS) + ['other']
STATE_EPOCHS = ['active', 'active_test', 'passive']
STATE_VARS = ['all', 'running', 'pupil', 'engaged']
N_LEVELS = 10
STATS = ['n', 'y', 'yy', 'p', 'pp', 'yp']            # p = prediction; per group: c, cc, cp (c = contribution)
GROUP_STATS = ['c', 'cc', 'cp']
SMOOTH_S = 1.0                                         # running and pupil smoothing before binning states
IG_TYPES = ['image', 'change_hit', 'change_miss', 'omission']
IG_OFFSETS_MS = np.arange(-96, 504, 24)


# ---- Shapley ---------------------------------------------------------------------------------

def coalitions(n):
    """(2**n, n) boolean: coalition k contains player i if bit i of k is set (0 = empty, 2**n - 1 = all)."""
    return np.array([[(k >> i) & 1 for i in range(n)] for k in range(2 ** n)], bool)


def shapley_matrix(n):
    """(n, 2**n) W with phi = W @ f for values f over coalitions (exact Shapley values)."""
    members = coalitions(n)
    size = members.sum(1)
    W = np.zeros((n, 2 ** n))
    for i in range(n):
        for k in np.flatnonzero(~members[:, i]):
            w = factorial(size[k]) * factorial(n - size[k] - 1) / factorial(n)
            W[i, k | (1 << i)] += w
            W[i, k] -= w
    return W


def filled(X, absent, mean):
    """X with the units in the boolean mask `absent` set to their mean (copy)."""
    Xs = X.copy() if isinstance(X, np.ndarray) else X.clone()
    Xs[..., absent] = mean[absent]
    return Xs


def coalition_predictions(model, X, unit_groups, mean):
    """(2**n, bins) predictions over the continuous X with each coalition of the groups present."""
    members = coalitions(len(unit_groups))
    out = []
    for keep in members:
        absent = np.zeros(X.shape[1], bool)
        for g, k in zip(unit_groups, keep):
            absent |= g & ~k
        out.append(model.evaluate(filled(X, absent, mean)).numpy())
    return np.stack(out)


# ---- states ----------------------------------------------------------------------------------

def behaviour_on_bins(session, bin_times, pupil):
    """Running speed and pupil (both smoothed SMOOTH_S, pupil z-scored) and engagement (0 / 1) at the bin centres."""
    rs = session.running_speed
    dt = np.median(np.diff(rs.timestamps))
    running = on_grid(rs.timestamps, uniform_filter1d(np.asarray(rs.speed, float), max(1, int(round(SMOOTH_S / dt)))), bin_times)
    pz = np.full(len(bin_times), np.nan)
    if pupil is not None:
        t, clean = pupil
        dtp = np.median(np.diff(t))
        smooth = pd.Series(clean).rolling(max(1, int(round(SMOOTH_S / dtp))), center=True, min_periods=1).mean().to_numpy().copy()
        smooth[np.isnan(clean)] = np.nan
        pz = on_grid(t, (smooth - np.nanmean(smooth)) / np.nanstd(smooth), bin_times)
    engaged = (event_rate(session.rewards.timestamps, bin_times, ENGAGED_WINDOW_S) * 60 > ENGAGED_REWARDS_PER_MIN).astype(float)
    return dict(running=running, pupil=pz, engaged=engaged)


def state_levels(values, edges):
    """Level index per bin for each state variable (-1 = undefined)."""
    out = {'all': np.zeros(len(values['running']), int)}
    for v in ('running', 'pupil'):
        x = values[v]
        out[v] = np.where(np.isfinite(x), np.clip(np.searchsorted(edges[v], x, side='right') - 1, 0, N_LEVELS - 1), -1)
    out['engaged'] = values['engaged'].astype(int)
    return out


def accumulate(stats, levels, y, p, contrib):
    """Add one model's sums to stats (vars x levels x stats) for one epoch; contrib: groups x bins."""
    n_g = contrib.shape[0]
    for v, var in enumerate(STATE_VARS):
        lev = levels[var]
        for level in np.unique(lev[lev >= 0]):
            m = lev == level
            yy, pp = y[m], p[m]
            stats[v, level, :len(STATS)] += [m.sum(), yy.sum(), (yy * yy).sum(), pp.sum(), (pp * pp).sum(), (yy * pp).sum()]
            c = contrib[:, m]
            stats[v, level, len(STATS):] += np.stack([c.sum(1), (c * c).sum(1), (c * pp).sum(1)], 1).reshape(n_g * 3)


# ---- run ---------------------------------------------------------------------------------------

def epoch_data(pipeline, epoch):
    """X (scaled spikes), Y (scaled targets), bin start times for 'active' or 'passive'."""
    if epoch == 'active':
        return pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat, pipeline.active_spikes.timestamps
    start, stop = pipeline.session.passive_times
    sp = pipeline.spikes(start, stop)
    Y = pipeline.target_scaler.transform(pipeline.lfp_targets(start, stop, sp.spkMat.shape[0]).lfpMat)
    return sp.spkMat, Y, sp.timestamps


def event_bins(events, timestamps, n_bins):
    out = {}
    for name, times in events.items():
        b = onset_bins(times, timestamps, config.BIN_SIZE)
        out[name] = b[(b >= 0) & (b < n_bins)]
    return out


def ig_check(pipeline, X, bins, groups_present, unit_groups, mean, chan, targets, n_events, rng):
    """IG (mean baseline) and exact Shapley region contributions on the same windows at one channel."""
    history = config.ATTR_HISTORY
    offsets = np.round(IG_OFFSETS_MS / 1e3 / config.BIN_SIZE).astype(int)
    shape = (len(IG_TYPES), len(offsets), len(GROUPS), config.N_TARGETS)
    out = dict(ig_contribution=np.full(shape, np.nan), ig_shapley=np.full(shape, np.nan),
               ig_completeness=np.full(config.N_TARGETS, np.nan), ig_n_events=np.zeros(len(IG_TYPES), int))
    Xt = torch.as_tensor(X, dtype=torch.float32, device=pipeline.device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=pipeline.device)
    W = torch.as_tensor(shapley_matrix(len(unit_groups)), dtype=torch.float32)
    members = coalitions(len(unit_groups))
    absent = [torch.as_tensor(np.any([g & ~k for g, k in zip(unit_groups, keep)], axis=0), device=pipeline.device)
              for keep in members]
    for i, name in enumerate(IG_TYPES):
        b = bins.get(name, np.array([], int))
        b = b[(b + offsets[0] >= history - 1) & (b + offsets[-1] < len(X))]
        if len(b) > n_events:
            b = np.sort(rng.choice(b, n_events, replace=False))
        out['ig_n_events'][i] = len(b)
        if not len(b):
            continue
        ends = (b[:, None] + offsets[None, :]).ravel()
        for band in targets:
            model = pipeline.models[chan, band].model.to(pipeline.device)
            res, err, steps = attribute_with_retries(model, Xt, ends, mean_t, pipeline.ig_batch)
            out['ig_completeness'][band] = np.nanmax([out['ig_completeness'][band], err])
            per_group = np.stack([res['total'][:, g].sum(1) for g in unit_groups], 1)          # windows x groups
            out['ig_contribution'][i, :, groups_present, band] = per_group.reshape(len(b), len(offsets), -1).mean(0).T
            ig = SequenceIG(model, n_steps=1)
            model.eval()
            f = []
            with torch.no_grad():
                for start in range(0, len(ends), 64):
                    idx = torch.as_tensor(window_index(ends[start:start + 64], history), device=pipeline.device)
                    xw = Xt[idx]
                    f.append(torch.stack([ig._f(torch.where(a, mean_t, xw), 1).cpu() for a in absent]))
            phi = (W @ torch.cat(f, 1)).numpy()                                                # groups x windows
            out['ig_shapley'][i, :, groups_present, band] = phi.T.reshape(len(b), len(offsets), -1).mean(0).T
            log(f'IG check {name} band {band}: {len(b)} events, {steps} steps, completeness error {err:.1e}')
    return out


def run(pipeline, targets=(0, 1), ig_events=100, seed=config.TORCH_SEED):
    rng = np.random.default_rng(seed)
    s = pipeline.session
    targets = list(targets)
    masks = region_masks(np.asarray(s.units.structure_acronym))
    present = [i for i, g in enumerate(GROUPS) if masks[g].any()]
    unit_groups = [masks[GROUPS[i]] for i in present]
    W = shapley_matrix(len(present))
    X_active = pipeline.active_spikes.spkMat
    mean = X_active[chunk_bins(pipeline.split['train'])].mean(axis=0)
    pre, post = int(round(-EVENT_WINDOW[0] / config.BIN_SIZE)), int(round(EVENT_WINDOW[1] / config.BIN_SIZE))
    lags = (np.arange(-pre, post) + 0.5) * config.BIN_SIZE * 1e3
    n_ch = pipeline.n_channels

    events = {'active': task_events(s.active, s.licks.timestamps, auto_rewarded_changes(s))[0],
              'passive': task_events(passive_presentations(s), s.licks.timestamps, prefix='passive_')[0]}
    eye = s.eye_tracking
    pupil = None
    if eye is not None and len(eye):
        t = eye.timestamps.to_numpy(float)
        pupil = (t, clean_pupil(t, eye.pupil_area.to_numpy(float), eye.likely_blink if 'likely_blink' in eye else None)['clean'])
    data = {e: epoch_data(pipeline, e) for e in ('active', 'passive')}
    behaviour = {e: behaviour_on_bins(s, ts + config.BIN_SIZE / 2, pupil) for e, (_, _, ts) in data.items()}
    edges = {v: np.nanquantile(np.concatenate([behaviour[e][v] for e in data]), np.linspace(0, 1, N_LEVELS + 1)[:-1])
             for v in ('running', 'pupil')}
    edges = {v: (e if np.isfinite(e).all() else np.zeros(N_LEVELS)) for v, e in edges.items()}
    levels = {e: state_levels(behaviour[e], edges) for e in data}
    levels['passive']['engaged'][:] = -1
    test = chunk_bins(pipeline.split['test'])
    bins = {e: event_bins(events[e], data[e][2], len(data[e][0])) for e in data}

    types = [k for k in ACTIVE_TYPES if k in bins['active']] + list(bins['passive'])
    n_stats = len(STATS) + len(GROUP_STATS) * len(present)
    shape = (len(types), pre + post, n_ch, config.N_TARGETS)
    out = {k: np.full(shape, np.nan, np.float32) for k in ('eta_target', 'eta_prediction', 'eta_base')}
    out['eta_contribution'] = np.full((len(types), pre + post, len(GROUPS), n_ch, config.N_TARGETS), np.nan, np.float32)
    active_types = [k for k in types if k in bins['active']]
    for k in ('eta_target_test', 'eta_prediction_test', 'eta_base_test'):
        out[k] = np.full((len(active_types),) + shape[1:], np.nan, np.float32)
    out['eta_contribution_test'] = np.full((len(active_types),) + out['eta_contribution'].shape[1:], np.nan, np.float32)
    stats = np.zeros((len(STATE_EPOCHS), len(STATE_VARS), N_LEVELS, n_stats, n_ch, config.N_TARGETS))
    n_events, n_events_test = np.zeros(len(types), int), np.zeros(len(active_types), int)

    for chan, band in sorted(k for k in pipeline.models if k[1] in targets):
        model = pipeline.models[chan, band]
        for e, (X, Y, ts) in data.items():
            f = coalition_predictions(model, X, unit_groups, mean)            # coalitions x bins
            n = f.shape[1]
            phi = W @ f                                                       # groups x bins
            y, p, base = Y[:n, chan, band], f[-1], f[0]
            assert np.allclose(phi.sum(0), p - base, atol=1e-3), 'Shapley efficiency'
            signals = np.concatenate([y[:, None], p[:, None], base[:, None], phi.T], 1)       # bins x (3 + groups)
            for name, b in bins[e].items():
                i = types.index(name)
                eta, n_events[i] = event_average(signals, b, pre, post, rng)
                out['eta_target'][i, :, chan, band], out['eta_prediction'][i, :, chan, band], out['eta_base'][i, :, chan, band] = eta[:, :3].T
                out['eta_contribution'][i][:, present, chan, band] = eta[:, 3:]
                if e == 'active':
                    j = active_types.index(name)
                    bt = b[(b - pre >= test[0]) & (b + post <= min(test[-1] + 1, n))]
                    eta, n_events_test[j] = event_average(signals, bt, pre, post, rng)
                    out['eta_target_test'][j, :, chan, band], out['eta_prediction_test'][j, :, chan, band], out['eta_base_test'][j, :, chan, band] = eta[:, :3].T
                    out['eta_contribution_test'][j][:, present, chan, band] = eta[:, 3:]
            lev = {v: levels[e][v][:n] for v in STATE_VARS}
            accumulate(stats[STATE_EPOCHS.index(e), :, :, :, chan, band], lev, y, p, phi)
            if e == 'active':
                in_test = (np.arange(n) >= test[0]) & (np.arange(n) <= test[-1])
                lev = {v: np.where(in_test, lev[v], -1) for v in STATE_VARS}
                accumulate(stats[STATE_EPOCHS.index('active_test'), :, :, :, chan, band], lev, y, p, phi)
        share = [np.cov(phi[g], p)[0, 1] / p.var() for g in range(len(present))]
        log(f'chan {chan} band {band}: Shapley share (passive) ' + ', '.join(f'{GROUPS[i]} {v:.2f}' for i, v in zip(present, share)))

    # stats: put the group sums at the full GROUPS positions (NaN for absent groups)
    full = np.full(stats.shape[:3] + (len(STATS) + len(GROUP_STATS) * len(GROUPS),) + stats.shape[4:], np.nan)
    full[:, :, :, :len(STATS)] = stats[:, :, :, :len(STATS)]
    for j, i in enumerate(present):
        for k in range(len(GROUP_STATS)):
            full[:, :, :, len(STATS) + i * len(GROUP_STATS) + k] = stats[:, :, :, len(STATS) + j * len(GROUP_STATS) + k]

    l4, borders = layer4_channel(pipeline.session_id, n_ch)
    out.update(event_types=np.array(types), active_types=np.array(active_types), eta_lags_ms=lags,
               n_events=n_events, n_events_test=n_events_test, groups=np.array(GROUPS),
               group_n_units=np.array([masks[g].sum() for g in GROUPS]), targets=np.array(targets),
               layer4_channel=l4, state_stats=full, state_epochs=np.array(STATE_EPOCHS), state_vars=np.array(STATE_VARS),
               stat_names=np.array(STATS + [f'{g}_{k}' for g in GROUPS for k in GROUP_STATS]),
               state_edges=np.stack([edges['running'], edges['pupil']]), ig_offsets_ms=IG_OFFSETS_MS, ig_types=np.array(IG_TYPES))
    if ig_events:
        out.update(ig_check(pipeline, X_active, bins['active'], present, unit_groups, mean, l4, targets, ig_events, rng))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--targets', default='0,1')
    parser.add_argument('--ig-events', type=int, default=100, help='events per type for the IG check (0 = off)')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, [int(t) for t in args.targets.split(',')], args.ig_events)
    f = pipeline.dirs.variables / 'region_events.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
