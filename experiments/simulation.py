"""
Ground-truth simulation (reviewer 3, comment 7; reviewer 2, comment 2): does attribution recover
known spike contributions to an LFP-like signal, and does it do better than simpler measures?

The session's real spike trains are kept (with their real rates, correlations and redundancy) and
a synthetic target is built from them, so the true contribution of every unit is known:

  drive(t)  = sum over N_DRIVERS randomly chosen units k (training rate >= MIN_RATE Hz) of
              w_k (h_k * c_k)(t),  c_k the unit's spike counts, h_k a causal alpha kernel peaking at
              latency d_k + tau_k (d_k ~ U[0, 200 ms], tau_k ~ log-U[4, 40 ms]), w_k = +-lognormal
  signal    = drive ('linear') or a leaky rectification of the z-scored drive ('nonlinear')
  target    = signal + AROUSAL_SHARE x running speed (the '_arousal' conditions: a component no spike
              causes, but which units modulated by running correlate with) + AR(1) noise making up
              NOISE_SHARE of the variance; z-scored with training statistics

The standard CNN-LSTM is trained on the target (fit_and_score, the session's blocked split) and
the training-block targets are attributed (SequenceIG, zero and training-mean baselines). Every
unit is scored by MEASURES:
  ig_zero, ig_zero_per_spike, ig_zero_recent   mean |attribution|, |attribution per spike| and mean
                                               |attribution over the last ATTR_RECENT_BINS lags| (an
                                               instantaneous-like measure), zero baseline
  ig_mean                                      mean |attribution|, training-mean baseline
  ridge_weight, stlfp, spike_lfp_correlation, population_coupling, firing_rate
                                               as in experiments/ranked_ablation.py
and compared with the truth, separately for each condition and draw of drivers:
  auc                   drivers vs non-drivers (ROC AUC)
  precision             fraction of drivers among the N_DRIVERS top-scored units
  spearman_total        among drivers, with the true mean |contribution| at the attributed targets
                        (for ig_mean: the true contribution minus its training mean)
  spearman_per_spike    among drivers, with the true effect of one spike |w_k sum h_k|
  confound_percentile   median percentile (0-1) of the CONFOUND_UNITS non-drivers most correlated
                        with running; 0.5 = no preference, higher = credit for the arousal component
  lag_spearman, lag_error_ms   (ig_zero and ig_mean only) peak lag of each driver's attribution
                        kernel vs the true peak latency: Spearman, median absolute error

usage (from the repo root):
    python -m experiments.simulation <session_dir> [--draws 3] [--conditions linear,nonlinear,...]

Needs the 'train' stage (for the session's split and attribution targets) and the covariates.
Saves variables/<session_id>/simulation.npz: each metric as conditions x draws x measures (lag
metrics: conditions x draws x LAG_MEASURES), r2 (conditions x draws, test R2 of the model),
conditions, measures, and per draw the truth and every score (scores: conditions x draws x
measures x units; true_total, true_per_spike, true_lag_ms, is_driver: conditions x draws x units).
"""
import argparse

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import config
import paths
from analysis import coupling_measures as cm
from analysis.sequence_attribution import unit_summaries
from experiments.attribution_baseline import attribute_with_retries
from experiments.ranked_ablation import coupling_scores
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins

CONDITIONS = {'linear': (False, False), 'nonlinear': (True, False),
              'linear_arousal': (False, True), 'nonlinear_arousal': (True, True)}   # (nonlinear, arousal)
MEASURES = ['ig_zero', 'ig_zero_per_spike', 'ig_zero_recent', 'ig_mean', 'ridge_weight', 'stlfp',
            'spike_lfp_correlation', 'population_coupling', 'firing_rate']
LAG_MEASURES = ['ig_zero', 'ig_mean']
METRICS = ['auc', 'precision', 'spearman_total', 'spearman_per_spike', 'confound_percentile']
N_DRIVERS = 40
MIN_RATE = 0.5                  # Hz, drivers' minimum training firing rate
KERNEL_LENGTH = 0.5             # s
LATENCY = (0.0, 0.2)            # s, uniform
TAU = (0.004, 0.04)             # s, log-uniform
WEIGHT_SD = 1.0                 # lognormal sigma of |w|
AROUSAL_SHARE = 0.3             # variance of the arousal component relative to the signal's
NOISE_SHARE = 0.3               # fraction of the target's variance that is noise
NOISE_AR = 0.9                  # AR(1) coefficient of the noise (per bin)
LEAK = 0.2                      # slope of the rectified drive below zero ('nonlinear')
CONFOUND_UNITS = 20


def alpha_kernel(latency, tau, length):
    """Causal kernel over lags 0..length-1 (bins): 0 before `latency`, then an alpha function peaking (at 1) tau later."""
    s = (np.arange(length) - latency) / tau
    return np.where(s > 0, s * np.exp(1 - s), 0.0)


def zscore(x, rows):
    sd = x[rows].std()
    return (x - x[rows].mean()) / (sd if sd > 0 else 1)


def simulate(counts, arousal, train, nonlinear, use_arousal, rng, n_drivers=N_DRIVERS):
    """The synthetic target (bins,) and the ground truth: drivers, weights, kernels and drives (bins x drivers)."""
    n_bins, n_units = counts.shape
    rate = cm.firing_rate(counts, train, config.BIN_SIZE)
    eligible = np.flatnonzero(rate >= MIN_RATE)
    drivers = np.sort(rng.choice(eligible, min(n_drivers, len(eligible)), replace=False))
    length = int(round(KERNEL_LENGTH / config.BIN_SIZE))
    latency = rng.uniform(*LATENCY, len(drivers)) / config.BIN_SIZE
    tau = np.exp(rng.uniform(*np.log(TAU), len(drivers))) / config.BIN_SIZE
    weight = rng.choice([-1, 1], len(drivers)) * rng.lognormal(0, WEIGHT_SD, len(drivers))
    kernels = np.stack([alpha_kernel(d, t, length) for d, t in zip(latency, tau)])
    drives = np.stack([w * lfilter(h, 1, np.asarray(counts[:, k], dtype=np.float64))
                       for k, w, h in zip(drivers, weight, kernels)], axis=1)
    signal = drives.sum(axis=1)
    if nonlinear:
        z = zscore(signal, train)
        signal = np.where(z > 0, z, LEAK * z)
    signal = zscore(signal, train)
    if use_arousal:
        signal = signal + np.sqrt(AROUSAL_SHARE) * zscore(arousal, train)
    noise = lfilter([1], [1, -NOISE_AR], rng.normal(size=n_bins))
    noise = zscore(noise, train) * np.sqrt(NOISE_SHARE / (1 - NOISE_SHARE)) * signal[train].std()
    target = zscore(signal + noise, train).astype(np.float32)
    return target, dict(drivers=drivers, weight=weight, kernels=kernels, drives=drives,
                        peak_lag=latency + tau)


def truth(sim, n_units, ends, train):
    """True per-unit total (|contribution| at the targets; also around the training mean), per-spike effect and peak lag."""
    d = sim['drivers']
    out = {k: np.zeros(n_units) for k in ('total', 'total_centred', 'per_spike')}
    out['total'][d] = np.abs(sim['drives'][ends]).mean(axis=0)
    out['total_centred'][d] = np.abs(sim['drives'][ends] - sim['drives'][train].mean(axis=0)).mean(axis=0)
    out['per_spike'][d] = np.abs(sim['weight'] * sim['kernels'].sum(axis=1))
    out['lag_ms'] = np.full(n_units, np.nan)
    out['lag_ms'][d] = sim['peak_lag'] * config.BIN_SIZE * 1e3
    out['is_driver'] = np.isin(np.arange(n_units), d)
    return out


def kernel_peak_lag_ms(kernel_abs_sum):
    """Lag (ms) of the largest summed |attribution| of each unit; kernels are (history x units), oldest lag first."""
    return (kernel_abs_sum.shape[0] - 1 - kernel_abs_sum.argmax(axis=0)) * config.BIN_SIZE * 1e3


def score_metrics(scores, true, confound_units):
    """METRICS for every measure: (len(METRICS), len(MEASURES))."""
    d = true['is_driver']
    out = np.full((len(METRICS), len(MEASURES)), np.nan)
    for i, m in enumerate(MEASURES):
        s = np.nan_to_num(scores[m], nan=-np.inf)
        total = true['total_centred'] if m == 'ig_mean' else true['total']
        top = np.argsort(-s, kind='stable')[:d.sum()]
        pct = (np.argsort(np.argsort(s, kind='stable'), kind='stable') + 0.5) / len(s)
        out[:, i] = [roc_auc_score(d, np.nan_to_num(scores[m], nan=np.nanmin(scores[m]))),
                     d[top].mean(),
                     spearmanr(scores[m][d], total[d])[0],
                     spearmanr(scores[m][d], true['per_spike'][d])[0],
                     np.median(pct[confound_units])]
    return out


def run(pipeline, conditions=tuple(CONDITIONS), draws=3, n_drivers=N_DRIVERS):
    counts = pipeline.active_counts.spkMat
    n_bins, n_units = counts.shape
    X = torch.as_tensor(pipeline.active_spikes.spkMat, device=pipeline.device)
    train = chunk_bins(pipeline.split['train'])
    t = pipeline.attr_targets
    ends, window_spikes = t['sampled_train'], t['sampled_train_spikes']
    arousal = pipeline.load_covariates('active')['behaviour']['running_speed'].fillna(0).to_numpy(np.float64)[:n_bins]
    mean_input = torch.as_tensor(pipeline.active_spikes.spkMat[train].mean(axis=0), dtype=torch.float32, device=pipeline.device)

    # non-drivers most correlated with running (fixed per draw below once drivers are known)
    c = cm.sparse_rows(counts, train)
    mean, sd = cm.column_moments(c)
    a = zscore(arousal, train)[train]
    arousal_corr = np.abs((np.asarray(c.T @ a).ravel() / len(train) - mean * a.mean()) / np.where(sd > 0, sd, np.inf))

    shape = (len(conditions), draws)
    out = {m: np.full(shape + (len(MEASURES),), np.nan) for m in METRICS}
    out.update(lag_spearman=np.full(shape + (len(LAG_MEASURES),), np.nan),
               lag_error_ms=np.full(shape + (len(LAG_MEASURES),), np.nan),
               r2=np.full(shape, np.nan), scores=np.full(shape + (len(MEASURES), n_units), np.nan, dtype=np.float32))
    for key in ('true_total', 'true_per_spike', 'true_lag_ms'):
        out[key] = np.full(shape + (n_units,), np.nan, dtype=np.float32)
    out['is_driver'] = np.zeros(shape + (n_units,), bool)

    for ci, name in enumerate(conditions):
        nonlinear, use_arousal = CONDITIONS[name]
        for draw in range(draws):
            rng = np.random.default_rng(config.TORCH_SEED + draw)       # same drivers in every condition
            y, sim = simulate(counts, arousal, train, nonlinear, use_arousal, rng, n_drivers)
            true = truth(sim, n_units, ends, train)
            model, out['r2'][ci, draw] = pipeline.fit_and_score(pipeline.active_spikes.spkMat, y)
            scores, lags = {}, {}
            for baseline_name, baseline in (('ig_zero', 0.0), ('ig_mean', mean_input)):
                res, err, steps = attribute_with_retries(model.model, X, ends, baseline, pipeline.ig_batch,
                                                         keep_kernel=True, recent_bins=config.ATTR_RECENT_BINS)
                summary = unit_summaries(res['total'], window_spikes)
                scores[baseline_name] = summary['mean_abs']
                lags[baseline_name] = kernel_peak_lag_ms(res['kernel_abs_sum'])
                if baseline_name == 'ig_zero':
                    scores['ig_zero_per_spike'] = np.abs(summary['per_spike'])
                    scores['ig_zero_recent'] = np.abs(res['recent']).mean(axis=0)
                log(f'{name} draw {draw}: {baseline_name} {steps} steps, completeness error {err:.1e}')
            scores.update({m: s[:, 0] for m, s in coupling_scores(pipeline, y[:, None].astype(np.float64)).items()})

            non_drivers = np.flatnonzero(~true['is_driver'])
            confound = non_drivers[np.argsort(-arousal_corr[non_drivers])[:CONFOUND_UNITS]]
            metrics = score_metrics(scores, true, confound)
            for mi, m in enumerate(METRICS):
                out[m][ci, draw] = metrics[mi]
            d = true['is_driver']
            for li, m in enumerate(LAG_MEASURES):
                out['lag_spearman'][ci, draw, li] = spearmanr(lags[m][d], true['lag_ms'][d])[0]
                out['lag_error_ms'][ci, draw, li] = np.median(np.abs(lags[m][d] - true['lag_ms'][d]))
            out['scores'][ci, draw] = np.stack([scores[m] for m in MEASURES])
            out['true_total'][ci, draw], out['true_per_spike'][ci, draw] = true['total'], true['per_spike']
            out['true_lag_ms'][ci, draw], out['is_driver'][ci, draw] = true['lag_ms'], d
            log(f'{name} draw {draw}: model R2 {out["r2"][ci, draw]:.3f}; AUC '
                + ', '.join(f'{m} {out["auc"][ci, draw, i]:.2f}' for i, m in enumerate(MEASURES)))
    out.update(conditions=np.array(conditions), measures=np.array(MEASURES), lag_measures=np.array(LAG_MEASURES),
               metrics=np.array(METRICS))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--conditions', default=','.join(CONDITIONS))
    parser.add_argument('--n-drivers', type=int, default=N_DRIVERS)
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.conditions.split(','), args.draws, args.n_drivers)
    f = pipeline.dirs.variables / 'simulation.npz'
    save_npz(f, compressed=True, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
