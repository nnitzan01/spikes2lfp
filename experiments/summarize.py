"""
Print summaries of the experiment results, over the pooled channels of each session
(first_channel_index to last_channel_index of tables/layer_4_borders_aligned.csv).

usage (from the repo root, after source cluster/env.sh):
    python -m experiments.summarize training_length [--session 1047969464]
    python -m experiments.summarize attribution_baseline [--session 1047969464]
    python -m experiments.summarize baselines          # every session with baselines.npz
    python -m experiments.summarize simulation [--session 1047969464]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import paths
from experiments.pool_sessions import ALIGNMENT_TABLE

TARGET_NAMES = ['broadband', '0.5-4 Hz', '4-8 Hz', '8-12 Hz', '12-25 Hz']
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def pooled_channels(session_id):
    row = pd.read_csv(ALIGNMENT_TABLE).set_index('session').loc[session_id]
    return slice(int(row['first_channel_index']), int(row['last_channel_index']) + 1)


def variables_dir(session_id):
    return paths.session_output_dirs(paths.DATA_ROOT, session_id).variables


def channel_median(a, ch):
    """Median over the pooled channels of a (... x channels x targets) array -> (... x targets)."""
    return np.nanmedian(a[..., ch, :], axis=-2)


def training_length(session_id):
    d, ch = np.load(variables_dir(session_id) / 'training_length.npz'), pooled_channels(session_id)
    epochs = int(d['num_epochs'])
    print(f'session {session_id}, channels {ch.start}-{ch.stop - 1}, retrained with up to {epochs} epochs')
    saved, warm_saved = channel_median(d['r2_saved'], ch), channel_median(d['r2_warm_saved'], ch)
    for inputs in [k[len('r2_warm_'):] for k in d if k.startswith('r2_warm_') and k != 'r2_warm_saved']:
        r2, warm = channel_median(d[f'r2_{inputs}'], ch), channel_median(d[f'r2_warm_{inputs}'], ch)
        best = d[f'best_epoch_{inputs}'][ch]
        for t in np.flatnonzero(np.isfinite(r2)):
            gain = d[f'r2_{inputs}'][ch, t] - d['r2_saved'][ch, t]
            print(f'{inputs:8s} {TARGET_NAMES[t]:10s} R2 {r2[t]:.3f} (saved 15-epoch units model {saved[t]:.3f}; '
                  f'paired gain over it {np.nanmedian(gain):+.3f}, higher on {np.sum(gain > 0)}/{len(gain)} channels) | '
                  f'after 1 s warm-up {warm[t]:.3f} (saved {warm_saved[t]:.3f}) | best epoch median '
                  f'{np.nanmedian(best[:, t]):.0f}, at the last 3 epochs {np.mean(best[:, t] >= epochs - 3):.0%}')


def ablation_table(d, ch, measures):
    full = channel_median(d['r2_full'], ch)
    fr = d['fractions']
    print('fractions removed', fr)
    for t in np.flatnonzero(np.isfinite(full))[:3]:
        print(f'--- {TARGET_NAMES[t]} (full {full[t]:.3f}): R2 drop at inference, mean fill')
        for i, m in enumerate(measures):
            print(f'{m:38s}', np.round(full[t] - channel_median(d['r2_ablated'][0, i], ch)[:, t], 3))
        rnd = np.nanmean(channel_median(d['r2_random'][0], ch), axis=0)
        print(f'{"random (mean of orderings)":38s}', np.round(full[t] - rnd[:, t], 3))
    if 'r2_retrained' in d:
        rf = channel_median(d['r2_retrained_full'], ch)[0, 0]
        print(f'--- broadband retrained without the top units (retrained full {rf:.3f}); fractions {d["retrain_fractions"]}')
        for i, m in enumerate(measures):
            print(f'{m:38s}', np.round(rf - channel_median(d['r2_retrained'][i], ch)[:, 0, 0], 3))
        print(f'{"random":38s}', np.round(rf - np.nanmean(channel_median(d['r2_retrained_random'], ch)[:, :, 0, 0], 0), 3))


def attribution_baseline(session_id):
    d, ch = np.load(variables_dir(session_id) / 'attribution_baseline.npz'), pooled_channels(session_id)
    print(f'session {session_id}, channels {ch.start}-{ch.stop - 1}')
    ablation_table(d, ch, list(d['measures']))
    targets = np.flatnonzero(np.isfinite(channel_median(d['r2_full'], ch)))
    print('--- mean-baseline attribution, by target', [TARGET_NAMES[t] for t in targets])
    print('completeness error (median)  ', channel_median(d['completeness'], ch)[targets])
    print('integration steps (max)      ', d['n_steps'][ch][:, targets].max(axis=0))
    print('Spearman with zero-baseline  ', channel_median(d['spearman_zero_baseline'], ch)[targets])
    print('Spearman with firing rate    ', channel_median(d['spearman_firing_rate'], ch)[targets])


def simulation(session_id):
    d = np.load(variables_dir(session_id) / 'simulation.npz')
    measures, lag_measures = list(d['measures']), list(d['lag_measures'])
    print(f'session {session_id}; mean over {d["r2"].shape[1]} draws of drivers')
    for ci, c in enumerate(d['conditions']):
        print(f'=== {c}: model test R2 {np.nanmean(d["r2"][ci]):.3f}')
        print(f'{"":24s}' + ''.join(f'{m:>20s}' for m in d['metrics']))
        for i, m in enumerate(measures):
            print(f'{m:24s}' + ''.join(f'{np.nanmean(d[k][ci, :, i]):20.3f}' for k in d['metrics']))
        for i, m in enumerate(lag_measures):
            print(f'{m:24s} peak lag vs true latency: Spearman {np.nanmean(d["lag_spearman"][ci, :, i]):.2f}, '
                  f'median error {np.nanmean(d["lag_error_ms"][ci, :, i]):.0f} ms')


def baselines():
    sessions = [int(s) for s in open(paths.REPO_DIR / 'cluster' / 'sessions.txt').read().split()]
    keys = ['model_r2', 'r2_units', 'r2_units_instant', 'r2_regions', 'r2_stimulus_behaviour',
            'r2_regions_stimulus_behaviour', 'r2_units_stimulus_behaviour', 'r2_lstm_regions']
    rows, missing = {}, []
    for sid in sessions:
        f = variables_dir(sid) / 'baselines.npz'
        if not f.exists():
            missing.append(sid)
            continue
        d, ch = np.load(f), pooled_channels(sid)
        rows[sid] = {k: channel_median(d[k], ch)[:3] for k in keys if k in d}
    print(f'{len(rows)} sessions with baselines.npz; {len(missing)} missing{": " + " ".join(map(str, missing)) if missing else ""}')
    print('median over sessions of the per-session median over pooled channels (broadband, 0.5-4, 4-8 Hz):')
    for k in keys:
        v = np.array([r[k] for r in rows.values() if k in r])
        if len(v):
            print(f'  {k:32s} {np.median(v, axis=0)}  (n = {len(v)})')
    print('CNN-LSTM minus each ridge, per session (median over sessions; sessions where the CNN-LSTM is higher; Wilcoxon p):')
    for k in keys[1:]:
        both = [r for r in rows.values() if k in r and 'model_r2' in r]
        if not both:
            continue
        diff = np.array([r['model_r2'] - r[k] for r in both])
        p = [wilcoxon(diff[:, t]).pvalue if np.any(diff[:, t]) else np.nan for t in range(3)]
        print(f'  {k:32s} {np.median(diff, axis=0)}  higher in {np.sum(diff > 0, axis=0)} of {len(diff)}  p {np.array(p)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=['training_length', 'attribution_baseline', 'baselines', 'simulation'])
    parser.add_argument('--session', type=int, default=1047969464)
    args = parser.parse_args()
    if args.what == 'baselines':
        baselines()
    else:
        globals()[args.what](args.session)


if __name__ == '__main__':
    main()
