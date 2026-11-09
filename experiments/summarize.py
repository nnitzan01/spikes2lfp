"""
Print summaries of the experiment results, over the pooled channels of each session
(first_channel_index to last_channel_index of tables/layer_4_borders_aligned.csv).

Over every session with the experiment's file (median over sessions of the per-session median, with
the paired Wilcoxon over sessions):
    python -m experiments.summarize baselines | ranked_ablation | attribution_baseline | region_ablation | stimulus_clock
One session (default 1047969464):
    python -m experiments.summarize training_length | simulation [--session 1047969464]
    python -m experiments.summarize attribution_baseline --session 1047969464
"""
import argparse
import warnings

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
    with warnings.catch_warnings():         # targets, regions or measures that were not run are all NaN
        warnings.simplefilter('ignore', RuntimeWarning)
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


def sessions():
    return [int(s) for s in open(paths.REPO_DIR / 'cluster' / 'sessions.txt').read().split()]


def over_sessions(name, per_session):
    """{session: per_session(npz, pooled channels)} for every session with variables/<session>/<name>.npz."""
    rows, missing = {}, []
    for sid in sessions():
        f = variables_dir(sid) / f'{name}.npz'
        if f.exists():
            rows[sid] = per_session(np.load(f), pooled_channels(sid))
        else:
            missing.append(sid)
    print(f'{len(rows)} sessions with {name}.npz; {len(missing)} missing{": " + " ".join(map(str, missing)) if missing else ""}')
    return rows


def paired(diff):
    """Median over sessions of a per-session difference, sessions above 0 and the Wilcoxon p (NaN sessions dropped)."""
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        return 'n/a'
    p = wilcoxon(diff).pvalue if np.any(diff) else np.nan
    return f'{np.median(diff):+.3f} ({np.sum(diff > 0)}/{len(diff)} > 0, p {p:.0e})'


def baselines():
    keys = ['model_r2', 'r2_units', 'r2_units_instant', 'r2_regions', 'r2_stimulus_behaviour',
            'r2_regions_stimulus_behaviour', 'r2_units_stimulus_behaviour', 'r2_lstm_regions']
    rows = over_sessions('baselines', lambda d, ch: {k: channel_median(d[k], ch)[:3] for k in keys if k in d})
    print('median over sessions of the per-session median over pooled channels (broadband, 0.5-4, 4-8 Hz):')
    for k in keys:
        v = np.array([r[k] for r in rows.values() if k in r])
        if len(v):
            print(f'  {k:32s} {np.median(v, axis=0)}  (n = {len(v)})')
    print('CNN-LSTM minus each ridge, per session (median over sessions; sessions where the CNN-LSTM is higher; Wilcoxon p):')
    for k in keys[1:]:
        both = [r for r in rows.values() if k in r and 'model_r2' in r]
        if both:
            diff = np.array([r['model_r2'] - r[k] for r in both])
            print(f'  {k:32s} ' + ' | '.join(paired(diff[:, t]) for t in range(3)))


def ablation_over_sessions(name, measures, reference='ridge_weight', max_fraction=0.2):
    """
    Ranked ablation (mean fill) over sessions: R2 drop per measure and fraction removed, and the mean drop
    over the fractions up to max_fraction compared with the reference measure's, per session.
    """
    def per_session(d, ch):
        out = dict(fractions=d['fractions'],
                   drop=channel_median(d['r2_full'] - d['r2_ablated'][0], ch),                    # measures x fractions x targets
                   random=np.nanmean(channel_median(d['r2_full'] - d['r2_random'][0], ch), 0))   # fractions x targets
        if 'r2_retrained' in d:
            full = d['r2_retrained_full'][0]
            out['retrained'] = channel_median(full - d['r2_retrained'][:, :, 0], ch)[..., 0]   # measures x fractions
            out['retrained_random'] = np.nanmean(channel_median(full - d['r2_retrained_random'][:, :, 0], ch), 0)[..., 0]
            out['retrain_fractions'] = d['retrain_fractions']
        if 'spearman_with_attribution' in d:
            out['spearman'] = channel_median(d['spearman_with_attribution'], ch)               # measures x targets
        return out

    rows = list(over_sessions(name, per_session).values())
    if not rows:
        return
    fr = rows[0]['fractions']
    drop, rnd = np.stack([r['drop'] for r in rows]), np.stack([r['random'] for r in rows])
    upto = fr <= max_fraction + 1e-9
    score = np.nanmean(drop[:, :, upto], axis=2)                                               # sessions x measures x targets
    ref = measures.index(reference)
    print('fractions removed', fr)
    for t in np.flatnonzero(np.isfinite(np.nanmedian(drop[:, 0, 0], axis=0)))[:3]:
        print(f'--- {TARGET_NAMES[t]}: R2 drop at inference, mean fill (median over sessions)')
        for i, m in enumerate(measures):
            print(f'{m:38s}', np.round(np.nanmedian(drop[:, i, :, t], axis=0), 3))
        print(f'{"random (mean of orderings)":38s}', np.round(np.nanmedian(rnd[:, :, t], axis=0), 3))
        best = np.nanargmax(np.where(np.isfinite(score[:, :, t]), score[:, :, t], -np.inf), axis=1)
        print(f'mean drop over {fr[upto].min():g}-{fr[upto].max():g} removed; minus {reference}; sessions where best:')
        for i, m in enumerate(measures):
            vs = '' if i == ref else paired(score[:, i, t] - score[:, ref, t])
            print(f'  {m:36s} {np.nanmedian(score[:, i, t]):.3f}  {vs:40s} best in {np.sum(best == i)}')
    if 'retrained' in rows[0]:
        re = np.stack([r['retrained'] for r in rows if 'retrained' in r])
        re_rnd = np.stack([r['retrained_random'] for r in rows if 'retrained' in r])
        print(f'--- broadband, retrained without the top units; fractions {rows[0]["retrain_fractions"]} '
              f'(median R2 drop; minus {reference})')
        for i, m in enumerate(measures):
            vs = '' if i == ref else '  '.join(paired(re[:, i, j] - re[:, ref, j]) for j in range(re.shape[2]))
            print(f'{m:38s}', np.round(np.nanmedian(re[:, i], axis=0), 3), vs)
        print(f'{"random":38s}', np.round(np.nanmedian(re_rnd, axis=0), 3))
    if 'spearman' in rows[0]:
        sp = np.stack([r['spearman'] for r in rows])
        print('Spearman of each measure with attribution (median over sessions; broadband, 0.5-4, 4-8 Hz)')
        for i, m in enumerate(measures):
            print(f'  {m:36s}', np.round(np.nanmedian(sp[:, i, :3], axis=0), 2))


def ranked_ablation():
    from experiments.ranked_ablation import MEASURES
    ablation_over_sessions('ranked_ablation', MEASURES)


def attribution_baseline_sessions():
    from experiments.attribution_baseline import MEASURES
    ablation_over_sessions('attribution_baseline', MEASURES)
    rows = list(over_sessions('attribution_baseline', lambda d, ch: {
        k: channel_median(d[k], ch)[:3] for k in ('completeness', 'spearman_zero_baseline', 'spearman_firing_rate')}).values())
    for k in rows[0] if rows else []:
        print(f'{k:24s} median over sessions (broadband, 0.5-4, 4-8 Hz)', np.round(np.nanmedian([r[k] for r in rows], 0), 3))


def region_ablation():
    def per_session(d, ch):
        out = dict(regions=d['regions'], n_units=d['region_n_units'],
                   removed=channel_median(d['r2_full'] - d['r2_removed'][0], ch)[:, :3],        # regions x targets
                   only=channel_median(d['r2_only'][0], ch)[:, :3])
        for k in ('retrained', 'cov'):
            full = d[f'r2_{k}_full'][0] if f'r2_{k}_full' in d else None
            if full is not None:
                key = 'r2_retrained_removed' if k == 'retrained' else 'r2_cov_removed'
                out[k] = channel_median(full - d[key][:, 0], ch)[:, 0]                          # regions (broadband)
        if 'r2_cov_full' in d:
            out['cov_gain'] = channel_median(d['r2_cov_full'][0] - d['r2_retrained_full'][0], ch)[0]
        return out

    rows = list(over_sessions('region_ablation', per_session).values())
    if not rows:
        return
    regions = [str(r) for r in rows[0]['regions']]
    stack = lambda k: np.stack([r[k] for r in rows if k in r])
    n_units, removed, only = stack('n_units').astype(float), stack('removed'), stack('only')
    print('R2 drop when the region is removed at inference (mean fill), per session with units in the region:')
    print(f'{"":22s} {"sessions":>8s} {"units":>6s}' + ''.join(f'{n:>28s}' for n in TARGET_NAMES[:3])
          + '   R2 of the region alone (broadband)')
    for i, r in enumerate(regions):
        has = n_units[:, i] > 0
        print(f'{r:22s} {has.sum():8d} {np.median(n_units[has, i]) if has.any() else 0:6.0f}'
              + ''.join(f'{paired(removed[has, i, t]):>28s}' for t in range(3))
              + f'   {np.nanmedian(only[has, i, 0]) if has.any() else np.nan:.3f}')
    for k, what in (('retrained', 'retrained without the region'), ('cov', 'retrained without the region, covariates given')):
        if any(k in r for r in rows):
            v, n = stack(k), np.stack([r['n_units'] for r in rows if k in r]).astype(float)
            print(f'--- broadband, {what}: R2 drop')
            for i, r in enumerate(regions):
                ok = (n[:, i] > 0) & np.isfinite(v[:, i])
                if ok.any():
                    print(f'  {r:22s} {paired(v[ok, i])}')
    if any('cov_gain' in r for r in rows):
        print(f'covariates added to the full model (broadband): {paired(stack("cov_gain"))}')


def stimulus_clock():
    keys = ['r2_model', 'r2_template', 'r2_residual', 'r2_residual_own', 'r2_trial_model', 'r2_trial_template',
            'r2_shuffle', 'r2_residual_shuffle', 'r2_shift', 'r2_residual_shift', 'r2_retrained_residual',
            'r2_spikes_plus_template']
    rows = list(over_sessions('stimulus_clock', lambda d, ch: {
        k: channel_median(np.nanmean(d[k], 0) if d[k].ndim == 3 else d[k], ch) for k in keys if k in d}).values())
    if not rows:
        return
    v = {k: np.stack([r[k] for r in rows if k in r]) for k in keys if any(k in r for r in rows)}
    targets = np.flatnonzero(np.isfinite(np.nanmedian(v['r2_model'], 0)))[:5]
    print('median over sessions;', ', '.join(TARGET_NAMES[t] for t in targets))
    for k, x in v.items():
        print(f'  {k:26s}', np.round(np.nanmedian(x[:, targets], 0), 3))
    print('per-session differences:')
    for a, b in (('r2_model', 'r2_template'), ('r2_model', 'r2_shuffle'), ('r2_model', 'r2_shift'),
                 ('r2_spikes_plus_template', 'r2_model')):
        if a in v and b in v:
            print(f'  {a} - {b}')
            for t in targets:
                print(f'    {TARGET_NAMES[t]:10s} {paired(v[a][:, t] - v[b][:, t])}')
    print('  template / model')
    for t in targets:
        print(f'    {TARGET_NAMES[t]:10s} {np.nanmedian(v["r2_template"][:, t] / v["r2_model"][:, t]):.2f}')


SESSION_SUMMARIES = {'training_length': training_length, 'attribution_baseline': attribution_baseline,
                     'simulation': simulation}
ALL_SESSIONS = {'baselines': baselines, 'ranked_ablation': ranked_ablation, 'region_ablation': region_ablation,
                'stimulus_clock': stimulus_clock, 'attribution_baseline': attribution_baseline_sessions}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=sorted(set(SESSION_SUMMARIES) | set(ALL_SESSIONS)))
    parser.add_argument('--session', type=int, help='one session (default: every session where possible, '
                                                    'otherwise 1047969464)')
    args = parser.parse_args()
    if args.what in SESSION_SUMMARIES and (args.session or args.what not in ALL_SESSIONS):
        SESSION_SUMMARIES[args.what](args.session or 1047969464)
    else:
        ALL_SESSIONS[args.what]()


if __name__ == '__main__':
    main()
