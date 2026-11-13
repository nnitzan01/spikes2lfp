"""
Print summaries of the experiment results, over the pooled channels of each session
(first_channel_index to last_channel_index of tables/layer_4_borders_aligned.csv).

Over every session with the experiment's file (median over sessions of the per-session median, with
the paired Wilcoxon over sessions):
    python -m experiments.summarize baselines | ranked_ablation | attribution_baseline | region_ablation | region_bands
                                    | region_clock | stimulus_clock | transfer_scaling | timescale_r2 | bleed_through
                                    | explore_task | region_events
                                    | simulation
One session (default 1047969464):
    python -m experiments.summarize training_length | simulation [--session 1047969464]
    python -m experiments.summarize simulation --pattern 'simulation_long*'     # some of the simulation files
    python -m experiments.summarize attribution_baseline --session 1047969464
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
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


def simulation(session_id, pattern='simulation*'):
    for f in sorted(variables_dir(session_id).glob(f'{pattern}.npz')):
        d = np.load(f)
        measures, lag_measures = list(d['measures']), list(d['lag_measures'])
        print(f'##### {f.name}: session {session_id}; mean over {d["r2"].shape[1]} draws of drivers')
        for ci, c in enumerate(d['conditions']):
            line = f'=== {c}: model test R2 {np.nanmean(d["r2"][ci]):.3f}'
            if 'r2_ceiling' in d:
                line += f' (noise-free ceiling {np.nanmean(d["r2_ceiling"][ci]):.3f})'
            if 'r2_ridge' in d:
                line += (f', lagged unit ridge {np.nanmean(d["r2_ridge"][ci]):.3f}; '
                         f'best epochs {d["best_epoch"][ci].astype(int).tolist()}')
            print(line)
            print(f'{"":24s}' + ''.join(f'{m:>20s}' for m in d['metrics']))
            for i, m in enumerate(measures):
                print(f'{m:24s}' + ''.join(f'{np.nanmean(d[k][ci, :, i]):20.3f}' for k in d['metrics']))
            for i, m in enumerate(lag_measures):
                line = (f'{m:24s} peak lag vs true latency: Spearman {np.nanmean(d["lag_spearman"][ci, :, i]):.2f}, '
                        f'median error {np.nanmean(d["lag_error_ms"][ci, :, i]):.0f} ms')
                if 'lag_ms' in d:
                    drv = d['is_driver'][ci]
                    est, true = d['lag_ms'][ci, :, i][drv], d['true_lag_ms'][ci][drv]
                    line += (f'; drivers\' peak lag median {np.median(est):.0f} ms (true {np.median(true):.0f}), '
                             f'at lag 0 for {np.mean(est == 0):.0%}, beyond 500 ms for {np.mean(est > 500):.0%}')
                print(line)


SIM_COMPARISONS = [('ig_mean', 'ridge_weight'), ('ig_zero', 'ridge_weight'), ('ig_zero_per_spike', 'ridge_weight'),
                   ('ig_mean', 'spike_lfp_correlation')]


def simulation_sessions(pattern='simulation_final*'):
    """
    Simulation files matching pattern in every session, pooled by condition: each draw of drivers (in any
    session) is one sample; means over draws, and paired differences between measures over draws.
    """
    draws, files = {}, {}                                   # condition: [per-draw dicts]; condition: {file names}
    for sid in sessions():
        for f in sorted(variables_dir(sid).glob(f'{pattern}.npz')):
            d = np.load(f)
            measures, lag_measures = list(d['measures']), list(d['lag_measures'])
            for ci, c in enumerate(d['conditions']):
                files.setdefault(c, set()).add(f.name)
                for k in range(d['r2'].shape[1]):
                    row = dict(session=sid, r2=d['r2'][ci, k], r2_ridge=d['r2_ridge'][ci, k],
                               r2_ceiling=d['r2_ceiling'][ci, k], best_epoch=d['best_epoch'][ci, k])
                    row.update({(m, metric): d[metric][ci, k, i] for metric in d['metrics'] for i, m in enumerate(measures)})
                    row.update({(m, 'lag_spearman'): d['lag_spearman'][ci, k, i] for i, m in enumerate(lag_measures)})
                    draws.setdefault(c, []).append(row)
    if not draws:
        print(f'no {pattern}.npz in any session')
        return
    for c, rows in draws.items():
        def col(key):
            return np.array([r.get(key, np.nan) for r in rows], float)

        metrics = list(dict.fromkeys(key[1] for key in rows[0] if isinstance(key, tuple) and key[1] != 'lag_spearman'))
        measures = list(dict.fromkeys(key[0] for key in rows[0] if isinstance(key, tuple)))
        print(f'=== {c}: {len(rows)} draws in {len({r["session"] for r in rows})} sessions ({", ".join(sorted(files[c]))})')
        print(f'  test R2, mean over draws: model {np.nanmean(col("r2")):.3f}, lagged unit ridge {np.nanmean(col("r2_ridge")):.3f}, '
              f'noise-free ceiling {np.nanmean(col("r2_ceiling")):.3f}; model - ridge {paired(col("r2") - col("r2_ridge"))}; '
              f'best epoch median {np.nanmedian(col("best_epoch")):.0f} (range {np.nanmin(col("best_epoch")):.0f}-{np.nanmax(col("best_epoch")):.0f})')
        print(f'  {"mean over draws":22s}' + ''.join(f'{m:>20s}' for m in metrics + ['lag_spearman']))
        for m in measures:
            print(f'  {m:22s}' + ''.join(f'{np.nanmean(col((m, k))):20.3f}' for k in metrics + ['lag_spearman']))
        print('  paired over draws (median difference, draws > 0, Wilcoxon p):')
        for a, b in SIM_COMPARISONS:
            for k in metrics:
                print(f'    {a + " - " + b:42s} {k:20s} {paired(col((a, k)) - col((b, k)))}')


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


BAND_REGIONS = ['VISp', 'visual_cortex_other', 'hippocampus', 'thalamus', 'midbrain', 'VISl', 'VISal', 'VISrl',
                'VISam', 'VISpm', 'CA1', 'DG', 'LG', 'LP', 'APN', 'MRN']
PROFILE_REGIONS = ['VISp', 'visual_cortex_other', 'hippocampus']
N_PROFILE = 16                                              # aligned channels, deepest first (pool_sessions.ALIGNED_CHANNELS)
MIN_PROFILE_R2 = 0.05                                       # channels with a lower full R2 are left out of the depth profile


def region_bands():
    """
    Local vs distributed (section D of REVISION_STATUS): the fraction of the full R2 lost when a region is
    removed at inference (mean fill), per band (median drop over the pooled channels / median full R2, per
    session), VISp vs the other visual areas per band, and the depth profile of the drop as a fraction of
    each channel's own full R2 (pooled channels aligned from the deepest, with the layer 4 borders).
    """
    def per_session(d, ch):
        regions = [str(r) for r in d['regions']]
        idx = [regions.index(r) for r in BAND_REGIONS]
        full = d['r2_full'][ch]                                          # channels x targets
        drop = (d['r2_full'] - d['r2_removed'][0])[idx][:, ch]            # regions x channels x targets
        profile = np.full((len(PROFILE_REGIONS), N_PROFILE, full.shape[1]), np.nan)
        n = min(N_PROFILE, ch.stop - ch.start)
        own = np.where(full[:n] > MIN_PROFILE_R2, full[:n], np.nan)              # each channel's own full R2
        profile[:, :n] = drop[[BAND_REGIONS.index(r) for r in PROFILE_REGIONS], :n] / own
        out = dict(n_units=d['region_n_units'][idx], full=np.nanmedian(full, 0),
                   fraction=np.nanmedian(drop, 1) / np.nanmedian(full, 0),
                   alone=np.nanmedian(d['r2_only'][0][idx][:, ch], 1) / np.nanmedian(full, 0), profile=profile)
        out.update(retrained_fractions(d, ch))
        return out

    def retrained_fractions(d, ch):
        regions = [str(r) for r in d['regions']]
        idx = [regions.index(r) for r in BAND_REGIONS]
        out = {'n_units': d['region_n_units'][idx]}
        for k in ('retrained', 'cov'):
            key = 'r2_retrained_removed' if k == 'retrained' else 'r2_cov_removed'
            if f'r2_{k}_full' in d:
                rfull = d[f'r2_{k}_full'][0][ch]
                out[k] = np.nanmedian((rfull - d[key][idx, 0][:, ch]), 1) / np.nanmedian(rfull, 0)
        return out

    rows = over_sessions('region_ablation', per_session)
    if not rows:
        return
    band_rows = over_sessions('region_ablation_bands', retrained_fractions)     # retrained for more bands (D)
    stack = lambda k: np.stack([r[k] for r in rows.values() if k in r])
    n_units, full, fraction, alone = stack('n_units').astype(float), stack('full'), stack('fraction'), stack('alone')
    targets = [t for t in range(min(5, full.shape[1])) if np.isfinite(full[:, t]).any()]
    print('full R2 (median over sessions): ' + ', '.join(f'{TARGET_NAMES[t]} {np.nanmedian(full[:, t]):.3f}' for t in targets))
    for name, v, what in (('fraction', fraction, 'removed at inference (mean fill)'),
                          ('alone', alone, 'alone (everyone else removed at inference)')):
        print(f'fraction of the full R2 {"lost" if name == "fraction" else "kept"}, region {what}; median over sessions with units:')
        print(f'  {"":22s} {"sessions":>8s} {"units":>6s}' + ''.join(f'{TARGET_NAMES[t]:>12s}' for t in targets))
        for i, r in enumerate(BAND_REGIONS):
            has = n_units[:, i] > 0
            if has.any():
                print(f'  {r:22s} {has.sum():8d} {np.median(n_units[has, i]):6.0f}'
                      + ''.join(f'{np.nanmedian(v[has, i, t]):12.3f}' for t in targets))
    for source, k, what in [(src, k, what) for src in ('region_ablation', 'region_ablation_bands') for k, what in
                            (('retrained', 'retrained without the region'), ('cov', 'retrained without the region, covariates given'))]:
        found = [r for r in (rows if source == 'region_ablation' else band_rows).values() if k in r]
        if found:
            v, n = np.stack([r[k] for r in found]), np.stack([r['n_units'] for r in found]).astype(float)
            done = [t for t in targets if np.isfinite(v[:, :, t]).any()]
            print(f'fraction of the full R2 lost, {what} ({source}.npz; per session; median, sessions > 0, Wilcoxon p):')
            for i, r in enumerate(BAND_REGIONS):
                ok = n[:, i] > 0
                if ok.any() and np.isfinite(v[ok, i][:, done]).any():
                    print(f'  {r:22s} ' + ' | '.join(f'{TARGET_NAMES[t]} {paired(v[ok, i, t])}' for t in done))
    a, b = BAND_REGIONS.index('VISp'), BAND_REGIONS.index('visual_cortex_other')
    both = (n_units[:, a] > 0) & (n_units[:, b] > 0)
    diff = fraction[both, a] - fraction[both, b]
    print(f'VISp minus other visual cortex, fraction lost at inference ({both.sum()} sessions with both):')
    for t in targets:
        print(f'  {TARGET_NAMES[t]:10s} {paired(diff[:, t])}')
    for t in targets[1:]:
        print(f'  (broadband - {TARGET_NAMES[t]}) of that difference: {paired(diff[:, 0] - diff[:, t])}')
    found = [r for r in band_rows.values() if 'retrained' in r]
    if found:
        v, n = np.stack([r['retrained'] for r in found]), np.stack([r['n_units'] for r in found]).astype(float)
        ok = (n[:, a] > 0) & (n[:, b] > 0)
        rdiff = v[ok, a] - v[ok, b]
        done = [t for t in targets if np.isfinite(rdiff[:, t]).any()]
        print(f'VISp minus other visual cortex, fraction lost retrained (region_ablation_bands.npz; {ok.sum()} sessions with both):')
        for t in done:
            print(f'  {TARGET_NAMES[t]:10s} {paired(rdiff[:, t])}')
        for t in done[1:]:
            print(f'  (broadband - {TARGET_NAMES[t]}) of that difference: {paired(rdiff[:, 0] - rdiff[:, t])}')
    borders =pd.read_csv(ALIGNMENT_TABLE).set_index('session').loc[list(rows)]
    lower, upper = (np.median(borders[k] - borders['first_channel_index']) for k in ('lower_border_index', 'upper_border_index'))
    print(f'depth profile: fraction of each channel\'s full R2 lost (channels with full R2 > {MIN_PROFILE_R2}; 0 = deepest pooled channel; '
          f'layer 4 at about {lower:.0f}-{upper:.0f}), median over sessions with units:')
    profile = stack('profile')
    for i, r in enumerate(PROFILE_REGIONS):
        has = n_units[:, BAND_REGIONS.index(r)] > 0
        print(f'  {r}')
        for t in targets:
            print(f'    {TARGET_NAMES[t]:10s}', np.round(np.nanmedian(profile[has, i, :, t], 0), 2))


def region_clock():
    """Evoked vs induced by region (section E of REVISION_STATUS): R2 and residual R2 lost per region."""
    def per_session(d, ch):
        swap, swap_res = np.nanmean(d['r2_swap'], 0), np.nanmean(d['r2_residual_swap'], 0)
        out = dict(regions=d['regions'], n_units=d['region_n_units'], model=channel_median(d['r2_model'], ch),
                   residual=channel_median(d['r2_residual'], ch),
                   removed=channel_median(d['r2_model'] - d['r2_removed'], ch),
                   removed_residual=channel_median(d['r2_residual'] - d['r2_residual_removed'], ch),
                   swap=channel_median(d['r2_model'] - swap, ch),
                   swap_residual=channel_median(d['r2_residual'] - swap_res, ch),
                   swap_all_residual=channel_median(d['r2_residual'] - np.nanmean(d['r2_residual_swap_all'], 0), ch))
        out.update({k: channel_median(d[k], ch) for k in d.files if k.startswith('share_')})
        return out

    rows = list(over_sessions('region_clock', per_session).values())
    if not rows:
        return
    stack = lambda k: np.stack([r[k] for r in rows if k in r])
    n_units, model = stack('n_units').astype(float), stack('model')
    regions = [str(r) for r in rows[0]['regions']]
    shown = [(r, regions.index(r)) for r in BAND_REGIONS if r in regions]
    targets = [t for t in range(5) if np.isfinite(model[:, t]).any()]
    print('median over sessions: R2 ' + ', '.join(f'{TARGET_NAMES[t]} {np.nanmedian(model[:, t]):.3f}' for t in targets)
          + '; residual R2 ' + ', '.join(f'{np.nanmedian(stack("residual")[:, t]):.3f}' for t in targets)
          + '; residual R2 lost when every unit is swapped ' + ', '.join(f'{np.nanmedian(stack("swap_all_residual")[:, t]):.3f}' for t in targets))
    for key, what in (('removed', 'R2 lost, region removed (mean fill)'),
                      ('removed_residual', 'residual R2 lost, region removed'),
                      ('swap', "R2 lost, region's spikes swapped between presentations of the same condition"),
                      ('swap_residual', "residual R2 lost, region's spikes swapped")):
        v = stack(key)
        print(f'{what} (per session; median, sessions > 0, Wilcoxon p):')
        for r, i in shown:
            has = n_units[:, i] > 0
            if has.any():
                print(f'  {r:22s} {has.sum():3d} sessions ' + ' | '.join(f'{TARGET_NAMES[t]} {paired(v[has, i, t])}' for t in targets))
    print('trial-specific fraction (R2 lost by swapping / R2 lost by removing; median over sessions where removing costs > 0.005):')
    removed, swap = stack('removed'), stack('swap')
    for r, i in shown:
        line = ''
        for t in targets:
            has = (n_units[:, i] > 0) & (removed[:, i, t] > 0.005)
            line += f'{TARGET_NAMES[t]:>12s} {np.median(swap[has, i, t] / removed[has, i, t]) if has.any() else np.nan:5.2f} ({has.sum()})'
        print(f'  {r:22s}{line}')
    for key in sorted({k for r in rows for k in r if k.startswith('share_')}):
        v = stack(key)
        print(f'{key[6:]}: share of the region (median over sessions with units; broadband, 0.5-4, 4-8 Hz)')
        for r, i in shown:
            has = n_units[:, i] > 0
            if has.any():
                print(f'  {r:22s}', np.round(np.nanmedian(v[has, i, :3], 0), 3))


def transfer_scaling():
    """Spontaneous / passive R2 under the active scalers vs per-epoch scaling, recalibration and r^2."""
    from experiments.transfer_scaling import KEYS
    rows = list(over_sessions('transfer_scaling', lambda d, ch: dict(
        epochs=d['epochs'], ratio=d['input_rate_ratio'], **{k: channel_median(d[k], ch) for k in KEYS})).values())
    if not rows:
        return
    stack = lambda k: np.stack([r[k] for r in rows])
    targets = [t for t in range(5) if np.isfinite(stack('r2_active')[:, 0, t]).any()]
    for e, epoch in enumerate(rows[0]['epochs']):
        print(f'=== {epoch}: input rate / active training rate, median over sessions {np.median(stack("ratio")[:, e]):.2f}')
        print(f'  {"":18s}' + ''.join(f'{TARGET_NAMES[t]:>12s}' for t in targets))
        for k in KEYS:
            print(f'  {k:18s}' + ''.join(f'{np.nanmedian(stack(k)[:, e, t]):12.3f}' for t in targets))
        for k in ('r2_epoch', 'r2_recalibrated'):
            print(f'  {k} - r2_active: ' + ' | '.join(f'{TARGET_NAMES[t]} {paired(stack(k)[:, e, t] - stack("r2_active")[:, e, t])}' for t in targets))


ALL_BANDS = ['broadband'] + [f'{lo:g}-{hi:g} Hz' for lo, hi in config.BANDS]
MODULATION_RANGES = [(0, 1), (1, 4), (4, 10), (10, 40)]     # Hz, coherence of target and prediction envelopes


def timescale_r2():
    """R2 by smoothing window, log / amplitude R2, coherence and transient AUC per band, against the time-shift null."""
    def per_session(d, ch):
        out = {k: channel_median(d[k], ch) for k in ('r2_timescale', 'r2_log', 'r2_amplitude', 'auc_transient')}
        out.update({f'{k}_shift': channel_median(np.nanmean(d[f'{k}_shift'], 0), ch)
                    for k in ('r2_timescale', 'r2_log', 'r2_amplitude', 'auc_transient')})
        f = d['coherence_freqs']
        for k in ('coherence', 'coherence_shift'):
            c = d[k] if k == 'coherence' else np.nanmean(d[k], 0)
            out[k] = np.stack([channel_median(c[(f >= lo) & (f < hi)].mean(0), ch) for lo, hi in MODULATION_RANGES])
        out['windows'] = d['windows_ms']
        return out

    rows = list(over_sessions('timescale_r2', per_session).values())
    if not rows:
        return
    stack = lambda k: np.stack([r[k] for r in rows])
    windows = rows[0]['windows']
    head = f'  {"":16s}' + ''.join(f'{b:>13s}' for b in ALL_BANDS)
    v, null = stack('r2_timescale'), stack('r2_timescale_shift')
    print('R2 after smoothing target and prediction (median over sessions; in brackets the time-shifted spikes):')
    print(head)
    for i, w in enumerate(windows):
        print(f'  {f"{w} ms":16s}' + ''.join(f'{np.nanmedian(v[:, i, t]):6.3f} ({np.nanmedian(null[:, i, t]):5.2f})' for t in range(len(ALL_BANDS))))
    print(f'sessions (of {len(rows)}) where the model beats the shifted spikes:')
    for i, w in enumerate(windows):
        print(f'  {f"{w} ms":16s}' + ''.join(f'{np.sum(v[:, i, t] > null[:, i, t]):>13d}' for t in range(len(ALL_BANDS))))
    print('band targets: R2 of log power, of amplitude, transient AUC (median over sessions; shifted spikes in brackets):')
    print(head)
    for key in ('r2_log', 'r2_amplitude', 'auc_transient'):
        v, null = stack(key), stack(key + '_shift')
        print(f'  {key:16s}' + ''.join(f'{np.nanmedian(v[:, t]):6.3f} ({np.nanmedian(null[:, t]):5.2f})' for t in range(len(ALL_BANDS))))
    v, null = stack('coherence'), stack('coherence_shift')
    print('coherence of target and prediction by modulation frequency (median over sessions; shifted spikes in brackets):')
    print(head)
    for i, (lo, hi) in enumerate(MODULATION_RANGES):
        print(f'  {f"{lo}-{hi} Hz":16s}' + ''.join(f'{np.nanmedian(v[:, i, t]):6.3f} ({np.nanmedian(null[:, i, t]):5.2f})' for t in range(len(ALL_BANDS))))


def bleed_through():
    """High-band prediction without the units near each LFP contact: inference (mean fill) and retrained."""
    def per_session(d, ch):
        f = d['coherence_freqs']
        lo = (f >= 1) & (f < 4)
        out = dict(groups=d['groups'], retrain_groups=d['retrain_groups'], n_excluded=np.nanmedian(d['n_excluded'][:, ch], 1),
                   nearest=np.nanmedian(d['nearest_um'][ch]), windows=d['windows_ms'])
        for p in ('inf', 'ret'):
            out[f'{p}_r2'] = channel_median(d[f'{p}_r2_timescale'], ch)                  # groups x windows x targets
            out[f'{p}_coh'] = channel_median(d[f'{p}_coherence'][:, lo].mean(1), ch)      # groups x targets, 1-4 Hz
            out[f'{p}_auc'] = channel_median(d[f'{p}_auc_transient'], ch)
        return out

    rows = list(over_sessions('bleed_through', per_session).values())
    if not rows:
        return
    stack = lambda k: np.stack([r[k] for r in rows])
    groups, retrain_groups = [str(g) for g in rows[0]['groups']], [str(g) for g in rows[0]['retrain_groups']]
    windows = list(rows[0]['windows'])
    print(f'nearest unit to the contact, median {np.median(stack("nearest")):.0f} um; units excluded per channel (median): '
          + ', '.join(f'{g} {np.median(stack("n_excluded")[:, i]):.0f}' for i, g in enumerate(groups[1:])))
    bands = range(len(ALL_BANDS))
    for p, names, what in (('inf', groups, 'inference, excluded units mean-filled'), ('ret', retrain_groups, 'retrained without them')):
        r2, coh, auc = stack(f'{p}_r2'), stack(f'{p}_coh'), stack(f'{p}_auc')
        if np.isnan(r2).all():
            continue
        print(f'=== {what} (median over sessions)')
        print(f'  {"":24s}' + ''.join(f'{b:>13s}' for b in ALL_BANDS))
        for w in (0, len(windows) - 1):
            for i, g in enumerate(names):
                print(f'  {f"R2 {windows[w]} ms, {g}":24s}' + ''.join(f'{np.nanmedian(r2[:, i, w, t]):13.3f}' for t in bands))
        for label, v in (('coherence 1-4 Hz', coh), ('transient AUC', auc)):
            for i, g in enumerate(names):
                print(f'  {f"{label}, {g}":24s}' + ''.join(f'{np.nanmedian(v[:, i, t]):13.3f}' for t in bands))
        print(f'  change from all units, R2 at {windows[-1]} ms (median, sessions > 0, Wilcoxon p):')
        for i, g in enumerate(names[1:], 1):
            for t in bands:
                diff = r2[:, i, -1, t] - r2[:, 0, -1, t]
                if np.isfinite(diff).any():
                    print(f'    {g:10s} {ALL_BANDS[t]:12s} {paired(diff)}')


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


EXPLORE_SCALARS = ['hit_rate', 'lick_latency_median', 'pupil_missing', 'pupil_outlier', 'pupil_stable_active',
                   'running_active', 'stationary_active', 'engaged_active', 'engaged_test', 'active_minutes']


def explore_task():
    """Per-session task, behaviour and pupil numbers, event counts, and the familiar vs novel event-locked LFP at layer 4."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def per_session(d, ch):
        row = {k: float(d[k]) for k in EXPLORE_SCALARS}
        types = [str(t) for t in d['event_types']]
        row.update({f'n_{t}': int(n) for t, n in zip(types, d['n_events'])})
        row.update({f'n_test_{t}': int(n) for t, n in zip(types, d['n_events_test']) if n >= 0})
        l4 = int(d['layer4_channel'])
        return row, {t: d['eta_lfp'][i, :, l4] for i, t in enumerate(types)}, d['eta_lags_ms']

    rows = over_sessions('explore_task', per_session)
    if not rows:
        return
    table = pd.DataFrame({sid: r[0] for sid, r in rows.items()}).T.rename_axis('session_id').reset_index()
    meta = pd.read_csv(paths.REPO_DIR / 'tables' / 'sessions.csv')[['session_id', 'mouse_id', 'experience_level', 'cre_line']]
    table = meta.merge(table, on='session_id', how='right')
    out_dir = variables_dir(sessions()[0]).parent
    table.to_csv(out_dir / 'explore_task_sessions.csv', index=False)
    print(f'table: {out_dir / "explore_task_sessions.csv"}')
    counts = [c for c in table if c.startswith('n_')]
    print('median (min-max) over sessions:')
    for c in EXPLORE_SCALARS + counts:
        v = table[c].astype(float)
        print(f'  {c:28s} {v.median():9.3f} ({v.min():.3f}-{v.max():.3f})')
    if table.experience_level.notna().any():
        print('by experience level (median):')
        print(table.groupby('experience_level')[EXPLORE_SCALARS + counts].median().T.round(3).to_string())

    lags = next(iter(rows.values()))[2]
    types = ['image', 'change_hit', 'change_miss', 'omission', 'post_omission', 'passive_image']
    names = ['broadband'] + [f'{lo:g}-{hi:g} Hz' for lo, hi in config.BANDS]
    level = dict(zip(table.session_id, table.experience_level))
    fig, ax = plt.subplots(len(types), 3, figsize=(14, 2.6 * len(types)), sharex=True, squeeze=False)
    for i, t in enumerate(types):
        for j, target in enumerate((0, 1, 7)):
            for lev, color in (('Familiar', 'tab:blue'), ('Novel', 'tab:red')):
                eta = [r[1][t][:, target] for sid, r in rows.items() if level.get(sid) == lev and t in r[1]]
                eta = [e for e in eta if np.isfinite(e).all()]
                if eta:
                    eta = np.stack(eta)
                    ax[i, j].plot(lags, np.median(eta, 0), color=color, label=f'{lev} ({len(eta)})')
                    ax[i, j].fill_between(lags, *np.percentile(eta, [25, 75], 0), color=color, alpha=0.15, lw=0)
            ax[i, j].axvline(0, color='k', lw=0.5)
            ax[i, j].set_title(f'{t}, {names[target]}', fontsize=9)
        ax[i, 0].legend(fontsize=7)
    for a in ax[-1]:
        a.set_xlabel('time from onset (ms)')
    fig.suptitle('event-locked targets at the layer 4 channel (z), median and IQR over sessions', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'explore_task_pooled.png', dpi=110)
    plt.close(fig)
    print(f'figure: {out_dir / "explore_task_pooled.png"}')


EVENT_WINDOWS_MS = {'early': (30, 100), 'late': (100, 250), 'post': (250, 750)}
EVENT_CONTRASTS = [('change_hit', 'image'), ('change_hit', 'change_miss'), ('image', 'passive_image'),
                   ('post_omission', 'image'), ('image_novel', 'image_familiar'), ('image_novel', 'image_shared')]
STATE_CONTRASTS = [('active', 'all', (0,), 'passive', 'all', (0,)),                 # (epoch, var, levels) vs (epoch, var, levels)
                   ('active', 'running', (7, 8, 9), 'active', 'running', (0, 1, 2)),
                   ('active', 'pupil', (7, 8, 9), 'active', 'pupil', (0, 1, 2)),
                   ('active', 'engaged', (1,), 'active', 'engaged', (0,)),
                   ('passive', 'running', (7, 8, 9), 'passive', 'running', (0, 1, 2)),
                   ('passive', 'pupil', (7, 8, 9), 'passive', 'pupil', (0, 1, 2))]


def window_rms_shares(eta, lags):
    """eta (lags x groups x ...), corrected by its pre-onset mean: RMS per EVENT_WINDOWS_MS window over the
    groups' summed |...| -> (windows x groups x ...) shares (NaN groups stay NaN)."""
    eta = eta - np.nanmean(eta[lags < 0], axis=0)
    rms = np.stack([np.sqrt(np.mean(eta[(lags >= lo) & (lags < hi)] ** 2, axis=0)) for lo, hi in EVENT_WINDOWS_MS.values()])
    return rms / np.nansum(rms, axis=1, keepdims=True)


def state_shares(stats, names, groups, e, var, levels):
    """Shapley shares Cov(c_g, prediction) / Var(prediction) from the summed statistics: groups x ..."""
    s = np.nansum(stats[e, var, list(levels)], axis=0)                          # stats x ...
    col = {k: i for i, k in enumerate(names)}
    n = s[col['n']]
    with np.errstate(invalid='ignore', divide='ignore'):
        var_p = s[col['pp']] / n - (s[col['p']] / n) ** 2
        return np.stack([(s[col[f'{g}_cp']] / n - s[col[f'{g}_c']] / n * s[col['p']] / n) / var_p for g in groups])


def region_events():
    """
    Region contributions to the task-evoked LFP (Phase 2 F; experiments/region_events.py), at the layer 4 channel:
    share of each group in the RMS of the event-locked contribution (baseline-corrected) per window and event type,
    and of the difference between event types; Shapley variance shares by state (H); IG vs Shapley agreement.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def per_session(d, ch):
        l4, lags, targets = int(d['layer4_channel']), d['eta_lags_ms'], [int(t) for t in d['targets']]
        types, groups = [str(t) for t in d['event_types']], [str(g) for g in d['groups']]
        C = d['eta_contribution'][:, :, :, l4][..., targets]                       # types x lags x groups x targets
        row = dict(n_events=dict(zip(types, d['n_events'])), eta={t: C[i] for i, t in enumerate(types)},
                   shares={t: window_rms_shares(C[i], lags) for i, t in enumerate(types)})
        row['contrast'] = {(a, b): window_rms_shares(C[types.index(a)] - C[types.index(b)], lags)
                           for a, b in EVENT_CONTRASTS if a in types and b in types}
        active = [str(t) for t in d['active_types']]
        if 'image' in active:
            row['shares_test_image'] = window_rms_shares(d['eta_contribution_test'][active.index('image'), :, :, l4][..., targets], lags)
        stats, names = d['state_stats'][..., l4, :][..., targets], [str(n) for n in d['stat_names']]
        E, V = [str(e) for e in d['state_epochs']], [str(v) for v in d['state_vars']]
        row['state'] = {c: (state_shares(stats, names, groups, E.index(c[0]), V.index(c[1]), c[2]),
                            state_shares(stats, names, groups, E.index(c[3]), V.index(c[4]), c[5])) for c in STATE_CONTRASTS}
        if 'ig_contribution' in d:
            off = d['ig_offsets_ms']
            ig, sh = d['ig_contribution'][..., targets], d['ig_shapley'][..., targets]      # types x offsets x groups x targets
            ig, sh = ig - np.nanmean(ig[:, off < 0], 1, keepdims=True), sh - np.nanmean(sh[:, off < 0], 1, keepdims=True)
            after = off >= 0
            corr = np.full(ig.shape[:1] + ig.shape[2:], np.nan)
            for idx in np.ndindex(*corr.shape):
                a, b = ig[idx[0], after, idx[1], idx[2]], sh[idx[0], after, idx[1], idx[2]]
                if np.isfinite(a).all() and np.isfinite(b).all() and a.std() > 0 and b.std() > 0:
                    corr[idx] = np.corrcoef(a, b)[0, 1]
            rms = lambda x: np.sqrt(np.mean(x[:, after] ** 2, 1))
            row['ig'] = dict(types=[str(t) for t in d['ig_types']], corr=corr,
                             share_ig=rms(ig) / np.nansum(rms(ig), 1, keepdims=True),
                             share_shapley=rms(sh) / np.nansum(rms(sh), 1, keepdims=True))
        # RMS (not share) of each group's baseline-corrected contribution per window, for comparisons across sessions
        row['rms'] = {t: np.stack([np.sqrt(np.mean((C[i] - np.nanmean(C[i][lags < 0], 0))[(lags >= lo) & (lags < hi)] ** 2, 0))
                                   for lo, hi in EVENT_WINDOWS_MS.values()]) for i, t in enumerate(types)}
        row['strategy'] = float(d['strategy_index']) if 'strategy_index' in d else np.nan
        return dict(row, lags=lags, groups=groups, targets=targets, units=d['group_n_units'])

    rows = over_sessions('region_events', per_session)
    if not rows:
        return
    first = next(iter(rows.values()))
    groups, targets, lags = first['groups'], first['targets'], first['lags']
    tnames = [TARGET_NAMES[t] for t in targets]
    has = np.stack([r['units'] for r in rows.values()]) > 0                        # sessions x groups
    windows = list(EVENT_WINDOWS_MS)

    def table(get, title):
        vals = [get(r) for r in rows.values()]
        keep = [v is not None for v in vals]
        if not any(keep):
            return
        v = np.stack([x for x in vals if x is not None])                           # sessions x windows x groups x targets
        h = has[keep]
        print(title + f' ({len(v)} sessions)')
        for j, tn in enumerate(tnames):
            print(f'  {tn:10s} {"":8s}' + ''.join(f'{g[:12]:>13s}' for g in groups))
            for w, wn in enumerate(windows):
                med = [np.nanmedian(v[h[:, g], w, g, j]) if h[:, g].any() else np.nan for g in range(len(groups))]
                print(f'  {"":10s} {wn:8s}' + ''.join(f'{m:13.2f}' for m in med))

    print(f'layer 4 channel; windows {EVENT_WINDOWS_MS} ms; share of each group in the RMS of its baseline-corrected '
          'event-locked contribution (median over sessions with units)')
    for t in ['image', 'change_hit', 'change_miss', 'omission', 'post_omission', 'passive_image', 'passive_change']:
        n = [r['n_events'].get(t, 0) for r in rows.values()]
        table(lambda r: r['shares'].get(t), f'--- {t} (events per session, median {np.median(n):.0f})')
    table(lambda r: r.get('shares_test_image'), '--- image, test block only (out of sample)')
    for a, b in EVENT_CONTRASTS:
        table(lambda r: r['contrast'].get((a, b)), f'--- {a} minus {b}: shares of the difference')

    meta = pd.read_csv(paths.REPO_DIR / 'tables' / 'sessions.csv').set_index('session_id')
    pairs = {}
    for sid in rows:
        if sid in meta.index:
            pairs.setdefault(meta.loc[sid, 'mouse_id'], {})[meta.loc[sid, 'experience_level']] = sid
    pairs = [(v['Familiar'], v['Novel']) for v in pairs.values() if {'Familiar', 'Novel'} <= set(v)]
    for t in ('image_shared', 'change_shared', 'passive_image_shared'):
        both = [(f, n) for f, n in pairs if t in rows[f]['rms'] and t in rows[n]['rms']]
        if not both:
            continue
        print(f'--- {t}: same images, same mouse, novel day minus familiar day; RMS of the contribution per window ({len(both)} mice)')
        for j, tn in enumerate(tnames):
            for g, gn in enumerate(groups):
                d = np.array([rows[n]['rms'][t][:, g, j] - rows[f]['rms'][t][:, g, j] for f, n in both])        # mice x windows
                if np.isfinite(d).any():
                    print(f'    {tn:10s} {gn:20s} ' + ' | '.join(f'{w} {paired(d[:, w_])}' for w_, w in enumerate(windows)))
    strat = np.array([r['strategy'] for r in rows.values()])
    if np.isfinite(strat).sum() > 5:
        from scipy.stats import spearmanr
        print(f'strategy index (visual - timing; median {np.nanmedian(strat):+.3f}, {np.sum(strat > 0)}/{np.isfinite(strat).sum()} visual): '
              'Spearman with each group\'s share of the change_hit and image responses (late window)')
        for t in ('change_hit', 'image'):
            for j, tn in enumerate(tnames):
                sh = np.array([r['shares'][t][1, :, j] if t in r['shares'] else np.full(len(groups), np.nan) for r in rows.values()])
                out = []
                for g, gn in enumerate(groups):
                    ok = np.isfinite(strat) & np.isfinite(sh[:, g]) & has[:, g]
                    if ok.sum() > 5:
                        rho, p = spearmanr(strat[ok], sh[ok, g])
                        out.append(f'{gn[:10]} {rho:+.2f} (p {p:.0e})')
                print(f'  {t:10s} {tn:10s} ' + ' | '.join(out))

    print('Shapley variance shares Cov(c, prediction) / Var(prediction) by state, layer 4 (median over sessions; paired difference):')
    for c in STATE_CONTRASTS:
        print(f'  {c[0]} {c[1]} {c[2]}  vs  {c[3]} {c[4]} {c[5]}')
        for j, tn in enumerate(tnames):
            for g, gn in enumerate(groups):
                a = np.array([r['state'][c][0][g, j] for r in rows.values()])
                b = np.array([r['state'][c][1][g, j] for r in rows.values()])
                ok = has[:, g] & np.isfinite(a) & np.isfinite(b)
                if ok.any():
                    print(f'    {tn:10s} {gn:20s} {np.median(a[ok]):6.2f} vs {np.median(b[ok]):6.2f}  diff {paired(a[ok] - b[ok])}')

    ig_rows = [r['ig'] for r in rows.values() if 'ig' in r]
    if ig_rows:
        print(f'IG (mean baseline) vs Shapley on the same windows, layer 4 ({len(ig_rows)} sessions): correlation of the '
              'baseline-corrected time courses after onset, and RMS shares (median over sessions)')
        for i, t in enumerate(ig_rows[0]['types']):
            for j, tn in enumerate(tnames):
                corr = np.stack([r['corr'][i, :, j] for r in ig_rows])
                s_ig = np.stack([r['share_ig'][i, :, j] for r in ig_rows])
                s_sh = np.stack([r['share_shapley'][i, :, j] for r in ig_rows])
                print(f'  {t:12s} {tn:10s} ' + ' | '.join(
                    f'{groups[g][:10]} r {np.nanmedian(corr[:, g]):.2f}, share {np.nanmedian(s_ig[:, g]):.2f}/{np.nanmedian(s_sh[:, g]):.2f}'
                    for g in range(len(groups)) if np.isfinite(corr[:, g]).any()))

    show = ['image', 'change_hit', 'change_miss', 'omission', 'passive_image']
    fig, ax = plt.subplots(len(targets), len(show), figsize=(3.2 * len(show), 2.8 * len(targets)), sharex=True, squeeze=False)
    colors = plt.cm.tab10(np.arange(len(groups)))
    for j, tn in enumerate(tnames):
        for i, t in enumerate(show):
            for g, gn in enumerate(groups):
                eta = [r['eta'][t][:, g, j] for k, r in enumerate(rows.values()) if t in r['eta'] and has[k, g]]
                eta = [e - np.nanmean(e[lags < 0]) for e in eta if np.isfinite(e).all()]
                if eta:
                    ax[j, i].plot(lags, np.median(eta, 0), color=colors[g], label=gn)
            ax[j, i].axvline(0, color='k', lw=0.5)
            ax[j, i].set_title(f'{t}, {tn}', fontsize=9)
    ax[0, 0].legend(fontsize=7)
    for a in ax[-1]:
        a.set_xlabel('time from onset (ms)')
    fig.suptitle('region contributions at the layer 4 channel (baseline-corrected, z), median over sessions', fontsize=10)
    fig.tight_layout()
    out_dir = variables_dir(sessions()[0]).parent
    fig.savefig(out_dir / 'region_events_pooled.png', dpi=110)
    plt.close(fig)
    print(f'figure: {out_dir / "region_events_pooled.png"}')


SESSION_SUMMARIES = {'training_length': training_length, 'attribution_baseline': attribution_baseline,
                     'simulation': simulation}
ALL_SESSIONS = {'baselines': baselines, 'ranked_ablation': ranked_ablation, 'region_ablation': region_ablation,
                'region_bands': region_bands, 'region_clock': region_clock, 'transfer_scaling': transfer_scaling,
                'timescale_r2': timescale_r2, 'bleed_through': bleed_through, 'explore_task': explore_task,
                'region_events': region_events,
                'stimulus_clock': stimulus_clock, 'attribution_baseline': attribution_baseline_sessions,
                'simulation': simulation_sessions}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=sorted(set(SESSION_SUMMARIES) | set(ALL_SESSIONS)))
    parser.add_argument('--session', type=int, help='one session (default: every session where possible, '
                                                    'otherwise 1047969464)')
    parser.add_argument('--pattern', help='simulation: the files to show (glob, without .npz; default simulation* '
                                          'for one session, simulation_final* pooled over sessions)')
    args = parser.parse_args()
    warnings.simplefilter('ignore', RuntimeWarning)     # targets, regions and measures not run are all-NaN slices
    kwargs = dict(pattern=args.pattern) if args.what == 'simulation' and args.pattern else {}
    if args.what in SESSION_SUMMARIES and (args.session or args.what not in ALL_SESSIONS):
        SESSION_SUMMARIES[args.what](args.session or 1047969464, **kwargs)
    else:
        ALL_SESSIONS[args.what](**kwargs)


if __name__ == '__main__':
    main()
