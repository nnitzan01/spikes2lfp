"""
The paper's quantitative claims at the mouse level (reviewer 3, comments 9 and minor 7-8), from the saved results.

  claims       every headline paired comparison, per session as in experiments/summarize.py, then averaged within
               mouse: median over mice, 95% CI from a cluster bootstrap over mice, mice > 0, Wilcoxon p over mice,
               and Benjamini-Hochberg q over all claims. Saved to <data root>/spikes2lfp_v2/statistics.csv.
  hierarchy    Fig. 7B redone: per session, the Spearman correlation across the higher visual areas (>= MIN_AREA_UNITS
               units, >= 3 areas) between area mean |attribution per spike| (broadband, mean over the aligned channels)
               and the anatomical hierarchy score; tested over mice. Also the area-level Pearson r over the pooled
               units with a mouse-bootstrap CI, and a mixed model log10|attribution| ~ hierarchy with random
               intercepts for mice and sessions. Figure figures_review/hierarchy.png.
  lognormal    the log-normality claim: log10 |attribution per spike| (broadband, layer 4 aligned channels),
               centred and scaled within each session, against the normal distribution (QQ, skewness, excess
               kurtosis) for VISp, LG, CA1 and all units. Figure figures_review/lognormal.png.

usage (on the cluster, from the repo root, after `source cluster/env.sh`):
    python -m experiments.statistics [claims hierarchy lognormal]       # default: all
"""
import argparse
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

import paths
from analysis import stats
from experiments import review_figures as rf
from experiments import summarize as S

OUT = paths.DATA_ROOT / paths.OUTPUT_SUBDIR
HIERARCHY = {'VISl': -0.093, 'VISrl': -0.059, 'VISal': 0.152, 'VISpm': 0.327, 'VISam': 0.441}   # Siegle et al. 2021
MIN_AREA_UNITS = 5
L4_ALIGNED = [8, 9, 10]                         # aligned channels around the upper L4 border (channel 10)


# ------------------------------------------------------------------------------------------------- claims

def per_session(name, f):
    """{session: f(npz, pooled channels)} over the sessions with <name>.npz."""
    return {sid: f(d, ch) for sid, d, ch in ((sid, np.load(S.variables_dir(sid) / f'{name}.npz'), S.pooled_channels(sid))
                                             for sid in S.sessions() if (S.variables_dir(sid) / f'{name}.npz').exists())}


def fraction_lost(d, ch, key_full, key_removed, regions):
    """Fraction of the full R2 lost per region (median over channels of the drop / median full), regions x targets."""
    names = [str(r) for r in d['regions']]
    idx = [names.index(r) for r in regions]
    full = d[key_full][ch] if d[key_full].ndim == 2 else d[key_full][0][ch]
    removed = d[key_removed][0][idx][:, ch] if key_removed == 'r2_removed' else d[key_removed][idx, 0][:, ch]
    n = d['region_n_units'][idx]
    out = np.nanmedian(full - removed, 1) / np.nanmedian(full, 0)
    out[n == 0] = np.nan
    return out


def ablation_score(d, ch, measure, max_fraction=0.2):
    measures = [str(m) for m in d['measures']]
    drop = S.channel_median(d['r2_full'] - d['r2_ablated'][0], ch)                 # measures x fractions x targets
    return np.nanmean(drop[measures.index(measure), d['fractions'] <= max_fraction + 1e-9], 0)


def claims():
    rows = []

    def add(family, claim, values):
        values = {s: v for s, v in values.items() if np.isfinite(v)}
        if values:
            rows.append(dict(family=family, claim=claim, sessions=list(values), values=np.array(list(values.values()))))

    b = per_session('baselines', lambda d, ch: S.channel_median(d['model_r2'], ch)[:3] - S.channel_median(d['r2_units'], ch)[:3])
    for t in range(3):
        add('prediction', f'{S.MODEL} - lagged unit ridge R2, {S.TARGET_NAMES[t]}', {s: v[t] for s, v in b.items()})
    b = per_session('baselines', lambda d, ch: S.channel_median(d['r2_units'], ch)[:3] - S.channel_median(d['r2_units_instant'], ch)[:3])
    add('prediction', 'lagged - current-bin ridge R2 (history), broadband', {s: v[0] for s, v in b.items()})
    c = per_session('stimulus_clock', lambda d, ch: dict(
        model=S.channel_median(d['r2_model'], ch)[:5], template=S.channel_median(d['r2_template'], ch)[:5],
        swapped=S.channel_median(np.nanmean(d['r2_shuffle'], 0), ch)[:5]))
    for t in range(5):
        add('stimulus clock', f'model - template R2, {S.TARGET_NAMES[t]}', {s: v['model'][t] - v['template'][t] for s, v in c.items()})
    add('stimulus clock', 'template - spikes swapped within condition R2, broadband',
        {s: v['template'][0] - v['swapped'][0] for s, v in c.items()})
    a = per_session('attribution_baseline', lambda d, ch: ablation_score(d, ch, 'attribution_mean_baseline') - ablation_score(d, ch, 'ridge_weight'))
    for t in range(3):
        add('attribution', f'R2 lost removing top units: mean-baseline attribution - ridge weight, {S.TARGET_NAMES[t]}',
            {s: v[t] for s, v in a.items()})
    r = per_session('region_ablation', lambda d, ch: fraction_lost(d, ch, 'r2_full', 'r2_removed', ['VISp', 'visual_cortex_other']))
    for t in range(5):
        add('two sources', f'fraction of R2 lost, VISp - other visual (inference), {S.TARGET_NAMES[t]}',
            {s: v[0, t] - v[1, t] for s, v in r.items()})
    add('two sources', 'broadband - 4-8 Hz of (VISp - other visual), inference',
        {s: (v[0, 0] - v[1, 0]) - (v[0, 2] - v[1, 2]) for s, v in r.items()})
    r = per_session('region_ablation_bands', lambda d, ch: fraction_lost(d, ch, 'r2_retrained_full', 'r2_retrained_removed',
                                                                         ['VISp', 'visual_cortex_other']))
    for t in range(3):
        add('two sources', f'fraction of R2 lost, VISp - other visual (retrained), {S.TARGET_NAMES[t]}',
            {s: v[0, t] - v[1, t] for s, v in r.items()})

    def clock(d, ch):
        names = [str(x) for x in d['regions']]
        removed = S.channel_median(d['r2_model'] - d['r2_removed'], ch)[:, :2]
        swapped = S.channel_median(d['r2_model'] - np.nanmean(d['r2_swap'], 0), ch)[:, :2]
        out = {g: (swapped[names.index(g)] - removed[names.index(g)]) if d['region_n_units'][names.index(g)] > 0
               else np.full(2, np.nan) for g in ('VISp', 'visual_cortex_other', 'hippocampus')}
        return out
    k = per_session('region_clock', clock)
    for g in ('VISp', 'visual_cortex_other', 'hippocampus'):
        add('trial-specific', f'R2 lost swapped - removed, {rf.label(g)}, broadband', {s: v[g][0] for s, v in k.items()})
    h = per_session('bleed_through', lambda d, ch: S.channel_median(d['ret_r2_timescale'], ch)[:, -1])
    for t, name in ((7, '100-200 Hz'), (8, '200-400 Hz')):
        add('high bands', f'R2 at 250 ms, retrained without VISp - all units, {name}', {s: v[2, t] - v[0, t] for s, v in h.items()})
    ts = per_session('timescale_r2', lambda d, ch: S.channel_median(d['r2_timescale'], ch)[-1]
                     - S.channel_median(np.nanmean(d['r2_timescale_shift'], 0), ch)[-1])
    for t, name in ((7, '100-200 Hz'), (8, '200-400 Hz')):
        add('high bands', f'R2 at 250 ms - time-shift null, {name}', {s: v[t] for s, v in ts.items()})

    ev = rf.events()
    groups = next(iter(ev.values()))['groups']
    for w, wn in ((0, 'early'), (1, 'late')):
        add('laminar profile', f'L4 fraction of CSD power, other visual - VISp, image {wn}',
            {s: r['depth']['frac']['image'][w, groups.index('visual_cortex_other'), 0, 1] - r['depth']['frac']['image'][w, groups.index('VISp'), 0, 1]
             for s, r in ev.items() if r['depth'] is not None and 'image' in r['depth']['frac']})
    for wn, window in S.LATENCY_WINDOWS_MS.items():
        add('timing', f'delay behind own firing, other visual - VISp, {wn}', {s: latency_difference(r, window) for s, r in ev.items()})
    for contrast, group, j, text in (
            (('active', 'running', (7, 8, 9), 'active', 'running', (0, 1, 2)), 'VISp', 1, 'running - stationary, VISp share, 0.5-4 Hz'),
            (('active', 'running', (7, 8, 9), 'active', 'running', (0, 1, 2)), 'midbrain', 1, 'running - stationary, midbrain share, 0.5-4 Hz'),
            (('active', 'all', (0,), 'passive', 'all', (0,)), 'midbrain', 1, 'active - passive, midbrain share, 0.5-4 Hz'),
            (('active', 'engaged', (1,), 'active', 'engaged', (0,)), 'VISp', 1, 'engaged - disengaged, VISp share, 0.5-4 Hz')):
        gi = groups.index(group)
        add('state', text, {s: r['state'][contrast][0][gi, j] - r['state'][contrast][1][gi, j] for s, r in ev.items() if r['has'][gi]})
    pairs = rf.mouse_pairs(ev)
    lags = next(iter(ev.values()))['lags']
    for t, j, g, (wn, (lo, hi)) in [('image_shared', 0, 'visual_cortex_other', w) for w in S.EVENT_WINDOWS_MS.items()] + \
                                   [('image_shared', 0, 'VISp', w) for w in S.EVENT_WINDOWS_MS.items()] + \
                                   [('passive_image_shared', 0, 'visual_cortex_other', ('early', S.EVENT_WINDOWS_MS['early']))]:
        sel = (lags >= lo) & (lags < hi)
        diff = {}
        for f, n in pairs:
            a_, b_ = rf.contribution(ev[f], t, g, j), rf.contribution(ev[n], t, g, j)
            if a_ is not None and b_ is not None:
                diff[n] = np.sqrt(np.mean(b_[sel] ** 2)) - np.sqrt(np.mean(a_[sel] ** 2))      # one value per mouse
        add('novelty', f'novel - familiar day RMS, shared images, {rf.label(g)}, {t}, {wn}', diff)
    rho = hierarchy_slopes()
    add('hierarchy (Fig. 7B)', 'Spearman of area |attribution per spike| with hierarchy score, per session', rho)

    rng = np.random.default_rng(0)
    table = []
    for row in rows:
        mice = stats.mouse_ids(row['sessions'])
        med, pos, n, p = stats.wilcoxon_mice(row['values'], mice)
        _, low, high = stats.cluster_bootstrap(row['values'], mice, rng=rng)
        table.append(dict(family=row['family'], claim=row['claim'], sessions=len(row['values']), mice=n,
                          median=med, ci_low=low, ci_high=high, mice_above_0=pos, p=p))
    table = pd.DataFrame(table)
    table['q_fdr'] = stats.fdr(table['p'])
    table.to_csv(OUT / 'statistics.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 95, 'display.max_rows', 200):
        print(table.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print(f'saved {OUT / "statistics.csv"}')


def latency_difference(r, window):
    """Contribution t50 minus own firing t50, other visual minus VISp (image, broadband), or NaN."""
    if 'rate' not in r or 'image' not in r['etypes']:
        return np.nan
    delay = []
    for g in ('VISp', 'visual_cortex_other'):
        c = rf.contribution(r, 'image', g)
        if c is None or g not in r['egroups']:
            return np.nan
        rate = r['rate'][r['etypes'].index('image'), :, r['egroups'].index(g)]
        delay.append(S.event_timing(c[:, None], r['lags'], window)[0, 0] - S.event_timing(rate[:, None], r['lags'], window)[0, 0])
    return delay[1] - delay[0]


# --------------------------------------------------------------------------------------- hierarchy (Fig. 7B)

def pooled_units(measure='per_spike'):
    """Pooled units (pooled/units.csv) with |attribution| (broadband) averaged over the aligned channels and at layer 4."""
    pooled = OUT / 'pooled'
    units = pd.read_csv(pooled / 'units.csv')
    a = np.abs(np.load(pooled / 'unit_attribution_active.npz')[measure][:, :, 0])           # units x aligned channels
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        units['attr'], units['attr_l4'] = np.nanmean(a, 1), np.nanmean(a[:, L4_ALIGNED], 1)
    units['mouse'] = stats.mouse_ids(units['session_id'].to_numpy())
    return units


def hierarchy_slopes(units=None):
    units = pooled_units() if units is None else units
    hva = units[units['area'].isin(list(HIERARCHY))]
    out = {}
    for sid, u in hva.groupby('session_id'):
        means = u.groupby('area')['attr'].agg(['mean', 'size'])
        means = means[means['size'] >= MIN_AREA_UNITS]
        if len(means) >= 3:
            out[sid] = st.spearmanr(means['mean'], [HIERARCHY[a] for a in means.index])[0]
    return out


def hierarchy():
    units = pooled_units()
    hva = units[units['area'].isin(list(HIERARCHY))].copy()
    hva['hierarchy'] = hva['area'].map(HIERARCHY)
    rho = hierarchy_slopes(units)
    mice = stats.mouse_ids(list(rho))
    print(f'per-session Spearman across areas (>= {MIN_AREA_UNITS} units, >= 3 areas; {len(rho)} sessions): '
          + stats.describe(np.array(list(rho.values())), mice))
    areas = list(HIERARCHY)
    area_mean = hva.groupby('area')['attr'].mean()[areas]
    r_pooled = st.pearsonr(area_mean, [HIERARCHY[a] for a in areas])
    rng, boots = np.random.default_rng(0), []
    by_mouse = {m: u for m, u in hva.groupby('mouse')}
    ids = list(by_mouse)
    for _ in range(2000):
        draw = pd.concat([by_mouse[ids[i]] for i in rng.integers(len(ids), size=len(ids))])
        m = draw.groupby('area')['attr'].mean().reindex(areas)
        if m.notna().all():
            boots.append(st.pearsonr(m, [HIERARCHY[a] for a in areas])[0])
    print(f'pooled area means (n = {len(areas)} areas): Pearson r {r_pooled[0]:+.2f} (p {r_pooled[1]:.2f}, the original '
          f'test); mouse-bootstrap 95% CI [{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}]')
    hva['log_attr'] = np.log10(hva['attr'].where(hva['attr'] > 0))
    hva['session'] = hva['session_id'].astype(str)
    try:
        fit = stats.mixed_model(hva[['log_attr', 'hierarchy', 'mouse', 'session']], 'log_attr ~ hierarchy')
        ci = fit.conf_int().loc['hierarchy']
        print(f'mixed model log10|attribution per spike| ~ hierarchy (+ mouse, session within mouse; {fit.nobs:.0f} units): '
              f'slope {fit.params["hierarchy"]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p {fit.pvalues["hierarchy"]:.1e}')
    except Exception as e:                   # noqa: BLE001  (convergence problems are reported, not fatal)
        print(f'mixed model failed: {e}')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    per_session = hva.groupby(['session_id', 'area'])['attr'].agg(['mean', 'size']).reset_index()
    per_session = per_session[per_session['size'] >= MIN_AREA_UNITS]
    for sid, u in per_session.groupby('session_id'):
        u = u.set_index('area').reindex([a for a in areas if a in u['area'].values])
        ax[0].plot([HIERARCHY[a] for a in u.index], u['mean'], color='gray', lw=0.4, alpha=0.5)
    ax[0].plot([HIERARCHY[a] for a in areas], per_session.groupby('area')['mean'].median()[areas], 'k-o', lw=2)
    ax[0].set_xticks([HIERARCHY[a] for a in areas], areas, rotation=45)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('anatomical hierarchy score')
    ax[0].set_ylabel('area mean |attribution per spike| (broadband)')
    ax[0].set_title('per session (grey) and median (black)', fontsize=9)
    rf.strip(ax[1], list(rho.values()), 0, 'k', width=0.4)
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_xticks([0], ['per-session Spearman'])
    ax[1].set_title(stats.describe(np.array(list(rho.values())), mice), fontsize=8)
    ax[2].hist(boots, 40, color='gray')
    ax[2].axvline(r_pooled[0], color='k')
    ax[2].set_xlabel('Pearson r over the 5 area means (mouse bootstrap)')
    ax[2].set_title(f'original test: r {r_pooled[0]:+.2f}, p {r_pooled[1]:.2f} (n = 5 areas)', fontsize=9)
    rf.save(fig, 'hierarchy', 'Fig. 7B at the session and mouse level: attribution vs anatomical hierarchy of the higher visual areas')


# ---------------------------------------------------------------------------------------------- log-normality

def lognormal():
    units = pooled_units()
    units = units[units['attr_l4'] > 0].copy()
    units['log'] = np.log10(units['attr_l4'])
    units['z'] = units.groupby('session_id')['log'].transform(lambda x: (x - x.mean()) / x.std())
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))
    for p, (name, sel) in enumerate((('VISp', units['area'] == 'VISp'), ('LG', units['area'] == 'LG'),
                                     ('CA1', units['area'] == 'CA1'), ('all units', np.ones(len(units), bool)))):
        z = units.loc[sel, 'z'].dropna().to_numpy()
        (theory, ordered), _ = st.probplot(z, dist='norm')
        ax[p].plot(theory, ordered, '.', ms=2, color='k', alpha=0.4)
        lim = [min(theory.min(), ordered.min()), max(theory.max(), ordered.max())]
        ax[p].plot(lim, lim, color='tab:red', lw=1)
        skew, kurt = st.skew(z), st.kurtosis(z)
        within = units.loc[sel].groupby('session_id')['log'].agg(lambda x: st.skew(x) if len(x) >= 20 else np.nan).dropna()
        ax[p].set_title(f'{name} ({len(z)} units): skew {skew:+.2f}, excess kurtosis {kurt:+.2f}\n'
                        f'per-session skew median {within.median():+.2f} ({len(within)} sessions)', fontsize=8)
        ax[p].set_xlabel('normal quantiles')
        ax[p].set_ylabel('log10 |attribution per spike|, z within session')
        print(f'{name}: {len(z)} units, skew {skew:+.2f}, excess kurtosis {kurt:+.2f}, per-session skew median {within.median():+.2f}')
    rf.save(fig, 'lognormal', 'Log-normality of attribution (broadband, layer 4): QQ of log10 |attribution per spike| against the normal')


PARTS = dict(claims=claims, hierarchy=hierarchy, lognormal=lognormal)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('parts', nargs='*', help=f'of {", ".join(PARTS)} (default: all)')
    args = parser.parse_args()
    if set(args.parts) - set(PARTS):
        parser.error(f'unknown parts: {" ".join(set(args.parts) - set(PARTS))}')
    warnings.simplefilter('ignore', RuntimeWarning)
    for name in args.parts or PARTS:
        PARTS[name]()


if __name__ == '__main__':
    main()
