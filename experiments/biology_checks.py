"""
Model-independence checks of the candidate biological results (REVISION_STATUS 2026-11-24), on the CNN (spikes2lfp_v3),
the CNN-LSTM (spikes2lfp_v2) and, where it exists, the lagged ridge or a model-free measure. Statistics at the mouse
level (analysis/stats.wilcoxon_mice). Figures go to --figures (default docs/figures).

    state      region contributions by brain state at the layer 4 channel: Shapley share Cov(c_g, p) / Var(p) and
               the absolute Cov(c_g, p) by running and pupil decile for the CNN and the CNN-LSTM (region_events.npz),
               the same for the lagged ridge, and the model-free correlation of each group's population rate with the
               target (state_ridge.npz, spikes2lfp_v3)
    novelty    the hippocampal contribution to the responses to the shared images, novel minus familiar day of the
               same mouse, against the day-to-day change in hippocampal unit count, firing rate (tables/units_info.csv)
               and event-locked firing (explore_task.npz eta_pop); per-unit normalisations; a regression of the log
               ratio on the log ratios of unit count and rate; model-free spike-LFP coupling of the hippocampal units
    overview   the other three candidates in both models: VISp minus other visual cortex in the fraction of R2 lost
               at inference by band (region_ablation.npz), the layer 4 fraction of the VISp- and other-visual-driven
               CSD (region_events.npz), and the hippocampal share of the hit-minus-miss difference

    python -m experiments.biology_checks state | novelty | overview | all [--figures docs/figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

import paths  # noqa: E402
from analysis import stats  # noqa: E402
from experiments import summarize as S  # noqa: E402
from experiments.pool_sessions import UNITS_INFO  # noqa: E402
from experiments.region_ablation import region_masks  # noqa: E402

TREES = {'CNN': 'spikes2lfp_v3', 'CNN-LSTM': 'spikes2lfp_v2'}
COLORS = {'VISp': 'tab:blue', 'visual_cortex_other': 'tab:orange', 'hippocampus': 'tab:green', 'thalamus': 'tab:red',
          'midbrain': 'tab:purple', 'other': 'tab:gray'}
SHOW_GROUPS = ['VISp', 'visual_cortex_other', 'hippocampus', 'thalamus', 'midbrain']
TARGETS = {0: 'broadband', 1: '0.5-4 Hz'}
LOW, HIGH = (0, 1, 2), (7, 8, 9)
WINDOWS = list(S.EVENT_WINDOWS_MS)


def vdir(tree, sid):
    return paths.DATA_ROOT / tree / 'variables' / str(sid)


def load(tree, name):
    """{session: npz} of every session with <tree>/variables/<session>/<name>.npz."""
    out = {}
    for sid in S.sessions():
        f = vdir(tree, sid) / f'{name}.npz'
        if f.exists():
            out[sid] = np.load(f)
    return out


def mouse_test(diff, sids):
    med, pos, n, p = stats.wilcoxon_mice(np.asarray(diff, float), stats.mouse_ids(np.asarray(sids)))
    return f'{med:+.3g} ({pos}/{n} mice > 0, p {p:.0e})'


# ---- state --------------------------------------------------------------------------------------

def moments(st, names, groups, e, v, levels):
    """From summed state statistics (epochs x vars x levels x stats x ...): Var(p), and per group Cov(c, p), Var(c)."""
    s = np.nansum(st[e, v, list(levels)], axis=0)
    col = {k: i for i, k in enumerate(names)}
    n = s[col['n']]
    with np.errstate(invalid='ignore', divide='ignore'):
        mp = s[col['p']] / n
        var_p = s[col['pp']] / n - mp ** 2
        cov = np.stack([s[col[f'{g}_cp']] / n - s[col[f'{g}_c']] / n * mp for g in groups])
        var_c = np.stack([s[col[f'{g}_cc']] / n - (s[col[f'{g}_c']] / n) ** 2 for g in groups])
    return var_p, cov, var_c


def state_curves(d, stats_key, l4=True, kind='share'):
    """(vars running/pupil) x levels x groups x targets of the share, absolute covariance or correlation (kind)."""
    names, groups = [str(n) for n in d['stat_names']], [str(g) for g in d['groups']]
    V, E = [str(v) for v in d['state_vars']], [str(e) for e in d['state_epochs']]
    st = d[stats_key]
    if l4:                                                       # region_events: ... x channels x targets
        st = st[..., int(d['layer4_channel']), :][..., [int(t) for t in d['targets']]]
    out = np.full((2, 10, len(groups), st.shape[-1]), np.nan)
    for i, var in enumerate(('running', 'pupil')):
        for lev in range(10):
            var_p, cov, var_c = moments(st, names, groups, E.index('active'), V.index(var), (lev,))
            out[i, lev] = {'share': cov / var_p, 'cov': cov, 'corr': cov / np.sqrt(var_c * var_p)}[kind]
    return out, groups


def state():
    print('=== state: region contributions by running / pupil decile, active epoch, layer 4 channel')
    sources = {}
    for model, tree in TREES.items():
        rows = load(tree, 'region_events')
        sources[model] = {sid: state_curves(d, 'state_stats') for sid, d in rows.items()}
        sources[model + ' (cov)'] = {sid: state_curves(d, 'state_stats', kind='cov') for sid, d in rows.items()}
    ridge = load(TREES['CNN'], 'state_ridge')
    sources['ridge'] = {sid: state_curves(d, 'ridge_stats', l4=False) for sid, d in ridge.items()}
    sources['ridge (cov)'] = {sid: state_curves(d, 'ridge_stats', l4=False, kind='cov') for sid, d in ridge.items()}
    sources['rate-target corr'] = {sid: state_curves(d, 'rate_stats', l4=False, kind='corr') for sid, d in ridge.items()}
    r2 = np.stack([d['ridge_r2_test'] for d in ridge.values()]) if ridge else None
    if r2 is not None:
        print(f'lagged ridge at layer 4, {len(ridge)} sessions: test R2 median ' + ', '.join(
            f'{TARGETS[t]} {np.nanmedian(r2[:, j]):.3f}' for j, t in enumerate(ridge[next(iter(ridge))]['targets'])))
    groups = None
    for name, rows in sources.items():
        if not rows:
            print(f'{name}: no sessions')
            continue
        sids = np.array(list(rows))
        curves = np.stack([c for c, _ in rows.values()])                  # sessions x vars x levels x groups x targets
        groups = next(iter(rows.values()))[1]
        print(f'--- {name} ({len(sids)} sessions): deciles {HIGH} minus {LOW}, per mouse')
        for i, var in enumerate(('running', 'pupil')):
            for j, tn in TARGETS.items():
                for g in ('midbrain', 'VISp', 'visual_cortex_other', 'hippocampus', 'thalamus'):
                    k = groups.index(g)
                    hi, lo = np.nanmean(curves[:, i, list(HIGH), k, j], 1), np.nanmean(curves[:, i, list(LOW), k, j], 1)
                    print(f'  {var:8s} {tn:10s} {g:20s} {np.nanmedian(lo):+.3f} -> {np.nanmedian(hi):+.3f}  diff {mouse_test(hi - lo, sids)}')
    if groups is None:
        return {}
    return sources


def state_figure(sources, out):
    panels = [k for k in ('CNN', 'CNN-LSTM', 'ridge', 'rate-target corr', 'CNN (cov)', 'ridge (cov)') if sources.get(k)]
    fig, ax = plt.subplots(2 * len(TARGETS), len(panels), figsize=(3.1 * len(panels), 2.6 * 2 * len(TARGETS)), squeeze=False)
    for c, name in enumerate(panels):
        rows = sources[name]
        curves = np.stack([v for v, _ in rows.values()])
        groups = next(iter(rows.values()))[1]
        for i, var in enumerate(('running', 'pupil')):
            for j, tn in TARGETS.items():
                a = ax[i * len(TARGETS) + j, c]
                for g in SHOW_GROUPS:
                    k = groups.index(g)
                    y = curves[:, i, :, k, j]
                    ok = np.isfinite(y).all(1)
                    if not ok.any():
                        continue
                    a.plot(np.arange(10), np.median(y[ok], 0), color=COLORS[g], lw=2 if g == 'midbrain' else 1, label=g)
                    if g == 'midbrain':
                        a.fill_between(np.arange(10), *np.percentile(y[ok], [25, 75], 0), color=COLORS[g], alpha=0.15, lw=0)
                a.axhline(0, color='k', lw=0.4)
                a.set_title(f'{name}\n{tn}, by {var} decile', fontsize=8)
                a.set_xticks([0, 9])
                a.set_xticklabels(['low', 'high'], fontsize=7)
                a.tick_params(labelsize=7)
                if c == 0:
                    a.set_ylabel({'CNN': 'Shapley share'}.get(name, ''), fontsize=7)
    ax[0, 0].legend(fontsize=6)
    fig.suptitle('Region contributions at layer 4 by state (active epoch; median over sessions, midbrain IQR shaded)\n'
                 'share = Cov(c, p) / Var(p); cov = Cov(c, p) in target z units; rate-target corr: model-free', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'state_midbrain.png', dpi=120)
    plt.close(fig)
    print(f'figure: {out / "state_midbrain.png"}')


# ---- novelty ------------------------------------------------------------------------------------

def mouse_pairs(sids):
    meta = pd.read_csv(paths.REPO_DIR / 'tables' / 'sessions.csv').set_index('session_id')
    pairs = {}
    for sid in sids:
        if sid in meta.index:
            pairs.setdefault(meta.loc[sid, 'mouse_id'], {})[meta.loc[sid, 'experience_level']] = sid
    return [(v['Familiar'], v['Novel']) for v in pairs.values() if {'Familiar', 'Novel'} <= set(v)]


def window_rms(x, lags):
    """x (lags x ...), baseline corrected: RMS per EVENT_WINDOWS_MS window -> windows x ..."""
    x = x - np.nanmean(x[lags < 0], axis=0)
    return np.stack([np.sqrt(np.nanmean(x[(lags >= lo) & (lags < hi)] ** 2, axis=0)) for lo, hi in S.EVENT_WINDOWS_MS.values()])


def hippocampus_session(sid, info):
    """Per session: hippocampal unit count, firing rates and model-free coupling (units_info), and the RMS of the
    event-locked hippocampal firing (explore_task.npz, per unit and population) per type and window."""
    units = pd.read_csv(vdir(TREES['CNN'], sid) / 'units.csv')
    hpc = region_masks(units['structure_acronym'].to_numpy())['hippocampus']
    u = units.loc[hpc, ['unit_id']].merge(info, on='unit_id', how='left')
    out = dict(n=int(hpc.sum()), rate_total=u['firing_rate'].sum(), rate_mean=u['firing_rate'].mean(),
               coupling=u['coupling_index'].median(), coher_2_5=u['coher_2_5'].median(),
               coupling_mean=u['coupling_index'].mean())
    e = np.load(vdir(TREES['CNN'], sid) / 'explore_task.npz')
    types, groups = [str(t) for t in e['event_types']], [str(g) for g in e['region_groups']]
    g = groups.index('hippocampus')
    out['pop'] = {t: window_rms(e['eta_pop'][i, :, g], e['eta_lags_ms']) for i, t in enumerate(types)}    # spikes/s per unit
    l4 = int(e['layer4_channel'])
    out['lfp'] = {t: window_rms(e['eta_lfp'][i, :, l4, :2], e['eta_lags_ms']) for i, t in enumerate(types)}   # windows x targets
    return out


def novelty():
    print('=== novelty: hippocampal contribution to the shared-image responses, novel minus familiar day (same mouse)')
    info = pd.read_csv(UNITS_INFO)
    contrib = {}
    for model, tree in TREES.items():
        for sid, d in load(tree, 'region_events').items():
            l4, lags, types = int(d['layer4_channel']), d['eta_lags_ms'], [str(t) for t in d['event_types']]
            g = [str(x) for x in d['groups']].index('hippocampus')
            C = d['eta_contribution'][:, :, g, l4][..., [int(t) for t in d['targets']]]      # types x lags x targets
            contrib.setdefault(model, {})[sid] = {t: window_rms(C[i], lags) for i, t in enumerate(types)}
    pairs = mouse_pairs(set(contrib['CNN']) & set(contrib['CNN-LSTM']))
    hpc = {sid: hippocampus_session(sid, info) for p in pairs for sid in p}
    pairs = [(f, n) for f, n in pairs if hpc[f]['n'] > 0 and hpc[n]['n'] > 0]
    fam = np.array([f for f, _ in pairs])
    print(f'{len(pairs)} mice with both days and hippocampal units on both')
    log_ratio = lambda k: np.log10([hpc[n][k] / hpc[f][k] for f, n in pairs])
    print('day-to-day changes (novel - familiar; log10 ratio for counts and rates):')
    for k in ('n', 'rate_total', 'rate_mean'):
        print(f'  {k:12s} familiar {np.median([hpc[f][k] for f, _ in pairs]):8.2f}, novel {np.median([hpc[n][k] for _, n in pairs]):8.2f}; '
              f'log10 ratio {mouse_test(log_ratio(k), fam)}')
    for k in ('coupling', 'coupling_mean', 'coher_2_5'):
        d = np.array([hpc[n][k] - hpc[f][k] for f, n in pairs])
        print(f'  {k:12s} (model free, median over hippocampal units) diff {mouse_test(d, fam)}')

    results = {}
    for t in ('image_shared', 'change_shared', 'passive_image_shared'):
        print(f'--- {t}')
        for w, wn in enumerate(WINDOWS):
            pop = np.array([hpc[n]['pop'][t][w] - hpc[f]['pop'][t][w] for f, n in pairs])
            popn = np.array([hpc[n]['pop'][t][w] * hpc[n]['n'] - hpc[f]['pop'][t][w] * hpc[f]['n'] for f, n in pairs])
            print(f'  {wn:5s} event-locked hippocampal firing (model free): per unit {mouse_test(pop, fam)}; population {mouse_test(popn, fam)}')
            for j, tn in TARGETS.items():
                lfp = np.array([hpc[n]['lfp'][t][w, j] - hpc[f]['lfp'][t][w, j] for f, n in pairs])
                print(f'  {wn:5s} {tn:10s} recorded layer 4 response, RMS diff {mouse_test(lfp, fam)}')
                for model in TREES:
                    a = np.array([contrib[model][f][t][w, j] for f, _ in pairs])
                    b = np.array([contrib[model][n][t][w, j] for _, n in pairs])
                    nf, nn = (np.array([hpc[s]['n'] for s in x]) for x in (fam, [n for _, n in pairs]))
                    pf = np.array([hpc[f]['pop'][t][w] * hpc[f]['n'] for f, _ in pairs])
                    pn = np.array([hpc[n]['pop'][t][w] * hpc[n]['n'] for _, n in pairs])
                    ratio = np.log10(b / a)
                    X = sm.add_constant(np.column_stack([np.log10(nn / nf), log_ratio('rate_mean')]))
                    ok = np.isfinite(ratio) & np.isfinite(X).all(1)
                    fit = sm.OLS(ratio[ok], X[ok]).fit()
                    matched = np.abs(np.log10(nn / nf)) < np.log10(1.5)
                    print(f'    {model:9s} raw {mouse_test(b - a, fam)} | per unit {mouse_test(b / nn - a / nf, fam)} | '
                          f'per population firing {mouse_test(b / pn - a / pf, fam)}')
                    print(f'    {"":9s} log10 ratio ~ 1 + log unit ratio + log rate ratio: intercept {fit.params[0]:+.3f} '
                          f'(p {fit.pvalues[0]:.0e}), unit slope {fit.params[1]:+.2f} (p {fit.pvalues[1]:.0e}), rate slope '
                          f'{fit.params[2]:+.2f} (p {fit.pvalues[2]:.0e}), n {ok.sum()} | unit count within 1.5x ({matched.sum()} mice): '
                          f'{mouse_test((b - a)[matched], fam[matched])}')
                    results[(t, wn, tn, model)] = dict(a=a, b=b, nf=nf, nn=nn, pf=pf, pn=pn, fit=fit)
    return dict(pairs=pairs, hpc=hpc, results=results)


def novelty_figure(nov, out):
    pairs, hpc, res = nov['pairs'], nov['hpc'], nov['results']
    t, wn, tn = 'image_shared', 'late', 'broadband'
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.4))
    for c, model in enumerate(TREES):
        r = res[(t, wn, tn, model)]
        a = ax[c]
        a.scatter(r['a'], r['b'], s=14, c=np.log2(r['nn'] / r['nf']), cmap='coolwarm', vmin=-1.5, vmax=1.5)
        lim = [0, np.nanmax([r['a'], r['b']]) * 1.05]
        a.plot(lim, lim, 'k', lw=0.5)
        a.set_xlabel('familiar day', fontsize=8)
        a.set_ylabel('novel day', fontsize=8)
        a.set_title(f'{model}: hippocampal contribution\n{t}, {wn} window, {tn} (RMS, z)', fontsize=8)
    r = res[(t, wn, tn, 'CNN')]
    a = ax[2]
    x = np.log10(r['nn'] / r['nf'])
    a.scatter(x, np.log10(r['b'] / r['a']), s=14, color='k')
    a.axhline(0, color='k', lw=0.4)
    a.axvline(0, color='k', lw=0.4)
    a.set_xlabel('log10 hippocampal units, novel / familiar', fontsize=8)
    a.set_ylabel('log10 contribution, novel / familiar (CNN)', fontsize=8)
    a.set_title(f'unit-count control: intercept {r["fit"].params[0]:+.2f} (p {r["fit"].pvalues[0]:.0e})', fontsize=8)
    a = ax[3]
    pf = np.array([hpc[f]['pop'][t][WINDOWS.index(wn)] for f, _ in pairs])
    pn = np.array([hpc[n]['pop'][t][WINDOWS.index(wn)] for _, n in pairs])
    a.scatter(pf, pn, s=14, color='tab:green')
    lim = [0, np.nanmax([pf, pn]) * 1.05]
    a.plot(lim, lim, 'k', lw=0.5)
    a.set_xlabel('familiar day', fontsize=8)
    a.set_ylabel('novel day', fontsize=8)
    a.set_title(f'model free: event-locked hippocampal firing\n{t}, {wn} window (RMS, spikes/s per unit)', fontsize=8)
    a = ax[4]
    for k, (key, label) in enumerate((('coupling', 'coupling index'), ('coher_2_5', '2-5 Hz coherence'))):
        d = np.array([hpc[n][key] - hpc[f][key] for f, n in pairs])
        a.scatter(np.full(len(d), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d, s=10, color='k')
        a.plot([k - 0.25, k + 0.25], [np.nanmedian(d)] * 2, color='r')
    a.axhline(0, color='k', lw=0.4)
    a.set_xticks([0, 1])
    a.set_xticklabels(['coupling index', '2-5 Hz coherence'], fontsize=8)
    a.set_title('model free: hippocampal spike-LFP coupling\nnovel - familiar (median over units)', fontsize=8)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.suptitle(f'Hippocampal contribution on the novel vs familiar day, same images, same mouse ({len(pairs)} mice; '
                 'colour: log2 unit-count ratio)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'novelty_hippocampus.png', dpi=120)
    plt.close(fig)
    print(f'figure: {out / "novelty_hippocampus.png"}')


# ---- overview -----------------------------------------------------------------------------------

def overview(out):
    print('=== overview: the other model-independent candidates, CNN vs CNN-LSTM')
    table = pd.read_csv(S.ALIGNMENT_TABLE).set_index('session')
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for m, (model, tree) in enumerate(TREES.items()):
        # (1) VISp minus other visual cortex, fraction of R2 lost at inference, by band
        diff, sids = [], []
        for sid, d in load(tree, 'region_ablation').items():
            ch = S.pooled_channels(sid)
            regions = [str(r) for r in d['regions']]
            a, b = regions.index('VISp'), regions.index('visual_cortex_other')
            if d['region_n_units'][a] == 0 or d['region_n_units'][b] == 0:
                continue
            full = np.nanmedian(d['r2_full'][ch], 0)
            drop = np.nanmedian((d['r2_full'] - d['r2_removed'][0])[[a, b]][:, ch], 1) / full
            diff.append(drop[0] - drop[1])
            sids.append(sid)
        diff = np.stack(diff)[:, :5]
        print(f'{model}: VISp - other visual, fraction lost at inference: ' + ' | '.join(
            f'{S.TARGET_NAMES[t]} {mouse_test(diff[:, t], sids)}' for t in range(5)))
        bp = ax[0].boxplot([diff[:, t] for t in range(5)], positions=np.arange(5) + (m - 0.5) * 0.35, widths=0.3,
                           patch_artist=True, showfliers=False)
        for p in bp['boxes']:
            p.set_facecolor(['tab:blue', 'tab:orange'][m])
            p.set_alpha(0.6)
        # (2) layer 4 fraction of the CSD driven by VISp and by the other visual areas (image, late)
        frac, sids2 = [], []
        # (3) hippocampal share of the hit-minus-miss difference (broadband, late)
        share, sids3 = [], []
        for sid, d in load(tree, 'region_events').items():
            groups = [str(g) for g in d['groups']]
            if sid in table.index:
                paths.OUTPUT_SUBDIR = tree                         # depth_session reads the tree's target scaler
                r = S.depth_session(d, sid, table)
                if 'image' in r['frac']:
                    f = r['frac']['image'][WINDOWS.index('late'), :, 0, 1]                    # groups (L4 fraction)
                    if d['group_n_units'][0] > 0 and d['group_n_units'][1] > 0:
                        frac.append(f[[0, 1]])
                        sids2.append(sid)
            types = [str(t) for t in d['event_types']]
            if 'change_hit' in types and 'change_miss' in types and d['group_n_units'][groups.index('hippocampus')] > 0:
                l4, lags = int(d['layer4_channel']), d['eta_lags_ms']
                C = d['eta_contribution'][:, :, :, l4][..., [int(t) for t in d['targets']]]
                sh = S.window_rms_shares(C[types.index('change_hit')] - C[types.index('change_miss')], lags)
                share.append(sh[WINDOWS.index('late'), groups.index('hippocampus'), 0])
                sids3.append(sid)
        frac = np.stack(frac)
        print(f'{model}: layer 4 fraction of the CSD (image, late): VISp {np.median(frac[:, 0]):.2f}, other visual '
              f'{np.median(frac[:, 1]):.2f}; other - VISp {mouse_test(frac[:, 1] - frac[:, 0], sids2)}')
        share = np.array(share)
        print(f'{model}: hippocampal share of hit - miss (broadband, late) median {np.nanmedian(share):.2f} (n {np.isfinite(share).sum()})')
        for k in range(2):
            ax[1].scatter(np.full(len(frac), 2 * m + k) + np.random.default_rng(k).uniform(-0.2, 0.2, len(frac)), frac[:, k], s=5,
                          color=COLORS[['VISp', 'visual_cortex_other'][k]], alpha=0.5)
            ax[1].plot([2 * m + k - 0.3, 2 * m + k + 0.3], [np.median(frac[:, k])] * 2, 'k')
        ax[2].hist(share[np.isfinite(share)], bins=np.linspace(0, 0.6, 25), histtype='step', lw=1.5, color=['tab:blue', 'tab:orange'][m], label=model)
    ax[0].axhline(0, color='k', lw=0.4)
    ax[0].set_xticks(range(5))
    ax[0].set_xticklabels(S.TARGET_NAMES, fontsize=7)
    ax[0].set_title('VISp - other visual: fraction of R2 lost at inference\n(blue CNN, orange CNN-LSTM; per session)', fontsize=8)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(['CNN\nVISp', 'CNN\nother vis', 'CNN-LSTM\nVISp', 'CNN-LSTM\nother vis'], fontsize=7)
    ax[1].set_title('layer 4 fraction of the region-driven CSD\n(image, 100-250 ms, broadband)', fontsize=8)
    ax[2].set_xlabel('hippocampal share', fontsize=8)
    ax[2].set_title('hippocampal share of hit - miss\n(broadband, 100-250 ms, per session)', fontsize=8)
    ax[2].legend(fontsize=7)
    for a in ax:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / 'candidates_overview.png', dpi=120)
    plt.close(fig)
    print(f'figure: {out / "candidates_overview.png"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('what', choices=['state', 'novelty', 'overview', 'all'])
    parser.add_argument('--figures', default=str(paths.REPO_DIR / 'docs' / 'figures'))
    args = parser.parse_args()
    out = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    if args.what in ('state', 'all'):
        sources = state()
        if sources:
            state_figure(sources, out)
    if args.what in ('novelty', 'all'):
        novelty_figure(novelty(), out)
    if args.what in ('overview', 'all'):
        overview(out)


if __name__ == '__main__':
    main()
