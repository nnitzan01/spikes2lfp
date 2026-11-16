"""
Figures for inspecting the 89-session results (REVISION_STATUS section 4), one PNG per finding, from the result
files the experiments already saved (nothing is rerun). Per-session values are medians over the pooled channels
(experiments/summarize.py); dots are sessions (or mice, or simulation draws), black bars and lines medians, shading
the interquartile range.

usage (on the cluster, from the repo root, after `source cluster/env.sh`):
    python -m experiments.review_figures [baselines stimulus_clock ...]      # default: all

Saves <data root>/spikes2lfp_v2/figures_review/<name>.png:
    baselines            CNN-LSTM vs ridge and stimulus / behaviour models (test R2)
    stimulus_clock       model vs stimulus-clock template, spikes swapped between presentations, time-shifted
    unit_ranking         R2 lost when the top units by each measure are removed (inference and retrained)
    simulation           ground-truth simulation: drivers found by attribution vs ridge weight and the rest
    region_bands         fraction of R2 lost per region and band (inference, retrained), by depth
    region_clock         R2 lost per region when removed vs when only its spikes are swapped (trial-specific part)
    high_bands           high-band R2 by smoothing window against the time-shift null; without nearby / VISp units
    event_contributions  Shapley contributions of each region to the event-locked LFP at layer 4, shares per window
    latency              timing of the contributions against the regions' own firing
    novelty              shared images, novel vs familiar day (same mouse)
    depth                CSD of the contributions (broadband) and their laminar fractions against the Allen atlas
    state                Shapley variance shares by running, pupil, engagement, active vs passive
    midbrain_licks       midbrain 0.5-4 Hz contribution against licking and running
"""
import argparse
import functools
import shutil
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from experiments import summarize as S
from experiments.pool_sessions import ALIGNMENT_TABLE

OUT = paths.DATA_ROOT / paths.OUTPUT_SUBDIR / 'figures_review'
COLORS = dict(VISp='tab:blue', visual_cortex_other='tab:orange', hippocampus='tab:green', thalamus='tab:red',
              midbrain='tab:purple', other='tab:brown')
LABELS = dict(visual_cortex_other='other visual')
REGIONS = ['VISp', 'visual_cortex_other', 'hippocampus', 'thalamus', 'midbrain']
BANDS = S.ALL_BANDS                                     # broadband, 0.5-4, ..., 200-400 Hz (target index)
rng = np.random.default_rng(0)


def label(g):
    return LABELS.get(g, g)


def strip(ax, values, x, color='gray', width=0.5, s=6):
    """Dots (jittered) and a black median bar at x."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    ax.scatter(x + rng.uniform(-width / 2, width / 2, len(v)), v, s=s, color=color, alpha=0.45, lw=0)
    ax.plot([x - width / 2, x + width / 2], [np.median(v)] * 2, color='k', lw=2)


def median_band(ax, x, rows, color, text=None, ls='-', shade=True):
    rows = np.asarray(rows, float)
    ax.plot(x, np.nanmedian(rows, 0), color=color, label=text, ls=ls)
    if shade:
        ax.fill_between(x, *np.nanpercentile(rows, [25, 75], 0), color=color, alpha=0.2, lw=0)


def identity(ax, x, y, color='k', text=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.scatter(x, y, s=8, color=color, alpha=0.6, lw=0, label=text)
    lo, hi = np.nanmin([x, y]), np.nanmax([x, y])
    ax.plot([lo, hi], [lo, hi], color='k', lw=0.5)


def save(fig, name, title):
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=120)
    plt.close(fig)
    print(f'saved {OUT / f"{name}.png"}')


def l4_relative(sid):
    """Pooled channels of a session relative to the layer 4 centre (channels, > 0 towards the surface)."""
    row = pd.read_csv(ALIGNMENT_TABLE).set_index('session').loc[sid]
    l4 = (row['upper_border_index'] + row['lower_border_index']) / 2
    return np.arange(row['first_channel_index'], row['last_channel_index'] + 1) - l4


def on_grid(values, rel, half=S.DEPTH_HALF_WIDTH):
    """values (channels x ...) at rel (channels from L4) -> (2 half + 1) x ... grid of channels -half..half."""
    grid = np.full((2 * half + 1,) + values.shape[1:], np.nan)
    pos = np.round(rel).astype(int) + half
    keep = (pos >= 0) & (pos < len(grid))
    grid[pos[keep]] = values[keep]
    return grid


# ---------------------------------------------------------------------------------------------------- prediction

BASELINE_MODELS = [('r2_units_instant', 'ridge, current bin'), ('r2_units', 'ridge, 1 s history'),
                   ('r2_units_stimulus_behaviour', 'ridge, 1 s + stim. + beh.'), ('r2_regions', 'ridge, region counts'),
                   ('r2_stimulus_behaviour', 'stimulus + behaviour'), ('model_r2', 'CNN-LSTM')]


def baselines():
    rows = list(S.over_sessions('baselines', lambda d, ch: {
        k: S.channel_median(d[k], ch)[:3] for k, _ in BASELINE_MODELS if k in d}).values())
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for t in range(3):
        for i, (k, _) in enumerate(BASELINE_MODELS):
            strip(ax[0, t], [r[k][t] for r in rows if k in r], i, 'k' if k == 'model_r2' else 'gray')
        ax[0, t].set_xticks(range(len(BASELINE_MODELS)), [n for _, n in BASELINE_MODELS], rotation=35, ha='right', fontsize=8)
        ax[0, t].set_title(f'{BANDS[t]}: test R² per session', fontsize=9)
        ax[0, t].set_ylim(bottom=-0.3)
        x, y = np.array([r['r2_units'][t] for r in rows]), np.array([r['model_r2'][t] for r in rows])
        identity(ax[1, t], x, y)
        ax[1, t].set_xlim(left=-0.3)
        ax[1, t].set_ylim(bottom=-0.3)
        ax[1, t].set_xlabel('ridge, 1 s history (axes clipped at -0.3)')
        ax[1, t].set_ylabel('CNN-LSTM')
        ax[1, t].set_title(f'{BANDS[t]}: CNN-LSTM − ridge {np.nanmedian(y - x):+.3f}, higher in {np.sum(y > x)}/{len(x)}', fontsize=9)
    save(fig, 'baselines', f'Baselines ({len(rows)} sessions): history is most of the gain over the v1 linear model; '
                           'the CNN-LSTM is a little but consistently better')


CLOCK_KEYS = [('r2_model', 'model'), ('r2_template', 'stimulus clock\n(template)'),
              ('r2_shuffle', 'spikes swapped,\nsame image'), ('r2_shift', 'spikes shifted\n30-120 s')]


def stimulus_clock():
    rows = list(S.over_sessions('stimulus_clock', lambda d, ch: {
        k: S.channel_median(np.nanmean(d[k], 0) if d[k].ndim == 3 else d[k], ch)[:5] for k, _ in CLOCK_KEYS}).values())
    fig, ax = plt.subplots(2, 5, figsize=(17, 7))
    for t in range(5):
        for i, (k, _) in enumerate(CLOCK_KEYS):
            strip(ax[0, t], [r[k][t] for r in rows], i, 'k' if i == 0 else 'gray')
        ax[0, t].axhline(0, color='k', lw=0.5)
        ax[0, t].set_ylim(-1, 1)
        ax[0, t].set_xticks(range(len(CLOCK_KEYS)), [n for _, n in CLOCK_KEYS], fontsize=7, rotation=30, ha='right')
        ax[0, t].set_title(f'{BANDS[t]}: test R²', fontsize=9)
        x, y = np.array([r['r2_template'][t] for r in rows]), np.array([r['r2_model'][t] for r in rows])
        identity(ax[1, t], x, y)
        ax[1, t].set_xlabel('stimulus clock (template)')
        ax[1, t].set_ylabel('model')
        ax[1, t].set_title(f'model higher in {np.sum(y > x)}/{len(x)}', fontsize=9)
    save(fig, 'stimulus_clock', f'Stimulus clock ({len(rows)} sessions): the model beats the condition-average template '
                                'and falls below it when given another presentation\'s spikes (trial-specific prediction)')


UNIT_MEASURES = [('attribution_baseline', 'attribution_mean_baseline', 'attribution, mean baseline', 'k'),
                 ('attribution_baseline', 'attribution', 'attribution, zero baseline', 'gray'),
                 ('attribution_baseline', 'ridge_weight', 'ridge weight', 'tab:red'),
                 ('attribution_baseline', 'spike_lfp_correlation', 'spike-LFP correlation', 'tab:orange'),
                 ('ranked_ablation', 'stlfp', 'stLFP', 'tab:green'),
                 ('ranked_ablation', 'population_coupling', 'population coupling', 'tab:purple'),
                 ('ranked_ablation', 'firing_rate', 'firing rate', 'tab:brown')]


def unit_ranking():
    from experiments.ranked_ablation import MEASURES

    def per_session(d, ch):
        out = dict(measures=[str(m) for m in d['measures']] if 'measures' in d else MEASURES, fractions=d['fractions'],
                   drop=S.channel_median(d['r2_full'] - d['r2_ablated'][0], ch),                # measures x fractions x targets
                   random=np.nanmean(S.channel_median(d['r2_full'] - d['r2_random'][0], ch), 0))
        if 'r2_retrained' in d:
            out['retrained'] = S.channel_median(d['r2_retrained_full'][0] - d['r2_retrained'][:, :, 0], ch)[..., 0]
            out['retrain_fractions'] = d['retrain_fractions']
        return out

    data = {name: list(S.over_sessions(name, per_session).values()) for name in ('attribution_baseline', 'ranked_ablation')}
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    for t in range(3):
        for source, m, text, color in UNIT_MEASURES:
            rows = data[source]
            fr = rows[0]['fractions']
            median_band(ax[t], fr * 100, [r['drop'][r['measures'].index(m), :, t] for r in rows], color, text, shade=color == 'k')
        rows = data['attribution_baseline']
        median_band(ax[t], rows[0]['fractions'] * 100, [r['random'][:, t] for r in rows], 'gray', 'random', ls=':', shade=False)
        ax[t].set_xscale('log')
        ax[t].set_xlabel('top units removed (%)')
        ax[t].set_ylabel('R² lost (inference, mean fill)')
        ax[t].set_title(BANDS[t], fontsize=9)
    ax[0].legend(fontsize=7)
    rows = [r for r in data['attribution_baseline'] if 'retrained' in r]
    measures = rows[0]['measures']
    for f, frac in enumerate(rows[0]['retrain_fractions']):
        for i, (_, m, text, color) in enumerate([u for u in UNIT_MEASURES if u[0] == 'attribution_baseline']):
            strip(ax[3], [r['retrained'][measures.index(m), f] for r in rows], f * 5 + i, color, width=0.7)
    ax[3].set_xticks([1.5, 6.5], [f'{100 * x:g}% removed' for x in rows[0]['retrain_fractions']])
    ax[3].set_ylabel('R² lost, retrained without them (broadband)')
    ax[3].set_title('retrained: ridge weight / correlation catch up', fontsize=9)
    save(fig, 'unit_ranking', 'Which units does the model need? Mean-baseline attribution is the best ranking at inference '
                              f'({len(rows)} sessions); retrained, ridge weights and correlation find the irreplaceable units')


def simulation():
    draws = {}
    for sid in S.sessions():
        for f in sorted(S.variables_dir(sid).glob('simulation_final*.npz')):
            d = np.load(f)
            measures = [str(m) for m in d['measures']]
            for ci, c in enumerate(d['conditions']):
                for k in range(d['r2'].shape[1]):
                    draws.setdefault(str(c), []).append({(m, metric): d[metric][ci, k, i] for metric in ('auc', 'spearman_total')
                                                         for i, m in enumerate(measures)} | dict(r2=d['r2'][ci, k], ridge=d['r2_ridge'][ci, k]))
    if not draws:
        print('no simulation_final*.npz')
        return
    conditions = list(draws)
    colors = dict(zip(conditions, plt.cm.tab10(np.arange(len(conditions)))))
    shown = ['ig_mean', 'ig_zero', 'ridge_weight', 'spike_lfp_correlation', 'stlfp', 'population_coupling', 'firing_rate']
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for c, rows in draws.items():
        for p, metric in enumerate(('auc', 'spearman_total')):
            identity(ax[p], [r[('ridge_weight', metric)] for r in rows], [r[('ig_mean', metric)] for r in rows], colors[c], c)
        identity(ax[2], [r['ridge'] for r in rows], [r['r2'] for r in rows], colors[c], c)
    for p, (metric, text) in enumerate((('auc', 'driver detection (AUC)'), ('spearman_total', 'Spearman with true total effect'))):
        ax[p].set_xlabel(f'ridge weight, {text}')
        ax[p].set_ylabel(f'attribution (mean baseline), {text}')
    ax[2].set_xlim(-0.5, 0.8)
    ax[2].set_ylim(-0.5, 0.8)
    ax[2].set_xlabel('ridge test R² (axes clipped at -0.5)')
    ax[2].set_ylabel('CNN-LSTM test R²')
    ax[0].legend(fontsize=7)
    for i, c in enumerate(conditions):
        for j, m in enumerate(shown):
            strip(ax[3], [r[(m, 'auc')] for r in draws[c]], j + i * 0.18 - 0.27, colors[c], width=0.15, s=4)
    ax[3].axhline(0.5, color='k', lw=0.5)
    ax[3].set_xticks(range(len(shown)), shown, rotation=35, ha='right', fontsize=8)
    ax[3].set_ylabel('driver detection (AUC)')
    save(fig, 'simulation', f'Ground-truth simulation (LFP-like regime, {sum(map(len, draws.values()))} draws): attribution '
                            'finds the drivers better than ridge weights and much better than correlational measures')


# ------------------------------------------------------------------------------------------------------ regions

def region_bands():
    def per_session(d, ch, sid):
        regions = [str(r) for r in d['regions']]
        idx = [regions.index(r) for r in REGIONS]
        full = d['r2_full'][ch][:, :5]
        drop = (d['r2_full'] - d['r2_removed'][0])[idx][:, ch][..., :5]                    # regions x channels x targets
        out = dict(n=d['region_n_units'][idx], frac=np.nanmedian(drop, 1) / np.nanmedian(full, 0))
        own = np.where(full > S.MIN_PROFILE_R2, full, np.nan)
        out['profile'] = np.moveaxis(on_grid(np.moveaxis(drop / own, 1, 0), l4_relative(sid)), 0, 1)   # regions x depth x targets
        if 'r2_retrained_full' in d:
            rfull = d['r2_retrained_full'][0][ch][:, :5]
            out['retrained'] = np.nanmedian(rfull - d['r2_retrained_removed'][idx, 0][:, ch][..., :5], 1) / np.nanmedian(rfull, 0)
        return out

    def over(name):
        rows = {}
        for sid in S.sessions():
            f = S.variables_dir(sid) / f'{name}.npz'
            if f.exists():
                rows[sid] = per_session(np.load(f), S.pooled_channels(sid), sid)
        return list(rows.values())

    rows, retrained = over('region_ablation'), over('region_ablation_bands')
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for i, g in enumerate(REGIONS):
        for t in range(5):
            strip(ax[0, 0], [r['frac'][i, t] for r in rows if r['n'][i] > 0], t + (i - 2) * 0.17, COLORS[g], width=0.14, s=4)
        for t in range(3):
            strip(ax[0, 1], [r['retrained'][i, t] for r in retrained if r['n'][i] > 0 and 'retrained' in r],
                  t + (i - 2) * 0.17, COLORS[g], width=0.14, s=4)
        ax[0, 0].plot([], [], 'o', color=COLORS[g], label=label(g))
    for a, n, title in ((ax[0, 0], 5, 'removed at inference (mean fill)'), (ax[0, 1], 3, 'retrained without the region')):
        a.set_xticks(range(n), BANDS[:n])
        a.axhline(0, color='k', lw=0.5)
        a.set_ylabel('fraction of the full R² lost')
        a.set_title(title + (' (a few outliers below -0.5 not shown)' if n == 5 else ''), fontsize=9)
    ax[0, 0].set_ylim(-0.5, 1.2)
    ax[0, 1].set_ylim(-0.2, 0.6)
    ax[0, 0].legend(fontsize=7)
    diff = np.array([r['frac'][0] - r['frac'][1] for r in rows if r['n'][0] > 0 and r['n'][1] > 0])
    for v in diff:
        ax[0, 2].plot(range(5), v, color='gray', lw=0.4, alpha=0.5)
    ax[0, 2].plot(range(5), np.median(diff, 0), color='k', lw=2)
    ax[0, 2].axhline(0, color='k', lw=0.5)
    ax[0, 2].set_xticks(range(5), BANDS[:5])
    ax[0, 2].set_ylim(-1.5, 1)
    ax[0, 2].set_ylabel('VISp − other visual, fraction lost')
    ax[0, 2].set_title(f'per session ({len(diff)}): VISp leads broadband, other visual leads every envelope', fontsize=9)
    depth = np.arange(-S.DEPTH_HALF_WIDTH, S.DEPTH_HALF_WIDTH + 1)
    for p, g in enumerate(['VISp', 'visual_cortex_other', 'hippocampus']):
        i = REGIONS.index(g)
        for t, color in zip((0, 1, 2, 3), ('k', 'tab:blue', 'tab:green', 'tab:red')):
            prof = np.array([r['profile'][i, :, t] for r in rows if r['n'][i] > 0])
            ax[1, p].plot(np.nanmedian(prof, 0), depth, color=color, label=BANDS[t])
        ax[1, p].axhspan(-2, 2, color='gray', alpha=0.1)
        ax[1, p].set_xlabel('fraction of the channel\'s R² lost (inference)')
        ax[1, p].set_ylabel('channels from L4 centre (40 µm; up = superficial)')
        ax[1, p].set_title(f'{label(g)} by depth (grey ≈ L4)', fontsize=9)
    ax[1, 0].legend(fontsize=7)
    save(fig, 'region_bands', f'Two sources ({len(rows)} sessions; retrained {len(retrained)}): VISp carries the broadband LFP, '
                              'the other visual areas the 0.5-25 Hz band envelopes')


def region_clock():
    def per_session(d, ch):
        swap = np.nanmean(d['r2_swap'], 0)
        return dict(regions=[str(r) for r in d['regions']], n=d['region_n_units'],
                    removed=S.channel_median(d['r2_model'] - d['r2_removed'], ch)[:, :2],
                    swapped=S.channel_median(d['r2_model'] - swap, ch)[:, :2])

    rows = list(S.over_sessions('region_clock', per_session).values())
    regions = rows[0]['regions']
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
    for p, (t, shown) in enumerate(((0, ['VISp', 'visual_cortex_other']), (0, ['hippocampus', 'midbrain', 'thalamus']),
                                    (1, ['VISp', 'visual_cortex_other']))):
        for g in shown:
            i = regions.index(g)
            ok = [r for r in rows if r['n'][i] > 0]
            identity(ax[p], [r['removed'][i, t] for r in ok], [r['swapped'][i, t] for r in ok], COLORS[g], label(g))
        ax[p].set_xlabel('R² lost, region removed')
        ax[p].set_ylabel('R² lost, its spikes swapped (same image)')
        ax[p].set_title(f'{BANDS[t]} (on the line: all of it trial-specific)', fontsize=9)
        ax[p].legend(fontsize=7)
    for i, g in enumerate(REGIONS):
        for t in range(2):
            v = [r['swapped'][regions.index(g), t] / r['removed'][regions.index(g), t] for r in rows
                 if r['n'][regions.index(g)] > 0 and r['removed'][regions.index(g), t] > 0.005]
            strip(ax[3], v, i + (t - 0.5) * 0.35, COLORS[g] if t == 0 else 'gray', width=0.3, s=5)
    ax[3].axhline(1, color='k', lw=0.5)
    ax[3].set_ylim(-0.5, 4)
    ax[3].set_xticks(range(len(REGIONS)), [label(g) for g in REGIONS], fontsize=8)
    ax[3].set_ylabel('swapped / removed (colour broadband, grey 0.5-4 Hz)')
    ax[3].set_title('trial-specific fraction (sessions where removing costs > 0.005)', fontsize=9)
    save(fig, 'region_clock', f'Evoked vs trial-specific by region ({len(rows)} sessions): most of each cortical contribution '
                              'is trial-specific; hippocampus helps broadband only through correctly timed spikes')


def high_bands():
    shown = [1, 5, 6, 7, 8]

    def per_timescale(d, ch):
        return dict(windows=d['windows_ms'], r2=S.channel_median(d['r2_timescale'], ch),
                    null=S.channel_median(np.nanmean(d['r2_timescale_shift'], 0), ch))

    def per_bleed(d, ch):
        return dict(groups=[str(g) for g in d['retrain_groups']], r2=S.channel_median(d['ret_r2_timescale'], ch)[:, -1])

    ts = list(S.over_sessions('timescale_r2', per_timescale).values())
    bt = list(S.over_sessions('bleed_through', per_bleed).values())
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(shown)))
    w = ts[0]['windows']
    for t, color in zip(shown, colors):
        median_band(ax[0], w, [r['r2'][:, t] for r in ts], color, BANDS[t])
        ax[0].plot(w, np.nanmedian([r['null'][:, t] for r in ts], 0), color=color, ls=':')
    ax[0].set_xscale('log')
    ax[0].set_ylim(-0.3, 0.7)
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_xlabel('smoothing window (ms)')
    ax[0].set_ylabel('test R² (dotted: spikes shifted 30-120 s)')
    ax[0].legend(fontsize=7)
    ax[0].set_title('high-band power: its slow modulation is predicted', fontsize=9)
    groups = bt[0]['groups']
    names = dict(all='all units', near_100='without units\nwithin 100 µm', visp='without\nany VISp unit')
    for j, t in enumerate(shown):
        for i, g in enumerate(groups):
            strip(ax[1], [r['r2'][i, t] for r in bt], j + (i - 1) * 0.27, colors[j], width=0.22, s=4)
    ax[1].set_xticks(range(len(shown)), [BANDS[t] for t in shown], fontsize=8)
    ax[1].set_ylabel('test R², 250 ms smoothing (retrained)')
    ax[1].set_title('per band: all units | without units within 100 µm | without VISp', fontsize=9)
    for j, t in enumerate(shown):
        for i, g in enumerate(groups[1:], 1):
            strip(ax[2], [r['r2'][i, t] - r['r2'][0, t] for r in bt], j + (i - 1.5) * 0.4, colors[j], width=0.3, s=4)
    ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set_xticks(range(len(shown)), [BANDS[t] for t in shown], fontsize=8)
    ax[2].set_ylabel('change in R² from all units')
    ax[2].set_title('left: without units within 100 µm; right: without VISp', fontsize=9)
    save(fig, 'high_bands', f'High bands ({len(ts)} sessions): slow modulation of 100-400 Hz power is predicted and is not '
                            f'spike bleed-through ({len(bt)} sessions retrained without nearby or VISp units)')


# ------------------------------------------------------------------------------------------- task events (F-H)

@functools.lru_cache(maxsize=1)
def events():
    """Per session, from region_events.npz (and explore_task.npz): the baseline-corrected event-locked contributions at
    layer 4 (types x lags x groups x targets), state shares, the depth (CSD) profiles, and each group's own firing
    and the behaviour around the same events."""
    table = pd.read_csv(ALIGNMENT_TABLE).set_index('session')
    rows = {}
    for sid in S.sessions():
        f = S.variables_dir(sid) / 'region_events.npz'
        if not f.exists():
            continue
        d = np.load(f)
        lags, l4 = d['eta_lags_ms'], int(d['layer4_channel'])
        targets, groups = [int(t) for t in d['targets']], [str(g) for g in d['groups']]
        C = d['eta_contribution'][:, :, :, l4][..., targets]
        C = C - np.nanmean(C[:, lags < 0], 1, keepdims=True)
        stats, names = d['state_stats'][..., l4, :][..., targets], [str(n) for n in d['stat_names']]
        E, V = [str(e) for e in d['state_epochs']], [str(v) for v in d['state_vars']]
        state = {c: (S.state_shares(stats, names, groups, E.index(c[0]), V.index(c[1]), c[2]),
                     S.state_shares(stats, names, groups, E.index(c[3]), V.index(c[4]), c[5])) for c in S.STATE_CONTRASTS}
        row = dict(types=[str(t) for t in d['event_types']], lags=lags, groups=groups, targets=targets, C=C,
                   has=d['group_n_units'] > 0, state=state, depth=S.depth_session(d, sid, table) if sid in table.index else None)
        e = S.variables_dir(sid) / 'explore_task.npz'
        if e.exists():
            e = np.load(e)
            if np.allclose(e['eta_lags_ms'], lags):
                row.update(etypes=[str(t) for t in e['event_types']], egroups=[str(g) for g in e['region_groups']],
                           rate=e['eta_pop'], behaviour=e['eta_behaviour'], bnames=[str(b) for b in e['behaviour_names']])
        rows[sid] = row
    print(f'{len(rows)} sessions with region_events.npz')
    return rows


def contribution(r, t, g, j=0):
    """Baseline-corrected contribution of group g to event type t (target j: 0 broadband, 1 0.5-4 Hz) or None."""
    if t not in r['types'] or g not in r['groups'] or not r['has'][r['groups'].index(g)]:
        return None
    x = r['C'][r['types'].index(t), :, r['groups'].index(g), j]
    return x if np.isfinite(x).all() else None


def event_contributions():
    rows = list(events().values())
    lags, groups = rows[0]['lags'], rows[0]['groups']
    types = ['image', 'change_hit', 'change_miss', 'passive_image']
    fig, ax = plt.subplots(2, len(types) + 1, figsize=(20, 8))
    for j, tn in enumerate(['broadband', '0.5-4 Hz']):
        for i, t in enumerate(types):
            for g in groups:
                x = [v for v in (contribution(r, t, g, j) for r in rows) if v is not None]
                if x:
                    median_band(ax[j, i], lags, x, COLORS[g], label(g), shade=g in ('VISp', 'visual_cortex_other', 'midbrain'))
            ax[j, i].axvline(0, color='k', lw=0.5)
            ax[j, i].axvspan(0, 250, color='gray', alpha=0.07)
            ax[j, i].set_title(f'{t}, {tn}', fontsize=9)
            ax[j, i].set_xlabel('time from onset (ms; grey: image on)')
        ax[j, 0].set_ylabel('contribution at L4 (z)')
        for gi, g in enumerate(['VISp', 'visual_cortex_other', 'midbrain', 'hippocampus']):
            for w, (wn, (lo, hi)) in enumerate(S.EVENT_WINDOWS_MS.items()):
                v = [S.window_rms_shares(r['C'][r['types'].index('image')], lags)[w, r['groups'].index(g), j]
                     for r in rows if r['has'][r['groups'].index(g)]]
                strip(ax[j, -1], v, w + (gi - 1.5) * 0.22, COLORS[g], width=0.18, s=4)
        ax[j, -1].set_xticks(range(3), [f'{wn}\n{lo}-{hi} ms' for wn, (lo, hi) in S.EVENT_WINDOWS_MS.items()])
        ax[j, -1].set_ylabel('share of the RMS contribution')
        ax[j, -1].set_title(f'image, {tn}: shares per window', fontsize=9)
    ax[0, 0].legend(fontsize=7)
    save(fig, 'event_contributions', f'Region contributions (exact Shapley) to the event-locked LFP at layer 4, {len(rows)} sessions: '
                                     'no shift from local to network over the response')


def latency():
    rows = [r for r in events().values() if 'rate' in r]
    lags = rows[0]['lags']
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))

    def normalised(x, window=(0, 750)):
        w = (lags >= window[0]) & (lags < window[1])
        x = np.abs(x - np.nanmean(x[lags < 0]))
        return x / np.nanmax(x[w])

    for p, g in enumerate(['VISp', 'visual_cortex_other']):
        rate = [normalised(r['rate'][r['etypes'].index('image'), :, r['egroups'].index(g)]) for r in rows
                if g in r['egroups'] and 'image' in r['etypes'] and contribution(r, 'image', g) is not None]
        for j, (tn, color) in enumerate((('broadband', COLORS[g]), ('0.5-4 Hz', 'tab:gray'))):
            x = [normalised(v) for v in (contribution(r, 'image', g, j) for r in rows) if v is not None]
            median_band(ax[p], lags, x, color, f'contribution, {tn}')
        median_band(ax[p], lags, rate, 'k', 'own firing (|Δ rate|)')
        ax[p].axvspan(0, 250, color='gray', alpha=0.07)
        ax[p].set_xlim(-100, 750)
        ax[p].set_xlabel('time from image onset (ms)')
        ax[p].set_ylabel('normalised magnitude')
        ax[p].set_title(f'{label(g)}: own firing vs contribution', fontsize=9)
        ax[p].legend(fontsize=7)
    for p, (wn, window) in enumerate(S.LATENCY_WINDOWS_MS.items(), 2):
        delays = []
        for r in rows:
            v = []
            for g in ('VISp', 'visual_cortex_other'):
                c = contribution(r, 'image', g)
                if c is None or g not in r['egroups']:
                    break
                rate = r['rate'][r['etypes'].index('image'), :, r['egroups'].index(g)]
                v.append(S.event_timing(c[:, None], lags, window)[0, 0] - S.event_timing(rate[:, None], lags, window)[0, 0])
            if len(v) == 2:
                delays.append(v)
        delays = np.array(delays, float)
        identity(ax[p], delays[:, 0], delays[:, 1])
        ax[p].set_xlabel('VISp: contribution t50 − own firing t50 (ms)')
        ax[p].set_ylabel('other visual: same (ms)')
        ax[p].set_title(f'{wn} {window} ms: other − VISp {np.nanmedian(delays[:, 1] - delays[:, 0]):+.0f} ms, '
                        f'{np.sum(delays[:, 1] > delays[:, 0])}/{len(delays)} above the line', fontsize=9)
    save(fig, 'latency', f'Timing against the regions\' own firing (image, broadband unless stated, {len(rows)} sessions): '
                         'the other visual areas\' contribution persists after the image (offset response)')


def mouse_pairs(rows):
    meta = pd.read_csv(paths.REPO_DIR / 'tables' / 'sessions.csv').set_index('session_id')
    pairs = {}
    for sid in rows:
        if sid in meta.index:
            pairs.setdefault(meta.loc[sid, 'mouse_id'], {})[meta.loc[sid, 'experience_level']] = sid
    return [(v['Familiar'], v['Novel']) for v in pairs.values() if {'Familiar', 'Novel'} <= set(v)]


def novelty():
    rows = events()
    pairs = mouse_pairs(rows)
    lags = next(iter(rows.values()))['lags']
    windows = list(S.EVENT_WINDOWS_MS.items())
    panels = [('image_shared', 0), ('passive_image_shared', 0), ('image_shared', 1)]
    fig, ax = plt.subplots(len(panels), 3, figsize=(17, 4 * len(panels)))
    for k, (t, j) in enumerate(panels):
        for p, g in enumerate(['VISp', 'visual_cortex_other']):
            both = [(contribution(rows[f], t, g, j), contribution(rows[n], t, g, j)) for f, n in pairs]
            both = [b for b in both if b[0] is not None and b[1] is not None]
            if not both:
                continue
            fam, nov = np.array([b[0] for b in both]), np.array([b[1] for b in both])
            median_band(ax[k, p], lags, fam, 'tab:blue', f'familiar day ({len(both)} mice)')
            median_band(ax[k, p], lags, nov, 'tab:red', 'novel day')
            ax[k, p].axvspan(0, 250, color='gray', alpha=0.07)
            ax[k, p].set_title(f'{label(g)}, {t}, {BANDS[j]}', fontsize=9)
            ax[k, p].set_xlabel('time from onset (ms)')
            for w, (wn, (lo, hi)) in enumerate(windows):
                sel = (lags >= lo) & (lags < hi)
                rms = lambda x: np.sqrt(np.mean(x[:, sel] ** 2, 1))
                strip(ax[k, 2], rms(nov) - rms(fam), w + (p - 0.5) * 0.4, COLORS[g], width=0.3, s=6)
        ax[k, 0].legend(fontsize=7)
        ax[k, 0].set_ylabel('contribution at L4 (z)')
        ax[k, 2].axhline(0, color='k', lw=0.5)
        ax[k, 2].set_xticks(range(len(windows)), [f'{wn}\n{lo}-{hi} ms' for wn, (lo, hi) in windows])
        ax[k, 2].set_ylabel('novel − familiar day, RMS per mouse')
        ax[k, 2].set_title('per mouse: VISp (blue) | other visual (orange)', fontsize=9)
    save(fig, 'novelty', f'Shared images, same mouse, familiar vs novel day ({len(pairs)} mice): the other visual areas\' '
                         'contribution is smaller on the novel day, VISp\'s unchanged')


def depth():
    rows = [r['depth'] for r in events().values() if r['depth'] is not None and 'image' in r['depth']['csd']]
    lags, groups = rows[0]['lags'], rows[0]['groups']
    has = np.array([r['units'] > 0 for r in rows])
    rel = np.arange(-S.DEPTH_HALF_WIDTH, S.DEPTH_HALF_WIDTH + 1)
    panels = [('recorded', None), ('prediction', None)] + [(g, groups.index(g)) for g in
                                                           ['VISp', 'visual_cortex_other', 'thalamus', 'midbrain', 'hippocampus']]
    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(2, len(panels), height_ratios=[1.2, 1], hspace=0.35)
    top = [fig.add_subplot(grid[0, p]) for p in range(len(panels))]
    for p, (name, g) in enumerate(panels):
        stack = []
        for k, r in enumerate(rows):
            if g is not None and not has[k, g]:
                continue
            C, Y, P = r['csd']['image']
            x = Y[:, 0] if name == 'recorded' else P[:, 0] if name == 'prediction' else C[:, g, 0]
            stack.append(on_grid(x.T, r['rel']))
        if not stack:
            top[p].set_title(f'{label(name)}\n(no sessions)', fontsize=8)
            continue
        m = np.nanmedian(np.stack(stack), 0)
        lim = np.nanmax(np.abs(m))
        top[p].imshow(m, aspect='auto', origin='lower', cmap='RdBu', vmin=-lim, vmax=lim,
                         extent=(lags[0], lags[-1], rel[0] - 0.5, rel[-1] + 0.5))
        top[p].axhline(0, color='k', lw=0.5, ls=':')
        top[p].axvline(0, color='k', lw=0.5)
        top[p].set_xlim(-100, 750)
        top[p].set_ylim(rel[0] - 0.5, np.flatnonzero(np.isfinite(m).any(1)).max() - S.DEPTH_HALF_WIDTH + 0.5)
        top[p].set_title(f'{label(name)}\n({len(stack)} sessions, |max| {lim:.1g})', fontsize=8)
        top[p].set_xlabel('time from image onset (ms)')
    top[0].set_ylabel('channels from L4 centre (40 µm; up = superficial)')
    ax = fig.add_subplot(grid[1, 1:])
    bars = []
    for g in ['VISp', 'visual_cortex_other', 'thalamus', 'midbrain']:
        gi = groups.index(g)
        for w, wn in ((0, 'early'), (1, 'late')):
            f = np.array([r['frac']['image'][w, gi, 0] for k, r in enumerate(rows) if has[k, gi]])
            if len(f):
                bars.append((f'{label(g)} contribution, {wn}', np.nanmedian(f, 0)))
    if S.ALLEN_TABLE.exists():
        t = pd.read_csv(S.ALLEN_TABLE)
        frac = t[S.COMPARTMENTS].div(t[S.COMPARTMENTS].sum(axis=1), axis=0)
        projects = t[S.COMPARTMENTS].mean(axis=1) >= S.ALLEN_MIN_DENSITY
        for name, sel in (('higher visual areas', t.group == 'visual_cortex_other'), ('LGd', t.source == 'LGd'),
                          ('LP', t.source == 'LP'), ('thalamus group (all sources)', t.group == 'thalamus')):
            ok = sel & projects
            bars.append((f'Allen atlas: {name} → VISp (n {ok.sum()})', frac.loc[ok].median().values))
    left = np.zeros(len(bars))
    for k, (c, color) in enumerate(zip(S.COMPARTMENTS, ('tab:cyan', 'tab:red', 'tab:gray'))):
        v = np.array([b[1][k] for b in bars])
        ax.barh(range(len(bars)), v, left=left, color=color, label=c)
        for y, (x0, x) in enumerate(zip(left, v)):
            ax.text(x0 + x / 2, y, f'{x:.2f}', ha='center', va='center', fontsize=7)
        left += v
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('fraction in L2/3 | L4 | L5/6 (model: mean squared CSD per channel; atlas: projection density)')
    ax.legend(fontsize=8, loc='lower right')
    save(fig, 'depth', f'Laminar profile of the contributions (image, broadband; {len(rows)} sessions; red = sink): VISp\'s carries '
                       'the L4 sink, the other visual areas\' avoids L4, as their axons do. The thalamus group is mostly MG')


def state():
    rows = list(events().values())
    groups = rows[0]['groups']
    contrasts = S.STATE_CONTRASTS
    shown = ['VISp', 'visual_cortex_other', 'hippocampus', 'thalamus', 'midbrain']
    fig, ax = plt.subplots(2, len(contrasts), figsize=(3.2 * len(contrasts), 8), sharey='row')
    for j, tn in enumerate(['broadband', '0.5-4 Hz']):
        for c, contrast in enumerate(contrasts):
            for i, g in enumerate(shown):
                gi = groups.index(g)
                strip(ax[j, c], [r['state'][contrast][0][gi, j] - r['state'][contrast][1][gi, j] for r in rows if r['has'][gi]],
                      i, COLORS[g], width=0.6, s=4)
            ax[j, c].axhline(0, color='k', lw=0.5)
            ax[j, c].set_xticks(range(len(shown)), [label(g) for g in shown], rotation=40, ha='right', fontsize=8)
            name = (f'{contrast[0]} {contrast[1]} {"top 30%" if len(contrast[2]) == 3 else contrast[2][0]} − '
                    f'{contrast[3]} {contrast[4]} {"bottom 30%" if len(contrast[5]) == 3 else contrast[5][0]}')
            ax[j, c].set_title(f'{name}\n{tn}', fontsize=8)
        ax[j, 0].set_ylabel('difference in variance share')
    ax[0, 0].set_ylim(-0.4, 0.4)
    ax[1, 0].set_ylim(-0.4, 0.4)
    save(fig, 'state', f'Shapley variance shares by state, layer 4 ({len(rows)} sessions): running and large pupil move the '
                       '0.5-4 Hz prediction towards VISp and away from midbrain; engagement changes nothing')


def midbrain_licks():
    rows = [r for r in events().values() if 'behaviour' in r]
    lags = rows[0]['lags']
    types = ['change_hit', 'change_miss', 'image', 'passive_image']
    fig, ax = plt.subplots(1, len(types) + 1, figsize=(20, 4.4))
    def behaviour(t, name):
        """Mean over sessions (licks are sparse: the median is mostly 0) of a behaviour trace; running as its slowing."""
        x = np.array([r['behaviour'][r['etypes'].index(t), :, r['bnames'].index(name)] for r in rows if t in r['etypes']])
        if name == 'running':
            x = -(x - np.nanmean(x[:, lags < 0], 1, keepdims=True))
        return x

    scale = {name: np.nanmax(np.nanmean(behaviour('change_hit', name), 0)) for name in ('licks', 'running')}
    corr = {}
    for p, t in enumerate(types):
        mid = [contribution(r, t, 'midbrain', 1) for r in rows]
        ok = [k for k, m in enumerate(mid) if m is not None and t in rows[k]['etypes']]
        if not ok:
            continue
        mid = np.array([mid[k] for k in ok])
        median_band(ax[p], lags, mid, COLORS['midbrain'], 'midbrain contribution, 0.5-4 Hz')
        vis = [contribution(rows[k], t, 'visual_cortex_other', 1) for k in ok]
        median_band(ax[p], lags, [v for v in vis if v is not None], COLORS['visual_cortex_other'],
                    'other visual contribution, 0.5-4 Hz', shade=False)
        a2 = ax[p].twinx()
        for name, color, ls, text in (('licks', 'k', '-', 'lick rate'), ('running', 'gray', '--', 'running slowing (−Δ speed)')):
            a2.plot(lags, np.nanmean(behaviour(t, name), 0) / scale[name], color=color, ls=ls, label=text)
        a2.set_ylim(-0.6, 1.6)
        a2.set_ylabel('behaviour (mean; 1 = hit maximum)', fontsize=8)
        ax[p].axvline(0, color='k', lw=0.5)
        ax[p].set_title(f'{t} ({len(ok)} sessions)', fontsize=9)
        ax[p].set_xlabel('time from onset (ms)')
        if p == 1:
            ax[p].legend(fontsize=7, loc='upper left')
            a2.legend(fontsize=7, loc='upper right')
        if t == 'change_hit':
            after = lags >= 0
            beh = {name: np.array([rows[k]['behaviour'][rows[k]['etypes'].index(t), :, rows[k]['bnames'].index(name)]
                                   for k in ok]) for name in ('licks', 'running')}
            for name, x in (('licks', beh['licks']), ('running slowing', -beh['running'])):
                for g, c in (('midbrain', list(mid)), ('other visual', vis)):
                    corr[(name, g)] = [np.corrcoef(c[k][after], x[k, after])[0, 1] for k in range(len(c))
                                       if c[k] is not None and np.isfinite(x[k, after]).all() and np.std(x[k, after]) > 0]
    for i, ((name, g), v) in enumerate(corr.items()):
        strip(ax[-1], v, i, COLORS['midbrain'] if g == 'midbrain' else COLORS['visual_cortex_other'])
    ax[-1].axhline(0, color='k', lw=0.5)
    ax[-1].set_xticks(range(len(corr)), [f'{g}\nvs {n}' for n, g in corr], fontsize=8, rotation=30, ha='right')
    ax[-1].set_ylabel('correlation over 0-750 ms, per session')
    ax[-1].set_title('hits: contribution time course vs behaviour', fontsize=9)
    save(fig, 'midbrain_licks', 'Is the midbrain 0.5-4 Hz contribution movement? Midbrain contribution against licking and running '
                                '(layer 4)')


FIGURES = dict(baselines=baselines, stimulus_clock=stimulus_clock, unit_ranking=unit_ranking, simulation=simulation,
               region_bands=region_bands, region_clock=region_clock, high_bands=high_bands,
               event_contributions=event_contributions, latency=latency, novelty=novelty, depth=depth, state=state,
               midbrain_licks=midbrain_licks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', help=f'figures to make, of {", ".join(FIGURES)} (default: all)')
    args = parser.parse_args()
    if set(args.names) - set(FIGURES):
        parser.error(f'unknown figures: {" ".join(set(args.names) - set(FIGURES))}')
    warnings.simplefilter('ignore', RuntimeWarning)
    for name in args.names or FIGURES:
        FIGURES[name]()
    explore = S.variables_dir(S.sessions()[0]).parent / 'explore_task_pooled.png'
    if explore.exists():
        shutil.copy(explore, OUT / 'explore_task.png')


if __name__ == '__main__':
    main()
