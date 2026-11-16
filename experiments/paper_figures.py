"""
Preliminary main figures 1-7 of the reframed manuscript ("local and network sources of the laminar LFP"), from the
saved results of all sessions (nothing is rerun). Panels whose inputs have not been computed yet say so.

  fig1  approach and prediction: example stretch, recorded vs predicted depth profile, R2 by depth, units per region
  fig2  the prediction follows single trials: linear baselines, stimulus clock, swapped and shifted spikes, transfer
  fig3  two sources: R2 lost per region and band (inference, retrained), per session, by depth, single visual areas
  fig4  laminar decomposition: CSD of each region's contribution vs the recorded CSD, layer fractions vs the Allen atlas
  fig5  trial-specific vs evoked contributions, and their timing against each region's own firing
  fig6  experience: shared images on the novel vs familiar day, contribution vs firing rate
  fig7  which neurons: attribution in the simulation and in unit removal, firing-rate control, hierarchy, layers

usage (on the cluster, from the repo root, after `source cluster/env.sh`):
    python -m experiments.paper_figures [fig1 fig2 ...]          # default: all

Saves <data root>/spikes2lfp_v2/figures_paper/<fig>.png.
"""
import argparse
import functools
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
from experiments import statistics as sx
from experiments import summarize as S
from experiments.pool_sessions import ALIGNMENT_TABLE, BELOW_L4
from experiments.region_ablation import region_masks

OUT = paths.DATA_ROOT / paths.OUTPUT_SUBDIR / 'figures_paper'
POOLED = paths.DATA_ROOT / paths.OUTPUT_SUBDIR / 'pooled'
EXAMPLE_SESSION = 1047969464
COLORS = dict(rf.COLORS, LGN='tab:cyan', LP='tab:olive', thalamus_other='tab:red', rest='tab:gray')
BANDS = S.ALL_BANDS


def save(fig, name, title):
    fig.suptitle(title, fontsize=11, x=0.01, ha='left')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT / f"{name}.png"}')


def letter(ax, text):
    ax.text(-0.12, 1.06, text, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')


def missing(ax, text):
    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes, fontsize=9, color='gray', wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def per_session(name, f):
    return sx.per_session(name, f)


# ------------------------------------------------------------------------------------------------- figure 1

def fig1():
    fig = plt.figure(figsize=(18, 11))
    g = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1], hspace=0.45, wspace=0.3)
    f = S.variables_dir(EXAMPLE_SESSION) / 'example_trace.npz'
    ax = fig.add_subplot(g[0, :])
    letter(ax, 'a')
    if f.exists():
        d = np.load(f)
        t, l4 = d['t'], int(d['layer4_channel'])
        names = [BANDS[int(x)] for x in d['targets']]
        offset = 0
        for j, name in enumerate(names):
            y, p = d['target'][j, :, l4], d['prediction'][j, :, l4]
            scale = np.nanstd(y) * 4
            ax.plot(t, y / scale + offset, color='k', lw=0.6)
            ax.plot(t, p / scale + offset, color='tab:red', lw=0.8)
            ax.text(t[-1] + 0.1, offset, name, fontsize=8, va='center')
            offset -= 1.4
        rate = d['rate']
        for i, grp in enumerate(d['groups']):
            r = rate[:, i]
            if np.isfinite(r).any():
                ax.plot(d['rate_t'], (r - np.nanmean(r)) / (np.nanstd(r) * 5 + 1e-9) + offset - 0.3 * i, color=COLORS[str(grp)], lw=0.6)
                ax.text(t[-1] + 0.1, offset - 0.3 * i, rf.label(str(grp)), fontsize=7, va='center', color=COLORS[str(grp)])
        for times, color in ((d['image_times'], '0.6'), (d['change_times'], 'tab:green'), (d['omission_times'], 'tab:orange')):
            for x in times:
                ax.axvspan(x, x + 0.25, color=color, alpha=0.15, lw=0)
        ax.plot(d['lick_times'], np.full(len(d['lick_times']), 1.0), '|', color='tab:blue', ms=8)
        ax.set_xlim(t[0], t[-1] + 1.2)
        ax.set_yticks([])
        ax.set_xlabel('time in the test block (s); grey: images, green: change, orange: omission, blue ticks: licks')
        ax.set_title(f'session {EXAMPLE_SESSION}, layer 4 channel: recorded (black) and predicted (red) targets; '
                     'firing rate of each region below', fontsize=9)
        ax2 = fig.add_subplot(g[1, 0:2])
        letter(ax2, 'b')
        seg = slice(0, 750)
        pooled = S.pooled_channels(EXAMPLE_SESSION)                       # leave out channels above the probe track
        y = d['target'][0, seg][:, pooled].T
        p = d['prediction'][0, seg][:, pooled].T
        lim = np.nanpercentile(np.abs(y), 99)
        both = np.vstack([y, np.full((1, y.shape[1]), np.nan), p])
        ax2.imshow(both, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                   extent=(t[seg][0], t[seg][-1], 0, both.shape[0]))
        ax2.set_yticks([y.shape[0] / 2, y.shape[0] + 1 + p.shape[0] / 2], ['recorded', 'predicted'])
        ax2.set_xlabel('time (s)')
        ax2.set_title('broadband, every channel (deep at the bottom of each block)', fontsize=9)
    else:
        missing(ax, 'example trace not computed yet (experiments/example_trace.py)')
    ax = fig.add_subplot(g[1, 2:])
    letter(ax, 'c')
    r2 = np.load(POOLED / 'r2.npz')
    depth = np.arange(r2['test'].shape[1]) - BELOW_L4
    for t, color in zip((0, 1, 2, 7), ('k', 'tab:blue', 'tab:green', 'tab:purple')):
        rows = r2['test'][:, :, t]
        ax.plot(np.nanmedian(rows, 0), depth, color=color, label=BANDS[t])
        ax.fill_betweenx(depth, *np.nanpercentile(rows, [25, 75], 0), color=color, alpha=0.15, lw=0)
    ax.axhspan(-3, 0, color='gray', alpha=0.1)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-0.2, 1)
    ax.set_xlabel('test R² (median, IQR over 89 sessions)')
    ax.set_ylabel('channels from the upper L4 border (grey ≈ L4)')
    ax.legend(fontsize=8)
    ax.set_title('prediction by depth and band (held-out block)', fontsize=9)
    ax = fig.add_subplot(g[2, 0:2])
    letter(ax, 'd')
    units = pd.read_csv(POOLED / 'units.csv')
    counts = []
    for sid, u in units.groupby('session_id'):
        m = region_masks(u['area'].astype(str).to_numpy())
        counts.append([m[x].sum() for x in rf.REGIONS])
    counts = np.array(counts, float)
    for i, grp in enumerate(rf.REGIONS):
        rf.strip(ax, np.where(counts[:, i] > 0, counts[:, i], np.nan), i, COLORS[grp])
    ax.set_yscale('log')
    ax.set_xticks(range(len(rf.REGIONS)), [rf.label(x) for x in rf.REGIONS])
    ax.set_ylabel('units per session')
    ax.set_title(f'{len(units)} units in {len(counts)} sessions, 48 mice (model inputs)', fontsize=9)
    ax = fig.add_subplot(g[2, 2:])
    letter(ax, 'e')
    missing(ax, 'schematic: brain-wide spikes (4 ms bins, 3 s history) → causal CNN-LSTM → each VISp channel × target\n'
                'blocked split: train | validation | test (last 20%), 6 s buffers; attribution: history-aware IG, '
                'region contributions: exact Shapley values (from v1 Fig. 1-2, redrawn)')
    save(fig, 'fig1', 'Figure 1 | Predicting the laminar LFP of VISp from brain-wide spiking')


# ------------------------------------------------------------------------------------------------- figure 2

def fig2():
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    models = [('r2_units_instant', 'ridge,\ncurrent bin'), ('r2_units', 'ridge,\n1 s history'),
              ('r2_units_stimulus_behaviour', 'ridge 1 s +\nstim. + beh.'), ('r2_stimulus_behaviour', 'stimulus +\nbehaviour'),
              ('r2_arch_cnn', 'CNN only'), ('model_r2', 'CNN-LSTM')]
    b = per_session('baselines', lambda d, ch: {k: S.channel_median(d[k], ch)[:3] for k, _ in models if k in d})
    a = ax[0, 0]
    letter(a, 'a')
    for t, color in zip(range(2), ('k', 'tab:blue')):
        for i, (k, _) in enumerate(models):
            rf.strip(a, [r[k][t] for r in b.values() if k in r], i + (t - 0.5) * 0.35, color, width=0.3, s=4)
    a.set_xticks(range(len(models)), [n for _, n in models], fontsize=8)
    a.set_ylim(-0.2, 0.85)
    a.set_ylabel('test R² (black broadband, blue 0.5-4 Hz)')
    a.set_title('models on the same split (dots: sessions)', fontsize=9)
    a = ax[0, 1]
    letter(a, 'b')
    for t in range(3):
        rf.strip(a, [r['model_r2'][t] - r['r2_units'][t] for r in b.values()], t, 'k', width=0.5)
    a.axhline(0, color='k', lw=0.5)
    a.set_xticks(range(3), BANDS[:3])
    a.set_ylim(-0.15, 0.3)
    a.set_ylabel('CNN-LSTM − ridge with 1 s history')
    a.set_title('nonlinear gain: small but in 75-90% of sessions', fontsize=9)
    keys = [('r2_model', 'model'), ('r2_template', 'stimulus\nclock'), ('r2_shuffle', 'spikes from another\npresentation'),
            ('r2_shift', 'spikes shifted\n30-120 s')]
    c = per_session('stimulus_clock', lambda d, ch: {k: S.channel_median(np.nanmean(d[k], 0) if d[k].ndim == 3 else d[k], ch)[:5]
                                                     for k, _ in keys})
    a = ax[0, 2]
    letter(a, 'c')
    for t, color in zip(range(2), ('k', 'tab:blue')):
        for i, (k, _) in enumerate(keys):
            rf.strip(a, [r[k][t] for r in c.values()], i + (t - 0.5) * 0.35, color, width=0.3, s=4)
    a.axhline(0, color='k', lw=0.5)
    a.set_ylim(-0.9, 0.85)
    a.set_xticks(range(len(keys)), [n for _, n in keys], fontsize=8)
    a.set_ylabel('test R² (black broadband, blue 0.5-4 Hz)')
    a.set_title('the prediction follows the single trial, not the average response', fontsize=9)
    a = ax[1, 0]
    letter(a, 'd')
    rf.identity(a, [r['r2_template'][0] for r in c.values()], [r['r2_model'][0] for r in c.values()])
    a.set_xlabel('stimulus clock (condition-average template) R²')
    a.set_ylabel('model R²')
    a.set_title('broadband: model above the template in 89/89 sessions', fontsize=9)
    a = ax[1, 1]
    letter(a, 'e')
    rf.identity(a, [r['r2_template'][0] for r in c.values()], [r['r2_shuffle'][0] for r in c.values()])
    a.set_xlabel('stimulus clock R²')
    a.set_ylabel('model R² with spikes from another presentation of the same image')
    a.set_title('swapping single-trial spikes drops below the template', fontsize=9)
    tr = per_session('transfer_scaling', lambda d, ch: {k: S.channel_median(d[k], ch)[:, :3] for k in ('r2_active', 'r2_recalibrated')})
    a = ax[1, 2]
    letter(a, 'f')
    shown = [('task (test block)', 'r2_active', 0), ('passive replay', 'r2_active', 1), ('spontaneous\n(recalibrated)', 'r2_recalibrated', 2)]
    for t, color in zip(range(2), ('k', 'tab:blue')):
        for i, (_, k, e) in enumerate(shown):
            rf.strip(a, [r[k][e, t] for r in tr.values()], i + (t - 0.5) * 0.35, color, width=0.3, s=4)
    a.axhline(0, color='k', lw=0.5)
    a.set_ylim(-0.3, 0.85)
    a.set_xticks(range(3), [n for n, _, _ in shown])
    a.set_ylabel('R² of the task-trained model')
    a.set_title('transfer within session: passive yes, spontaneous only broadband', fontsize=9)
    save(fig, 'fig2', 'Figure 2 | The prediction is trial-specific, not a stimulus clock, and needs history more than nonlinearity')


# ------------------------------------------------------------------------------------------------- figure 3

AREAS = ['VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'LG', 'LP', 'CA1', 'DG', 'APN', 'MRN']


def fig3():
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    inf = per_session('region_ablation', lambda d, ch: sx.fraction_lost(d, ch, 'r2_full', 'r2_removed', rf.REGIONS + AREAS))
    ret = per_session('region_ablation_bands', lambda d, ch: sx.fraction_lost(d, ch, 'r2_retrained_full', 'r2_retrained_removed',
                                                                              rf.REGIONS))
    a = ax[0, 0]
    letter(a, 'a')
    for i, grp in enumerate(rf.REGIONS):
        for t in range(5):
            rf.strip(a, [v[i, t] for v in inf.values()], t + (i - 2) * 0.17, COLORS[grp], width=0.14, s=3)
        a.plot([], [], 'o', color=COLORS[grp], label=rf.label(grp))
    a.set_xticks(range(5), BANDS[:5])
    a.set_ylim(-0.3, 1.1)
    a.axhline(0, color='k', lw=0.5)
    a.set_ylabel('fraction of the R² lost, region removed')
    a.set_title('removed at inference (mean fill)', fontsize=9)
    a.legend(fontsize=7)
    a = ax[0, 1]
    letter(a, 'b')
    for i, grp in enumerate(rf.REGIONS[:3]):
        for t in range(3):
            rf.strip(a, [v[i, t] for v in ret.values()], t + (i - 1) * 0.25, COLORS[grp], width=0.2, s=4)
    a.set_xticks(range(3), BANDS[:3])
    a.set_ylim(-0.2, 0.5)
    a.axhline(0, color='k', lw=0.5)
    a.set_ylabel('fraction of the R² lost')
    a.set_title('retrained without the region (other units can take over)', fontsize=9)
    a = ax[0, 2]
    letter(a, 'c')
    diff = np.array([v[0, :5] - v[1, :5] for v in inf.values()])
    rdiff = np.array([v[0, :3] - v[1, :3] for v in ret.values()])
    for v in diff:
        a.plot(range(5), v, color='gray', lw=0.3, alpha=0.5)
    a.plot(range(5), np.nanmedian(diff, 0), 'k-o', lw=2, label='inference')
    a.plot(range(3), np.nanmedian(rdiff, 0), 's--', color='tab:red', lw=2, label='retrained (median)')
    a.axhline(0, color='k', lw=0.5)
    a.set_ylim(-1.2, 1)
    a.set_xticks(range(5), BANDS[:5])
    a.set_ylabel('VISp − other visual, fraction lost')
    a.set_title('per session: VISp leads broadband, the network every envelope', fontsize=9)
    a.legend(fontsize=7)
    prof = _profile_rows()
    depth = np.arange(-S.DEPTH_HALF_WIDTH, S.DEPTH_HALF_WIDTH + 1)
    for p, grp in enumerate(['VISp', 'visual_cortex_other']):
        a = ax[1, p]
        letter(a, 'de'[p])
        for t, color in zip((0, 1, 2, 3), ('k', 'tab:blue', 'tab:green', 'tab:red')):
            rows = np.array([v[p, :, t] for v in prof.values()])
            a.plot(np.nanmedian(rows, 0), depth, color=color, label=BANDS[t])
        a.axhspan(-2, 2, color='gray', alpha=0.1)
        a.set_xlim(0, 0.85)
        a.set_xlabel("fraction of the channel's R² lost (inference)")
        a.set_ylabel('channels from the L4 centre (grey ≈ L4)')
        a.set_title(f'{rf.label(grp)} by depth', fontsize=9)
        a.legend(fontsize=7)
    a = ax[1, 2]
    letter(a, 'f')
    offset = len(rf.REGIONS)
    for i, area in enumerate(AREAS):
        for t, color in ((0, 'k'), (2, 'tab:green')):
            rf.strip(a, [v[offset + i, t] for v in inf.values()], i + (0.18 if t else -0.18), color, width=0.3, s=3)
    a.axhline(0, color='k', lw=0.5)
    a.set_ylim(-0.1, 0.4)
    a.set_xticks(range(len(AREAS)), AREAS, rotation=45)
    a.set_ylabel('fraction of the R² lost')
    a.set_title('single areas (black broadband, green 4-8 Hz)', fontsize=9)
    save(fig, 'fig3', 'Figure 3 | Two sources: the local population carries the broadband LFP, the visual network the band envelopes')


def profile(d, ch):
    """Fraction of each channel's R2 lost (inference) for VISp and other visual cortex, on channels from the L4 centre."""
    names = [str(x) for x in d['regions']]
    idx = [names.index(x) for x in ('VISp', 'visual_cortex_other')]
    full = d['r2_full'][ch][:, :5]
    drop = (d['r2_full'] - d['r2_removed'][0])[idx][:, ch][..., :5]
    own = np.where(full > S.MIN_PROFILE_R2, full, np.nan)
    sid = profile.sid
    return np.moveaxis(rf.on_grid(np.moveaxis(drop / own, 1, 0), rf.l4_relative(sid)), 0, 1)


def _profile_rows():
    out = {}
    for sid in S.sessions():
        f = S.variables_dir(sid) / 'region_ablation.npz'
        if f.exists():
            profile.sid = sid
            out[sid] = profile(np.load(f), S.pooled_channels(sid))
    return out


# ------------------------------------------------------------------------------------------------- figure 4

@functools.lru_cache(maxsize=1)
def thalamus_depth():
    """Per session, depth profiles (summarize.depth_session) from region_events_thalamus.npz (LGN and LP as groups)."""
    table = pd.read_csv(ALIGNMENT_TABLE).set_index('session')
    rows = {}
    for sid in S.sessions():
        f = S.variables_dir(sid) / 'region_events_thalamus.npz'
        if f.exists() and sid in table.index:
            rows[sid] = S.depth_session(np.load(f), sid, table)
    print(f'{len(rows)} sessions with region_events_thalamus.npz')
    return rows


def csd_median(rows, name):
    rel = np.arange(-S.DEPTH_HALF_WIDTH, S.DEPTH_HALF_WIDTH + 1)
    stack = []
    for r in rows:
        if 'image' not in r['csd']:
            continue
        C, Y, P = r['csd']['image']
        if name == 'recorded':
            x = Y[:, 0]
        elif name == 'prediction':
            x = P[:, 0]
        elif name in r['groups'] and r['units'][r['groups'].index(name)] > 0:
            x = C[:, r['groups'].index(name), 0]
        else:
            continue
        stack.append(rf.on_grid(x.T, r['rel']))
    return (np.nanmedian(np.stack(stack), 0), len(stack)) if stack else (None, 0), rel


def fig4():
    regions = [r['depth'] for r in rf.events().values() if r['depth'] is not None]
    thal = list(thalamus_depth().values())
    lags = regions[0]['lags']
    fig = plt.figure(figsize=(20, 11))
    g = fig.add_gridspec(2, 6, height_ratios=[1.1, 1], hspace=0.4)
    panels = [('recorded', regions), ('prediction', regions), ('VISp', regions), ('visual_cortex_other', regions),
              ('LGN', thal), ('LP', thal)]
    for p, (name, rows) in enumerate(panels):
        a = fig.add_subplot(g[0, p])
        if p == 0:
            letter(a, 'a')
        (m, n), rel = csd_median(rows, name)
        if m is None:
            missing(a, f'{name}: needs region_events --grouping thalamus (running)')
            continue
        lim = np.nanmax(np.abs(m))
        a.imshow(m, aspect='auto', origin='lower', cmap='RdBu', vmin=-lim, vmax=lim, extent=(lags[0], lags[-1], rel[0] - 0.5, rel[-1] + 0.5))
        top = np.flatnonzero(np.isfinite(m).any(1)).max() - S.DEPTH_HALF_WIDTH
        a.set_ylim(rel[0] - 0.5, top + 0.5)
        a.axhline(0, color='k', lw=0.5, ls=':')
        a.axvline(0, color='k', lw=0.5)
        a.set_xlim(-50, 600)
        a.set_title(f'{rf.label(name)} ({n} sessions)', fontsize=9)
        a.set_xlabel('ms from image onset')
        if p == 0:
            a.set_ylabel('channels from L4 centre (40 µm)')
    a = fig.add_subplot(g[1, 0:3])
    letter(a, 'b')
    bars = []
    for rows, grp in ((regions, 'VISp'), (regions, 'visual_cortex_other'), (thal, 'LGN'), (thal, 'LP')):
        f = [r['frac']['image'][0, r['groups'].index(grp), 0] for r in rows
             if 'image' in r['frac'] and grp in r['groups'] and r['units'][r['groups'].index(grp)] > 0]
        if f:
            bars.append((f'{rf.label(grp)} contribution (early, {len(f)} sessions)', np.nanmedian(f, 0)))
    t = pd.read_csv(S.ALLEN_TABLE)
    frac = t[S.COMPARTMENTS].div(t[S.COMPARTMENTS].sum(axis=1), axis=0)
    projects = t[S.COMPARTMENTS].mean(axis=1) >= S.ALLEN_MIN_DENSITY
    for name, sel in (('higher visual areas', t.group == 'visual_cortex_other'), ('LGd', t.source == 'LGd'), ('LP', t.source == 'LP')):
        bars.append((f'Allen atlas: {name} → VISp axons (n {int((sel & projects).sum())})', frac.loc[sel & projects].median().values))
    left = np.zeros(len(bars))
    for k, (c, color) in enumerate(zip(S.COMPARTMENTS, ('tab:cyan', 'tab:red', 'tab:gray'))):
        v = np.array([x[1][k] for x in bars])
        a.barh(range(len(bars)), v, left=left, color=color, label=c)
        for y, (x0, x) in enumerate(zip(left, v)):
            a.text(x0 + x / 2, y, f'{x:.2f}', ha='center', va='center', fontsize=7)
        left += v
    a.set_yticks(range(len(bars)), [x[0] for x in bars], fontsize=8)
    a.invert_yaxis()
    a.legend(fontsize=7, loc='lower right')
    a.set_xlabel('fraction in L2/3 | L4 | L5/6')
    a = fig.add_subplot(g[1, 3:5])
    letter(a, 'c')
    pairs = [(regions, 'visual_cortex_other', 'other visual'), (thal, 'LGN', 'LGN'), (thal, 'LP', 'LP')]
    for i, (rows, grp, text) in enumerate(pairs):
        for w, color in ((0, 'k'), (1, 'gray')):
            v = [r['frac']['image'][w, r['groups'].index(grp), 0, 1] - r['frac']['image'][w, r['groups'].index('VISp'), 0, 1]
                 for r in rows if 'image' in r['frac'] and grp in r['groups'] and r['units'][r['groups'].index(grp)] > 0
                 and r['units'][r['groups'].index('VISp')] > 0]
            rf.strip(a, v, i + (w - 0.5) * 0.35, color, width=0.3, s=5)
    a.axhline(0, color='k', lw=0.5)
    a.set_xticks(range(3), [x[2] + ' − VISp' for x in pairs])
    a.set_ylabel('L4 fraction of CSD power (black early, grey late)')
    a.set_title('per session: below 0 = less L4 than VISp', fontsize=9)
    a = fig.add_subplot(g[1, 5])
    letter(a, 'd')
    match = [r['match']['image'][0] for r in regions if 'image' in r['match']]
    rf.strip(a, match, 0, 'k', width=0.5)
    a.set_xticks([0], ['predicted vs\nrecorded CSD'])
    a.set_ylabel('correlation, 0-750 ms (broadband)')
    a.set_ylim(0, 1)
    save(fig, 'fig4', 'Figure 4 | Each source has its own laminar profile, matching where its axons end')


# ------------------------------------------------------------------------------------------------- figure 5

def fig5():
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    def clock(d, ch):
        names = [str(x) for x in d['regions']]
        swap = np.nanmean(d['r2_swap'], 0)
        rem, swp = S.channel_median(d['r2_model'] - d['r2_removed'], ch)[:, :2], S.channel_median(d['r2_model'] - swap, ch)[:, :2]
        return {g: (rem[names.index(g)], swp[names.index(g)], d['region_n_units'][names.index(g)]) for g in rf.REGIONS}
    k = per_session('region_clock', clock)
    for p, shown in enumerate((['VISp', 'visual_cortex_other'], ['hippocampus', 'midbrain', 'thalamus'])):
        a = ax[0, p]
        letter(a, 'ab'[p])
        for grp in shown:
            ok = [v[grp] for v in k.values() if v[grp][2] > 0]
            rf.identity(a, [x[0][0] for x in ok], [x[1][0] for x in ok], COLORS[grp], rf.label(grp))
        a.set_xlabel('R² lost, region removed')
        a.set_ylabel('R² lost, only its single-trial timing removed\n(spikes swapped between presentations of the same image)')
        a.legend(fontsize=7)
        a.set_title('broadband; on the line: all of the contribution is trial-specific', fontsize=9)
    a = ax[0, 2]
    letter(a, 'c')
    for i, grp in enumerate(rf.REGIONS):
        for t, color in ((0, COLORS[grp]), (1, 'gray')):
            v = [x[grp][1][t] / x[grp][0][t] for x in k.values() if x[grp][2] > 0 and x[grp][0][t] > 0.005]
            rf.strip(a, v, i + (t - 0.5) * 0.35, color, width=0.3, s=4)
    a.axhline(1, color='k', lw=0.5)
    a.set_ylim(-0.3, 3.5)
    a.set_xticks(range(len(rf.REGIONS)), [rf.label(x) for x in rf.REGIONS], fontsize=8)
    a.set_ylabel('trial-specific / total (colour broadband, grey 0.5-4 Hz)')
    ev = rf.events()
    rows = [r for r in ev.values() if 'rate' in r]
    lags = rows[0]['lags']
    a = ax[1, 0]
    letter(a, 'd')
    for grp in rf.REGIONS:
        x = [v for v in (rf.contribution(r, 'image', grp) for r in rows) if v is not None]
        rf.median_band(a, lags, x, COLORS[grp], rf.label(grp), shade=grp in ('VISp', 'visual_cortex_other'))
    a.axvspan(0, 250, color='gray', alpha=0.07)
    a.set_xlim(-100, 750)
    a.set_xlabel('ms from image onset (grey: image on)')
    a.set_ylabel('contribution at L4 (z)')
    a.legend(fontsize=7)
    a.set_title('image, broadband: VISp carries the onset, the network the offset', fontsize=9)
    a = ax[1, 1]
    letter(a, 'e')

    def normalised(x):
        x = np.abs(x - np.nanmean(x[lags < 0]))
        return x / np.nanmax(x[(lags >= 0) & (lags < 750)])
    for grp, ls in (('VISp', '-'), ('visual_cortex_other', '-')):
        c = [normalised(v) for v in (rf.contribution(r, 'image', grp) for r in rows) if v is not None]
        fr = [normalised(r['rate'][r['etypes'].index('image'), :, r['egroups'].index(grp)]) for r in rows
              if grp in r['egroups'] and rf.contribution(r, 'image', grp) is not None]
        a.plot(lags, np.nanmedian(c, 0), color=COLORS[grp], lw=2, label=f'{rf.label(grp)} contribution')
        a.plot(lags, np.nanmedian(fr, 0), color=COLORS[grp], lw=1, ls=':', label=f'{rf.label(grp)} firing')
    a.axvspan(0, 250, color='gray', alpha=0.07)
    a.set_xlim(-100, 750)
    a.set_xlabel('ms from image onset')
    a.set_ylabel('normalised magnitude (median over sessions)')
    a.legend(fontsize=7)
    a.set_title('contribution vs the region\'s own firing', fontsize=9)
    a = ax[1, 2]
    letter(a, 'f')
    for (wn, window), color in zip(S.LATENCY_WINDOWS_MS.items(), ('k', 'tab:red')):
        v = [sx.latency_difference(r, window) for r in ev.values()]
        rf.strip(a, v, list(S.LATENCY_WINDOWS_MS).index(wn), color, width=0.5)
    a.axhline(0, color='k', lw=0.5)
    a.set_xticks(range(2), [f'{wn}\n{w[0]}-{w[1]} ms' for wn, w in S.LATENCY_WINDOWS_MS.items()])
    a.set_ylim(-200, 400)
    a.set_ylabel('extra delay behind own firing,\nother visual − VISp (ms)')
    a.set_title('the network contribution persists after the image', fontsize=9)
    save(fig, 'fig5', 'Figure 5 | Contributions are trial-specific; hippocampus contributes through timing only')


# ------------------------------------------------------------------------------------------------- figure 6

@functools.lru_cache(maxsize=1)
def event_rates():
    rows = {}
    for sid in S.sessions():
        f = S.variables_dir(sid) / 'event_rates.npz'
        if f.exists():
            d = np.load(f)
            rows[sid] = dict(types=[str(x) for x in d['event_types']], groups=[str(x) for x in d['groups']], rate=d['eta_rate'],
                             lags=d['eta_lags_ms'])
    print(f'{len(rows)} sessions with event_rates.npz')
    return rows


def rate_trace(rates, sid, t, grp):
    r = rates.get(sid)
    if r is None or t not in r['types'] or grp not in r['groups']:
        return None
    x = r['rate'][r['types'].index(t), :, r['groups'].index(grp)]
    return x - np.nanmean(x[r['lags'] < 0]) if np.isfinite(x).all() else None


def novelty_rows(t='image_shared', j=0):
    """Per mouse (familiar, novel session): window RMS of each group's contribution and its evoked firing rate."""
    ev, rates = rf.events(), event_rates()
    lags = next(iter(ev.values()))['lags']
    out = []
    for f, n in rf.mouse_pairs(ev):
        row = dict(mouse=stats.mouse_ids([n])[0])
        for grp in ('VISp', 'visual_cortex_other'):
            for wn, (lo, hi) in S.EVENT_WINDOWS_MS.items():
                sel = (lags >= lo) & (lags < hi)
                c = [rf.contribution(ev[s], t, grp, j) for s in (f, n)]
                r = [rate_trace(rates, s, t, grp) for s in (f, n)]
                if all(v is not None for v in c):
                    row[(grp, wn, 'contribution')] = [np.sqrt(np.mean(x[sel] ** 2)) for x in c]
                if all(v is not None for v in r):
                    row[(grp, wn, 'rate')] = [np.mean(x[sel]) for x in r]
        out.append(row)
    return out, lags


def fig6():
    ev, rates = rf.events(), event_rates()
    pairs = rf.mouse_pairs(ev)
    lags = next(iter(ev.values()))['lags']
    fig, ax = plt.subplots(2, 4, figsize=(22, 9.5))
    for p, grp in enumerate(['visual_cortex_other', 'VISp']):
        for q, (kind, text) in enumerate((('contribution', 'contribution at L4 (z)'), ('rate', 'firing rate change (spikes/s per unit)'))):
            a = ax[p, q]
            letter(a, 'abcd'[2 * p + q])
            fam, nov = [], []
            for f, n in pairs:
                x = ([rf.contribution(ev[s], 'image_shared', grp) for s in (f, n)] if kind == 'contribution'
                     else [rate_trace(rates, s, 'image_shared', grp) for s in (f, n)])
                if all(v is not None for v in x):
                    fam.append(x[0])
                    nov.append(x[1])
            if not fam:
                missing(a, 'firing rates around shared images not computed yet (experiments/event_rates.py)')
                continue
            rf.median_band(a, lags, fam, 'tab:blue', f'familiar day ({len(fam)} mice)')
            rf.median_band(a, lags, nov, 'tab:red', 'novel day')
            a.axvspan(0, 250, color='gray', alpha=0.07)
            a.set_xlabel('ms from onset of a shared image')
            a.set_ylabel(text)
            a.set_title(f'{rf.label(grp)}: {kind}', fontsize=9)
            a.legend(fontsize=7)
    rows, _ = novelty_rows()
    for p, grp in enumerate(['visual_cortex_other', 'VISp']):
        a = ax[p, 2]
        letter(a, 'ef'[p])
        for w, (wn, color) in enumerate(zip(S.EVENT_WINDOWS_MS, ('tab:purple', 'tab:green', 'k'))):
            keys = (grp, wn, 'contribution'), (grp, wn, 'rate')
            ok = [r for r in rows if keys[0] in r and keys[1] in r]
            if not ok:
                continue
            dc = np.array([r[keys[0]][1] - r[keys[0]][0] for r in ok])
            dr = np.array([r[keys[1]][1] - r[keys[1]][0] for r in ok])
            a.scatter(dr, dc, s=10, color=color, label=wn)
            fit = st.linregress(dr, dc)
            xs = np.linspace(dr.min(), dr.max(), 10)
            a.plot(xs, fit.intercept + fit.slope * xs, color=color, lw=1)
        a.axhline(0, color='k', lw=0.5)
        a.axvline(0, color='k', lw=0.5)
        a.set_xlabel('novel − familiar, evoked firing rate (spikes/s per unit)')
        a.set_ylabel('novel − familiar, contribution RMS')
        a.set_title(f'{rf.label(grp)}: per mouse; does the contribution follow firing?', fontsize=9)
        a.legend(fontsize=7)
        a = ax[p, 3]
        letter(a, 'gh'[p])
        text = []
        for w, wn in enumerate(S.EVENT_WINDOWS_MS):
            keys = (grp, wn, 'contribution'), (grp, wn, 'rate')
            ok = [r for r in rows if keys[0] in r and keys[1] in r]
            if len(ok) < 5:
                continue
            dc = np.array([r[keys[0]][1] - r[keys[0]][0] for r in ok])
            dr = np.array([r[keys[1]][1] - r[keys[1]][0] for r in ok])
            fit = st.linregress(dr, dc)
            residual = dc - fit.slope * dr                     # change not explained by the change in firing (intercept + noise)
            boot = []
            rng = np.random.default_rng(0)
            for _ in range(2000):
                i = rng.integers(len(dc), size=len(dc))
                if np.ptp(dr[i]) > 0:
                    boot.append(st.linregress(dr[i], dc[i]).intercept)
            rf.strip(a, residual, w, 'k', width=0.5)
            text.append(f'{wn}: intercept {fit.intercept:+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}], '
                        f'slope p {fit.pvalue:.2f}')
        if text:
            a.axhline(0, color='k', lw=0.5)
            a.set_xticks(range(3), list(S.EVENT_WINDOWS_MS))
            a.set_ylabel('contribution change not explained by firing change')
            a.set_title('\n'.join(text), fontsize=7)
        else:
            missing(a, 'needs event_rates.npz')
    save(fig, 'fig6', 'Figure 6 | Experience: for the same images on the novel day, the network contribution shrinks; '
                      'is it more than the change in firing?')


# ------------------------------------------------------------------------------------------------- figure 7

def fig7():
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    draws = {}
    for sid in S.sessions():
        for f in sorted(S.variables_dir(sid).glob('simulation_final*.npz')):
            d = np.load(f)
            measures = [str(m) for m in d['measures']]
            for ci, c in enumerate(d['conditions']):
                for kk in range(d['r2'].shape[1]):
                    draws.setdefault(str(c), []).append({(m, metric): d[metric][ci, kk, i] for metric in ('auc', 'spearman_total')
                                                         for i, m in enumerate(measures)})
    a = ax[0, 0]
    letter(a, 'a')
    shown = ['ig_mean', 'ridge_weight', 'spike_lfp_correlation', 'stlfp', 'firing_rate']
    for i, m in enumerate(shown):
        rf.strip(a, [r[(m, 'auc')] for rows in draws.values() for r in rows], i, 'k' if m == 'ig_mean' else 'gray')
    a.axhline(0.5, color='k', lw=0.5)
    a.set_xticks(range(len(shown)), ['attribution', 'ridge weight', 'spike-LFP corr.', 'stLFP', 'firing rate'], fontsize=8)
    a.set_ylabel('finding the true drivers (AUC)')
    a.set_title(f'ground-truth simulation ({sum(map(len, draws.values()))} draws, 4 conditions)', fontsize=9)
    rk = per_session('attribution_baseline', lambda d, ch: dict(fr=d['fractions'], measures=[str(m) for m in d['measures']],
                                                              drop=S.channel_median(d['r2_full'] - d['r2_ablated'][0], ch)))
    ra = per_session('ranked_ablation', lambda d, ch: dict(fr=d['fractions'], measures=[str(m) for m in d['measures']],
                                                         drop=S.channel_median(d['r2_full'] - d['r2_ablated'][0], ch)))
    a = ax[0, 1]
    letter(a, 'b')
    for src, m, text, color in ((rk, 'attribution_mean_baseline', 'attribution', 'k'), (rk, 'ridge_weight', 'ridge weight', 'tab:red'),
                                (rk, 'spike_lfp_correlation', 'spike-LFP correlation', 'tab:orange'), (ra, 'stlfp', 'stLFP', 'tab:green'),
                                (ra, 'firing_rate', 'firing rate', 'tab:brown')):
        rows = list(src.values())
        rf.median_band(a, rows[0]['fr'] * 100, [r['drop'][r['measures'].index(m), :, 2] for r in rows], color, text, shade=color == 'k')
    a.set_xscale('log')
    a.set_xlabel('top units removed (%)')
    a.set_ylabel('R² lost (4-8 Hz, inference)')
    a.legend(fontsize=7)
    a.set_title('real data: removing the top units by each measure', fontsize=9)
    a = ax[0, 2]
    letter(a, 'c')
    frc = per_session('firing_rate_control', lambda d, ch: dict((k, d[k]) for k in d.files))
    if frc:
        top = np.concatenate([v['attr_top'][0] for v in frc.values()])
        ctrl = np.concatenate([v['attr_control'][0] for v in frc.values()])
        ratio = np.concatenate([v['rate_control'][0] / v['rate_top'][0] for v in frc.values()])
        ok = np.abs(np.log(ratio)) < np.log(1.25)
        a.scatter(ctrl[ok], top[ok], s=8, color='k', label=f'rate within 25% ({ok.sum()} pairs)')
        a.scatter(ctrl[~ok], top[~ok], s=8, color='gray', alpha=0.5, label='looser match')
        lim = [min(top.min(), ctrl.min()), max(top.max(), ctrl.max())]
        a.plot(lim, lim, color='k', lw=0.5)
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('rate-matched control unit, mean |attribution| (test block)')
        a.set_ylabel('top unit (selected on the training block)')
        a.legend(fontsize=7)
        a.set_title(f'firing-rate control, broadband ({len(frc)} sessions)', fontsize=9)
    else:
        missing(a, 'firing-rate control running (experiments/firing_rate_contribution.py)')
    units = sx.pooled_units()
    a = ax[1, 0]
    letter(a, 'd')
    rho = sx.hierarchy_slopes(units)
    hva = units[units['area'].isin(list(sx.HIERARCHY))]
    per = hva.groupby(['session_id', 'area'])['attr'].agg(['mean', 'size']).reset_index()
    per = per[per['size'] >= sx.MIN_AREA_UNITS]
    areas = list(sx.HIERARCHY)
    for sid, u in per.groupby('session_id'):
        u = u.set_index('area')
        u = u.reindex([x for x in areas if x in u.index])
        a.plot([sx.HIERARCHY[x] for x in u.index], u['mean'], color='gray', lw=0.3, alpha=0.5)
    a.plot([sx.HIERARCHY[x] for x in areas], per.groupby('area')['mean'].median()[areas], 'k-o', lw=2)
    a.set_yscale('log')
    a.set_xticks([sx.HIERARCHY[x] for x in areas], areas, rotation=45)
    a.set_xlabel('anatomical hierarchy score')
    a.set_ylabel('area mean |attribution per spike|')
    a.set_title('per session Spearman ' + stats.describe(np.array(list(rho.values())), stats.mouse_ids(list(rho))), fontsize=8)
    a = ax[1, 1]
    letter(a, 'e')
    visp = units[(units['area'] == 'VISp') & units['layer'].isin([2, 4, 5, 6])]
    per = visp.groupby(['session_id', 'layer'])['attr_l4'].median().reset_index()
    for i, layer in enumerate([2, 4, 5, 6]):
        rf.strip(a, per.loc[per['layer'] == layer, 'attr_l4'], i, 'tab:blue')
    a.set_yscale('log')
    a.set_xticks(range(4), ['L2/3', 'L4', 'L5', 'L6'])
    a.set_ylabel('|attribution per spike| at L4 channels (session median)')
    a.set_title('VISp units by layer (broadband)', fontsize=9)
    a = ax[1, 2]
    letter(a, 'f')
    rob = per_session('robustness', lambda d, ch: dict((k, d[k]) for k in ('attr_seeds', 'channels') if k in d))
    if rob:
        rho = []
        for v in rob.values():
            a_ = v['attr_seeds']
            for chan in v['channels']:
                for s in range(1, a_.shape[0]):
                    x, y = a_[0, chan, 0], a_[s, chan, 0]
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() > 10:
                        rho.append(st.spearmanr(x[ok], y[ok])[0])
        rf.strip(a, rho, 0, 'k')
        a.set_ylim(0, 1)
        a.set_xticks([0], ['seed s vs seed 0'])
        a.set_ylabel('Spearman of unit attribution (broadband)')
        a.set_title(f'rank stability across initialisations ({len(rob)} sessions)', fontsize=9)
    else:
        missing(a, 'rank stability across seeds running (experiments/robustness.py)')
    save(fig, 'fig7', 'Figure 7 | Which neurons: attribution finds the units the prediction relies on, beyond firing rate')


FIGURES = dict(fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', help=f'of {", ".join(FIGURES)} (default: all)')
    args = parser.parse_args()
    if set(args.names) - set(FIGURES):
        parser.error(f'unknown figures: {" ".join(set(args.names) - set(FIGURES))}')
    warnings.simplefilter('ignore')
    for name in args.names or FIGURES:
        FIGURES[name]()


if __name__ == '__main__':
    main()
