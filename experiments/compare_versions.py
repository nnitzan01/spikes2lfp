"""
Compare the new pipeline (spikes2lfp_v2) with the first, published version (spikes2lfp) for one session.

usage (from the repo root, after run_session.py has finished the session):
    python -m experiments.compare_versions <session_dir>

Prints a summary and writes <data_root>/spikes2lfp_v2/plots/<session_id>/compare_v1_v2.pdf and .json.

What differs by design between the versions (so some differences are expected):
    v1: LSTM without conv, random 3 s chunks as test set, spikes min-max scaled per time bin,
        step-by-step ("instantaneous") integrated gradients, per-unit score = IPI (signed share of
        the co-active units' |attribution|, averaged over the unit's spike bins)
    v2: causal CNN-LSTM, last 20% of the session as test set, spikes scaled per unit, history-aware
        integrated gradients over 3 s windows; per-unit mean |attribution|, per spike and share

Compared:
    1. test R2 per channel x band, and its depth profile (Fig. 3C/F), plus v2 train/validation R2
    2. passive / spontaneous transfer R2, where v1 values exist
    3. per-unit attribution: rank correlation between v1 IPI and the v2 measures per channel x band,
       overlap of the top-10 units, and agreement of the area ranking (Figs. 7, S7)
    4. v2 only: lag kernels by area, completeness error and integration steps of every model
    5. baseline-referenced Jaccard time course (v1 used tie-affected ranks and K=40 of all units)
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

import config
import paths
from run_session import completeness_error

warnings.filterwarnings('ignore', 'All-NaN slice')    # bands that were not attributed give NaN rows

TOP = 10                    # size of the top-unit sets compared
MIN_AREA_UNITS = 5          # areas with fewer units are left out of the area ranking
BAND_NAMES = ['broadband'] + [f'{lo}-{hi} Hz' for lo, hi in config.BANDS]


# ---- loading ------------------------------------------------------------------------------------

def first_existing(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


def load_v1(data_root, sid):
    """v1 outputs of one session: per-session folders first, then the gathered folders used by pool_sessions."""
    base = Path(data_root) / 'spikes2lfp'
    files = dict(
        r2=first_existing(base / 'variables' / sid / 'r2_scores.npy', base / 'r2_all' / f'{sid}_r2_scores.npy'),
        r2_spont=first_existing(base / 'variables' / sid / 'r2_scores_spont.npy', base / 'r2_all' / f'{sid}_r2_scores_spont.npy'),
        r2_passive=first_existing(base / 'variables' / sid / 'r2_scores_passive.npy', base / 'r2_all' / f'{sid}_r2_scores_passive.npy'),
        ipi=first_existing(base / 'attrs' / sid / 'attribution_scores_ipi_mean.npy',
                           base / 'avr_attrs' / f'{sid}_attribution_scores_ipi_mean.npy'),
        ipi_spont=first_existing(base / 'attrs_spontaneous' / sid / 'attribution_scores_spontaneous_ipi_mean.npy',
                                 base / 'avr_attrs_spont' / f'{sid}_attribution_scores_spontaneous_ipi_mean.npy'),
        jaccard_ref=first_existing(base / 'variables' / sid / 'jaccard_ref_snippets.npy',
                                   base / 'Jaccard_analysis' / f'{sid}_jaccard_ref_snippets.npy'),
    )
    return {k: (np.load(f) if f else None) for k, f in files.items()}, {k: (str(f) if f else None) for k, f in files.items()}


def load_v2(dirs):
    v = dirs.variables
    out = dict(r2=np.load(v / 'r2_scores.npy'),
               r2_val=np.load(v / 'r2_scores_val.npy')[0], r2_train=np.load(v / 'r2_scores_train.npy')[0],
               r2_passive=np.load(v / 'r2_scores_passive.npy') if (v / 'r2_scores_passive.npy').exists() else None,
               r2_spont=np.load(v / 'r2_scores_spont.npy') if (v / 'r2_scores_spont.npy').exists() else None,
               areas=pd.read_csv(v / 'units.csv')['structure_acronym'].to_numpy(),
               jaccard_ref=np.load(v / 'jaccard_ref_snippets.npy') if (v / 'jaccard_ref_snippets.npy').exists() else None,
               peri_times=np.load(v / 'peri_times.npy') if (v / 'peri_times.npy').exists() else None)
    for s in ('active', 'spont'):
        f = dirs.attrs / f'unit_attribution_{s}.npz'
        out[f'units_{s}'] = dict(np.load(f)) if f.exists() else None
    n_ch = out['r2'].shape[0]
    out['completeness'] = np.full((n_ch, config.N_TARGETS), np.nan)
    out['n_steps'] = np.full((n_ch, config.N_TARGETS), np.nan)
    out['kernels'] = {}
    for c in range(n_ch):
        for b in range(config.N_TARGETS):
            f = paths.attribution_file(dirs, c, b)
            if f.exists():
                a = np.load(f)
                out['completeness'][c, b] = completeness_error(a)
                out['n_steps'][c, b] = int(a['n_steps']) if 'n_steps' in a else config.IG_N_STEPS
                if b == 0:
                    out['kernels'][c] = a['kernel_abs_mean']
    return out


# ---- comparisons --------------------------------------------------------------------------------

def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return spearmanr(x[ok], y[ok])[0]


def attribution_agreement(ipi, units, areas):
    """Per channel x band agreement between v1 IPI and the v2 unit measures."""
    n_ch = min(ipi.shape[0], units['mean_abs'].shape[0])
    measures = {'mean |attr|': np.abs(units['mean_abs']), 'per spike': units['per_spike'], 'share': units['share']}
    rho = {m: np.full((n_ch, config.N_TARGETS), np.nan) for m in measures}
    top_overlap = np.full((n_ch, config.N_TARGETS), np.nan)
    area_rho = np.full((n_ch, config.N_TARGETS), np.nan)
    keep_areas = [a for a in np.unique(areas) if np.sum(areas == a) >= MIN_AREA_UNITS]
    for c in range(n_ch):
        for b in range(config.N_TARGETS):
            v1 = ipi[c, b]
            if not np.isfinite(units['mean_abs'][c, b]).any():
                continue                                   # band not attributed in v2
            rho['mean |attr|'][c, b] = spearman(np.abs(v1), measures['mean |attr|'][c, b])
            rho['per spike'][c, b] = spearman(v1, measures['per spike'][c, b])           # signed vs signed
            rho['share'][c, b] = spearman(np.abs(v1), measures['share'][c, b])
            top1 = set(np.argsort(-np.abs(v1))[:TOP])
            top2 = set(np.argsort(-measures['mean |attr|'][c, b])[:TOP])
            top_overlap[c, b] = len(top1 & top2) / TOP
            a1 = [np.nanmean(np.abs(v1[areas == a])) for a in keep_areas]
            a2 = [np.nanmean(measures['mean |attr|'][c, b][areas == a]) for a in keep_areas]
            area_rho[c, b] = spearman(np.array(a1), np.array(a2))
    return rho, top_overlap, area_rho, keep_areas


# ---- report -------------------------------------------------------------------------------------

def _band_axes(ax, n_channels):
    """Band names on x, integer channel indices on y (0 at the bottom)."""
    ax.set_xticks(range(config.N_TARGETS))
    ax.set_xticklabels(BAND_NAMES, rotation=45, ha='right')
    ax.set_yticks(range(0, n_channels, max(1, n_channels // 12)))
    ax.set_ylabel('channel')


def compare(data_root, sid):
    dirs = paths.session_output_dirs(data_root, sid)
    v1, v1_files = load_v1(data_root, sid)
    v2 = load_v2(dirs)
    n_ch = v2['r2'].shape[0]
    best = int(np.nanargmax(v2['r2'][:, 0]))                       # channel with the best broadband test R2
    summary = dict(session=sid, v1_files=v1_files, n_channels_v2=n_ch, best_broadband_channel_v2=best)

    # 1. performance
    summary['r2_test_median_v2'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(v2['r2'], axis=0), 3).tolist()))
    summary['r2_test_max_v2'] = dict(zip(BAND_NAMES, np.round(np.nanmax(v2['r2'], axis=0), 3).tolist()))
    summary['r2_val_median_v2'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(v2['r2_val'], axis=0), 3).tolist()))
    summary['r2_train_median_v2'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(v2['r2_train'], axis=0), 3).tolist()))
    if v1['r2'] is not None:
        m = min(n_ch, v1['r2'].shape[0])
        summary['r2_test_median_v1'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(v1['r2'], axis=0), 3).tolist()))
        summary['r2_test_max_v1'] = dict(zip(BAND_NAMES, np.round(np.nanmax(v1['r2'], axis=0), 3).tolist()))
        summary['r2_depth_profile_correlation'] = dict(zip(BAND_NAMES, [
            round(float(np.corrcoef(v1['r2'][:m, b], v2['r2'][:m, b])[0, 1]), 3) for b in range(config.N_TARGETS)]))
        summary['n_channels_v1'] = int(v1['r2'].shape[0])
    for k in ('r2_passive', 'r2_spont'):
        if v2[k] is not None:
            summary[f'{k}_median_v2'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(v2[k], axis=0), 3).tolist()))
        if v1[k] is not None:
            summary[f'{k}_median_v1'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(v1[k], axis=0), 3).tolist()))

    # 3. attribution
    agreement = None
    if v1['ipi'] is not None and v2['units_active'] is not None:
        if v1['ipi'].shape[2] != len(v2['areas']):
            summary['attribution'] = f"unit counts differ (v1 {v1['ipi'].shape[2]}, v2 {len(v2['areas'])}); not compared"
        else:
            agreement = attribution_agreement(v1['ipi'], v2['units_active'], v2['areas'])
            rho, top_overlap, area_rho, keep_areas = agreement
            summary['attribution_median_spearman'] = {m: dict(zip(BAND_NAMES, np.round(np.nanmedian(r, axis=0), 3).tolist()))
                                                     for m, r in rho.items()}
            summary['top10_overlap_median'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(top_overlap, axis=0), 2).tolist()))
            summary['area_ranking_spearman_median'] = dict(zip(BAND_NAMES, np.round(np.nanmedian(area_rho, axis=0), 3).tolist()))
            summary['areas_ranked'] = keep_areas

    # 4. completeness
    comp = v2['completeness']
    summary['completeness'] = dict(models=int(np.isfinite(comp).sum()), median=float(np.nanmedian(comp)),
                                   above_1_percent=int(np.nansum(comp > 0.01)),
                                   steps_used={int(s): int(np.sum(v2['n_steps'] == s)) for s in np.unique(v2['n_steps'][np.isfinite(v2['n_steps'])])})

    pdf_path = dirs.plots / 'compare_v1_v2.pdf'
    dirs.plots.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _page_performance(pdf, v1, v2, sid)
        if agreement is not None:
            _page_attribution(pdf, agreement, v1['ipi'], v2, best)
        _page_v2_only(pdf, v2, best)
        if v1['jaccard_ref'] is not None or v2['jaccard_ref'] is not None:
            _page_jaccard(pdf, v1['jaccard_ref'], v2, best)
    (dirs.plots / 'compare_v1_v2.json').write_text(json.dumps(summary, indent=2, default=float))
    return summary, pdf_path


def _page_performance(pdf, v1, v2, sid):
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    vmin, vmax = -0.1, 0.8
    for i, (name, r2) in enumerate((('v1 test R2', v1['r2']), ('v2 test R2', v2['r2']))):
        if r2 is not None:
            im = ax[0, i].imshow(r2, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            fig.colorbar(im, ax=ax[0, i])
        ax[0, i].set_title(name)
        _band_axes(ax[0, i], (r2 if r2 is not None else v2['r2']).shape[0])
    if v1['r2'] is not None:
        m = min(v1['r2'].shape[0], v2['r2'].shape[0])
        im = ax[0, 2].imshow(v2['r2'][:m] - v1['r2'][:m], aspect='auto', cmap='RdBu_r', vmin=-0.4, vmax=0.4, origin='lower')
        fig.colorbar(im, ax=ax[0, 2])
        _band_axes(ax[0, 2], m)
    ax[0, 2].set_title('v2 - v1')
    for i, b in enumerate((0, 1, 2)):
        a = ax[1, i]
        ch = np.arange(v2['r2'].shape[0])
        if v1['r2'] is not None:
            a.plot(v1['r2'][:, b], np.arange(v1['r2'].shape[0]), 'k-', label='v1 test')
        a.plot(v2['r2'][:, b], ch, 'r-', label='v2 test')
        a.plot(v2['r2_val'][:, b], ch, 'r--', label='v2 validation')
        a.plot(v2['r2_train'][:, b], ch, 'r:', label='v2 train')
        for key, style in (('r2_passive', 'b-'), ('r2_spont', 'g-')):
            if v2[key] is not None:
                a.plot(v2[key][:, b], ch, style, label=f'v2 {key[3:]}')
            if v1[key] is not None:
                a.plot(v1[key][:, b], np.arange(v1[key].shape[0]), style.replace('-', '--'), alpha=0.6, label=f'v1 {key[3:]}')
        a.axvline(0, color='gray', lw=0.5)
        a.set_xlim(-0.3, 1.0)                                       # very negative R2 values are clipped
        a.set_title(f'{BAND_NAMES[b]} R2 by channel')
        a.set_xlabel('R2')
        a.set_ylabel('channel')
    ax[1, 0].legend(fontsize=7)
    fig.suptitle(f'Session {sid}: model performance (v1 = published models, v2 = new pipeline)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_attribution(pdf, agreement, ipi, v2, best):
    rho, top_overlap, area_rho, keep_areas = agreement
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    for i, (m, r) in enumerate(rho.items()):
        im = ax[0, i].imshow(r, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
        fig.colorbar(im, ax=ax[0, i])
        ax[0, i].set_title(f'Spearman: v1 IPI vs v2 {m}')
        _band_axes(ax[0, i], r.shape[0])
    im = ax[1, 0].imshow(top_overlap, aspect='auto', cmap='viridis', vmin=0, vmax=1, origin='lower')
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title(f'top-{TOP} unit overlap (v1 |IPI| vs v2 mean |attr|)')
    im = ax[1, 1].imshow(area_rho, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title(f'area ranking agreement ({len(keep_areas)} areas)')
    for a in ax[1, :2]:
        _band_axes(a, top_overlap.shape[0])
    # area means for the best broadband channel
    areas = v2['areas']
    a1 = np.array([np.nanmean(np.abs(ipi[best, 0][areas == a])) for a in keep_areas])
    a2 = np.array([np.nanmean(v2['units_active']['mean_abs'][best, 0][areas == a]) for a in keep_areas])
    ax[1, 2].scatter(a1, a2)
    for a, x, y in zip(keep_areas, a1, a2):
        ax[1, 2].annotate(a, (x, y), fontsize=7)
    ax[1, 2].set_xlabel('v1 mean |IPI|')
    ax[1, 2].set_ylabel('v2 mean |attribution|')
    ax[1, 2].set_title(f'area means, broadband, channel {best}')
    fig.suptitle('Per-unit attribution: v1 vs v2')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_v2_only(pdf, v2, best):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    im = ax[0].imshow(v2['completeness'], aspect='auto', cmap='magma', origin='lower',
                      norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=1))
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('v2 completeness error (models > 1% are redone with more steps)')
    im = ax[1].imshow(v2['n_steps'], aspect='auto', cmap='viridis', origin='lower')
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title('v2 integration steps used')
    for a in ax[:2]:
        _band_axes(a, v2['completeness'].shape[0])
    if best in v2['kernels']:
        kernel = v2['kernels'][best]                               # (lag bins, oldest first) x units
        lag_ms = (np.arange(kernel.shape[0])[::-1] + 0.5) * config.LAG_KERNEL_BIN * config.BIN_SIZE * 1000
        areas = v2['areas']
        names = [a for a in np.unique(areas) if np.sum(areas == a) >= MIN_AREA_UNITS]
        totals = {a: kernel[:, areas == a].mean(axis=1) for a in names}
        for a in sorted(totals, key=lambda a: -totals[a].sum())[:8]:
            ax[2].plot(lag_ms, totals[a], label=a)
        ax[2].set_xscale('log')
        ax[2].set_xlabel('lag before the predicted bin (ms)')
        ax[2].set_ylabel('mean |attribution| per unit')
        ax[2].set_title(f'v2 lag kernels, broadband, channel {best}')
        if totals:
            ax[2].legend(fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_jaccard(pdf, v1_jac, v2, best):
    fig, ax = plt.subplots(figsize=(7, 5))
    if v1_jac is not None and best < v1_jac.shape[2]:
        t1 = np.arange(v1_jac.shape[1]) * config.BIN_SIZE + config.STIM_WINDOW[0]
        ax.plot(t1, np.nanmean(v1_jac[:, :, best], axis=0), 'k', label='v1 (ties among silent units, all trials)')
    if v2['jaccard_ref'] is not None:
        ax.plot(v2['peri_times'], np.nanmean(v2['jaccard_ref'][:, :, best], axis=0), 'r',
                label=f'v2 ({v2["jaccard_ref"].shape[0]} images)')
    ax.axvline(0, color='gray', ls='--')
    ax.axvline(0.25, color='gray', ls='--')
    ax.set_xlabel('time from image onset (s)')
    ax.set_ylabel(f'Jaccard similarity to baseline top-{config.TOP_K}')
    ax.set_title(f'Baseline-referenced Jaccard, broadband, channel {best}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def print_summary(s):
    print(f"Session {s['session']}: v2 has {s['n_channels_v2']} channels, best broadband channel {s['best_broadband_channel_v2']}")
    print('v1 files found:', {k: bool(v) for k, v in s['v1_files'].items()})
    rows = ['r2_test_median_v1', 'r2_test_median_v2', 'r2_test_max_v1', 'r2_test_max_v2', 'r2_val_median_v2',
            'r2_train_median_v2', 'r2_passive_median_v1', 'r2_passive_median_v2', 'r2_spont_median_v1',
            'r2_spont_median_v2', 'r2_depth_profile_correlation', 'top10_overlap_median', 'area_ranking_spearman_median']
    table = pd.DataFrame({r: s[r] for r in rows if r in s}).T
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(table.round(3))
        if 'attribution_median_spearman' in s:
            print('\nmedian Spearman over channels, v1 IPI vs v2 measure:')
            print(pd.DataFrame(s['attribution_median_spearman']).T.round(3))
    if isinstance(s.get('attribution'), str):
        print(s['attribution'])
    print('\ncompleteness:', s['completeness'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    summary, pdf_path = compare(data_root, str(session_id))
    print_summary(summary)
    print(f'\nFigures: {pdf_path}')


if __name__ == '__main__':
    main()
