"""
Is a result model independent? The same per-unit attribution summaries (attrs/<session>/unit_attribution_active.npz,
run_session stage 'summaries') from two output trees, by default spikes2lfp_v3 (CNN) and spikes2lfp_v2 (CNN-LSTM),
compared per session over the pooled channels.

Per session, measure (per_spike, mean_abs) and target, with each unit's score the median over the pooled channels of
|attribution|:
    cross-model     Spearman of the unit scores, top-10% Jaccard (chance about 0.05), Spearman of the area means
                    (areas with >= MIN_AREA_UNITS units)
    per model       log10(VISp / other visual cortex) of the area means; Spearman of the higher visual areas' means with
                    the anatomical hierarchy (Fig. 7B); Spearman of the unit scores with model-free measures and firing
                    rate (tables/units_info.csv: coupling_index, coher_2_5, coher_5_10, firing_rate); test R2 (median
                    over pooled channels)
Saved as a long table (session, measure, target, metric, model, value) to <a>/pooled/model_agreement.csv.

    python -m experiments.model_agreement [--a spikes2lfp_v3] [--b spikes2lfp_v2] [--targets 0,1,2,3,4]
    python -m experiments.model_agreement --summary-only           # summarize the saved table
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import paths
from analysis import stats
from experiments import summarize as S
from experiments.pool_sessions import UNITS_INFO, unit_areas

MEASURES = ['per_spike', 'mean_abs']
MIN_AREA_UNITS = 5
TOP = 0.10
MODEL_FREE = ['coupling_index', 'coher_2_5', 'coher_5_10', 'firing_rate']
VISUAL_OTHER = ['VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'VISli', 'VISpl', 'VISpor']
HIERARCHY = {'VISl': -0.093, 'VISrl': -0.059, 'VISal': 0.152, 'VISpm': 0.327, 'VISam': 0.441}   # Siegle et al. 2021


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[ok], b[ok])[0] if ok.sum() > 10 else np.nan


def top_jaccard(a, b, top=TOP):
    ok = np.isfinite(a) & np.isfinite(b)
    k = max(1, int(top * ok.sum()))
    ta, tb = set(np.argsort(-a[ok])[:k]), set(np.argsort(-b[ok])[:k])
    return len(ta & tb) / len(ta | tb)


def area_means(score, areas):
    s = pd.DataFrame(dict(score=score, area=areas)).groupby('area')['score'].agg(['mean', 'size'])
    return s.loc[s['size'] >= MIN_AREA_UNITS, 'mean']


def per_model(score, areas, info, r2):
    """Per-model metrics of one session, measure and target (dict)."""
    m = area_means(score, areas)
    other = m.reindex(VISUAL_OTHER).dropna()
    hva = m.reindex(list(HIERARCHY)).dropna()
    out = dict(r2=r2,
               visp_vs_other=np.log10(m['VISp'] / other.mean()) if 'VISp' in m and len(other) else np.nan,
               hierarchy=spearmanr(hva, [HIERARCHY[a] for a in hva.index])[0] if len(hva) >= 3 else np.nan)
    for k in MODEL_FREE:
        out[f'vs_{k}'] = spearman(score, info[k].to_numpy(float))
    return out


def session_rows(sid, a, b, targets, measures=MEASURES):
    """Long-format rows of one session, or [] when either tree lacks its summaries."""
    da, db = (paths.DATA_ROOT / t for t in (a, b))
    files = [d / 'attrs' / str(sid) / 'unit_attribution_active.npz' for d in (da, db)]
    if not all(f.exists() for f in files):
        return []
    units = pd.read_csv(da / 'variables' / str(sid) / 'units.csv')
    ub = pd.read_csv(db / 'variables' / str(sid) / 'units.csv')
    if not np.array_equal(units['unit_id'], ub['unit_id']):
        raise ValueError(f'session {sid}: the two trees have different units')
    areas = unit_areas(units)
    info = units[['unit_id']].merge(pd.read_csv(UNITS_INFO), on='unit_id', how='left')
    ch = S.pooled_channels(sid)
    attr = [np.load(f) for f in files]
    r2 = [np.nanmedian(np.load(d / 'variables' / str(sid) / 'r2_scores.npy')[ch], 0) for d in (da, db)]
    rows = []
    for measure in measures:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            scores = [np.nanmedian(np.abs(x[measure][ch]), 0) for x in attr]           # targets x units
        for t in targets:
            sa, sb = scores[0][t], scores[1][t]
            ma, mb = area_means(sa, areas), area_means(sb, areas)
            cross = dict(unit_spearman=spearman(sa, sb), top_jaccard=top_jaccard(sa, sb),
                         area_spearman=spearmanr(ma, mb.reindex(ma.index))[0] if len(ma) >= 3 else np.nan)
            rows += [(sid, measure, t, k, 'cross', v) for k, v in cross.items()]
            for model, s, r in (('a', sa, r2[0][t]), ('b', sb, r2[1][t])):
                rows += [(sid, measure, t, k, model, v) for k, v in per_model(s, areas, info, r).items()]
    return rows


def run(a, b, targets):
    rows = []
    for sid in S.sessions():
        rows += session_rows(sid, a, b, targets)
    table = pd.DataFrame(rows, columns=['session', 'measure', 'target', 'metric', 'model', 'value'])
    out = paths.DATA_ROOT / a / 'pooled'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'model_agreement.csv', index=False)
    print(f'{table["session"].nunique()} sessions with both trees; saved {out / "model_agreement.csv"}')
    return table


def summary(table, a='a', b='b'):
    """Medians over sessions; for per-model metrics, each model's value at the mouse level and their agreement."""
    names = {'a': a, 'b': b}
    for measure, tm in table.groupby('measure', sort=False):
        print(f'\n===== {measure} ({tm["session"].nunique()} sessions) =====')
        cross = tm[tm['model'] == 'cross'].pivot_table(index='metric', columns='target', values='value', aggfunc='median')
        print('cross-model agreement (median over sessions; top-10% Jaccard chance about 0.05):')
        print(cross.round(3).rename(columns=lambda t: S.ALL_BANDS[t]).to_string())
        own = tm[tm['model'] != 'cross']
        for metric, g in own.groupby('metric', sort=False):
            print(f'--- {metric}')
            for t, gt in g.groupby('target'):
                w = gt.pivot(index='session', columns='model', values='value')
                mice = stats.mouse_ids(w.index.to_numpy())
                parts = []
                for m in ('a', 'b'):
                    med, pos, n, p = stats.wilcoxon_mice(w[m].to_numpy(), mice)
                    parts.append(f'{names[m]} {med:+.3f} ({pos}/{n} mice > 0, p {p:.0e})')
                ok = w.notna().all(axis=1)
                rho = spearmanr(w.loc[ok, 'a'], w.loc[ok, 'b'])[0] if ok.sum() > 5 else np.nan
                print(f'  {S.ALL_BANDS[t]:10s} ' + ' | '.join(parts) + f' | across sessions rho {rho:+.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', default='spikes2lfp_v3', help='output tree of the first model (default: the CNN)')
    parser.add_argument('--b', default='spikes2lfp_v2', help='output tree of the second model (default: the CNN-LSTM)')
    parser.add_argument('--targets', default='0,1,2,3,4')
    parser.add_argument('--summary-only', action='store_true', help='summarize <a>/pooled/model_agreement.csv')
    args = parser.parse_args()
    if args.summary_only:
        table = pd.read_csv(paths.DATA_ROOT / args.a / 'pooled' / 'model_agreement.csv')
    else:
        table = run(args.a, args.b, [int(t) for t in args.targets.split(',')])
    summary(table, args.a, args.b)


if __name__ == '__main__':
    main()
