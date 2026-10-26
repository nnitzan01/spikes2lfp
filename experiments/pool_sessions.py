"""
Check that every session's run_session.py outputs are complete, then pool them across sessions.

usage (from the repository root, with SPIKES2LFP_DATA_ROOT set, e.g. by cluster/env.sh):
    python -m experiments.pool_sessions check        # per-session status and what to resubmit
    python -m experiments.pool_sessions pool         # pool the complete sessions

check   lists, for every session in cluster/sessions.txt, what is missing or stale, and prints the
        sbatch commands that redo it. Also written to spikes2lfp_v2/pooled/status.csv.

pool    aligns each complete session's VISp channels on the CSD-derived upper L4 border
        (tables/layer_4_borders_aligned.csv; Methods: Cross-session alignment) and concatenates
        sessions (per-session arrays) or units (per-unit arrays). Outputs in spikes2lfp_v2/pooled/:

          sessions.csv              one row per pooled session: channels, units, alignment, flagged models
          units.csv                 one row per pooled unit (row order of every per-unit array):
                                    session_id, unit_id, area, and the columns of tables/units_info.csv
          r2.npz                    test, val, train, passive, spont R2, attribution completeness error,
                                    best epoch: sessions x channels x targets
          unit_attribution_{active,spont}[_recent].npz
                                    mean, mean_abs, per_spike, share: units x channels x targets
                                    (_recent: from the last ATTR_RECENT_BINS lags)
          components.npz            mean attribution at the four CSD components over ATTR_EVENTS
                                    images: total, recent: units x channels x targets x components
          peri.npz                  broadband peri-stimulus attribution (mean over PERI_EVENTS images):
                                    total, recent: units x times x channels; psth: units x bins
          jaccard.npz               top-K Jaccard (broadband, mean over images): raw, ref, raw_recent,
                                    ref_recent: sessions x times x channels
          area_kernels.npz          lag kernels averaged over the units of each area in each session:
                                    abs_mean, mean: rows x channels x targets x lags (rows = session x area)
          regional_dominance.npz    regional dominance (broadband): dominance, dominance_recent:
                                    rows x times x channels (rows = session x region)

        Channel axes have ALIGNED_CHANNELS entries, deepest first; aligned channel BELOW_L4 is the upper
        L4 border. Channels a session does not have are NaN. Each session is first pooled into
        pooled/sessions/<session_id>.npz, which is reused until its inputs change.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
import paths
from run_session import atomic, save_csv, save_npz

ALIGNED_CHANNELS = 16          # VISp channels kept per session
BELOW_L4 = 10                  # ...from 10 channels below the upper L4 border to 5 above it
UNIT_SETS = ('active', 'spont')                # unit_attribution_<set>.npz files pooled
MEASURES = ('mean', 'mean_abs', 'per_spike', 'share')
AREA_GROUPS = {'SC': ('SCig', 'SCiw')}         # pooled into one area
NUCLEI = ('MG', 'LG')                          # subdivisions (LGd, LGv, MGd, ...) pooled, as in process_session
TARGET_NAMES = ['broadband'] + [f'{lo}-{hi} Hz' for lo, hi in config.BANDS]
SESSION_LIST = paths.REPO_DIR / 'cluster' / 'sessions.txt'
ALIGNMENT_TABLE = paths.REPO_DIR / 'tables' / 'layer_4_borders_aligned.csv'
UNITS_INFO = paths.REPO_DIR / 'tables' / 'units_info.csv'


def pooled_dir(data_root):
    return Path(data_root) / paths.OUTPUT_SUBDIR / 'pooled'


# ---- check -------------------------------------------------------------------------------------

def downstream_files(dirs):
    """Outputs of the summaries, peri and ipd stages."""
    sets = ('train', 'test', 'active', 'spont')
    return ([dirs.attrs / f'unit_attribution_{k}{s}.npz' for k in ('', 'recent_') for s in sets]
            + [dirs.attrs / 'attribution_completeness.npy', dirs.variables / 'psth.npy', dirs.variables / 'peri_times.npy']
            + [dirs.variables / f'{n}{k}.npy' for k in ('', '_recent')
               for n in ('rank_snippets', 'jaccard_raw_snippets', 'jaccard_ref_snippets')]
            + [dirs.variables / f'regional_dominance_results{k}.npz' for k in ('', '_recent')])


def session_status(data_root, session_id):
    """
    What is missing from one session's outputs. Returns a dict: n_channels, n_units, missing (list
    of short descriptions), stages (run_session stages that would fill the gaps), flagged (models
    above IG_MAX_ERROR) and best broadband test R2.
    """
    dirs = paths.session_output_dirs(data_root, session_id)
    s = dict(session_id=session_id, n_channels=np.nan, n_units=np.nan, missing=[], stages=set(), flagged=np.nan,
             r2_broadband_max=np.nan)

    def miss(what, *stages):
        s['missing'].append(what)
        s['stages'].update(stages)

    channels = dirs.variables / 'channels.csv'
    if not channels.exists():
        miss('not trained', 'train')
        return s
    n_ch = s['n_channels'] = len(pd.read_csv(channels))
    n_models = n_ch * config.N_TARGETS
    targets = range(config.N_TARGETS) if config.ATTR_TARGETS is None else config.ATTR_TARGETS
    n_attr = n_ch * len(targets)

    models = sum(paths.model_file(dirs, c, b).exists() for c in range(n_ch) for b in range(config.N_TARGETS))
    if models < n_models:
        miss(f'models {models}/{n_models}', 'train')
    for f in ('r2_scores.npy', 'r2_scores_val.npy', 'r2_scores_train.npy'):
        if not (dirs.variables / f).exists():
            miss(f, 'train')
    for f in ('r2_scores_passive.npy', 'r2_scores_spont.npy'):
        if not (dirs.variables / f).exists():
            miss(f, 'transfer')
    if (dirs.variables / 'r2_scores.npy').exists():
        s['r2_broadband_max'] = float(np.nanmax(np.load(dirs.variables / 'r2_scores.npy')[:, 0]))

    units = dirs.variables / 'units.csv'
    if not units.exists():
        miss('units.csv', 'variables')
    else:
        columns = pd.read_csv(units, nrows=0).columns
        s['n_units'] = len(pd.read_csv(units, usecols=[columns[0]]))
        if 'unit_id' not in columns:
            miss('unit ids in units.csv', 'variables')

    attr_files = [paths.attribution_file(dirs, c, b) for c in range(n_ch) for b in targets]
    present = [f for f in attr_files if f.exists()]
    if len(present) < n_attr:
        miss(f'attributions {len(present)}/{n_attr}', 'attr')
    old = sum('recent_bins' not in np.load(f).files for f in present)
    if old:
        miss(f'{old} attributions without recent-lag sums', 'attr')

    outputs = downstream_files(dirs)
    absent = [f.name for f in outputs if not f.exists()]
    if absent:
        stage_of = lambda n: 'summaries' if n.startswith(('unit_', 'attribution_')) else 'ipd' if n.startswith('regional') else 'peri'
        miss(f'{len(absent)} downstream files (e.g. {absent[0]})', *{stage_of(n) for n in absent})
    elif present and min(f.stat().st_mtime for f in outputs[:9]) < max(f.stat().st_mtime for f in present):
        miss('summaries older than attributions', 'summaries', 'peri', 'ipd')
    completeness = dirs.attrs / 'attribution_completeness.npy'
    if completeness.exists():
        s['flagged'] = int(np.nansum(np.load(completeness) > config.IG_MAX_ERROR))
    return s


def check(data_root, session_ids):
    rows = []
    for task, sid in enumerate(session_ids, start=1):
        s = session_status(data_root, sid)
        rows.append(dict(task=task, session_id=sid, n_channels=s['n_channels'], n_units=s['n_units'],
                         complete=not s['missing'], flagged=s['flagged'], r2_broadband_max=s['r2_broadband_max'],
                         stages=','.join(st for st in ('train', 'transfer', 'variables', 'attr', 'summaries', 'peri', 'ipd')
                                         if st in s['stages']),
                         missing='; '.join(s['missing'])))
    return pd.DataFrame(rows)


def print_check(status):
    with pd.option_context('display.width', 200, 'display.max_rows', 200, 'display.max_colwidth', 80):
        incomplete = status[~status.complete]
        print(f'{status.complete.sum()}/{len(status)} sessions complete; '
              f'{int(np.nansum(status.flagged))} models above the completeness limit in total')
        if len(incomplete):
            print('\nincomplete:')
            print(incomplete[['task', 'session_id', 'stages', 'missing']].to_string(index=False))
            gpu = incomplete[incomplete.stages.str.contains('train|transfer|attr')]
            cpu = incomplete[~incomplete.index.isin(gpu.index)]
            print('\nto redo them:')
            if len(gpu):
                print(f'  sbatch --array={",".join(map(str, gpu.task))}%8 cluster/run_sessions.sbatch')
            if len(cpu):
                stages = sorted(set(','.join(cpu.stages).split(',')), key=['variables', 'summaries', 'peri', 'ipd'].index)
                print(f'  STAGES={",".join(stages)} sbatch --partition=cpu_short --gres=none --mem=32G --time=02:00:00 '
                      f'--array={",".join(map(str, cpu.task))}%20 cluster/run_sessions.sbatch')


# ---- pool --------------------------------------------------------------------------------------

def channel_alignment(row, n_channels):
    """Source channel indices (into the session's VISp channels, deepest first) and their aligned positions."""
    first, last, start = (int(row[k]) for k in ('first_channel_index', 'last_channel_index', 'start_index'))
    if last >= n_channels:
        raise ValueError(f'session {row["session"]}: alignment needs channel {last}, only {n_channels} recorded')
    src = np.arange(first, last + 1)
    return src, start + np.arange(len(src))


def align(x, src, dst, axis=0):
    """Move channels `src` of x (along `axis`) to positions `dst` of an ALIGNED_CHANNELS-long axis (NaN elsewhere)."""
    x = np.moveaxis(np.asarray(x, dtype=np.float32), axis, 0)
    out = np.full((ALIGNED_CHANNELS,) + x.shape[1:], np.nan, dtype=np.float32)
    out[dst] = x[src]
    return np.moveaxis(out, 0, axis)


def unit_areas(units):
    area = units['structure_acronym'].astype(str).copy()
    for nucleus in NUCLEI:
        area[area.str.contains(nucleus)] = nucleus
    for group, members in AREA_GROUPS.items():
        area[area.isin(members)] = group
    return area.to_numpy(dtype=str)


def check_channel_order(channels):
    """The alignment table indexes channels deepest first; check that against the saved channel positions."""
    if 'probe_vertical_position' in channels:
        pos = channels['probe_vertical_position'].to_numpy()
        if not np.all(np.diff(pos) >= 0):
            raise ValueError('VISp channels are not ordered deepest first (probe_vertical_position not increasing)')


def pool_session(data_root, session_id, alignment_row):
    """Aligned arrays of one session (see the module docstring), as a dict."""
    dirs = paths.session_output_dirs(data_root, session_id)
    v, a = dirs.variables, dirs.attrs
    channels = pd.read_csv(v / 'channels.csv')
    check_channel_order(channels)
    n_ch = len(channels)
    src, dst = channel_alignment(alignment_row, n_ch)
    al = lambda x, axis=0: align(x, src, dst, axis)
    units = pd.read_csv(v / 'units.csv')
    areas = unit_areas(units)
    area_names = np.unique(areas)
    out = dict(unit_id=units['unit_id'].to_numpy(), area=areas, area_names=area_names, n_channels=np.array(n_ch),
               src=src, dst=dst)

    # performance (channels x targets)
    out['r2_test'] = al(np.load(v / 'r2_scores.npy'))
    for k in ('val', 'train'):
        out[f'r2_{k}'] = al(np.load(v / f'r2_scores_{k}.npy')[0])
    for k in ('passive', 'spont'):
        out[f'r2_{k}'] = al(np.load(v / f'r2_scores_{k}.npy'))
    out['completeness'] = al(np.load(a / 'attribution_completeness.npy'))
    out['best_epoch'] = al(np.load(v / 'best_epoch.npy')[0])

    # per-unit summaries: (channels x targets x units) -> units x channels x targets
    for kind in ('', 'recent_'):
        for s in UNIT_SETS:
            u = np.load(a / f'unit_attribution_{kind}{s}.npz')
            for m in MEASURES:
                out[f'units_{kind}{s}_{m}'] = al(u[m]).transpose(2, 0, 1)

    # per-model files: CSD components and lag kernels
    n_u, n_comp = len(units), len(config.COMPONENT_TIMES)
    comp = {k: np.full((n_ch, config.N_TARGETS, n_u, n_comp), np.nan, np.float32) for k in ('total', 'recent')}
    kernels = {k: None for k in ('abs_mean', 'mean')}
    members = [areas == name for name in area_names]
    for c in range(n_ch):
        for b in range(config.N_TARGETS):
            f = paths.attribution_file(dirs, c, b)
            if not f.exists():
                continue
            m = np.load(f)
            for k in comp:
                comp[k][c, b] = m[f'components_{k}'].astype(np.float32).mean(axis=0).T     # mean over images
            for k in kernels:
                kern = m[f'kernel_{k}']                                                     # lags x units
                if kernels[k] is None:
                    kernels[k] = np.full((n_ch, config.N_TARGETS, len(area_names), kern.shape[0]), np.nan, np.float32)
                kernels[k][c, b] = np.stack([kern[:, sel].mean(axis=1) for sel in members])
    for k in comp:
        out[f'components_{k}'] = al(comp[k]).transpose(2, 0, 1, 3)           # units x channels x targets x components
    for k, kern in kernels.items():
        out[f'kernel_{k}'] = al(kern).transpose(2, 0, 1, 3)                  # areas x channels x targets x lags

    # broadband peri-stimulus maps (times x units per channel) -> units x times x channels
    peri = {k: [] for k in ('peri_mean', 'peri_recent_mean')}
    for c in range(n_ch):
        m = np.load(paths.attribution_file(dirs, c, 0))
        for k in peri:
            peri[k].append(m[k])
    out['peri_total'] = al(np.stack(peri['peri_mean'])).transpose(2, 1, 0)
    out['peri_recent'] = al(np.stack(peri['peri_recent_mean'])).transpose(2, 1, 0)
    out['peri_times'] = np.load(v / 'peri_times.npy')
    out['psth'] = np.load(v / 'psth.npy').astype(np.float32)

    # Jaccard (events x times x channels) -> times x channels; regional dominance
    for k in ('', '_recent'):
        for name in ('raw', 'ref'):
            with warnings.catch_warnings():             # times without a value (e.g. the first raw Jaccard)
                warnings.simplefilter('ignore', RuntimeWarning)
                jac = np.nanmean(np.load(v / f'jaccard_{name}_snippets{k}.npy'), axis=0)
            out[f'jaccard_{name}{k}'] = al(jac, axis=1)
        d = np.load(v / f'regional_dominance_results{k}.npz', allow_pickle=True)
        out[f'dominance{k}'] = al(d['average_dominance'], axis=2)            # regions x times x channels
        out['dominance_regions'] = d['region_names'].astype(str)
    return out


def session_inputs(data_root, session_id):
    dirs = paths.session_output_dirs(data_root, session_id)
    return [f for d in (dirs.variables, dirs.attrs) for f in d.iterdir() if f.is_file()]


def pooled_session(data_root, session_id, alignment_row, force=False):
    """pool_session, cached in pooled/sessions/<session_id>.npz until any input changes."""
    f = pooled_dir(data_root) / 'sessions' / f'{session_id}.npz'
    if not force and f.exists() and f.stat().st_mtime >= max(i.stat().st_mtime for i in session_inputs(data_root, session_id)):
        return dict(np.load(f, allow_pickle=False))
    out = pool_session(data_root, session_id, alignment_row)
    save_npz(f, **out)
    return out


def stack_rows(per_session, key, names_key):
    """Concatenate per-session (rows x ...) arrays whose rows are named by names_key, with the session of each row."""
    arrays = [s[key] for s in per_session.values()]
    sessions = np.concatenate([np.full(len(s[names_key]), sid) for sid, s in per_session.items()])
    names = np.concatenate([s[names_key] for s in per_session.values()])
    return np.concatenate(arrays), sessions, names


def pool(data_root, session_ids, alignment=None, units_info=None, force=False):
    """Pool the given (complete) sessions into pooled/ (see the module docstring)."""
    alignment = pd.read_csv(ALIGNMENT_TABLE) if alignment is None else alignment
    alignment = alignment.set_index('session')
    out_dir = pooled_dir(data_root)
    per_session, rows = {}, []
    for sid in session_ids:
        t0 = time.time()
        s = per_session[sid] = pooled_session(data_root, sid, alignment.loc[sid], force)
        rows.append(dict(session_id=sid, n_channels=int(s['n_channels']), n_units=len(s['unit_id']),
                         first_channel=int(s['src'][0]), last_channel=int(s['src'][-1]),
                         first_aligned=int(s['dst'][0]), upper_l4_channel=int(alignment.loc[sid, 'upper_border_index']),
                         flagged=int(np.nansum(s['completeness'] > config.IG_MAX_ERROR))))
        print(f'  {sid}: {rows[-1]["n_units"]} units, {time.time() - t0:.0f} s', flush=True)
    sids = list(per_session)
    save_csv(pd.DataFrame(rows), out_dir / 'sessions.csv')

    # units table
    units = pd.DataFrame({'session_id': np.concatenate([np.full(len(s['unit_id']), sid) for sid, s in per_session.items()]),
                          'unit_id': np.concatenate([s['unit_id'] for s in per_session.values()]),
                          'area': np.concatenate([s['area'] for s in per_session.values()])})
    info = pd.read_csv(UNITS_INFO) if units_info is None else units_info
    units = units.merge(info.drop(columns=['session_id'], errors='ignore'), on='unit_id', how='left', validate='many_to_one')
    save_csv(units, out_dir / 'units.csv')

    stack = lambda key: np.stack([per_session[sid][key] for sid in sids])
    cat = lambda key: np.concatenate([per_session[sid][key] for sid in sids])
    save_npz(out_dir / 'r2.npz', session_ids=np.array(sids), targets=np.array(TARGET_NAMES),
             **{k: stack(f'r2_{k}') for k in ('test', 'val', 'train', 'passive', 'spont')},
             completeness=stack('completeness'), best_epoch=stack('best_epoch'))
    for kind in ('', 'recent_'):
        for s in UNIT_SETS:
            save_npz(out_dir / f'unit_attribution_{s}{"_recent" if kind else ""}.npz',
                     **{m: cat(f'units_{kind}{s}_{m}') for m in MEASURES})
    save_npz(out_dir / 'components.npz', total=cat('components_total'), recent=cat('components_recent'),
             components=np.array(list(config.COMPONENT_TIMES)), times=np.array(list(config.COMPONENT_TIMES.values())))
    save_npz(out_dir / 'peri.npz', total=cat('peri_total'), recent=cat('peri_recent'), times=per_session[sids[0]]['peri_times'],
             psth=cat('psth'))
    save_npz(out_dir / 'jaccard.npz', times=per_session[sids[0]]['peri_times'], session_ids=np.array(sids),
             **{f'{n}{k}': stack(f'jaccard_{n}{k}') for n in ('raw', 'ref') for k in ('', '_recent')})
    kern = {k: stack_rows(per_session, f'kernel_{k}', 'area_names') for k in ('abs_mean', 'mean')}
    n_lags = kern['abs_mean'][0].shape[-1]
    lag_ms = (np.arange(n_lags)[::-1] + 0.5) * config.LAG_KERNEL_BIN * config.BIN_SIZE * 1000   # oldest lag first
    save_npz(out_dir / 'area_kernels.npz', abs_mean=kern['abs_mean'][0], mean=kern['mean'][0],
             session_ids=kern['abs_mean'][1], areas=kern['abs_mean'][2], lag_ms=lag_ms,
             n_units=np.concatenate([[np.sum(s['area'] == a) for a in s['area_names']] for s in per_session.values()]))
    dom, dom_sessions, regions = stack_rows(per_session, 'dominance', 'dominance_regions')
    save_npz(out_dir / 'regional_dominance.npz', dominance=dom, dominance_recent=stack_rows(per_session, 'dominance_recent', 'dominance_regions')[0],
             session_ids=dom_sessions, regions=regions, times=per_session[sids[0]]['peri_times'])
    with atomic(out_dir / 'README.json') as tmp:
        tmp.write_text(json.dumps(dict(sessions=len(sids), units=len(units), aligned_channels=ALIGNED_CHANNELS,
                                       upper_l4_aligned_channel=BELOW_L4, recent_lag_ms=config.ATTR_RECENT_BINS * config.BIN_SIZE * 1000,
                                       description=__doc__), indent=2))
    return units


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['check', 'pool'])
    parser.add_argument('--force', action='store_true', help='re-pool sessions even if their cached arrays are current')
    args = parser.parse_args()

    data_root = paths.DATA_ROOT
    session_ids = [int(line) for line in SESSION_LIST.read_text().split()]
    status = check(data_root, session_ids)
    save_csv(status, pooled_dir(data_root) / 'status.csv')
    print_check(status)
    if args.command == 'pool':
        complete = status.session_id[status.complete].tolist()
        print(f'\npooling {len(complete)} sessions into {pooled_dir(data_root)}')
        units = pool(data_root, complete, force=args.force)
        print(f'done: {len(complete)} sessions, {len(units)} units')


if __name__ == '__main__':
    main()
