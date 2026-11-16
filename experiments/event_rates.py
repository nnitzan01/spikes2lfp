"""
Event-locked firing rate of each region group, for the events of experiments/region_events.py (active and passive
epochs, including the shared images of the novelty comparison), so that changes in a group's contribution to the
LFP can be compared with changes in its own firing (is a smaller contribution just fewer spikes?).

usage (from the repo root):
    python -m experiments.event_rates <session_dir> [--grouping regions]

Needs the session's spikes only (no models). Saves variables/<session_id>/event_rates.npz:
  eta_rate        event types x lags x groups, mean firing rate per unit of the group (spikes/s)
  eta_rate_units  event types x units, each unit's mean rate 30-250 ms after onset minus the 250 ms before (spikes/s)
  event_types, n_events, eta_lags_ms, groups, group_n_units, unit_group (index into groups, -1 = none)
"""
import argparse

import numpy as np
import torch

import config
import paths
from experiments.explore_task import (ACTIVE_TYPES, EVENT_WINDOW, auto_rewarded_changes, event_average,
                                      passive_presentations, task_events)
from experiments.region_ablation import region_masks
from experiments.region_events import GROUPINGS, event_bins, grouping_masks
from run_session import SessionPipeline, save_npz

RESPONSE_MS = (30, 250)


def run(pipeline, grouping='regions', seed=config.TORCH_SEED):
    rng = np.random.default_rng(seed)
    s = pipeline.session
    masks = grouping_masks(region_masks(np.asarray(s.units.structure_acronym)), grouping)
    groups = GROUPINGS[grouping]
    unit_group = np.full(len(s.units), -1)
    for i, g in enumerate(groups):
        unit_group[masks[g]] = i
    pre, post = int(round(-EVENT_WINDOW[0] / config.BIN_SIZE)), int(round(EVENT_WINDOW[1] / config.BIN_SIZE))
    lags = (np.arange(-pre, post) + 0.5) * config.BIN_SIZE * 1e3
    events = {'active': task_events(s.active, s.licks.timestamps, auto_rewarded_changes(s))[0],
              'passive': task_events(passive_presentations(s), s.licks.timestamps, prefix='passive_')[0]}
    start, stop = s.passive_times
    counts = {'active': pipeline.active_counts, 'passive': pipeline.spikes(start, stop, scaled=False)}
    bins = {e: event_bins(events[e], counts[e].timestamps, counts[e].spkMat.shape[0]) for e in counts}
    types = [k for k in ACTIVE_TYPES if k in bins['active']] + list(bins['passive'])
    rate = np.full((len(types), pre + post, len(groups)), np.nan)
    unit_rate = np.full((len(types), len(unit_group)), np.nan)
    n_events = np.zeros(len(types), int)
    window = (lags >= RESPONSE_MS[0]) & (lags < RESPONSE_MS[1])
    for e, sp in counts.items():
        X = np.asarray(sp.spkMat, np.float32) / config.BIN_SIZE                       # spikes/s per unit
        for name, b in bins[e].items():
            i = types.index(name)
            eta, n_events[i] = event_average(X, b, pre, post, rng)                     # lags x units
            unit_rate[i] = eta[window].mean(0) - eta[lags < 0].mean(0)
            for g in range(len(groups)):
                if (unit_group == g).any():
                    rate[i, :, g] = eta[:, unit_group == g].mean(1)
    return dict(eta_rate=rate, eta_rate_units=unit_rate, event_types=np.array(types), n_events=n_events,
                eta_lags_ms=lags, groups=np.array(groups), group_n_units=np.array([masks[g].sum() for g in groups]),
                unit_group=unit_group)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--grouping', default='regions', choices=list(GROUPINGS))
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    pipeline = SessionPipeline(data_root, session_id, torch.device('cpu'))
    out = run(pipeline, args.grouping)
    f = pipeline.dirs.variables / ('event_rates.npz' if args.grouping == 'regions' else f'event_rates_{args.grouping}.npz')
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
