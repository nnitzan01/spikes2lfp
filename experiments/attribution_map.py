"""
A continuous attribution map for the example figure (v1 Figure 4A, redrawn with history-aware attribution): every
bin of a stretch of the test block is attributed (SequenceIG over the full ATTR_HISTORY of input, the attr stage's
baseline), giving each unit's signed contribution to the prediction at every bin.

The stretch is the one of experiments/example_trace.py (its start_chunk) when that file exists, so the map lines up
with the example predictions; otherwise --start-chunk; or an absolute window of session time (--start-time, --end-time;
v1 Figure 4A: session 1119946360, channel 6, 27-30 s). Bins with less than ATTR_HISTORY bins of input before them are
attributed over a window padded with no spikes (as at the start of a training sequence).

usage (from the repo root):
    python -m experiments.attribution_map <session_dir> [--channel 5] [--band 0] [--seconds 3] [--start-chunk N]
        [--step 1] [--start-time 27 --end-time 30]

Needs the 'train' stage. Saves variables/<session_id>/attribution_map.npz:
  attribution  bins x units, summed over each target's input history (signed)
  t (s from the start of the stretch), target, prediction (bins; the model output for each attributed window),
  completeness (relative error), areas, layers (units), block (bins: 'train', 'val', 'test' or 'none'), session_times
  (s), channel, band, start_chunk, step
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.sequence_attribution import attribute_targets, relative_completeness_error
from integrated_gradients import SequenceIG
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins


def blocks(pipeline, bins):
    """Which block of the split each bin belongs to."""
    out = np.full(len(bins), 'none', dtype=object)
    for part in ('train', 'val', 'test'):
        out[np.isin(bins, chunk_bins(pipeline.split[part]))] = part
    return out.astype(str)


def run(pipeline, channel=5, band=0, seconds=3.0, start_chunk=None, step=1, start_time=None, end_time=None):
    ts = pipeline.active_spikes.timestamps
    if start_time is not None:
        bins = np.flatnonzero((ts >= start_time) & (ts < end_time))[::step]
        start_chunk = -1
    else:
        if start_chunk is None:
            f = pipeline.dirs.variables / 'example_trace.npz'
            start_chunk = int(np.load(f)['start_chunk']) if f.exists() else 0
        n = int(round(seconds / config.BIN_SIZE))
        chunks = pipeline.split['test'][start_chunk:start_chunk + int(np.ceil(n / config.SEQLENGTH))]
        bins = chunk_bins(chunks)[:n:step]                                 # targets, in the continuous active epoch
    pad = max(0, config.ATTR_HISTORY - 1 - int(bins.min()))                # windows reaching before the epoch: no spikes
    spikes = np.vstack([np.zeros((pad, pipeline.active_spikes.spkMat.shape[1]), np.float32), pipeline.active_spikes.spkMat])
    X = torch.as_tensor(spikes, dtype=torch.float32, device=pipeline.device)
    model = pipeline.models[channel, band].model.to(pipeline.device)
    ig = SequenceIG(model, n_steps=config.IG_N_STEPS, max_sequences=config.IG_MAX_SEQUENCES)
    res = attribute_targets(ig, X, bins + pad, config.ATTR_HISTORY, baseline=pipeline.ig_baseline(X.shape[1]),
                            batch_windows=pipeline.ig_batch)
    err = relative_completeness_error(res['f_x'], res['f_b'], res['delta'])
    log(f'attribution map: {len(bins)} bins x {X.shape[1]} units, completeness error {err:.1e}')
    units = pipeline.session.units
    return dict(attribution=res['total'].astype(np.float32), t=(bins - bins[0]) * config.BIN_SIZE,
                target=pipeline.active_lfp.lfpMat[bins, channel, band], prediction=np.asarray(res['f_x']),
                completeness=np.array(err), areas=units['structure_acronym'].to_numpy().astype(str),
                layers=units['layer'].to_numpy() if 'layer' in units else np.full(len(units), np.nan),
                block=blocks(pipeline, bins), session_times=ts[bins], channel=np.array(channel),
                band=np.array(band), start_chunk=np.array(start_chunk), step=np.array(step))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--channel', type=int, default=5)
    parser.add_argument('--band', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--start-chunk', type=int, default=None)
    parser.add_argument('--step', type=int, default=1, help='attribute every step-th bin')
    parser.add_argument('--start-time', type=float, default=None, help='absolute session time (s); with --end-time')
    parser.add_argument('--end-time', type=float, default=None)
    parser.add_argument('--output', default='attribution_map', help='file name (without .npz) in variables/<session_id>/')
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.channel, args.band, args.seconds, args.start_chunk, args.step, args.start_time, args.end_time)
    f = pipeline.dirs.variables / f'{args.output}.npz'
    save_npz(f, compressed=True, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
