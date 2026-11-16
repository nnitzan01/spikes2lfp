"""
A continuous attribution map for the example figure (v1 Figure 4A, redrawn with history-aware attribution): every
bin of a stretch of the test block is attributed (SequenceIG over the full ATTR_HISTORY of input, the attr stage's
baseline), giving each unit's signed contribution to the prediction at every bin.

The stretch is the one of experiments/example_trace.py (its start_chunk) when that file exists, so the map lines up
with the example predictions; otherwise --start-chunk.

usage (from the repo root):
    python -m experiments.attribution_map <session_dir> [--channel 5] [--band 0] [--seconds 3] [--start-chunk N]
        [--step 1]

Needs the 'train' stage. Saves variables/<session_id>/attribution_map.npz:
  attribution  bins x units, summed over each target's input history (signed)
  t (s from the start of the stretch), target, prediction (bins), completeness (relative error),
  areas (units), channel, band, start_chunk, step
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.sequence_attribution import attribute_targets, relative_completeness_error
from integrated_gradients import SequenceIG
from preprocess_data import take_chunks
from run_session import SessionPipeline, log, save_npz
from splits import chunk_bins


def run(pipeline, channel=5, band=0, seconds=3.0, start_chunk=None, step=1):
    if start_chunk is None:
        f = pipeline.dirs.variables / 'example_trace.npz'
        start_chunk = int(np.load(f)['start_chunk']) if f.exists() else 0
    n = int(round(seconds / config.BIN_SIZE))
    chunks = pipeline.split['test'][start_chunk:start_chunk + int(np.ceil(n / config.SEQLENGTH))]
    bins = chunk_bins(chunks)[:n:step]                                     # targets, in the continuous active epoch
    X = torch.as_tensor(pipeline.active_spikes.spkMat, dtype=torch.float32, device=pipeline.device)
    model = pipeline.models[channel, band].model.to(pipeline.device)
    ig = SequenceIG(model, n_steps=config.IG_N_STEPS, max_sequences=config.IG_MAX_SEQUENCES)
    res = attribute_targets(ig, X, bins, config.ATTR_HISTORY, baseline=pipeline.ig_baseline(X.shape[1]),
                            batch_windows=pipeline.ig_batch)
    err = relative_completeness_error(res['f_x'], res['f_b'], res['delta'])
    log(f'attribution map: {len(bins)} bins x {X.shape[1]} units, completeness error {err:.1e}')
    Xc = torch.as_tensor(take_chunks(pipeline.active_spikes.spkMat, chunks), dtype=torch.float32)
    prediction = pipeline.models[channel, band].evaluate(Xc).numpy()[:n:step]
    target = take_chunks(pipeline.active_lfp.lfpMat[:, channel, band], chunks).ravel()[:n:step]
    return dict(attribution=res['total'].astype(np.float32), t=np.arange(0, n, step) * config.BIN_SIZE, target=target,
                prediction=prediction, completeness=np.array(err),
                areas=pipeline.session.units['structure_acronym'].to_numpy().astype(str), channel=np.array(channel),
                band=np.array(band), start_chunk=np.array(start_chunk), step=np.array(step))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--channel', type=int, default=5)
    parser.add_argument('--band', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=3.0)
    parser.add_argument('--start-chunk', type=int, default=None)
    parser.add_argument('--step', type=int, default=1, help='attribute every step-th bin')
    args = parser.parse_args()
    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.channel, args.band, args.seconds, args.start_chunk, args.step)
    f = pipeline.dirs.variables / 'attribution_map.npz'
    save_npz(f, compressed=True, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
