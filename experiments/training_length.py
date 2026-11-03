"""
Are the models undertrained? Most broadband models of the timing session keep their last epoch
(NUM_EPOCHS = 15) with the validation loss still falling. This retrains the standard CNN-LSTM with
a higher epoch cap (same patience, same split and seed) and compares test R2 with the saved models.

Also measures the cost of the zero initial state: every 3 s sequence starts with an empty LSTM
memory, so its first bins are predicted without history (the lagged ridge always has its full
history). R2 is reported over all test bins and over the bins at least WARMUP s into a sequence.

For every channel x --targets, and each of --inputs (units: the model's input; regions: the
region-count baseline of baselines.py):
  r2_<inputs>, r2_warm_<inputs>, best_epoch_<inputs>   retrained with --epochs
and for the saved models (units, NUM_EPOCHS):
  r2_saved, r2_warm_saved, best_epoch_saved

usage (from the repo root):
    python -m experiments.training_length <session_dir> [--epochs 40] [--targets 0,1,2] [--inputs units,regions]

Needs the 'train' and 'covariates' stages. Saves variables/<session_id>/training_length.npz
(channels x targets arrays, NaN for targets not retrained).
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import r2_score

import config
import paths
from experiments.baselines import lstm_inputs
from preprocess_data import take_chunks
from run_session import SessionPipeline, save_npz
from splits import chunk_bins

WARMUP = 1.0                    # s of each sequence left out of r2_warm


def warm_r2(model, X, y, chunks):
    """Test R2 over the bins at least WARMUP s into each sequence."""
    skip = int(round(WARMUP / config.BIN_SIZE))
    prediction = model.evaluate(take_chunks(X, chunks)).numpy().reshape(len(chunks), config.SEQLENGTH)
    return r2_score(take_chunks(y, chunks)[:, skip:].ravel(), prediction[:, skip:].ravel())


def run(pipeline, targets=(0, 1, 2), num_epochs=40, input_sets=('units', 'regions')):
    Y = pipeline.active_lfp.lfpMat
    n_ch = Y.shape[1]
    test = pipeline.split['test']
    shape = (n_ch, config.N_TARGETS)
    out = {f'{k}_{s}': np.full(shape, np.nan) for k in ('r2', 'r2_warm', 'best_epoch') for s in ('saved',) + tuple(input_sets)}

    X_units = pipeline.active_spikes.spkMat
    out['best_epoch_saved'] = np.load(pipeline.dirs.variables / 'best_epoch.npy')[0].astype(float)
    out['best_epoch_saved'][:, [t for t in range(config.N_TARGETS) if t not in targets]] = np.nan
    saved = pipeline.models
    for band in targets:
        for chan in range(n_ch):
            model, y = saved[chan, band], Y[:, chan, band]
            out['r2_saved'][chan, band] = r2_score(take_chunks(y, test).ravel(), model.evaluate(take_chunks(X_units, test)).numpy())
            out['r2_warm_saved'][chan, band] = warm_r2(model, X_units, y, test)

    train = chunk_bins(pipeline.split['train'])
    for inputs in input_sets:
        X = X_units if inputs == 'units' else lstm_inputs(pipeline, inputs, train)
        for band in targets:
            for chan in range(n_ch):
                y = Y[:, chan, band]
                model, out[f'r2_{inputs}'][chan, band] = pipeline.fit_and_score(X, y, num_epochs=num_epochs)
                out[f'r2_warm_{inputs}'][chan, band] = warm_r2(model, X, y, test)
                out[f'best_epoch_{inputs}'][chan, band] = model.best_epoch
                print(f'{inputs} chan {chan} band {band}: R2 {out[f"r2_{inputs}"][chan, band]:.3f} '
                      f'(saved model {out["r2_saved"][chan, band]:.3f}), after warm-up {out[f"r2_warm_{inputs}"][chan, band]:.3f}, '
                      f'best epoch {model.best_epoch}', flush=True)
        for band in targets:
            print(f'{inputs}, target {band}, median over channels: R2 {np.median(out[f"r2_{inputs}"][:, band]):.3f} '
                  f'(saved {np.median(out["r2_saved"][:, band]):.3f}), after warm-up '
                  f'{np.median(out[f"r2_warm_{inputs}"][:, band]):.3f} (saved {np.median(out["r2_warm_saved"][:, band]):.3f}), '
                  f'best epoch {np.median(out[f"best_epoch_{inputs}"][:, band]):.0f}', flush=True)
    out['num_epochs'] = np.array(num_epochs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--targets', default='0,1,2')
    parser.add_argument('--inputs', default='units,regions')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, [int(t) for t in args.targets.split(',')], args.epochs, args.inputs.split(','))
    f = pipeline.dirs.variables / 'training_length.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
