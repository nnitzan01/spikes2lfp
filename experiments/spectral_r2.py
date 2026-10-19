"""
How well do different groups of units predict each frequency band of the broadband VISp LFP?

For each unit group (all units, visual cortex, visual cortex without VISp, hippocampal formation,
everything else) train the standard broadband LSTM on every channel, then compute test R2 of the
broadband prediction and of band-passed versions of the recording and the prediction.

usage (from the repo root):
    python -m experiments.spectral_r2 <session_dir>

Saves to variables/<session_id>/:
    r2_results_{full_model,cortex,cortex_no_visp,hpc,subcortical}.csv   (bands x channels)
    units_counts_and_rates.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

import config
import models
import paths
from preprocess_data import chunk_and_reshape, get_data_loaders
from run_session import SessionPipeline
from spectral_analysis import bandpass_filter

HIPPOCAMPAL = ('CA', 'DG', 'SUB', 'HPF', 'ProS', 'POST')   # substrings of hippocampal-formation acronyms
# row names of units_counts_and_rates.csv (as read by pool_sessions.ipynb)
COUNT_ROWS = {'cortex': 'cortex', 'cortex_no_visp': 'cortex_no_visp', 'hpc': 'hippocampus', 'subcortical': 'subcortical'}
R2_BANDS = [b for b in config.BANDS if b[1] < 0.5 / config.BIN_SIZE]   # bands below the 125 Hz Nyquist


def unit_groups(areas):
    """Boolean unit masks of the groups compared in this experiment."""
    visual = np.char.startswith(areas.astype(str), 'VIS')
    hpc = np.array([any(h in a for h in HIPPOCAMPAL) for a in areas])
    return {
        'full_model': np.ones(len(areas), dtype=bool),
        'cortex': visual,
        'cortex_no_visp': visual & (areas != 'VISp'),   # exact match: VISpm, VISpl, VISpor stay in
        'hpc': hpc,
        'subcortical': ~visual & ~hpc,                   # everything else
    }


def band_r2(lfp, prediction, test_bins):
    """R2 on the test bins of the broadband signal and of each band, filtering the continuous signals."""
    fs = 1 / config.BIN_SIZE
    scores = {f'{low}-{high} Hz': r2_score(bandpass_filter(lfp, (low, high), fs)[test_bins],
                                           bandpass_filter(prediction, (low, high), fs)[test_bins])
              for low, high in R2_BANDS}
    scores['broadband'] = r2_score(lfp[test_bins], prediction[test_bins])
    return pd.Series(scores)


def run(pipeline):
    spikes = pipeline.active_spikes.spkMat
    lfp = pipeline.active_lfp.lfpMat[:, :, 0]                       # broadband target
    units = pipeline.session.units
    areas = units['structure_acronym'].to_numpy()
    torch.manual_seed(config.TORCH_SEED)

    results, counts = {}, {}
    for name, mask in unit_groups(areas).items():
        print(f'{name}: {mask.sum()} units', flush=True)
        if name in COUNT_ROWS:
            counts[COUNT_ROWS[name]] = [mask.sum(), units.loc[mask, 'firing_rate'].mean()]
        X_train, X_test, y_train, y_test, _, test_idx = chunk_and_reshape(
            spikes[:, mask], lfp, config.SEQLENGTH, return_indices=True)
        # bins of the test chunks in the continuous recording
        test_bins = (test_idx[:, None] * config.SEQLENGTH + np.arange(config.SEQLENGTH)).ravel()
        per_channel = {}
        for chan in range(lfp.shape[1]):
            train_dl, test_dl = get_data_loaders(X_train, X_test, y_train[:, :, chan], y_test[:, :, chan])
            model = models.build_lstm(int(mask.sum()), pipeline.device)
            model.train(train_dl, test_dl, config.NUM_EPOCHS)
            prediction = model.evaluate(spikes[:, mask]).numpy()     # whole epoch, chunk by chunk
            per_channel[f'channel_{chan}'] = band_r2(lfp[:len(prediction), chan], prediction, test_bins)
        results[name] = pd.DataFrame(per_channel)
    counts = pd.DataFrame.from_dict(counts, orient='index', columns=['num_units', 'mean_firing_rate'])
    return results, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    results, counts = run(pipeline)

    out = pipeline.dirs.variables
    out.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(out / f'r2_results_{name}.csv', index=True)
    counts.to_csv(out / 'units_counts_and_rates.csv', index=True)
    print('Saved to', out)


if __name__ == '__main__':
    main()
