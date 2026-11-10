"""
Stimulus clock vs single-trial prediction (reviewers 1 and 3, comment 1): how much of each
model's test performance is the stimulus-locked mean response, and how much is trial-specific?

For every channel x model target of the session's seed-0 models, on the test block
(definitions in analysis/stimulus_clock.py):

  r2_model             R2 of the model
  r2_template          R2 of the template (mean target per image condition and time since onset,
                       fitted on the training block): the stimulus-clock predictor
  r2_residual          fraction of the variance left by the template that the model explains
  r2_residual_own      R2 between the residuals of data and prediction, each minus its own template
  r2_trial_model       median R2 within single presentations (onset to next onset), model
  r2_trial_template    the same for the template
  r2_shuffle           model R2 when each test presentation's spikes are swapped with those of
                       another test presentation of the same condition (stimulus alignment kept,
                       single-trial pairing broken); one row per shuffle
  r2_residual_shuffle  r2_residual of those predictions
  r2_shift             model R2 with the spikes shifted 30-120 s back in time; one row per shift
  r2_residual_shift    r2_residual of those predictions

With --retrain-targets (the default: broadband and bands up to 25 Hz), models are also trained
on this session's split (fit_and_score) for every channel of those targets:
  r2_retrained_residual     a model trained to predict the residual y - T (R2 on the residual)
  r2_spikes_plus_template   a model given the template as an extra input next to the spikes

usage (from the repo root):
    python -m experiments.stimulus_clock <session_dir> [--n-shuffles 5] [--n-shifts 5]
                                         [--retrain-targets 0,1,2,3,4 | none]

Needs the 'train' stage. Saves variables/<session_id>/stimulus_clock.npz.
"""
import argparse

import numpy as np
import torch

import config
import paths
from analysis.stimulus_clock import (apply_template, bin_conditions, fit_template, onset_bins,
                                     per_presentation_r2, presentation_conditions, presentation_segments,
                                     r2, residual_r2, shuffle_presentations)
from preprocess_data import take_chunks
from run_session import SessionPipeline, save_npz
from splits import chunk_bins

MAX_LAG = int(round(config.CLOCK_MAX_LAG / config.BIN_SIZE))      # bins after an onset that get a template
SHIFT_RANGE = (30, 120)                                            # s, time-shift null
DEFAULT_RETRAIN = [0] + [i + 1 for i, (_, high) in enumerate(config.BANDS) if high <= 25]


def contiguous(bins):
    if len(bins) and not np.array_equal(bins, np.arange(bins[0], bins[0] + len(bins))):
        raise ValueError('expected a contiguous test block (splits.holdout_split)')
    return bins


def predict(model, X, split_chunks):
    """Model prediction on the given chunks of X (bins x inputs), chunk by chunk as in training."""
    return model.evaluate(take_chunks(X, split_chunks)).numpy()


def clock(pipeline):
    """
    The stimulus template and the test-block presentations: a dict with X, Y (all active bins), T (template,
    bins x channels x targets), cond, lag, names, train_bins, test_bins, segments (test-block presentations,
    relative to its start) and seg_cond (their conditions).
    """
    X, Y = pipeline.active_spikes.spkMat, pipeline.active_lfp.lfpMat          # bins x units, bins x channels x targets
    n_bins = X.shape[0]
    split = pipeline.split
    train_bins, test_bins = chunk_bins(split['train']), contiguous(chunk_bins(split['test']))

    stim = pipeline.session.stimulus_presentation
    stim = stim[stim.active].sort_values('start_time')
    onsets = onset_bins(stim.start_time.to_numpy(), pipeline.active_counts.timestamps, config.BIN_SIZE)
    labels = presentation_conditions(stim)
    inside = (onsets >= 0) & (onsets < n_bins)
    onsets, labels = onsets[inside], labels[inside]
    cond, lag, names = bin_conditions(onsets, labels, n_bins, MAX_LAG)
    table, fallback = fit_template(Y, cond, lag, train_bins, len(names), MAX_LAG)
    T = apply_template(table, fallback, cond, lag).astype(np.float32)        # bins x channels x targets

    # test-block presentations (positions relative to the start of the test block)
    t0 = test_bins[0]
    segments = presentation_segments(onsets, n_bins, MAX_LAG // 2, MAX_LAG)
    in_test = (segments[:, 0] >= t0) & (segments[:, 1] <= test_bins[-1] + 1)
    seg_cond = labels[np.searchsorted(onsets, segments[in_test, 0])]
    segments = segments[in_test] - t0
    return dict(X=X, Y=Y, T=T, cond=cond, lag=lag, names=names, train_bins=train_bins, test_bins=test_bins,
                segments=segments, seg_cond=seg_cond)


def run(pipeline, n_shuffles=5, n_shifts=5, retrain_targets=DEFAULT_RETRAIN, seed=config.TORCH_SEED):
    rng = np.random.default_rng(seed)
    c = clock(pipeline)
    X, Y, T, cond, lag, names = c['X'], c['Y'], c['T'], c['cond'], c['lag'], c['names']
    train_bins, test_bins, segments, seg_cond = c['train_bins'], c['test_bins'], c['segments'], c['seg_cond']
    n_bins, n_ch = X.shape[0], Y.shape[1]
    split = pipeline.split
    Yt, Tt = Y[test_bins], T[test_bins]
    shifts = rng.integers(*(np.array(SHIFT_RANGE) / config.BIN_SIZE).astype(int), size=n_shifts)

    shape = (n_ch, config.N_TARGETS)
    out = {k: np.full(shape, np.nan) for k in ('r2_model', 'r2_template', 'r2_residual', 'r2_residual_own',
                                                'r2_trial_model', 'r2_trial_template',
                                                'r2_retrained_residual', 'r2_spikes_plus_template')}
    for k, n in (('shuffle', n_shuffles), ('shift', n_shifts)):
        out[f'r2_{k}'] = np.full((n,) + shape, np.nan)
        out[f'r2_residual_{k}'] = np.full((n,) + shape, np.nan)

    predictions = {}
    for (chan, band), model in pipeline.models.items():
        y, tmpl = Yt[:, chan, band], Tt[:, chan, band]
        yhat = predict(model, X, split['test'])
        own_table, own_fallback = fit_template(predict(model, X, split['train']), cond[train_bins], lag[train_bins],
                                               np.arange(len(train_bins)), len(names), MAX_LAG)
        own = apply_template(own_table, own_fallback, cond[test_bins], lag[test_bins])
        out['r2_model'][chan, band] = r2(y, yhat)
        out['r2_template'][chan, band] = r2(y, tmpl)
        out['r2_residual'][chan, band] = residual_r2(y, yhat, tmpl)
        out['r2_residual_own'][chan, band] = r2(y - tmpl, yhat - own)
        out['r2_trial_model'][chan, band] = np.nanmedian(per_presentation_r2(y, yhat, segments))
        out['r2_trial_template'][chan, band] = np.nanmedian(per_presentation_r2(y, tmpl, segments))
        predictions[chan, band] = yhat

    # nulls: the same models on test-block spikes with the single-trial pairing broken
    Xt = X[test_bins]
    n_test_chunks = len(split['test'])
    for i in range(n_shuffles):
        Xs = shuffle_presentations(Xt, segments, seg_cond, rng)
        for (chan, band), model in pipeline.models.items():
            yhat = predict(model, Xs, np.arange(n_test_chunks))
            out['r2_shuffle'][i, chan, band] = r2(Yt[:, chan, band], yhat)
            out['r2_residual_shuffle'][i, chan, band] = residual_r2(Yt[:, chan, band], yhat, Tt[:, chan, band])
    for i, shift in enumerate(shifts):
        Xs = X[(test_bins - shift) % n_bins]
        for (chan, band), model in pipeline.models.items():
            yhat = predict(model, Xs, np.arange(n_test_chunks))
            out['r2_shift'][i, chan, band] = r2(Yt[:, chan, band], yhat)
            out['r2_residual_shift'][i, chan, band] = residual_r2(Yt[:, chan, band], yhat, Tt[:, chan, band])

    # retrained: predict the residual; spikes plus the template as input
    for band in retrain_targets or []:
        for chan in range(n_ch):
            y, tmpl = Y[:, chan, band], T[:, chan, band]
            _, out['r2_retrained_residual'][chan, band] = pipeline.fit_and_score(X, y - tmpl)
            _, out['r2_spikes_plus_template'][chan, band] = pipeline.fit_and_score(np.hstack([X, tmpl[:, None]]), y)
            print(f'retrained chan {chan} band {band}: residual R2 {out["r2_retrained_residual"][chan, band]:.3f}, '
                  f'spikes + template R2 {out["r2_spikes_plus_template"][chan, band]:.3f}', flush=True)

    out.update(conditions=np.array(names), max_lag=np.array(MAX_LAG), shifts_s=shifts * config.BIN_SIZE,
               n_test_presentations=np.array(len(segments)),
               template_fraction_of_bins=np.array(np.mean(cond[test_bins] >= 0)))
    return out


def parse_targets(text):
    return [] if text == 'none' else [int(t) for t in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--n-shuffles', type=int, default=5)
    parser.add_argument('--n-shifts', type=int, default=5)
    parser.add_argument('--retrain-targets', default=','.join(map(str, DEFAULT_RETRAIN)),
                        help="model targets to retrain on the residual and with the template input, or 'none'")
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    out = run(pipeline, args.n_shuffles, args.n_shifts, parse_targets(args.retrain_targets))
    f = pipeline.dirs.variables / 'stimulus_clock.npz'
    save_npz(f, **out)
    print('Saved', f)


if __name__ == '__main__':
    main()
