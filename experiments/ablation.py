"""
Ablation: how much does test R2 drop when units are removed from the model input?

For one channel and target, retrain the standard LSTM after removing the top X% of units ranked
by (a) firing rate, (b) mean |attribution|, and (c) at random (config-seeded permutations).

usage (from the repo root):
    python -m experiments.ablation <session_dir> [--channel 5] [--band 0] [--n-perm 20]

Needs the 'train' and 'attr' stages of run_session.py (attributions of the 'active' window).
Saves to variables/<session_id>/ (steps x 1, or steps x permutations):
    r2_scores_fr.npy, r2_scores_attr.npy, r2_scores_rand.npy, r2_scores_ablation_steps.npy (% removed),
    r2_scores_ablation_full.npy (the same model trained on all units, i.e. 0% removed)
"""
import argparse

import numpy as np
import torch

import config
import models
import paths
from preprocess_data import chunk_and_reshape, get_data_loaders
from run_session import SessionPipeline, load_sparse, r2

PERCENT_REMOVED = np.arange(5, 100, 5)


def test_r2(spikes, lfp, keep, device):
    """Train the standard LSTM on the kept units only and return its test R2."""
    X_train, X_test, y_train, y_test = chunk_and_reshape(spikes[:, keep], lfp, config.SEQLENGTH)
    train_dl, test_dl = get_data_loaders(X_train, X_test, y_train, y_test)
    model = models.build_lstm(len(keep), device)
    model.train(train_dl, test_dl, config.NUM_EPOCHS)
    return r2(model, test_dl)


def run_ablation(pipeline, chan, band, n_perm):
    spikes = pipeline.active_spikes.spkMat
    lfp = pipeline.active_lfp.lfpMat[:, chan, band]
    n_units = spikes.shape[1]
    torch.manual_seed(config.TORCH_SEED)
    rng = np.random.default_rng(config.TORCH_SEED)

    # unit orders, most important first
    by_rate = np.argsort(-pipeline.session.units['firing_rate'].to_numpy(), kind='stable')
    attrs = load_sparse(paths.attribution_file(pipeline.dirs, 'active', chan, band))
    by_attr = np.argsort(-np.mean(np.abs(attrs), axis=0), kind='stable')   # mean |attribution|, so signs don't cancel

    r2_full = test_r2(spikes, lfp, np.arange(n_units), pipeline.device)
    r2_rate, r2_attr = np.zeros((len(PERCENT_REMOVED), 1)), np.zeros((len(PERCENT_REMOVED), 1))
    r2_rand = np.zeros((len(PERCENT_REMOVED), n_perm))
    for i, percent in enumerate(PERCENT_REMOVED):
        n_removed = int(percent / 100 * n_units)
        print(f'{percent}% removed ({n_removed} units)', flush=True)
        r2_rate[i, 0] = test_r2(spikes, lfp, np.sort(by_rate[n_removed:]), pipeline.device)
        r2_attr[i, 0] = test_r2(spikes, lfp, np.sort(by_attr[n_removed:]), pipeline.device)
        for perm in range(n_perm):
            r2_rand[i, perm] = test_r2(spikes, lfp, np.sort(rng.permutation(n_units)[n_removed:]), pipeline.device)
    return r2_full, r2_rate, r2_attr, r2_rand


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir')
    parser.add_argument('--channel', type=int, default=5)
    parser.add_argument('--band', type=int, default=0, help='model target; 0 = broadband')
    parser.add_argument('--n-perm', type=int, default=20, help='random removals per step')
    args = parser.parse_args()

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = SessionPipeline(data_root, session_id, device)
    r2_full, r2_rate, r2_attr, r2_rand = run_ablation(pipeline, args.channel, args.band, args.n_perm)

    out = pipeline.dirs.variables
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'r2_scores_fr.npy', r2_rate)
    np.save(out / 'r2_scores_attr.npy', r2_attr)
    np.save(out / 'r2_scores_rand.npy', r2_rand)
    np.save(out / 'r2_scores_ablation_steps.npy', PERCENT_REMOVED)
    np.save(out / 'r2_scores_ablation_full.npy', r2_full)
    print('Saved ablation R2 scores to', out)


if __name__ == '__main__':
    main()
