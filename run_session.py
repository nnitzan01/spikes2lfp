"""
Per-session pipeline: every result that is computed for one session and saved to disk.

usage:
    python run_session.py <data_root>/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>
                          [--stages train,plots,...] [--retrain] [--seeds N] [--ig-batch N] [--device cuda|cpu]

Outputs go to <data_root>/spikes2lfp_v2/<kind>/<session_id>/ (see paths.py). Stages run in the order
below. Each one skips outputs that already exist, so an interrupted job can simply be resubmitted,
and later stages load what earlier ones saved.

Data handling: the active epoch is cut into 3 s chunks and split in time (splits.holdout_split):
train | buffer | validation | buffer | test, with the test set the last 20%. Spike counts are
min-max scaled per unit and targets z-scored per channel and target, both with statistics from the
training chunks only (variables/spike_scaler.npz, target_scaler.npz, split.npz, split.json).

  train      one causal CNN-LSTM per VISp channel x model target x seed, trained on the training
             chunks; the epoch with the lowest validation loss is kept (models are loaded instead
             when all files exist, unless --retrain)
             models/     lstm_model_chan{c}_band{b}_seed{s}.pt
             variables/  r2_scores.npy (test R2 of seed 0, channels x targets), r2_scores_seeds.npy,
                         r2_scores_{val,train}.npy (seeds x channels x targets), losses_{train,val}.npy,
                         best_epoch.npy, model_info.json, channels.csv
  transfer   R2 of the active-epoch models on the passive replay and spontaneous epochs
             variables/  r2_scores_passive.npy, r2_scores_spont.npy
  plots      diagnostic figures of the models on the active epoch -> plots/
  variables  model input and metadata -> variables/: spikes_sparse.npz (scaled active-epoch
             spikes), units.csv, timestamps.npy, stimulus_presentation_active.csv
  attr       history-aware attribution (integrated_gradients.SequenceIG) of every model target:
             each target bin's prediction is attributed to every unit at every lag of the preceding
             3 s. Targets (the same for all models; variables/attr_targets.npz): random bins of the
             training block, the test block and the spontaneous epoch; the four CSD-component
             windows after ATTR_EVENTS image onsets; a peri-stimulus grid for PERI_EVENTS onsets.
             attrs/  seqig_chan{c}_band{b}.npz (per-target unit totals, lag kernels, peri maps,
                     f(x), f(b) and completeness error of every target)
  summaries  per-unit summaries (mean, mean |.|, per spike, share; channels x targets x units)
             attrs/  unit_attribution_{train,test,active,spont}.npz, attribution_completeness.npy
  peri       broadband peri-stimulus ranks and top-K Jaccard stability -> variables/: psth.npy,
             peri_times.npy, rank_snippets.npy, jaccard_{raw,ref}_snippets.npy
  ipd        regional instantaneous predictive dominance -> variables/regional_dominance_results.npz
"""
import argparse
import json
import os
import time
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

import config
import models
import paths
import plots
from analysis.peri_events import (event_start_indices, extract_snippets, proportional_influence,
                                  regional_dominance, snippet_time_axis)
from analysis.ranking import (baseline_reference_set, jaccard_consecutive, jaccard_to_reference,
                              rank_active_neurons, top_k_mask)
from analysis.sequence_attribution import (attribute_targets, coarsen_lags, relative_completeness_error,
                                           unit_summaries, valid_ends, window_spike_counts)
from integrated_gradients import SequenceIG
from preprocess_data import SpikeScaler, Standardizer, pre_process_lfp, pre_process_spikes, take_chunks
from splits import chunk_bins, describe, holdout_split

STAGES = ['train', 'transfer', 'plots', 'variables', 'attr', 'summaries', 'peri', 'ipd']

def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def model_seed(seed, chan, band):
    """Seed of one model (weights and batch order), independent of the order in which models are trained."""
    return config.TORCH_SEED + 10_000 * seed + 100 * chan + band


@contextmanager
def atomic(path):
    """
    Yield a temporary file next to `path`, moved into place only when the block completes. An
    interrupted job therefore never leaves a partial file that a resubmitted job would skip as done.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.part')
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_npy(path, array):
    with atomic(path) as tmp, open(tmp, 'wb') as fh:
        np.save(fh, array)


def save_npz(path, compressed=False, **arrays):
    with atomic(path) as tmp, open(tmp, 'wb') as fh:
        (np.savez_compressed if compressed else np.savez)(fh, **arrays)


def save_text(path, text):
    with atomic(path) as tmp:
        tmp.write_text(text)


def save_csv(frame, path):
    with atomic(path) as tmp:
        frame.to_csv(tmp, index=False)


def save_sparse(path, matrix):
    with atomic(path) as tmp, open(tmp, 'wb') as fh:
        sparse.save_npz(fh, sparse.coo_matrix(matrix), compressed=True)


def make_loader(X, y, shuffle):
    """DataLoader over (chunks x seqlength x units) inputs and (chunks x seqlength x 1) targets."""
    return DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
                      batch_size=config.BATCH_SIZE, shuffle=shuffle)


def r2(model, dataloader):
    """R2 over all bins of all sequences in the dataloader."""
    X, y = dataloader.dataset.tensors
    return r2_score(y.numpy().reshape(-1, 1), model.evaluate(X).numpy())


class SessionPipeline:
    def __init__(self, data_root, session_id, device, retrain=False, ig_batch=config.IG_BATCH_WINDOWS,
                 n_seeds=config.N_SEEDS):
        self.data_root = data_root
        self.n_seeds = n_seeds
        self.session_id = session_id
        self.device = device
        self.retrain = retrain
        self.ig_batch = ig_batch
        self.dirs = paths.session_output_dirs(data_root, session_id)

    # ---- inputs, loaded on first use --------------------------------------------------------

    @cached_property
    def session(self):
        from process_session import session    # imported here because it pulls in allensdk
        units_table = pd.read_csv(paths.REPO_DIR / config.UNITS_TABLE)
        return session(self.session_id, units_table, self.data_root)

    def spike_counts(self, start, stop):
        """Binned spike counts (bins x units) from start to stop (s), truncated to whole sequences."""
        sp = pre_process_spikes(self.session.units, self.session.spike_times)
        sp.getSpkMat(start, stop)
        sp.truncate(config.SEQLENGTH)
        return sp

    def spikes(self, start, stop, scaled=True):
        """Binned spikes from start to stop (s); scaled with the training-data SpikeScaler (model input) if scaled."""
        sp = self.spike_counts(start, stop)
        if scaled:
            sp.spkMat = self.spike_scaler.transform(sp.spkMat)
        return sp

    @cached_property
    def active_counts(self):
        return self.spike_counts(*self.session.active_times)

    @cached_property
    def split(self):
        """Blocked split of the active epoch's 3 s chunks (splits.holdout_split), saved on first use."""
        f = self.dirs.variables / 'split.npz'
        if f.exists():
            return dict(np.load(f))
        split = holdout_split(self.active_counts.spkMat.shape[0] // config.SEQLENGTH)
        f.parent.mkdir(parents=True, exist_ok=True)
        save_npz(f, **split)
        save_text(f.parent / 'split.json', json.dumps(describe(split), indent=2))
        return split

    @cached_property
    def spike_scaler(self):
        """Per-unit min-max scaling fitted on the training chunks of the active epoch, saved on first use."""
        f = self.dirs.variables / 'spike_scaler.npz'
        if f.exists():
            return SpikeScaler.from_state(np.load(f))
        scaler = SpikeScaler().fit(self.active_counts.spkMat[chunk_bins(self.split['train'])])
        save_npz(f, **scaler.state())
        return scaler

    @cached_property
    def active_spikes(self):
        sp = self.spike_counts(*self.session.active_times)
        sp.spkMat = self.spike_scaler.transform(sp.spkMat)
        return sp

    def lfp_targets(self, start, stop, n_bins):
        """
        Model targets from start to stop (s): (bins x VISp channels x targets), raw (broadband LFP and
        band power), anti-alias decimated to the spike bin rate and trimmed to n_bins.
        """
        lfp = pre_process_lfp(self.session_id, self.session.channels, start, stop, self.data_root)
        lfp.filter_lfp(take_power=True, zscore_targets=False, downsample=config.LFP_DOWNSAMPLE)
        lfp.align_lfp(n_bins)
        return lfp

    @cached_property
    def target_scaler(self):
        """z-scoring of every channel x target, fitted on the training chunks of the active epoch."""
        f = self.dirs.variables / 'target_scaler.npz'
        if f.exists():
            return Standardizer.from_state(np.load(f))
        raw = self.lfp_targets(*self.session.active_times, self.active_spikes.spkMat.shape[0]).lfpMat
        scaler = Standardizer().fit(raw[chunk_bins(self.split['train'])])
        save_npz(f, **scaler.state())
        return scaler

    @cached_property
    def active_lfp(self):
        """Standardised model targets for the active epoch (bins x VISp channels x targets), aligned with active_spikes."""
        lfp = self.lfp_targets(*self.session.active_times, self.active_spikes.spkMat.shape[0])
        lfp.lfpMat = self.target_scaler.transform(lfp.lfpMat)
        return lfp

    @property
    def n_channels(self):
        saved = self.dirs.variables / 'channels.csv'     # written by the train stage
        return len(pd.read_csv(saved)) if saved.exists() else len(self.active_lfp.channels)

    @cached_property
    def unit_areas(self):
        saved = self.dirs.variables / 'units.csv'        # written by the variables stage
        units = pd.read_csv(saved) if saved.exists() else self.session.units
        return units['structure_acronym'].values

    def load_models(self, seed=0):
        """Trained models {(channel, target): process_model} of one seed, loaded from disk."""
        loaded = {}
        for chan in range(self.n_channels):
            for band in range(config.N_TARGETS):
                f = paths.model_file(self.dirs, chan, band, seed)
                if not f.exists():
                    raise FileNotFoundError(f'{f} not found; run the train stage first')
                model = models.build_lstm(len(self.session.units), self.device)
                model.load_model(f)
                loaded[chan, band] = model
        return loaded

    @cached_property
    def models(self):
        """Seed-0 models, used by the attribution stages."""
        return self.load_models(seed=0)

    def fit_and_score(self, X, y, seed=0):
        """
        Train the standard model on inputs X (bins x features) for one target y (bins,) with this
        session's blocked split and validation-based epoch selection. Returns (model, test R2).
        Used by the experiments that retrain on modified inputs.
        """
        torch.manual_seed(config.TORCH_SEED + seed)
        loaders = {part: make_loader(take_chunks(X, self.split[part]), take_chunks(y[:, None], self.split[part]),
                                     shuffle=(part == 'train')) for part in ('train', 'val', 'test')}
        model = models.build_lstm(X.shape[1], self.device)
        model.train(loaders['train'], loaders['val'], config.NUM_EPOCHS, select_best=True, patience=config.PATIENCE)
        return model, r2(model, loaders['test'])

    # ---- stages ------------------------------------------------------------------------------

    def train(self):
        """
        Train (or load) one model per channel x target x seed on the training chunks, keeping the
        epoch with the lowest validation loss, and score train / validation / test R2.

        Each model has its own seed (model_seed) and is saved as soon as it is trained, so a job
        killed part-way can be resubmitted: existing models are loaded and only missing ones are
        trained, with the same result as an uninterrupted run.
        """
        spikes, lfp, split = self.active_spikes.spkMat, self.active_lfp.lfpMat, self.split
        n_channels, input_size = lfp.shape[1], spikes.shape[1]
        save_csv(self.active_lfp.channels, self.dirs.variables / 'channels.csv')
        X = {part: take_chunks(spikes, split[part]) for part in ('train', 'val', 'test')}

        shape = (self.n_seeds, n_channels, config.N_TARGETS)
        r2_scores = {part: np.full(shape, np.nan) for part in ('train', 'val', 'test')}
        losses = {part: np.full(shape + (config.NUM_EPOCHS,), np.nan) for part in ('train', 'val')}
        best_epoch = np.full(shape, -1)
        for seed in range(self.n_seeds):
            fitted = {}
            for band in range(config.N_TARGETS):
                y = {part: take_chunks(lfp[:, :, band], split[part]) for part in ('train', 'val', 'test')}
                for chan in range(n_channels):
                    f = paths.model_file(self.dirs, chan, band, seed)
                    train_dl, val_dl, test_dl = (make_loader(X[part], y[part][..., chan:chan + 1], shuffle=(part == 'train'))
                                                 for part in ('train', 'val', 'test'))
                    if f.exists() and not self.retrain:
                        model = models.build_lstm(input_size, self.device)
                        history = model.load_model(f)
                    else:
                        torch.manual_seed(model_seed(seed, chan, band))    # before building: covers weight init too
                        model = models.build_lstm(input_size, self.device)
                        t0 = time.time()
                        loss_tr, loss_val = model.train(train_dl, val_dl, config.NUM_EPOCHS,
                                                        select_best=True, patience=config.PATIENCE)
                        # plain Python numbers, so the file loads with torch.load(weights_only=True)
                        history = dict(train_losses=loss_tr.tolist(), val_losses=loss_val.tolist(),
                                       best_epoch=int(model.best_epoch), seconds=float(time.time() - t0))
                        with atomic(f) as tmp:
                            model.save_model(tmp, **history)
                    if 'train_losses' in history:
                        losses['train'][seed, chan, band] = history['train_losses']
                        losses['val'][seed, chan, band] = history['val_losses']
                        best_epoch[seed, chan, band] = history['best_epoch']
                    for part, dl in (('train', train_dl), ('val', val_dl), ('test', test_dl)):
                        r2_scores[part][seed, chan, band] = r2(model, dl)
                    if 'seconds' in history:
                        log(f'seed {seed} chan {chan} band {band}: test R2 {r2_scores["test"][seed, chan, band]:.3f}, '
                            f'best epoch {history["best_epoch"]}, {history["seconds"]:.0f} s')
                    fitted[chan, band] = model
            if seed == 0:
                self.__dict__['models'] = fitted     # reuse in later stages instead of reloading

        out = self.dirs.variables
        save_npy(out / 'r2_scores.npy', r2_scores['test'][0])          # seed 0: channels x targets
        save_npy(out / 'r2_scores_seeds.npy', r2_scores['test'])       # seeds x channels x targets
        save_npy(out / 'r2_scores_val.npy', r2_scores['val'])
        save_npy(out / 'r2_scores_train.npy', r2_scores['train'])
        save_npy(out / 'losses_train.npy', losses['train'])            # seeds x channels x targets x epochs
        save_npy(out / 'losses_val.npy', losses['val'])
        save_npy(out / 'best_epoch.npy', best_epoch)
        save_text(out / 'model_info.json', json.dumps(dict(
            architecture=config.LSTM_KWARGS, n_parameters=models.build_lstm(input_size, 'cpu').n_parameters(),
            n_units=int(input_size), seeds=self.n_seeds, max_epochs=config.NUM_EPOCHS, patience=config.PATIENCE,
            batch_size=config.BATCH_SIZE), indent=2))

    def transfer(self):
        """
        Within-session transfer: R2 of the seed-0 active-epoch models on the passive replay and
        spontaneous epochs, with inputs and targets scaled by the active training-data scalers.
        """
        for epoch, name in (('passive', 'r2_scores_passive.npy'), ('spontaneous', 'r2_scores_spont.npy')):
            if (self.dirs.variables / name).exists():
                continue
            start, stop = {'passive': self.session.passive_times, 'spontaneous': self.session.spontaneous_times}[epoch]
            spikes = self.spikes(start, stop).spkMat
            lfp = self.target_scaler.transform(self.lfp_targets(start, stop, spikes.shape[0]).lfpMat)
            scores = np.full((self.n_channels, config.N_TARGETS), np.nan)
            for (chan, band), model in self.models.items():
                prediction = model.evaluate(spikes).numpy()
                scores[chan, band] = r2_score(lfp[:len(prediction), chan, band], prediction)
            save_npy(self.dirs.variables / name, scores)

    def plots(self):
        channels, lfp = self.active_lfp.channels, self.active_lfp.lfpMat
        yhat = np.zeros_like(lfp)
        for (chan, band), model in self.models.items():
            yhat[:, chan, band] = model.evaluate(self.active_spikes.spkMat).numpy()
        kw = dict(show_plot=False, save_fig=True, output_dir=self.dirs.plots, session_id=self.session_id)

        r2_test = np.load(self.dirs.variables / 'r2_scores.npy')
        plots.plot_r2(r2_test, channels, config.BANDS, fig_name='r2_scores.png', **kw)
        losses_file = self.dirs.variables / 'losses_val.npy'
        if losses_file.exists():
            val_losses = np.moveaxis(np.load(losses_file)[0], 2, 1)   # seed 0: channels x epochs x targets
            for band in range(config.N_TARGETS):
                plots.plot_all_channel_loss(channels, config.BANDS, band, val_losses,
                                            fig_name=f'all_channel_loss_band{band}.png', **kw)
        for chan in range(len(channels)):
            plots.plot_lfp_prediction(lfp, yhat, chan, config.BANDS, start_time=20, end_time=30,
                                      fig_name=f'lfp_prediction_chan{chan}.png', **kw)
            plots.plot_abs_error_change(lfp, yhat, self.session, self.active_spikes.timestamps, chan, config.BANDS,
                                        fig_name=f'abs_error_change_chan{chan}.png', **kw)
            plots.plot_psd(lfp, yhat, chan, fig_name=f'psd_chan{chan}.png', **kw)

    def variables(self):
        out = self.dirs.variables
        save_sparse(out / 'spikes_sparse.npz', self.active_spikes.spkMat)
        save_csv(self.session.units, out / 'units.csv')
        save_npy(out / 'timestamps.npy', self.active_spikes.timestamps)
        save_csv(self.session.active, out / 'stimulus_presentation_active.csv')

    @cached_property
    def attr_targets(self):
        """
        The attribution targets shared by every model of the session (saved to
        variables/attr_targets.npz on first use, so all channels and targets use the same ones):

          sampled_{train,test,spont}   random target bins in the training block, the test block
                                       and the spontaneous epoch (single-bin targets)
          events                       image onsets (s) of ATTR_EVENTS random active, non-omitted
                                       presentations, with their split partition, used for the
                                       CSD-component targets; the first PERI_EVENTS also get
                                       peri-stimulus maps
          *_spikes                     spike counts of every unit in each target's window
        """
        f = self.dirs.variables / 'attr_targets.npz'
        if f.exists():
            return dict(np.load(f))
        L, rng = config.ATTR_HISTORY, np.random.default_rng(config.TORCH_SEED)
        counts = self.active_counts.spkMat
        n_bins = counts.shape[0]

        def sample(candidates, n):
            candidates = valid_ends(candidates, n_bins, L)
            return np.sort(rng.choice(candidates, min(n, len(candidates)), replace=False))

        t = {'sampled_train': sample(chunk_bins(self.split['train']), config.ATTR_SAMPLED),
             'sampled_test': sample(chunk_bins(self.split['test']), config.ATTR_SAMPLED)}
        spont = self.spike_counts(*self.session.spontaneous_times).spkMat
        spont_ends = valid_ends(np.arange(spont.shape[0]), spont.shape[0], L)
        t['sampled_spont'] = np.sort(rng.choice(spont_ends, min(config.ATTR_SAMPLED, len(spont_ends)), replace=False))

        # image onsets whose component and peri-stimulus windows all fit inside the active epoch
        stim = self.session.stimulus_presentation
        onsets = stim.start_time[stim.active & (stim.image_name != 'omitted')].values
        onset_bins = np.round((onsets - self.active_counts.timestamps[0]) / config.BIN_SIZE).astype(int)
        first = onset_bins + int(np.round(config.STIM_WINDOW[0] / config.BIN_SIZE))
        last = onset_bins + max(self.component_end_offsets().max(), self.peri_offsets().max())
        ok = (first - L + 1 >= 0) & (last < n_bins)
        pick = np.sort(rng.choice(np.flatnonzero(ok), min(config.ATTR_EVENTS, ok.sum()), replace=False))
        rng.shuffle(pick)                               # random order: the first PERI_EVENTS are a random subset
        t['event_onsets'], t['event_onset_bins'] = onsets[pick], onset_bins[pick]
        chunk_of = t['event_onset_bins'] // config.SEQLENGTH
        t['event_partition'] = np.select([np.isin(chunk_of, self.split[p]) for p in ('train', 'val', 'test')],
                                         ['train', 'val', 'test'], 'buffer')
        t['peri_times'] = self.peri_offsets() * config.BIN_SIZE

        for name in ('sampled_train', 'sampled_test'):
            t[name + '_spikes'] = window_spike_counts(counts, t[name], L)
        t['sampled_spont_spikes'] = window_spike_counts(spont, t['sampled_spont'], L)
        f.parent.mkdir(parents=True, exist_ok=True)
        save_npz(f, **t)
        return t

    @staticmethod
    def component_end_offsets():
        """Bin offset from image onset to the last bin of each component target window."""
        half = (config.COMPONENT_TARGET_BINS - 1) // 2
        return np.array([int(np.round(tc / config.BIN_SIZE)) + half for tc in config.COMPONENT_TIMES.values()])

    @staticmethod
    def peri_offsets():
        """Bin offsets from image onset of the peri-stimulus targets."""
        first, last = (int(np.round(x / config.BIN_SIZE)) for x in config.STIM_WINDOW)
        return np.arange(first, last, config.PERI_STEP_BINS)

    def ig_baseline(self, n_units):
        """IG baseline per config.IG_BASELINE: 0 (no spikes) or each unit's mean scaled input in the training chunks."""
        if config.IG_BASELINE == 'zero':
            return 0.0
        if config.IG_BASELINE == 'mean':
            mean = self.active_spikes.spkMat[chunk_bins(self.split['train'])].mean(axis=0)
            return torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        raise ValueError(f'unknown IG_BASELINE {config.IG_BASELINE!r}')

    def attr(self):
        """History-aware attribution (SequenceIG) of every model target in config.ATTR_TARGETS."""
        targets = range(config.N_TARGETS) if config.ATTR_TARGETS is None else config.ATTR_TARGETS
        todo = [(c, b) for c in range(self.n_channels) for b in targets
                if not paths.attribution_file(self.dirs, c, b).exists()]
        if not todo:
            print('Attributions already saved', flush=True)
            return
        t, L = self.attr_targets, config.ATTR_HISTORY
        X = torch.as_tensor(self.active_spikes.spkMat, device=self.device)
        X_spont = torch.as_tensor(self.spikes(*self.session.spontaneous_times).spkMat, device=self.device)
        baseline = self.ig_baseline(X.shape[1])
        comp_ends = t['event_onset_bins'][:, None] + self.component_end_offsets()[None, :]     # events x components
        peri_on = t['event_onset_bins'][:config.PERI_EVENTS]
        peri_ends = peri_on[:, None] + self.peri_offsets()[None, :]                             # events x times
        run = dict(history=L, baseline=baseline, batch_windows=self.ig_batch)
        log(f'Attributing {len(todo)} models')
        for chan, band in todo:
            t0 = time.time()
            ig = SequenceIG(self.models[chan, band].model.to(self.device), n_steps=config.IG_N_STEPS)
            out = {}
            sampled = {}
            for name, data in (('sampled_train', X), ('sampled_test', X), ('sampled_spont', X_spont)):
                sampled[name] = attribute_targets(ig, data, t[name], keep_kernel=(name != 'sampled_spont'), **run)
            comp = attribute_targets(ig, X, comp_ends.ravel(), target_bins=config.COMPONENT_TARGET_BINS,
                                     keep_kernel=True, **run)
            peri = attribute_targets(ig, X, peri_ends.ravel(), **run)
            for name, res in sampled.items():
                out[name + '_total'] = res['total'].astype(np.float16)
            n_ev, n_comp = comp_ends.shape
            out['components_total'] = comp['total'].reshape(n_ev, n_comp, -1).astype(np.float16)
            peri_total = peri['total'].reshape(len(peri_on), peri_ends.shape[1], -1)
            out['peri_mean'], out['peri_abs_mean'] = peri_total.mean(axis=0), np.abs(peri_total).mean(axis=0)
            if band in config.PERI_PER_EVENT_TARGETS:
                out['peri_total'] = peri_total.astype(np.float16)
            n_kernel = len(t['sampled_train']) + len(t['sampled_test'])
            k = config.LAG_KERNEL_BIN
            out['kernel_mean'] = coarsen_lags(sampled['sampled_train']['kernel_sum'] + sampled['sampled_test']['kernel_sum'], k) / n_kernel
            out['kernel_abs_mean'] = coarsen_lags(sampled['sampled_train']['kernel_abs_sum'] + sampled['sampled_test']['kernel_abs_sum'], k) / n_kernel
            out['components_kernel_mean'] = coarsen_lags(comp['kernel_sum'], k) / len(comp['total'])
            for name, res in list(sampled.items()) + [('components', comp), ('peri', peri)]:
                for key in ('f_x', 'f_b', 'delta'):
                    out[f'{name}_{key}'] = res[key].astype(np.float32)
            save_npz(paths.attribution_file(self.dirs, chan, band), compressed=True, **out)
            err = relative_completeness_error(*(np.concatenate([out[f'{n}_{k}'] for n in ('sampled_train', 'sampled_test')])
                                                for k in ('f_x', 'f_b', 'delta')))
            n_windows = sum(len(t[n]) for n in sampled) + comp_ends.size + peri_ends.size
            log(f'attr chan {chan} band {band}: {n_windows} windows, {time.time() - t0:.0f} s, '
                f'median completeness error {err:.1e}')

    def summaries(self):
        """
        Per-unit attribution summaries (analysis/sequence_attribution.py) for each target set,
        as (channels x model targets x units) arrays, and the completeness error of every model.
          attrs/unit_attribution_{train,test,active,spont}.npz   keys: mean, mean_abs, per_spike, share
          attrs/attribution_completeness.npy                      channels x targets (median relative error)
        """
        t = self.attr_targets
        n_units = t['sampled_train_spikes'].shape[1]
        sets = {'train': ['sampled_train'], 'test': ['sampled_test'], 'active': ['sampled_train', 'sampled_test'],
                'spont': ['sampled_spont']}
        files = {s: self.dirs.attrs / f'unit_attribution_{s}.npz' for s in sets}
        if all(f.exists() for f in files.values()):
            return
        shape = (self.n_channels, config.N_TARGETS, n_units)
        result = {s: {k: np.full(shape, np.nan) for k in ('mean', 'mean_abs', 'per_spike', 'share')} for s in sets}
        completeness = np.full(shape[:2], np.nan)
        for chan in range(self.n_channels):
            for band in range(config.N_TARGETS):
                f = paths.attribution_file(self.dirs, chan, band)
                if not f.exists():
                    continue
                a = np.load(f)
                for s, names in sets.items():
                    totals = np.concatenate([a[n + '_total'] for n in names])
                    spikes = np.concatenate([t[n + '_spikes'] for n in names])
                    for k, v in unit_summaries(totals, spikes).items():
                        result[s][k][chan, band] = v
                completeness[chan, band] = relative_completeness_error(
                    *(np.concatenate([a[f'{n}_{key}'] for n in ('sampled_train', 'sampled_test', 'components')])
                      for key in ('f_x', 'f_b', 'delta')))
        for s, f in files.items():
            save_npz(f, **result[s])
        save_npy(self.dirs.attrs / 'attribution_completeness.npy', completeness)

    def peri(self):
        """
        Peri-stimulus analyses from the broadband per-event attribution maps (PERI_EVENTS images,
        targets every PERI_STEP_BINS bins over STIM_WINDOW), per channel:
          variables/  psth.npy (units x bins, all active non-omitted images), peri_times.npy,
                      rank_snippets.npy (events x times x units x channels; 1 = largest |attribution|),
                      jaccard_raw_snippets.npy (consecutive peri times), jaccard_ref_snippets.npy
                      (vs. the K units with the largest mean share over the pre-stimulus times)
        """
        out = self.dirs.variables
        if (out / 'jaccard_ref_snippets.npy').exists():
            return
        t = self.attr_targets
        stim = self.session.stimulus_presentation
        counts = self.active_counts
        tax = snippet_time_axis(config.STIM_WINDOW, config.BIN_SIZE)
        starts, _ = event_start_indices(counts.timestamps, stim.start_time[stim.active & (stim.image_name != 'omitted')].values,
                                        config.STIM_WINDOW, len(tax))
        save_npy(out / 'psth.npy', (extract_snippets(counts.spkMat, starts, len(tax)).mean(axis=0).T / config.BIN_SIZE).astype(np.float32))
        save_npy(out / 'peri_times.npy', t['peri_times'])

        n_base = int(np.sum(t['peri_times'] < config.STIM_BASELINE[1]))
        maps = [np.load(paths.attribution_file(self.dirs, c, 0))['peri_total'].astype(np.float32) for c in range(self.n_channels)]
        n_ev, n_t, n_u = maps[0].shape
        ranks = np.zeros((n_ev, n_t, n_u, self.n_channels), dtype=np.uint16)
        jac_raw = np.full((n_ev, n_t, self.n_channels), np.nan, dtype=np.float32)
        jac_ref = np.full((n_ev, n_t, self.n_channels), np.nan, dtype=np.float32)
        for ci, totals in enumerate(maps):
            flat = totals.reshape(n_ev * n_t, n_u)
            r = rank_active_neurons(flat)
            top = top_k_mask(r, config.TOP_K)
            reference = baseline_reference_set(flat, np.arange(n_ev) * n_t, (0, n_base), config.TOP_K)
            ranks[..., ci] = r.reshape(n_ev, n_t, n_u)
            jac_ref[..., ci] = jaccard_to_reference(top, reference).reshape(n_ev, n_t)
            for e in range(n_ev):                       # consecutive times within each event only
                jac_raw[e, :, ci] = jaccard_consecutive(top[e * n_t:(e + 1) * n_t])
        save_npy(out / 'rank_snippets.npy', ranks)
        save_npy(out / 'jaccard_raw_snippets.npy', jac_raw)
        save_npy(out / 'jaccard_ref_snippets.npy', jac_ref)

    def ipd(self):
        """Regional instantaneous predictive dominance from the peri-stimulus ranks -> variables/regional_dominance_results.npz."""
        out = self.dirs.variables
        if not (out / 'rank_snippets.npy').exists() or (out / 'regional_dominance_results.npz').exists():
            return
        ranks = np.load(out / 'rank_snippets.npy')
        per_channel = []
        for ci in range(ranks.shape[-1]):
            dominance, regions = regional_dominance(ranks[..., ci], ranks[..., ci] > 0, self.unit_areas)
            per_channel.append(dominance.mean(axis=0))                  # average over events
        average_dominance = np.stack(per_channel, axis=-1)                # regions x times x channels
        save_npz(out / 'regional_dominance_results.npz', compressed=True, average_dominance=average_dominance,
                            proportional_influence=proportional_influence(average_dominance),
                            region_names=regions, t=np.load(out / 'peri_times.npy'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir', help='.../visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>')
    parser.add_argument('--stages', default=','.join(STAGES), help=f'comma-separated subset of {STAGES}')
    parser.add_argument('--retrain', action='store_true', help='train even if saved models exist')
    parser.add_argument('--ig-batch', type=int, default=config.IG_BATCH_WINDOWS, help='windows attributed per batch')
    parser.add_argument('--seeds', type=int, default=config.N_SEEDS, help='independently initialised fits per model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    stages = args.stages.split(',')
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f'unknown stages: {sorted(unknown)}')

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    if os.environ.get('SLURM_CPUS_PER_TASK'):        # use the CPUs the cluster allocated, not every core on the node
        torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    gpu = torch.cuda.get_device_name(0) if torch.device(args.device).type == 'cuda' else 'none'
    log(f'Session {session_id}, data root {data_root}, device {args.device} (GPU: {gpu}), '
        f'{torch.get_num_threads()} CPU threads, outputs in {paths.OUTPUT_SUBDIR}/')
    pipeline = SessionPipeline(data_root, session_id, torch.device(args.device), args.retrain, args.ig_batch, args.seeds)
    for stage in STAGES:                    # always in pipeline order
        if stage in stages:
            log(f'--- {stage}')
            t0 = time.time()
            getattr(pipeline, stage)()
            log(f'--- {stage} done in {(time.time() - t0) / 60:.1f} min')
    log('Session complete.')


if __name__ == '__main__':
    main()
