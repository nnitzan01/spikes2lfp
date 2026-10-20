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
  attr       raw integrated-gradient attributions over each window in config.ATTR_WINDOWS
             attrs/<window>/attribution_raw_chan{c}_band{b}.npz (time bins x units)
  summaries  per-unit summaries (channels x targets x units) of the 'active' and 'spontaneous' windows
             attrs/  attribution_scores{,_spontaneous}_{mean,relative_mean,ipi_mean}.npy
  peri       snippets around image onsets, and around HVS peaks when HVS were detected, from
             the 'session' window broadband attributions -> variables/ (suffix _hvs for HVS):
             psth, rank_snippets, jaccard_{raw,ref}_snippets, active_mask_snippet_sparse +
             active_mask_shape, mean_raw_attribution_snippet, mean_raw_potency_snippet
  ipd        regional instantaneous predictive dominance from the peri snippets
             variables/  regional_dominance_results{,_hvs}.npz
"""
import argparse
import json
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, load_npz, save_npz
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

import config
import models
import paths
import plots
from analysis.attribution_summaries import ipi_mean_attribution, mean_attribution, relative_mean_attribution
from analysis.peri_events import (event_start_indices, extract_snippets, proportional_influence,
                                  regional_dominance, snippet_time_axis)
from analysis.ranking import (EPSILON, baseline_reference_set, jaccard_consecutive, jaccard_to_reference,
                              rank_active_neurons, top_k_mask)
from integrated_gradients import IntegratedGradient
from preprocess_data import SpikeScaler, Standardizer, pre_process_lfp, pre_process_spikes, take_chunks
from splits import chunk_bins, describe, holdout_split

STAGES = ['train', 'transfer', 'plots', 'variables', 'attr', 'summaries', 'peri', 'ipd']

# file-name prefix of the per-unit summaries of each summarised attribution window
SUMMARY_PREFIX = {'active': 'attribution_scores', 'spontaneous': 'attribution_scores_spontaneous'}
SUMMARY_KINDS = {'mean': mean_attribution, 'relative_mean': relative_mean_attribution,
                 'ipi_mean': ipi_mean_attribution}

# peri-event analyses: output-file suffix, snippet window and Jaccard baseline (s, relative to the event)
EVENT_TYPES = {
    'stim': dict(suffix='', window=config.STIM_WINDOW, baseline=config.STIM_BASELINE),
    'hvs': dict(suffix='_hvs', window=config.HVS_WINDOW, baseline=config.HVS_BASELINE),
}


def save_sparse(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, coo_matrix(matrix), compressed=True)


def load_sparse(path):
    return load_npz(path).toarray()


def make_loader(X, y, shuffle):
    """DataLoader over (chunks x seqlength x units) inputs and (chunks x seqlength x 1) targets."""
    return DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
                      batch_size=config.BATCH_SIZE, shuffle=shuffle)


def r2(model, dataloader):
    """R2 over all bins of all sequences in the dataloader."""
    X, y = dataloader.dataset.tensors
    return r2_score(y.numpy().reshape(-1, 1), model.evaluate(X).numpy())


class SessionPipeline:
    def __init__(self, data_root, session_id, device, retrain=False, ig_batch=config.IG_N_BATCH,
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

    def window_times(self, window):
        """[start, stop] in seconds of an attribution window from config.ATTR_WINDOWS."""
        spec = config.ATTR_WINDOWS[window]
        s = self.session
        start, stop = {'active': s.active_times, 'spontaneous': s.spontaneous_times,
                       'session': [s.active_times[0], s.passive_times[1]]}[spec['epoch']]
        if spec['duration'] is not None:
            stop = min(stop, start + spec['duration'])
        return start, stop

    def window_targets(self, window):
        targets = config.ATTR_WINDOWS[window]['targets']
        return range(config.N_TARGETS) if targets is None else targets

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
        np.savez(f, **split)
        (f.parent / 'split.json').write_text(json.dumps(describe(split), indent=2))
        return split

    @cached_property
    def spike_scaler(self):
        """Per-unit min-max scaling fitted on the training chunks of the active epoch, saved on first use."""
        f = self.dirs.variables / 'spike_scaler.npz'
        if f.exists():
            return SpikeScaler.from_state(np.load(f))
        scaler = SpikeScaler().fit(self.active_counts.spkMat[chunk_bins(self.split['train'])])
        np.savez(f, **scaler.state())
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
        np.savez(f, **scaler.state())
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
        """
        spikes, lfp, split = self.active_spikes.spkMat, self.active_lfp.lfpMat, self.split
        n_channels, input_size = lfp.shape[1], spikes.shape[1]
        self.dirs.models.mkdir(parents=True, exist_ok=True)
        self.active_lfp.channels.to_csv(self.dirs.variables / 'channels.csv', index=False)
        X = {part: take_chunks(spikes, split[part]) for part in ('train', 'val', 'test')}

        shape = (self.n_seeds, n_channels, config.N_TARGETS)
        r2_scores = {part: np.full(shape, np.nan) for part in ('train', 'val', 'test')}
        losses = {part: np.full(shape + (config.NUM_EPOCHS,), np.nan) for part in ('train', 'val')}
        best_epoch = np.full(shape, -1)
        for seed in range(self.n_seeds):
            files = {(c, b): paths.model_file(self.dirs, c, b, seed) for c in range(n_channels) for b in range(config.N_TARGETS)}
            train_now = self.retrain or not all(f.exists() for f in files.values())
            print(f'seed {seed}: ' + ('training models' if train_now else 'loading saved models'), flush=True)
            torch.manual_seed(config.TORCH_SEED + seed)
            fitted = {}
            for band in range(config.N_TARGETS):
                y = {part: take_chunks(lfp[:, :, band], split[part]) for part in ('train', 'val', 'test')}
                for chan in range(n_channels):
                    train_dl, val_dl, test_dl = (make_loader(X[part], y[part][..., chan:chan + 1], shuffle=(part == 'train'))
                                                 for part in ('train', 'val', 'test'))
                    model = models.build_lstm(input_size, self.device)
                    if train_now:
                        loss_tr, loss_val = model.train(train_dl, val_dl, config.NUM_EPOCHS,
                                                        select_best=True, patience=config.PATIENCE)
                        losses['train'][seed, chan, band], losses['val'][seed, chan, band] = loss_tr, loss_val
                        best_epoch[seed, chan, band] = model.best_epoch
                        model.save_model(files[chan, band])
                    else:
                        model.load_model(files[chan, band])
                    for part, dl in (('train', train_dl), ('val', val_dl), ('test', test_dl)):
                        r2_scores[part][seed, chan, band] = r2(model, dl)
                    fitted[chan, band] = model
            if seed == 0:
                self.__dict__['models'] = fitted     # reuse in later stages instead of reloading

        out = self.dirs.variables
        np.save(out / 'r2_scores.npy', r2_scores['test'][0])          # seed 0: channels x targets
        np.save(out / 'r2_scores_seeds.npy', r2_scores['test'])       # seeds x channels x targets
        np.save(out / 'r2_scores_val.npy', r2_scores['val'])
        np.save(out / 'r2_scores_train.npy', r2_scores['train'])
        if np.any(best_epoch >= 0):
            np.save(out / 'losses_train.npy', losses['train'])         # seeds x channels x targets x epochs
            np.save(out / 'losses_val.npy', losses['val'])
            np.save(out / 'best_epoch.npy', best_epoch)
        (out / 'model_info.json').write_text(json.dumps(dict(
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
            np.save(self.dirs.variables / name, scores)

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
        out.mkdir(parents=True, exist_ok=True)
        save_sparse(out / 'spikes_sparse.npz', self.active_spikes.spkMat)
        self.session.units.to_csv(out / 'units.csv', index=False)
        np.save(out / 'timestamps.npy', self.active_spikes.timestamps)
        self.session.active.to_csv(out / 'stimulus_presentation_active.csv', index=False)

    def attr(self):
        for window in config.ATTR_WINDOWS:
            todo = [(c, b) for c in range(self.n_channels) for b in self.window_targets(window)
                    if not paths.attribution_file(self.dirs, window, c, b).exists()]
            if not todo:
                print(f'Attributions for window "{window}" already saved', flush=True)
                continue
            start, stop = self.window_times(window)
            print(f'Attributions for window "{window}" ({start}-{stop} s): {len(todo)} models', flush=True)
            X = torch.tensor(self.spikes(start, stop).spkMat).float().to(self.device)
            for chan, band in todo:
                # train() mode is required for the cuDNN RNN backward pass; LSTMnet has no dropout
                net = self.models[chan, band].model.train().to(self.device)
                ig = IntegratedGradient(net, method='last time point', seqlength=config.SEQLENGTH)
                attrs = ig.run(X, baselines=config.IG_BASELINE, n_batch=self.ig_batch, n_steps=config.IG_N_STEPS)
                save_sparse(paths.attribution_file(self.dirs, window, chan, band), attrs.numpy())
            del X

    def summaries(self):
        n_units = len(self.unit_areas)
        for window, prefix in SUMMARY_PREFIX.items():
            out = {kind: self.dirs.attrs / f'{prefix}_{kind}.npy' for kind in SUMMARY_KINDS}
            if all(f.exists() for f in out.values()):
                continue
            # targets that were not attributed stay NaN
            result = {kind: np.full((self.n_channels, config.N_TARGETS, n_units), np.nan) for kind in SUMMARY_KINDS}
            for chan in range(self.n_channels):
                for band in self.window_targets(window):
                    attrs = load_sparse(paths.attribution_file(self.dirs, window, chan, band))
                    for kind, summarise in SUMMARY_KINDS.items():
                        result[kind][chan, band] = summarise(attrs)
            for kind, f in out.items():
                np.save(f, result[kind])

    def event_times(self):
        """Event times (s) for each type in EVENT_TYPES; None where unavailable."""
        stim = self.session.stimulus_presentation
        events = {'stim': stim.start_time[stim.active & (stim.image_name != 'omitted')].values, 'hvs': None}

        hvs_table = pd.read_csv(paths.REPO_DIR / 'tables' / 'hvs_detection.csv')
        flagged = hvs_table.loc[hvs_table['session'] == self.session_id, 'hvs']
        hvs_file = paths.hvs_events_file(self.dirs)
        legacy_file = self.data_root / f'session_{self.session_id}' / 'detected_hvs_events.npy'
        if flagged.empty or not flagged.iloc[0]:
            print('Session not marked as having HVS (tables/hvs_detection.csv); skipping HVS events')
        elif hvs_file.exists() or legacy_file.exists():
            events['hvs'] = np.load(hvs_file if hvs_file.exists() else legacy_file)[:, 1]   # peak times
        else:
            print(f'No HVS events file at {hvs_file}; run experiments/detect_cortical_oscillations.py first')
        return events

    def peri(self):
        channels = [c for c in range(self.n_channels) if paths.attribution_file(self.dirs, 'session', c, 0).exists()]
        if not channels:
            raise FileNotFoundError('No "session" window attributions; run the attr stage first')
        spikes = self.spikes(*self.window_times('session'), scaled=False)    # raw counts for the PSTH
        out = self.dirs.variables
        out.mkdir(parents=True, exist_ok=True)

        for name, times in self.event_times().items():
            spec = EVENT_TYPES[name]
            sfx = spec['suffix']
            if times is None or (out / f'mean_raw_potency_snippet{sfx}.npy').exists():
                continue
            t = snippet_time_axis(spec['window'], config.BIN_SIZE)
            n_bins = len(t)
            starts, _ = event_start_indices(spikes.timestamps, times, spec['window'], n_bins)
            baseline_bins = tuple(np.searchsorted(t, spec['baseline']))
            n_events, n_units, n_ch = len(starts), spikes.spkMat.shape[1], len(channels)
            print(f'{name}: {n_events} events, {n_ch} channels', flush=True)

            psth = extract_snippets(spikes.spkMat, starts, n_bins).mean(axis=0).T / config.BIN_SIZE   # units x bins, spikes/s
            rank_snippets = np.zeros((n_events, n_bins, n_units, n_ch), dtype=np.uint16)
            active_snippets = np.zeros((n_events, n_bins, n_units, n_ch), dtype=bool)
            jaccard_raw = np.zeros((n_events, n_bins, n_ch), dtype=np.float32)
            jaccard_ref = np.zeros((n_events, n_bins, n_ch), dtype=np.float32)
            mean_attr = np.zeros((n_bins, n_units, n_ch), dtype=np.float32)
            mean_potency = np.zeros((n_bins, n_units, n_ch), dtype=np.float32)

            for ci, chan in enumerate(channels):
                attrs = load_sparse(paths.attribution_file(self.dirs, 'session', chan, 0)).astype(np.float32)
                ranks = rank_active_neurons(attrs)
                top_k = top_k_mask(ranks, config.TOP_K)
                reference = baseline_reference_set(attrs, starts, baseline_bins, config.TOP_K)
                jaccard_raw[..., ci] = extract_snippets(jaccard_consecutive(top_k), starts, n_bins)
                jaccard_ref[..., ci] = extract_snippets(jaccard_to_reference(top_k, reference), starts, n_bins)
                rank_snippets[..., ci] = extract_snippets(ranks, starts, n_bins)
                del ranks, top_k

                attr_snippets = extract_snippets(attrs, starts, n_bins)
                active = np.abs(attr_snippets) > EPSILON
                active_snippets[..., ci] = active
                mean_attr[:, :, ci] = attr_snippets.mean(axis=0)
                # potency: mean attribution per spike (over the events in which the unit fired in that bin)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean_potency[:, :, ci] = attr_snippets.sum(axis=0) / active.sum(axis=0)
                del attrs, attr_snippets, active

            np.save(out / f'psth{sfx}.npy', psth.astype(np.float32))
            np.save(out / f'rank_snippets{sfx}.npy', rank_snippets)
            np.save(out / f'jaccard_raw_snippets{sfx}.npy', jaccard_raw)
            np.save(out / f'jaccard_ref_snippets{sfx}.npy', jaccard_ref)
            save_sparse(out / f'active_mask_snippet_sparse{sfx}.npz', active_snippets.reshape(-1, n_ch))
            np.save(out / f'active_mask_shape{sfx}.npy', active_snippets.shape)
            np.save(out / f'mean_raw_attribution_snippet{sfx}.npy', mean_attr)
            np.save(out / f'mean_raw_potency_snippet{sfx}.npy', mean_potency)

    def ipd(self):
        out = self.dirs.variables
        for name, spec in EVENT_TYPES.items():
            sfx = spec['suffix']
            rank_file, result_file = out / f'rank_snippets{sfx}.npy', out / f'regional_dominance_results{sfx}.npz'
            if not rank_file.exists() or result_file.exists():
                continue
            rank_snippets = np.load(rank_file)
            shape = tuple(np.load(out / f'active_mask_shape{sfx}.npy'))
            active_snippets = load_npz(out / f'active_mask_snippet_sparse{sfx}.npz').toarray().reshape(shape).astype(bool)

            per_channel = []
            for ci in range(rank_snippets.shape[-1]):
                dominance, regions = regional_dominance(rank_snippets[..., ci], active_snippets[..., ci], self.unit_areas)
                per_channel.append(dominance.mean(axis=0))                  # average over events
            average_dominance = np.stack(per_channel, axis=-1)                # regions x bins x channels
            np.savez_compressed(result_file,
                                average_dominance=average_dominance,
                                proportional_influence=proportional_influence(average_dominance),
                                region_names=regions,
                                t=snippet_time_axis(spec['window'], config.BIN_SIZE))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('session_dir', help='.../visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>')
    parser.add_argument('--stages', default=','.join(STAGES), help=f'comma-separated subset of {STAGES}')
    parser.add_argument('--retrain', action='store_true', help='train even if saved models exist')
    parser.add_argument('--ig-batch', type=int, default=config.IG_N_BATCH, help='sequences per IG batch')
    parser.add_argument('--seeds', type=int, default=config.N_SEEDS, help='independently initialised fits per model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    stages = args.stages.split(',')
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f'unknown stages: {sorted(unknown)}')

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    print(f'Session {session_id}, data root {data_root}, device {args.device}', flush=True)
    pipeline = SessionPipeline(data_root, session_id, torch.device(args.device), args.retrain, args.ig_batch, args.seeds)
    for stage in STAGES:                    # always in pipeline order
        if stage in stages:
            print(f'--- {stage}', flush=True)
            getattr(pipeline, stage)()
    print('Session complete.', flush=True)


if __name__ == '__main__':
    main()
