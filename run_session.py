"""
Per-session pipeline: every result that is computed for one session and saved to disk.

usage:
    python run_session.py <data_root>/visual-behavior-neuropixels-0.4.0/behavior_ecephys_sessions/<session_id>
                          [--stages train,plots,...] [--retrain] [--ig-batch N] [--device cuda|cpu]

Outputs go to <data_root>/spikes2lfp/<kind>/<session_id>/ (see paths.py). Stages run in the order
below. Each one skips outputs that already exist, so an interrupted job can simply be resubmitted,
and later stages load what earlier ones saved.

  train      one LSTM per VISp channel x model target, trained on the active epoch
             (loaded instead when all model files exist, unless --retrain)
             models/     lstm_model_chan{c}_band{b}.pt
             variables/  r2_scores.npy (test R2, channels x targets), r2_scores_train.npy,
                         losses_{train,test}.npy (channels x epochs x targets; only when trained),
                         split_{train,test}_idx.npy (3 s chunks in each set), channels.csv
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
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, load_npz, save_npz
from sklearn.metrics import r2_score

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
from preprocess_data import chunk_and_reshape, get_data_loaders, pre_process_lfp, pre_process_spikes

STAGES = ['train', 'plots', 'variables', 'attr', 'summaries', 'peri', 'ipd']

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


def r2(model, dataloader):
    """R2 over all bins of all sequences in the dataloader."""
    X, y = dataloader.dataset.tensors
    return r2_score(y.numpy().reshape(-1, 1), model.evaluate(X).numpy())


class SessionPipeline:
    def __init__(self, data_root, session_id, device, retrain=False, ig_batch=config.IG_N_BATCH):
        self.data_root = data_root
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

    def spikes(self, start, stop, scaled=True):
        """Binned spikes from start to stop (s), truncated to whole sequences; min-max scaled (model input) if scaled."""
        sp = pre_process_spikes(self.session.units, self.session.spike_times)
        sp.getSpkMat(start, stop)
        sp.truncate(config.SEQLENGTH)
        if scaled:
            sp.minmax()
        return sp

    @cached_property
    def active_spikes(self):
        return self.spikes(*self.session.active_times)

    @cached_property
    def active_lfp(self):
        """Model targets for the active epoch (bins x VISp channels x targets), aligned with active_spikes."""
        lfp = pre_process_lfp(self.session_id, self.session.channels, *self.session.active_times, self.data_root)
        lfp.filter_lfp(take_power=True)
        lfp.downsample_lfp(config.LFP_DOWNSAMPLE)
        lfp.truncate(config.SEQLENGTH)
        lfp.align_lfp(self.active_spikes.spkMat.shape[0])
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

    @cached_property
    def models(self):
        """Trained models {(channel, target): process_model} loaded from disk."""
        loaded = {}
        for chan in range(self.n_channels):
            for band in range(config.N_TARGETS):
                f = paths.model_file(self.dirs, chan, band)
                if not f.exists():
                    raise FileNotFoundError(f'{f} not found; run the train stage first')
                model = models.build_lstm(len(self.session.units), self.device)
                model.load_model(f)
                loaded[chan, band] = model
        return loaded

    # ---- stages ------------------------------------------------------------------------------

    def train(self):
        spikes, lfp = self.active_spikes.spkMat, self.active_lfp.lfpMat
        n_channels, input_size = lfp.shape[1], spikes.shape[1]
        self.dirs.models.mkdir(parents=True, exist_ok=True)
        self.dirs.variables.mkdir(parents=True, exist_ok=True)
        self.active_lfp.channels.to_csv(self.dirs.variables / 'channels.csv', index=False)

        files = {(c, b): paths.model_file(self.dirs, c, b) for c in range(n_channels) for b in range(config.N_TARGETS)}
        train_now = self.retrain or not all(f.exists() for f in files.values())
        print('Training models' if train_now else 'Loading saved models', flush=True)

        torch.manual_seed(config.TORCH_SEED)
        r2_test, r2_train = (np.zeros((n_channels, config.N_TARGETS)) for _ in range(2))
        losses_train, losses_test = (np.zeros((n_channels, config.NUM_EPOCHS, config.N_TARGETS)) for _ in range(2))
        trained = {}
        for band in range(config.N_TARGETS):
            X_train, X_test, y_train, y_test, train_idx, test_idx = chunk_and_reshape(
                spikes, lfp[:, :, band], config.SEQLENGTH, return_indices=True)
            for chan in range(n_channels):
                train_dl, test_dl = get_data_loaders(X_train, X_test, y_train[:, :, chan], y_test[:, :, chan])
                model = models.build_lstm(input_size, self.device)
                if train_now:
                    losses_train[chan, :, band], losses_test[chan, :, band] = model.train(train_dl, test_dl, config.NUM_EPOCHS)
                    model.save_model(files[chan, band])
                else:
                    model.load_model(files[chan, band])
                r2_test[chan, band], r2_train[chan, band] = r2(model, test_dl), r2(model, train_dl)
                trained[chan, band] = model
        self.__dict__['models'] = trained     # reuse in later stages instead of reloading

        np.save(self.dirs.variables / 'r2_scores.npy', r2_test)
        np.save(self.dirs.variables / 'r2_scores_train.npy', r2_train)
        np.save(self.dirs.variables / 'split_train_idx.npy', train_idx)   # identical for every target
        np.save(self.dirs.variables / 'split_test_idx.npy', test_idx)
        if train_now:
            np.save(self.dirs.variables / 'losses_train.npy', losses_train)
            np.save(self.dirs.variables / 'losses_test.npy', losses_test)

    def plots(self):
        channels, lfp = self.active_lfp.channels, self.active_lfp.lfpMat
        yhat = np.zeros_like(lfp)
        for (chan, band), model in self.models.items():
            yhat[:, chan, band] = model.evaluate(self.active_spikes.spkMat).numpy()
        kw = dict(show_plot=False, save_fig=True, output_dir=self.dirs.plots, session_id=self.session_id)

        r2_test = np.load(self.dirs.variables / 'r2_scores.npy')
        plots.plot_r2(r2_test, channels, config.BANDS, fig_name='r2_scores.png', **kw)
        losses_file = self.dirs.variables / 'losses_test.npy'
        if losses_file.exists():
            for band in range(config.N_TARGETS):
                plots.plot_all_channel_loss(channels, config.BANDS, band, np.load(losses_file),
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
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    stages = args.stages.split(',')
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f'unknown stages: {sorted(unknown)}')

    data_root, session_id = paths.parse_session_dir(args.session_dir)
    print(f'Session {session_id}, data root {data_root}, device {args.device}', flush=True)
    pipeline = SessionPipeline(data_root, session_id, torch.device(args.device), args.retrain, args.ig_batch)
    for stage in STAGES:                    # always in pipeline order
        if stage in stages:
            print(f'--- {stage}', flush=True)
            getattr(pipeline, stage)()
    print('Session complete.', flush=True)


if __name__ == '__main__':
    main()
