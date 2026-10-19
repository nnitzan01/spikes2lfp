"""End-to-end run of every run_session.py stage on a small synthetic session (no AllenSDK or NWB files)."""
import sys
import types

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import config
import paths
import preprocess_data

N_UNITS, SESSION_LEN, N_CHANNELS = 12, 120, 2
HVS_SESSION = 1049273528      # marked as having HVS in tables/hvs_detection.csv


class FakeSession:
    def __init__(self, session_id, df, output_dir, target_areas=None):
        rng = np.random.default_rng(session_id % 1000)
        areas = np.array(['VISp', 'VISl', 'LGd', 'CA1'])
        self.units = pd.DataFrame({'structure_acronym': areas[np.arange(N_UNITS) % 4]},
                                  index=np.arange(500, 500 + N_UNITS))
        self.spike_times = {u: np.sort(rng.uniform(0, SESSION_LEN, rng.integers(300, 3000))) for u in self.units.index}
        starts = np.arange(2.0, SESSION_LEN - 2, 0.75)
        n = len(starts)
        self.stimulus_presentation = pd.DataFrame({
            'start_time': starts, 'active': starts < 60, 'is_change': np.arange(n) % 7 == 0,
            'image_name': np.where(np.arange(n) % 10 == 0, 'omitted', 'im')})
        self.active = self.stimulus_presentation[self.stimulus_presentation.active]
        self.active_times, self.spontaneous_times, self.passive_times = [0, 60], [60, 75], [75, SESSION_LEN]
        self.channels = pd.DataFrame({'structure_acronym': ['VISp'] * N_CHANNELS + ['VISl'],
                                      'dorsal_ventral_ccf_coordinate': [100.0, 200.0, 300.0]})


def fake_load_lfp(data_root, session_id, channels, start, stop):
    t = np.arange(start, stop, 1 / config.LFP_FS)
    lfp = np.stack([np.sin(2 * np.pi * f * t) + 0.1 * np.random.default_rng(f).normal(size=len(t))
                    for f in (6, 40, 9)], axis=1)
    visp = (channels['structure_acronym'] == 'VISp').to_numpy()
    return channels[visp], lfp[:, visp], t


@pytest.fixture
def pipeline_module(monkeypatch):
    monkeypatch.setitem(sys.modules, 'process_session', types.SimpleNamespace(session=FakeSession))
    monkeypatch.setattr(preprocess_data.pp, 'load_lfp', fake_load_lfp)
    monkeypatch.setattr(config, 'NUM_EPOCHS', 1)
    monkeypatch.setattr(config, 'IG_N_STEPS', 3)
    monkeypatch.setattr(config, 'ATTR_WINDOWS', {
        'active': dict(epoch='active', duration=9, targets=None),
        'spontaneous': dict(epoch='spontaneous', duration=6, targets=None),
        'session': dict(epoch='session', duration=None, targets=[0])})
    import run_session
    return run_session


def test_all_stages(pipeline_module, tmp_path):
    run_session = pipeline_module
    session_dir = tmp_path / 'visual-behavior-neuropixels-0.4.0' / 'behavior_ecephys_sessions' / str(HVS_SESSION)
    session_dir.mkdir(parents=True)
    data_root, session_id = paths.parse_session_dir(session_dir)
    dirs = paths.session_output_dirs(data_root, session_id)

    pipe = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4)
    pipe.train()
    # HVS events, as written by experiments/detect_cortical_oscillations.py: [start, peak, end] in s
    np.save(paths.hvs_events_file(dirs), np.array([[10.0, 10.5, 11.0], [80.0, 80.4, 81.0]]))
    for stage in run_session.STAGES[1:]:
        getattr(run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4), stage)()

    r2 = np.load(dirs.variables / 'r2_scores.npy')
    assert r2.shape == (N_CHANNELS, config.N_TARGETS) and np.isfinite(r2).all()
    assert len(list(dirs.models.glob('*.pt'))) == N_CHANNELS * config.N_TARGETS
    assert np.load(dirs.attrs / 'attribution_scores_ipi_mean.npy').shape == (N_CHANNELS, config.N_TARGETS, N_UNITS)
    assert np.load(dirs.attrs / 'attribution_scores_spontaneous_mean.npy').shape == (N_CHANNELS, config.N_TARGETS, N_UNITS)
    for sfx in ('', '_hvs'):
        ranks = np.load(dirs.variables / f'rank_snippets{sfx}.npy')
        assert ranks.ndim == 4 and ranks.shape[2:] == (N_UNITS, N_CHANNELS)
        jac = np.load(dirs.variables / f'jaccard_ref_snippets{sfx}.npy')
        assert np.nanmax(jac) <= 1
        ipd = np.load(dirs.variables / f'regional_dominance_results{sfx}.npz')
        assert ipd['average_dominance'].shape == (4, ranks.shape[1], N_CHANNELS)
    assert len(list(dirs.plots.glob('*.png'))) > 0

    # a rerun skips trained models and existing outputs
    rerun = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4)
    before = (dirs.models / 'lstm_model_chan0_band0.pt').stat().st_mtime_ns
    rerun.train()
    assert (dirs.models / 'lstm_model_chan0_band0.pt').stat().st_mtime_ns == before
    np.testing.assert_allclose(np.load(dirs.variables / 'r2_scores.npy'), r2, rtol=1e-6)
