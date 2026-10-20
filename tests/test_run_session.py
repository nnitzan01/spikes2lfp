"""End-to-end run of every run_session.py stage on a small synthetic session (no AllenSDK or NWB files)."""
import numpy as np

import config
import paths
from conftest import N_CHANNELS, N_UNITS


def test_all_stages(pipeline_module, session_dir):
    run_session = pipeline_module
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
    for f in ('split.npz', 'split.json', 'spike_scaler.npz', 'target_scaler.npz', 'model_info.json', 'best_epoch.npy'):
        assert (dirs.variables / f).exists(), f
    split = np.load(dirs.variables / 'split.npz')
    assert split['train'].max() < split['val'].min() < split['val'].max() < split['test'].min()   # past -> future
    for f in ('r2_scores_passive.npy', 'r2_scores_spont.npy'):
        assert np.load(dirs.variables / f).shape == (N_CHANNELS, config.N_TARGETS)

    # a rerun skips trained models and existing outputs
    rerun = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4)
    before = (dirs.models / 'lstm_model_chan0_band0_seed0.pt').stat().st_mtime_ns
    rerun.train()
    assert (dirs.models / 'lstm_model_chan0_band0_seed0.pt').stat().st_mtime_ns == before
    np.testing.assert_allclose(np.load(dirs.variables / 'r2_scores.npy'), r2, rtol=1e-6)
