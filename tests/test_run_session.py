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
    for stage in run_session.STAGES[1:]:
        getattr(run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4), stage)()

    r2 = np.load(dirs.variables / 'r2_scores.npy')
    assert r2.shape == (N_CHANNELS, config.N_TARGETS) and np.isfinite(r2).all()
    assert len(list(dirs.models.glob('*.pt'))) == N_CHANNELS * config.N_TARGETS
    assert len(list(dirs.attrs.glob('seqig_*.npz'))) == N_CHANNELS * config.N_TARGETS

    a = np.load(paths.attribution_file(dirs, 0, 0))
    t = np.load(dirs.variables / 'attr_targets.npz')
    assert a['sampled_test_total'].shape == (len(t['sampled_test']), N_UNITS)
    assert a['components_total'].shape == (len(t['event_onsets']), len(config.COMPONENT_TIMES), N_UNITS)
    assert a['peri_total'].shape == (config.PERI_EVENTS, len(t['peri_times']), N_UNITS)
    assert a['kernel_mean'].shape == (config.ATTR_HISTORY // config.LAG_KERNEL_BIN, N_UNITS)
    assert set(t['event_partition']) <= {'train', 'val', 'test', 'buffer'}
    assert np.nanmax(np.load(dirs.attrs / 'attribution_completeness.npy')) < 0.05

    for s in ('train', 'test', 'active', 'spont'):
        u = np.load(dirs.attrs / f'unit_attribution_{s}.npz')
        assert u['mean_abs'].shape == (N_CHANNELS, config.N_TARGETS, N_UNITS) and np.isfinite(u['share']).all()
    ranks = np.load(dirs.variables / 'rank_snippets.npy')
    assert ranks.shape == (config.PERI_EVENTS, len(t['peri_times']), N_UNITS, N_CHANNELS)
    assert np.nanmax(np.load(dirs.variables / 'jaccard_ref_snippets.npy')) <= 1
    ipd = np.load(dirs.variables / 'regional_dominance_results.npz')
    assert ipd['average_dominance'].shape == (4, len(t['peri_times']), N_CHANNELS)
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
