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
    assert a['peri_recent'].shape == a['peri_total'].shape
    assert a['sampled_test_recent'].shape == a['sampled_test_total'].shape
    assert a['components_recent'].shape == a['components_total'].shape
    assert t['sampled_test_recent_spikes'].shape == t['sampled_test_spikes'].shape
    assert (t['sampled_test_recent_spikes'] <= t['sampled_test_spikes']).all()
    assert a['kernel_mean'].shape == (config.ATTR_HISTORY // config.LAG_KERNEL_BIN, N_UNITS)
    assert set(t['event_partition']) <= {'train', 'val', 'test', 'buffer'}
    assert np.nanmax(np.load(dirs.attrs / 'attribution_completeness.npy')) < 0.05

    for s in ('train', 'test', 'active', 'spont', 'recent_train', 'recent_active', 'recent_spont'):
        u = np.load(dirs.attrs / f'unit_attribution_{s}.npz')
        assert u['mean_abs'].shape == (N_CHANNELS, config.N_TARGETS, N_UNITS) and np.isfinite(u['share']).all()
    for suffix in ('', '_recent'):
        ranks = np.load(dirs.variables / f'rank_snippets{suffix}.npy')
        assert ranks.shape == (config.PERI_EVENTS, len(t['peri_times']), N_UNITS, N_CHANNELS)
        assert np.nanmax(np.load(dirs.variables / f'jaccard_ref_snippets{suffix}.npy')) <= 1
        ipd = np.load(dirs.variables / f'regional_dominance_results{suffix}.npz')
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


def test_resume_retrains_only_missing_models_identically(pipeline_module, session_dir):
    import torch
    run_session = pipeline_module
    data_root, session_id = paths.parse_session_dir(session_dir)
    dirs = paths.session_output_dirs(data_root, session_id)
    run_session.SessionPipeline(data_root, session_id, device='cpu').train()

    lost, kept = paths.model_file(dirs, 1, 3), paths.model_file(dirs, 0, 0)
    original = torch.load(lost)['state_dict']
    kept_mtime = kept.stat().st_mtime_ns
    lost.unlink()                                                   # as if the job died before saving it
    run_session.SessionPipeline(data_root, session_id, device='cpu').train()
    assert kept.stat().st_mtime_ns == kept_mtime                    # existing models are not retrained
    retrained = torch.load(lost)['state_dict']
    for k in original:                                              # per-model seed: same weights as before
        torch.testing.assert_close(retrained[k], original[k])
    assert 'best_epoch' in torch.load(lost)


def test_atomic_write_leaves_nothing_when_interrupted(tmp_path):
    import pytest
    import run_session
    target = tmp_path / 'out.npy'
    with pytest.raises(RuntimeError):
        with run_session.atomic(target) as tmp:
            tmp.write_text('partial')
            raise RuntimeError('job killed')
    assert not target.exists() and list(tmp_path.iterdir()) == []
    run_session.save_npy(target, np.arange(3))
    np.testing.assert_array_equal(np.load(target), [0, 1, 2])


def test_attr_raises_integration_steps_until_complete(pipeline_module, session_dir, monkeypatch):
    import config
    run_session = pipeline_module
    data_root, session_id = paths.parse_session_dir(session_dir)
    dirs = paths.session_output_dirs(data_root, session_id)
    monkeypatch.setattr(config, 'ATTR_TARGETS', [0])
    pipe = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=8)
    pipe.train()

    monkeypatch.setattr(config, 'IG_N_STEPS', 2)
    monkeypatch.setattr(config, 'IG_MAX_ERROR', 1e-12)       # unreachable: steps escalate to the cap
    monkeypatch.setattr(config, 'IG_MAX_STEPS', 8)
    pipe.attr()
    assert int(np.load(paths.attribution_file(dirs, 0, 0))['n_steps']) == 8

    # an existing file that fails the check (and has fewer steps than the cap) is recomputed
    monkeypatch.setattr(config, 'IG_MAX_STEPS', 32)
    pipe.attr()
    assert int(np.load(paths.attribution_file(dirs, 0, 0))['n_steps']) == 32
    # one that passes is left alone
    monkeypatch.setattr(config, 'IG_MAX_ERROR', 1.0)
    before = paths.attribution_file(dirs, 0, 0).stat().st_mtime_ns
    pipe.attr()
    assert paths.attribution_file(dirs, 0, 0).stat().st_mtime_ns == before


def test_files_without_recent_sums_are_upgraded(pipeline_module, session_dir, monkeypatch):
    import config
    run_session = pipeline_module
    data_root, session_id = paths.parse_session_dir(session_dir)
    dirs = paths.session_output_dirs(data_root, session_id)
    monkeypatch.setattr(config, 'ATTR_TARGETS', [0])
    monkeypatch.setattr(config, 'IG_MAX_ERROR', 1.0)          # no step escalation in this test
    pipe = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=8)
    pipe.train()
    pipe.attr()

    # make the files look like ones saved before the recent-lag sums existed, with 8 steps
    f, ft = paths.attribution_file(dirs, 0, 0), dirs.variables / 'attr_targets.npz'
    old = {k: v for k, v in np.load(f).items() if 'recent' not in k}
    old['n_steps'] = np.array(8)
    run_session.save_npz(f, **old)
    targets = {k: v for k, v in np.load(ft).items() if 'recent' not in k}
    run_session.save_npz(ft, **targets)

    run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=8).attr()
    a, t = np.load(f), np.load(ft)
    assert int(a['recent_bins']) == config.ATTR_RECENT_BINS and int(a['n_steps']) == 8    # redone with its own steps
    assert 'peri_recent' in a and 'sampled_spont_recent_spikes' in t
    for k, v in targets.items():                                    # the targets themselves are unchanged
        np.testing.assert_array_equal(t[k], v)


def test_downstream_stages_redo_when_attributions_change(pipeline_module, session_dir, monkeypatch):
    import os
    import config
    run_session = pipeline_module
    data_root, session_id = paths.parse_session_dir(session_dir)
    dirs = paths.session_output_dirs(data_root, session_id)
    monkeypatch.setattr(config, 'ATTR_TARGETS', [0])
    pipe = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=8)
    for stage in ('train', 'attr', 'summaries', 'peri', 'ipd'):
        getattr(pipe, stage)()
    outputs = [dirs.attrs / 'unit_attribution_active.npz', dirs.variables / 'jaccard_ref_snippets.npy',
               dirs.variables / 'regional_dominance_results.npz']
    stamps = [o.stat().st_mtime_ns for o in outputs]

    for stage in ('summaries', 'peri', 'ipd'):                     # nothing changed: all skipped
        getattr(pipe, stage)()
    assert [o.stat().st_mtime_ns for o in outputs] == stamps

    f = paths.attribution_file(dirs, 0, 0)                          # as if attr had repaired this model
    later = max(o.stat().st_mtime for o in outputs) + 10
    os.utime(f, (later, later))
    for stage in ('summaries', 'peri', 'ipd'):
        getattr(pipe, stage)()
    assert all(o.stat().st_mtime_ns != s for o, s in zip(outputs, stamps))   # all three redone
