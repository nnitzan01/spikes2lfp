"""The experiments run end to end on the synthetic session from conftest.py."""
import numpy as np

import paths
from conftest import N_UNITS


def _trained_pipeline(run_session, session_dir):
    data_root, session_id = paths.parse_session_dir(session_dir)
    pipeline = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4)
    pipeline.train()
    pipeline.attr()
    return pipeline


def test_ablation(pipeline_module, session_dir, monkeypatch):
    from experiments import ablation
    monkeypatch.setattr(ablation, 'PERCENT_REMOVED', np.array([25, 50]))
    pipeline = _trained_pipeline(pipeline_module, session_dir)
    r2_full, r2_rate, r2_attr, r2_rand = ablation.run_ablation(pipeline, chan=1, band=0, n_perm=2)
    assert np.isfinite(r2_full) and r2_rate.shape == (2, 1) and r2_rand.shape == (2, 2)


def test_firing_rate_contribution(pipeline_module, session_dir):
    from experiments import firing_rate_contribution
    pipeline = _trained_pipeline(pipeline_module, session_dir)
    results = firing_rate_contribution.run(pipeline, chan=0, n_top=3, n_iter=2)
    assert len(results['top_neuron_indices']) == 3
    assert all(len(d) == 2 for d in results['downsampled_attributions'])


def test_spectral_r2(pipeline_module, session_dir):
    from experiments import spectral_r2
    data_root, session_id = paths.parse_session_dir(session_dir)
    results, counts = spectral_r2.run(pipeline_module.SessionPipeline(data_root, session_id, device='cpu'))
    assert set(results) == {'full_model', 'cortex', 'cortex_no_visp', 'hpc', 'subcortical'}
    assert list(counts.index) == ['cortex', 'cortex_no_visp', 'hippocampus', 'subcortical']
    assert results['full_model'].shape == (len(spectral_r2.R2_BANDS) + 1, 2)
    areas = np.array(['VISp', 'VISpm', 'VISl', 'CA1', 'LGd'])
    np.testing.assert_array_equal(spectral_r2.unit_groups(areas)['cortex_no_visp'], [0, 1, 1, 0, 0])
    assert counts.loc['cortex', 'num_units'] == N_UNITS // 2      # VISp + VISl in the fake session


def test_hvs_detection_finds_bursts():
    from experiments.detect_cortical_oscillations import detect_oscillations
    fs = 1250
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(0)
    lfp = rng.normal(size=len(t))
    centers = [20, 50, 90]
    for c in centers:
        lfp += 4 * np.exp(-((t - c) / 0.8) ** 2) * np.sin(2 * np.pi * 4 * t)
    events = detect_oscillations(lfp, fs)
    assert len(events) == 3
    np.testing.assert_allclose(events[:, 1] / fs, centers, atol=0.3)
