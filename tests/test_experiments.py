"""The experiments run end to end on the synthetic session from conftest.py."""
import numpy as np

import paths
from conftest import N_UNITS


def _trained_pipeline(run_session, session_dir):
    import config
    config.ATTR_TARGETS = [0]          # the experiments only use the broadband model (reset by the fixture)
    data_root, session_id = paths.parse_session_dir(session_dir)
    pipeline = run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=4)
    pipeline.train()
    pipeline.attr()
    pipeline.summaries()
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
    assert all(len(d) == 2 for d in results['downsampled_per_spike'])


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


def test_compare_versions(pipeline_module, session_dir, capsys):
    from experiments import compare_versions
    pipeline = _trained_pipeline(pipeline_module, session_dir)
    for stage in ('transfer', 'variables', 'peri', 'ipd'):
        getattr(pipeline, stage)()
    data_root, sid = pipeline.data_root, str(pipeline.session_id)
    n_ch, n_units = np.load(pipeline.dirs.variables / 'r2_scores.npy').shape[0], N_UNITS

    # v1 outputs with the published shapes
    rng = np.random.default_rng(0)
    v1 = data_root / 'spikes2lfp'
    (v1 / 'variables' / sid).mkdir(parents=True)
    (v1 / 'attrs' / sid).mkdir(parents=True)
    np.save(v1 / 'variables' / sid / 'r2_scores.npy', rng.random((n_ch, 9)))
    np.save(v1 / 'variables' / sid / 'jaccard_ref_snippets.npy', rng.random((5, 188, n_ch)))
    np.save(v1 / 'attrs' / sid / 'attribution_scores_ipi_mean.npy', rng.normal(size=(n_ch, 9, n_units)))

    summary, pdf = compare_versions.compare(data_root, sid)
    compare_versions.print_summary(summary)
    assert pdf.exists() and (pdf.parent / 'compare_v1_v2.json').exists()
    assert summary['v1_files']['r2'] and summary['v1_files']['ipi'] and not summary['v1_files']['r2_spont']
    assert set(summary['attribution_median_spearman']) == {'mean |attr|', 'per spike', 'share'}
    assert summary['completeness']['models'] == n_ch          # broadband only in this test
    assert 'r2_test_median_v1' in capsys.readouterr().out
