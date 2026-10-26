"""experiments/pool_sessions.py on the synthetic session from conftest.py."""
import numpy as np
import pandas as pd

import config
import paths
from conftest import HVS_SESSION, N_CHANNELS, N_UNITS


def _run_all_stages(run_session, session_dir):
    data_root, session_id = paths.parse_session_dir(session_dir)
    for stage in run_session.STAGES:
        getattr(run_session.SessionPipeline(data_root, session_id, device='cpu', ig_batch=8), stage)()
    return data_root, paths.session_output_dirs(data_root, session_id)


def test_align_places_channels_and_fills_nan():
    from experiments.pool_sessions import ALIGNED_CHANNELS, align, channel_alignment
    row = pd.Series(dict(session=1, first_channel_index=2, last_channel_index=4, start_index=0))
    src, dst = channel_alignment(row, n_channels=6)
    np.testing.assert_array_equal(src, [2, 3, 4])
    x = np.arange(6 * 2).reshape(2, 6)                       # channels on axis 1
    out = align(x, src, dst, axis=1)
    assert out.shape == (2, ALIGNED_CHANNELS)
    np.testing.assert_array_equal(out[:, :3], x[:, 2:5])
    assert np.isnan(out[:, 3:]).all()


def test_check_and_pool(pipeline_module, session_dir):
    from experiments import pool_sessions as ps
    data_root, dirs = _run_all_stages(pipeline_module, session_dir)
    assert 'unit_id' in pd.read_csv(dirs.variables / 'units.csv').columns

    status = ps.check(data_root, [HVS_SESSION])
    assert status.complete.all(), status.missing.tolist()

    # upper L4 border on the session's channel 1, which becomes aligned channel BELOW_L4
    alignment = pd.DataFrame(dict(session=[HVS_SESSION], upper_border_index=[1], first_channel_index=[0],
                                  last_channel_index=[1], start_index=[ps.BELOW_L4 - 1]))
    info = pd.DataFrame(dict(unit_id=np.arange(500, 500 + N_UNITS), layer=np.arange(N_UNITS) % 4))
    units = ps.pool(data_root, [HVS_SESSION], alignment=alignment, units_info=info)
    out = ps.pooled_dir(data_root)

    assert len(units) == N_UNITS and (units.layer.to_numpy() == np.arange(N_UNITS) % 4).all()
    assert set(units.area) == {'VISp', 'VISl', 'LG', 'CA1'}
    r2 = np.load(out / 'r2.npz')
    assert r2['test'].shape == (1, ps.ALIGNED_CHANNELS, config.N_TARGETS)
    np.testing.assert_allclose(r2['test'][0, ps.BELOW_L4 - 1:ps.BELOW_L4 + 1], np.load(dirs.variables / 'r2_scores.npy'), rtol=1e-6)
    assert np.isnan(r2['test'][0, :ps.BELOW_L4 - 1]).all()

    u = np.load(out / 'unit_attribution_active.npz')
    raw = np.load(dirs.attrs / 'unit_attribution_active.npz')
    np.testing.assert_allclose(u['per_spike'][:, ps.BELOW_L4], raw['per_spike'][1].T, rtol=1e-6)
    assert np.load(out / 'unit_attribution_spont_recent.npz')['mean_abs'].shape == (N_UNITS, ps.ALIGNED_CHANNELS, config.N_TARGETS)
    comp = np.load(out / 'components.npz')
    assert comp['total'].shape == (N_UNITS, ps.ALIGNED_CHANNELS, config.N_TARGETS, len(config.COMPONENT_TIMES))
    peri = np.load(out / 'peri.npz')
    assert peri['recent'].shape == (N_UNITS, len(peri['times']), ps.ALIGNED_CHANNELS)
    assert np.load(out / 'jaccard.npz')['ref_recent'].shape == (1, len(peri['times']), ps.ALIGNED_CHANNELS)
    kern = np.load(out / 'area_kernels.npz')
    assert kern['abs_mean'].shape[:3] == (4, ps.ALIGNED_CHANNELS, config.N_TARGETS) and kern['n_units'].sum() == N_UNITS
    dom = np.load(out / 'regional_dominance.npz')
    assert dom['dominance'].shape == (len(dom['regions']), len(peri['times']), ps.ALIGNED_CHANNELS)
    assert pd.read_csv(out / 'sessions.csv').n_channels.tolist() == [N_CHANNELS]

    # the cached per-session arrays are reused until an input changes
    cache = out / 'sessions' / f'{HVS_SESSION}.npz'
    before = cache.stat().st_mtime_ns
    ps.pool(data_root, [HVS_SESSION], alignment=alignment, units_info=info)
    assert cache.stat().st_mtime_ns == before


def test_check_reports_what_to_redo(pipeline_module, session_dir):
    from experiments import pool_sessions as ps
    data_root, dirs = _run_all_stages(pipeline_module, session_dir)
    paths.attribution_file(dirs, 1, 4).unlink()
    units = pd.read_csv(dirs.variables / 'units.csv')
    units.drop(columns='unit_id').to_csv(dirs.variables / 'units.csv', index=False)   # as written before the fix

    s = ps.session_status(data_root, HVS_SESSION)
    assert {'attr', 'variables'} <= s['stages']
    assert any(m.startswith('attributions 17/18') for m in s['missing'])
    assert 'unit ids in units.csv' in s['missing']
