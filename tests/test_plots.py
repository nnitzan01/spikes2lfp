"""Smoke tests: every plotting function runs on small synthetic inputs and saves a file."""
import types

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

import plots

BANDS = [(1, 4), (4, 8), (8, 12)]
N_T, N_CH, N_U = 2000, 3, 12


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    areas = np.array(['VISp', 'VISl', 'LGd'])[np.arange(N_U) % 3]
    units = pd.DataFrame({'structure_acronym': areas}, index=np.arange(100, 100 + N_U))
    starts = np.arange(0.5, 7, 0.75)
    stim = pd.DataFrame({'start_time': starts, 'is_change': np.arange(len(starts)) % 3 == 0})
    session = types.SimpleNamespace(units=units, active=stim, stimulus_presentation=stim)
    channels = pd.DataFrame({'dorsal_ventral_ccf_coordinate': [100.0, 200.0, 300.0]})
    return dict(rng=rng, session=session, channels=channels, timestamps=np.arange(N_T) * 0.004,
                y=rng.normal(size=(N_T, N_CH, len(BANDS) + 1)), attr=rng.normal(size=(N_T, N_U)),
                lfp_obj=types.SimpleNamespace(channels=channels))


def test_all_plots_run_and_save(data, tmp_path):
    d, kw = data, dict(show_plot=False, save_fig=True, output_dir=tmp_path)
    plots.plot_r2(d['rng'].random((N_CH, len(BANDS) + 1)), d['channels'], BANDS, fig_name='r2.png', **kw)
    plots.plot_lfp_prediction(d['y'], d['y'] * 0.5, 1, BANDS, fig_name='pred.pdf', **kw)
    plots.plot_lfp_prediction(d['y'], (d['y'] * 0.5)[:, 1, :], 1, BANDS, start_time=1, end_time=2,
                              fig_name='pred2.pdf', **kw)
    plots.plot_all_channel_loss(d['channels'], BANDS, 1, d['rng'].random((N_CH, 5, len(BANDS) + 1)),
                                fig_name='loss.pdf', **kw)
    plots.plot_abs_error_change(d['y'], d['y'] * 0.5, d['session'], d['timestamps'], 0, BANDS,
                                fig_name='err.pdf', **kw)
    plots.plot_psd(d['y'], d['y'] * 0.5, 0, fig_name='psd.pdf', **kw)
    plots.plot_attr_corrmat(d['attr'], d['session'], fig_name='corr.pdf', **kw)
    mean_attr = d['rng'].normal(size=(N_CH, len(BANDS) + 1, N_U))
    plots.plot_attr_density_plots(mean_attr, d['session'], BANDS, 'VIS', fig_name='dens.pdf', **kw)
    plots.plot_mean_attr_areas(d['rng'].random((3, len(BANDS) + 1, N_CH)), d['lfp_obj'], d['session'], BANDS,
                               fig_name='areas.pdf', **kw)
    df = pd.DataFrame({'unit_id': d['session'].units.index, 'pyr': np.arange(N_U) % 2})
    plots.plot_mean_attr_pyr_int(mean_attr, d['session'], 'VIS', df, BANDS, 0, fig_name='pyr.pdf', **kw)
    plots.plot_attr_matrix(d['attr'], d['y'][:, 0, 0], d['timestamps'], d['session'], [1, 5], plot_stim=True,
                           min_neurons=2, fig_name='mat.pdf', **kw)
    plots.plot_attr_lfp_corr(d['attr'], d['y'][:, 0, 0], d['session'], fig_name='lc.pdf', **kw)
    plots.plot_attr_lfp_corr(d['attr'], d['y'][:, 0, 0], d['session'], sorting='area', fig_name='lc2.pdf', **kw)
    plots.plot_mean_attr_clusters(d['rng'].random((6, len(BANDS) + 1, N_CH)), d['lfp_obj'], BANDS,
                                  fig_name='clu.pdf', **kw)
    plots.plot_peri_stim_attr(d['attr'], d['y'][:, 0, 0], d['session'], d['timestamps'], [-0.25, 0.5],
                              fig_name='peri.pdf', **kw)
    plots.plot_unit_attr_fr_corr(d['rng'].random((N_U, N_CH, len(BANDS) + 1)), d['session'], BANDS,
                                 fig_name='fr.pdf', **kw)
    assert (tmp_path / 'r2.png').read_bytes()[:4] == b'\x89PNG'   # format follows the extension
    assert len(list(tmp_path.iterdir())) == 16
