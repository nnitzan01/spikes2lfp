import numpy as np

import config
from experiments import summarize
from experiments.attribution_baseline import MEASURES as AB_MEASURES
from experiments.ranked_ablation import FRACTIONS, MEASURES as RA_MEASURES
from experiments.region_ablation import REGIONS

N_CH, T = 6, config.N_TARGETS


def r2(rng, *shape):
    return rng.uniform(0.1, 0.6, shape + (N_CH, T))


def ablation_file(rng, measures):
    m = len(measures)
    return dict(r2_full=r2(rng) + 0.3, r2_ablated=r2(rng, 2, m, len(FRACTIONS)), r2_random=r2(rng, 2, 5, len(FRACTIONS)),
                fractions=np.array(FRACTIONS), retrain_fractions=np.array([0.05, 0.2]), measures=np.array(measures),
                r2_retrained=r2(rng, m, 2, 1), r2_retrained_random=r2(rng, 2, 2, 1), r2_retrained_full=r2(rng, 1) + 0.3,
                spearman_with_attribution=r2(rng, m), completeness=r2(rng), spearman_zero_baseline=r2(rng),
                spearman_firing_rate=r2(rng))


def region_file(rng):
    n = len(REGIONS)
    retrained = np.full((n, 1, N_CH, T), np.nan)
    retrained[:6] = r2(rng, 6, 1)
    return dict(regions=np.array(REGIONS), region_n_units=rng.integers(0, 30, n), r2_full=r2(rng) + 0.3,
                r2_removed=r2(rng, 2, n), r2_only=r2(rng, 2, n), r2_retrained_full=r2(rng, 1) + 0.3,
                r2_cov_full=r2(rng, 1) + 0.3, r2_retrained_removed=retrained, r2_cov_removed=retrained.copy())


def clock_file(rng):
    out = {k: r2(rng) for k in ('r2_model', 'r2_template', 'r2_residual', 'r2_residual_own', 'r2_trial_model',
                                'r2_trial_template', 'r2_retrained_residual', 'r2_spikes_plus_template')}
    out.update({k: r2(rng, 5) for k in ('r2_shuffle', 'r2_residual_shuffle', 'r2_shift', 'r2_residual_shift')})
    return out


def region_clock_file(rng):
    n = len(REGIONS)
    out = {k: r2(rng) for k in ('r2_model', 'r2_residual')}
    out.update(r2_removed=r2(rng, n), r2_residual_removed=r2(rng, n), r2_swap=r2(rng, 2, n), r2_residual_swap=r2(rng, 2, n),
               r2_swap_all=r2(rng, 2), r2_residual_swap_all=r2(rng, 2), share_stlfp=r2(rng, n),
               share_attribution_mean_baseline=r2(rng, n), regions=np.array(REGIONS), region_n_units=rng.integers(0, 30, n))
    return out


def test_summaries_over_sessions(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    sids = [1, 2, 3, 4]
    for sid in sids[:3]:                                     # the last session has no files
        d = tmp_path / str(sid)
        d.mkdir()
        np.savez(d / 'ranked_ablation.npz', **ablation_file(rng, RA_MEASURES))
        np.savez(d / 'attribution_baseline.npz', **ablation_file(rng, AB_MEASURES))
        np.savez(d / 'region_ablation.npz', **region_file(rng))
        np.savez(d / 'region_ablation_bands.npz', **region_file(rng))
        np.savez(d / 'stimulus_clock.npz', **clock_file(rng))
        np.savez(d / 'region_clock.npz', **region_clock_file(rng))
    monkeypatch.setattr(summarize, 'sessions', lambda: sids)
    monkeypatch.setattr(summarize, 'variables_dir', lambda sid: tmp_path / str(sid))
    monkeypatch.setattr(summarize, 'pooled_channels', lambda sid: slice(1, 5))
    borders = tmp_path / 'borders.csv'
    borders.write_text('session,lower_border_index,upper_border_index,first_channel_index\n'
                       + ''.join(f'{sid},2,3,1\n' for sid in sids))
    monkeypatch.setattr(summarize, 'ALIGNMENT_TABLE', borders)
    for f in (summarize.ranked_ablation, summarize.attribution_baseline_sessions, summarize.region_ablation,
              summarize.region_bands, summarize.region_clock, summarize.stimulus_clock):
        f()
        out = capsys.readouterr().out
        assert '3 sessions with' in out and 'missing: 4' in out
    assert summarize.paired([0.1, 0.2, np.nan, -0.05]).startswith('+0.100 (2/3 > 0')


def test_simulation_pooled_over_sessions(tmp_path, monkeypatch, capsys):
    from experiments.simulation import LAG_MEASURES, MEASURES, METRICS
    rng = np.random.default_rng(0)
    for sid in (1, 2):
        d = tmp_path / str(sid)
        d.mkdir()
        for c in ('linear', 'gain'):
            out = {k: rng.uniform(size=(1, 3, len(MEASURES))) for k in METRICS}
            out.update({k: rng.uniform(size=(1, 3)) for k in ('r2', 'r2_ridge', 'r2_ceiling')}, best_epoch=np.full((1, 3), 50.),
                       lag_spearman=rng.uniform(size=(1, 3, len(LAG_MEASURES))), conditions=np.array([c]),
                       measures=np.array(MEASURES), lag_measures=np.array(LAG_MEASURES), metrics=np.array(METRICS))
            np.savez(d / f'simulation_final_{c}.npz', **out)
    monkeypatch.setattr(summarize, 'sessions', lambda: [1, 2, 3])
    monkeypatch.setattr(summarize, 'variables_dir', lambda sid: tmp_path / str(sid))
    summarize.simulation_sessions()
    out = capsys.readouterr().out
    assert '=== linear: 6 draws in 2 sessions (simulation_final_linear.npz)' in out
    assert 'ig_mean - ridge_weight' in out and '/6 > 0' in out
