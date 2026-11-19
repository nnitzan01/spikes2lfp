import numpy as np
import pandas as pd

import paths
from experiments import model_agreement as MA

N_CH, T, U = 6, 9, 60
AREAS = ['VISp'] * 15 + ['VISl'] * 10 + ['VISal'] * 10 + ['VISpm'] * 10 + ['CA1'] * 15


def write_tree(root, sid, units, attr, rng):
    v, a = root / 'variables' / str(sid), root / 'attrs' / str(sid)
    v.mkdir(parents=True)
    a.mkdir(parents=True)
    units.to_csv(v / 'units.csv', index=False)
    np.save(v / 'r2_scores.npy', rng.uniform(0.1, 0.6, (N_CH, T)))
    np.savez(a / 'unit_attribution_active.npz', **{m: attr * (1 + i) for i, m in enumerate(['per_spike', 'mean_abs', 'mean', 'share'])})


def test_model_agreement(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    sids = [1, 2, 3]
    info = []
    for sid in sids:
        units = pd.DataFrame(dict(unit_id=sid * 1000 + np.arange(U), structure_acronym=AREAS))
        rate = rng.lognormal(1, 1, U)
        info.append(pd.DataFrame(dict(unit_id=units['unit_id'], firing_rate=rate, coupling_index=rng.normal(size=U),
                                      coher_2_5=rng.uniform(size=U), coher_5_10=rng.uniform(size=U))))
        base = rate[None, None, :] * rng.lognormal(0, 0.3, (N_CH, T, U))
        write_tree(tmp_path / 'A', sid, units, base, rng)
        write_tree(tmp_path / 'B', sid, units, base * rng.lognormal(0, 0.1, base.shape), rng)     # nearly the same ranking
    pd.concat(info).to_csv(tmp_path / 'units_info.csv', index=False)
    (tmp_path / 'B' / 'attrs' / '3' / 'unit_attribution_active.npz').unlink()                  # one session only in A

    monkeypatch.setattr(paths, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(MA, 'UNITS_INFO', tmp_path / 'units_info.csv')
    monkeypatch.setattr(MA.S, 'sessions', lambda: sids)
    monkeypatch.setattr(MA.S, 'pooled_channels', lambda sid: slice(1, 5))

    table = MA.run('A', 'B', [0, 2])
    assert sorted(table['session'].unique()) == [1, 2]
    assert (tmp_path / 'A' / 'pooled' / 'model_agreement.csv').exists()
    cross = table[(table['model'] == 'cross') & (table['metric'] == 'unit_spearman')]['value']
    assert (cross > 0.9).all()
    rate = table[(table['metric'] == 'vs_firing_rate') & (table['model'] == 'a')]['value']
    assert (rate > 0.5).all()                       # scores were built from the firing rate
    assert table.query("metric == 'hierarchy'")['value'].notna().all()
    MA.summary(table, 'A', 'B')
    out = capsys.readouterr().out
    assert 'cross-model agreement' in out and 'visp_vs_other' in out
