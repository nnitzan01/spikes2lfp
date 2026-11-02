import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from analysis.decoders import (SESSION_TABLE, cross_validate, evaluate, fixed_forest, grouped_folds, median_params,
                               mouse_ids, permute_within, tuned_forest)
from sklearn.metrics import balanced_accuracy_score

SMALL = dict(n_estimators=30, max_depth=6)


def real_sessions(n_mice):
    """Both sessions of the first n_mice mice recorded twice in tables/sessions.csv."""
    table = pd.read_csv(SESSION_TABLE)
    twice = table.groupby('mouse_id').session_id.count().loc[lambda c: c == 2].index[:n_mice]
    return table[table.mouse_id.isin(twice)].session_id.to_numpy()


def test_mouse_ids():
    table = pd.read_csv(SESSION_TABLE)
    np.testing.assert_array_equal(mouse_ids(table.session_id[:3]), table.mouse_id[:3])
    with pytest.raises(KeyError):
        mouse_ids([1])


def test_folds_keep_mice_together_and_classes_in_every_fold():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 30)
    y = rng.integers(0, 3, len(groups))
    for stratify in (True, False):
        folds = grouped_folds(y, groups, 5, seed=1, stratify=stratify)
        tested = np.concatenate([test for _, test in folds])
        assert sorted(tested) == list(range(len(y)))
        for train, test in folds:
            assert not set(groups[train]) & set(groups[test])
            if stratify:
                assert set(y[test]) == {0, 1, 2}


def test_permute_within_keeps_each_group_composition():
    rng = np.random.default_rng(2)
    groups = np.repeat([10, 11, 12], 50)
    y = rng.integers(0, 2, 150)
    y_null = permute_within(y, groups, rng)
    for g in (10, 11, 12):
        assert sorted(y_null[groups == g]) == sorted(y[groups == g])
    assert (y_null != y).any()


def test_grouped_cv_removes_the_mouse_leak():
    """Labels are a property of each mouse and features only identify the mouse: unit folds look
    accurate, mouse-grouped folds are at chance."""
    rng = np.random.default_rng(3)
    sessions = np.repeat(real_sessions(10), 40)
    mice = mouse_ids(sessions)
    _, mouse_index = np.unique(mice, return_inverse=True)
    mouse_label = np.arange(10) % 2
    y = mouse_label[mouse_index]
    X = rng.normal(size=(10, 5))[mouse_index] + 0.05 * rng.normal(size=(len(y), 5))

    decoder = fixed_forest('classifier', SMALL)
    unit_folds = [balanced_accuracy_score(y[te], decoder(X[tr], y[tr], None, X[te]))
                  for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]
    assert np.mean(unit_folds) > 0.95
    grouped = cross_validate(X, y, mice, decoder, balanced_accuracy_score, verbose=False)
    assert grouped['fold_scores'].mean() < 0.75


def test_evaluate_finds_real_signal_and_its_null_does_not():
    rng = np.random.default_rng(4)
    sessions = np.repeat(real_sessions(8), 60)
    y = rng.integers(0, 2, len(sessions))
    X = np.column_stack([y + 0.7 * rng.normal(size=len(y)), rng.normal(size=(len(y), 3))])
    X[rng.random(X.shape) < 0.05] = np.nan                                   # imputed inside each fold
    decoder = tuned_forest('classifier', n_trials=2, space={'n_estimators': (20, 30), 'max_depth': (3, 6)})
    out = evaluate('toy', X, y, sessions, decoder, 'classifier', n_permutations=9)
    assert out['fold_scores'].mean() > 0.7
    assert out['null'].mean() < 0.6 and out['p'] == pytest.approx(0.1)
    assert len(decoder.params) == 5 and set(median_params(decoder.params)) == {'n_estimators', 'max_depth'}


def test_evaluate_regression():
    rng = np.random.default_rng(5)
    sessions = np.repeat(real_sessions(6), 50)
    X = rng.normal(size=(len(sessions), 3))
    y = 2 * X[:, 0] + 0.3 * rng.normal(size=len(sessions))
    out = evaluate('toy regression', X, y, sessions, fixed_forest('regressor', SMALL), 'regressor')
    assert out['fold_scores'].mean() > 0.8 and out['prediction'].dtype == float
