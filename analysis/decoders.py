"""
Decoding unit identity from attribution with cross-validation grouped by mouse (reviewer comments on
Figs. 5, 6 and 8).

Units of one session, and of the two sessions of one mouse, share a probe track, a channel
alignment and a model, so their attribution vectors are not independent samples. Folds over units
put units of the same mouse in training and test sets, which inflates accuracy. Here every unit of a
mouse is in the same fold, and hyperparameters are tuned on a split that is grouped the same way.

A decoder is a function fit_predict(X_train, y_train, groups_train, X_test) -> predictions; it may
tune itself on (X_train, y_train) using groups_train for its inner split.

Null: labels permuted among the units of each session, which keeps every session's class
composition (and so any session-level confound) and breaks only the unit-to-label pairing.
p = (1 + #(null >= observed)) / (1 + n_permutations).
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import paths

SESSION_TABLE = paths.REPO_DIR / 'tables' / 'sessions.csv'
# random-forest hyperparameters searched by tuned_forest: name -> (low, high)
FOREST_SPACE = {'n_estimators': (50, 300), 'max_depth': (5, 30), 'min_samples_split': (2, 20)}
SCORES = {'classifier': balanced_accuracy_score, 'regressor': r2_score}


def mouse_ids(session_ids, table=SESSION_TABLE):
    """Mouse of each session id (tables/sessions.csv); raises if a session is not in the table."""
    mouse = pd.read_csv(table).set_index('session_id')['mouse_id']
    session_ids = pd.Series(np.asarray(session_ids))
    missing = set(session_ids) - set(mouse.index)
    if missing:
        raise KeyError(f'sessions not in {table}: {sorted(missing)}')
    return session_ids.map(mouse).to_numpy()


def grouped_folds(y, groups, n_splits=5, seed=0, stratify=True):
    """(train, test) index pairs with every group in one test fold; class-stratified when stratify."""
    X = np.zeros((len(y), 1))
    if stratify:
        return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    order = np.random.default_rng(seed).permutation(np.unique(groups))      # GroupKFold has no shuffle
    rank = dict(zip(order, range(len(order))))
    return list(GroupKFold(n_splits).split(X, y, [rank[g] for g in groups]))


def inner_split(y, groups, seed=0, stratify=True):
    """One grouped split of about 80 / 20 for tuning: (train, validation) indices."""
    return grouped_folds(y, groups, 5, seed, stratify)[0]


def permute_within(y, groups, rng):
    """y with its entries shuffled among the members of each group."""
    y = np.asarray(y)
    out = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = y[rng.permutation(idx)]
    return out


def cross_validate(X, y, groups, fit_predict, score, n_splits=5, seed=0, stratify=True, verbose=True):
    """Out-of-fold predictions and the score of each fold: dict(prediction, fold_scores, folds)."""
    y = np.asarray(y)
    prediction = None
    folds = grouped_folds(y, groups, n_splits, seed, stratify)
    scores = []
    for k, (train, test) in enumerate(folds):
        fold_prediction = np.asarray(fit_predict(X[train], y[train], groups[train], X[test]))
        if prediction is None:
            prediction = np.empty(len(y), dtype=fold_prediction.dtype)
        prediction[test] = fold_prediction
        scores.append(score(y[test], prediction[test]))
        if verbose:
            print(f'  fold {k + 1}: {len(np.unique(groups[test]))} mice, {len(test)} units, score {scores[-1]:.3f}',
                  flush=True)
    return dict(prediction=prediction, fold_scores=np.array(scores), folds=folds)


def permutation_test(X, y, groups, permute_groups, fit_predict, score, observed, n_permutations=50, n_splits=5,
                     seed=0, stratify=True):
    """
    Mean fold score of each of n_permutations refits on labels permuted within permute_groups (the
    folds stay grouped by `groups`), and the one-sided p-value of `observed`.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        y_null = permute_within(y, permute_groups, rng)
        null[i] = cross_validate(X, y_null, groups, fit_predict, score, n_splits, seed, stratify,
                                 verbose=False)['fold_scores'].mean()
    return null, (1 + np.sum(null >= observed)) / (1 + n_permutations)


def forest(kind, params, seed=42):
    """Median imputation, standardisation and a random forest ('classifier' class-balanced, or 'regressor')."""
    if kind == 'classifier':
        model = RandomForestClassifier(**params, class_weight='balanced', random_state=seed, n_jobs=-1)
    else:
        model = RandomForestRegressor(**params, random_state=seed, n_jobs=-1)
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('forest', model)])


def fixed_forest(kind, params, seed=42):
    """A decoder that fits forest(kind, params) as is (for permutation nulls)."""
    def fit_predict(X_train, y_train, groups_train, X_test):
        return forest(kind, params, seed).fit(X_train, y_train).predict(X_test)
    return fit_predict


def tuned_forest(kind, n_trials=30, space=FOREST_SPACE, seed=42):
    """
    A decoder that picks forest hyperparameters by Optuna on a mouse-grouped inner split of its
    training set, then refits on the whole training set. The chosen parameters and fitted models of
    each call are kept in .params and .models.
    """
    score = SCORES[kind]

    def fit_predict(X_train, y_train, groups_train, X_test):
        fit, val = inner_split(y_train, groups_train, seed, stratify=kind == 'classifier')

        def objective(trial):
            params = {name: trial.suggest_int(name, lo, hi) for name, (lo, hi) in space.items()}
            model = forest(kind, params, seed).fit(X_train[fit], y_train[fit])
            return score(y_train[val], model.predict(X_train[val]))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials)
        model = forest(kind, study.best_params, seed).fit(X_train, y_train)
        fit_predict.params.append(study.best_params)
        fit_predict.models.append(model)
        return model.predict(X_test)

    fit_predict.params, fit_predict.models = [], []
    return fit_predict


def median_params(params):
    """Per-parameter median (rounded) of a list of integer hyperparameter dicts."""
    return {k: int(round(np.median([p[k] for p in params]))) for k in params[0]}


def evaluate(name, X, y, session_ids, decoder, kind, n_permutations=0, null_decoder=None, n_splits=5, seed=0):
    """
    Mouse-grouped cross-validation of `decoder` on units from `session_ids` (one per row of X), and
    optionally the within-session permutation null. The null refits null_decoder, by default a forest
    with the median of the hyperparameters the tuned decoder chose (tuning on shuffled labels
    cannot find structure, so it is left out to save time).
    Returns the cross_validate dict plus groups (mouse ids), null, p and the printed summary line.
    """
    session_ids = np.asarray(session_ids)
    groups = mouse_ids(session_ids)
    stratify = kind == 'classifier'
    cv = cross_validate(X, y, groups, decoder, SCORES[kind], n_splits, seed, stratify)
    null = p = None
    if n_permutations:
        null_decoder = null_decoder or fixed_forest(kind, median_params(decoder.params))
        null, p = permutation_test(X, y, groups, session_ids, null_decoder, SCORES[kind], cv['fold_scores'].mean(),
                                   n_permutations, n_splits, seed, stratify)
    cv.update(groups=groups, null=null, p=p, line=summary(name, cv['fold_scores'], null, p))
    return cv


def summary(name, observed_folds, null=None, p=None):
    """One printed line: mean +/- sd over folds, and the null when there is one."""
    line = f'{name}: {observed_folds.mean():.3f} +/- {observed_folds.std():.3f} over {len(observed_folds)} mouse-grouped folds'
    if null is not None:
        line += f'; within-session label shuffle {null.mean():.3f} (95th pct {np.percentile(null, 95):.3f}), p = {p:.3g}'
    print(line, flush=True)
    return line
