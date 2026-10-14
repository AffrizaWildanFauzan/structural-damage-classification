import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from scripts.feature_engineering import LocationTargetEncoder
from scripts.preprocess import CAT_COLS, GRADES, ID_COL, SEED, TARGET, add_unsupervised_features, load_data

CAT_FEATURES = CAT_COLS + ['mat_found', 'mat_found_roof']
EXCLUDE = [ID_COL, TARGET, 'skey']
SKEWED = ['plinth_area_sq_ft', 'height_ft_pre_eq', 'vol', 'area_per_floor', 'height_per_floor']

LGBM_PARAMS = dict(n_estimators=450, learning_rate=0.05, num_leaves=127, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                   random_state=SEED, verbose=-1)


def numeric_columns(X):
    return [c for c in X.columns if c not in EXCLUDE + CAT_FEATURES]


def categorical_columns(X):
    return [c for c in CAT_FEATURES if c in X.columns]


def _is_skewed(c):
    return c in SKEWED or c.startswith('cnt_') or c.endswith('_n')


def skewed_columns(X):
    return [c for c in numeric_columns(X) if _is_skewed(c)]


def unskewed_columns(X):
    return [c for c in numeric_columns(X) if not _is_skewed(c)]


class ColumnSubset:
    def __init__(self, base, keep=None):
        self.base = base
        self.keep = keep

    def __call__(self, X):
        cols = self.base(X)
        return cols if self.keep is None else [c for c in cols if c in set(self.keep)]


def signed_log1p(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor(kind='tree', keep=None):
    num = ColumnSubset(numeric_columns, keep)
    cat = ColumnSubset(categorical_columns, keep)
    if kind == 'tree':
        ct = ColumnTransformer([
            ('num', 'passthrough', num),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                   encoded_missing_value=-1), cat),
        ], verbose_feature_names_out=False)
    else:
        skew = ColumnSubset(skewed_columns, keep)
        rest = ColumnSubset(unskewed_columns, keep)
        ct = ColumnTransformer([
            ('skew', Pipeline([('imp', SimpleImputer(strategy='median')),
                               ('log', FunctionTransformer(signed_log1p, feature_names_out='one-to-one')),
                               ('sc', StandardScaler())]), skew),
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), rest),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50, sparse_output=False), cat),
        ], verbose_feature_names_out=False)
    return ct.set_output(transform='pandas')


class ClassWeightedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, class_weights=None):
        """Menyimpan parameter tanpa validasi (konvensi scikit-learn)."""
        self.estimator = estimator
        self.class_weights = class_weights

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.le_.transform(y))
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        w = np.ones(len(self.classes_)) if self.class_weights is None else np.asarray(self.class_weights)
        return self.classes_[(self.predict_proba(X) * w).argmax(1)]


def build_pipeline(model, kind='tree', keep=None):
    return Pipeline([
        ('target_features', LocationTargetEncoder()),
        ('columns', build_preprocessor(kind, keep)),
        ('model', ClassWeightedClassifier(model)),
    ])


def macro_f1(y_true, y_pred):
    cm = np.bincount((np.asarray(y_true) - 1) * 5 + (np.asarray(y_pred) - 1), minlength=25).reshape(5, 5)
    tp = np.diag(cm); denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0)))


def fit_class_weights(proba, y, iters=3, grid=np.linspace(0.4, 3.0, 27)):
    w = np.ones(5)
    for _ in range(iters):
        for k in range(5):
            sc = [macro_f1(y, (proba * np.r_[w[:k], v, w[k + 1:]]).argmax(1) + 1) for v in grid]
            w[k] = grid[int(np.argmax(sc))]
    return w / w[0]


def crossfit_macro_f1(proba, y, folds):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        pred[b] = (proba[b] * fit_class_weights(proba[a], y[a], iters=2)).argmax(1) + 1
    return macro_f1(y, pred)


def make_folds(X, y, n_splits=5):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y))


def oof_proba(pipeline, X, y, folds):
    return cross_val_predict(pipeline, X, y, cv=folds, method='predict_proba')


def tune_n_estimators(X, y, params, folds, max_rounds=3000):
    import lightgbm as lgb
    a, b = folds[0]
    enc = LocationTargetEncoder()
    Xa = enc.fit_transform(X.iloc[a], y[a]); Xb = enc.transform(X.iloc[b])
    ct = build_preprocessor('tree').fit(Xa)
    m = lgb.LGBMClassifier(**{**params, 'n_estimators': max_rounds})
    m.fit(ct.transform(Xa), y[a] - 1, eval_set=[(ct.transform(Xb), y[b] - 1)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return int(m.best_iteration_)


def main():
    import lightgbm as lgb
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--model-out', default='models/pipeline_lgbm.pkl')
    p.add_argument('--n-estimators', type=int, default=LGBM_PARAMS['n_estimators'])
    a = p.parse_args()
    t0 = time.time()
    train, test, y = load_data(a.data_dir)
    X, _ = add_unsupervised_features(train, test)
    folds = make_folds(X, y)
    pipe = build_pipeline(lgb.LGBMClassifier(**{**LGBM_PARAMS, 'n_estimators': a.n_estimators}))
    P = oof_proba(pipe, X, y, folds)
    w = fit_class_weights(P, y)
    print(f'OOF Macro F1 argmax={macro_f1(y, P.argmax(1) + 1):.5f} crossfit={crossfit_macro_f1(P, y, folds):.5f} w={np.round(w, 3)}')
    pipe.set_params(model__class_weights=w).fit(X, y)
    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, a.model_out, compress=3)
    print(f'tersimpan: {a.model_out} ({(time.time() - t0) / 60:.1f} menit)')

main()