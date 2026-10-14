"""Fitur berbasis label yang fold-safe, dibungkus sebagai transformer scikit-learn.

``LocationTargetEncoder`` bekerja seperti ``sklearn.preprocessing.TargetEncoder``:

* ``fit_transform(X, y)`` (dipanggil ``Pipeline.fit``) memakai *cross-fitting* internal,
  sehingga fitur setiap baris latih dihitung tanpa melihat labelnya sendiri;
* ``transform(X)`` (dipanggil ``Pipeline.predict``) memakai statistik dari seluruh data latih.

Dengan begitu pipeline yang sama bisa dipakai di ``cross_val_predict`` (validasi jujur)
dan untuk prediksi test.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from scripts.preprocess import GRADES, ID_COL, LOC_LEVELS, SEED

INTERACTIONS = [('p6', 'superstructure'), ('ward_id', 'mat_found_roof'),
                ('p10', 'weak_mat'), ('vdcmun_id', 'skey')]


def hash_keys(values):
    """Mengubah kunci (string/angka) menjadi hash uint64 agar tabel statistik kecil saat disimpan.

    Parameters
    ----------
    values : array-like or pd.Series
        Kunci yang akan di-hash.

    Returns
    -------
    np.ndarray
        Hash uint64 per baris.
    """
    return pd.util.hash_pandas_object(pd.Series(values).astype(str), index=False).values


def _sum_count(keys, v):
    return pd.DataFrame({'k': keys, 'v': v}).groupby('k').v.agg(['sum', 'count'])


def _lookup_mean(table, keys, prior, m):
    s = np.nan_to_num(table['sum'].reindex(keys).values)
    n = np.nan_to_num(table['count'].reindex(keys).values)
    return (s + m * prior) / (n + m), n


def _knn(ids_sorted, w6_sorted, vals_sorted, q_ids, q_w6, kmax):
    pos = np.searchsorted(ids_sorted, q_ids)
    cy, cd = [], []
    for j in range(1, kmax + 1):
        for idx in (pos - j, pos + j - 1):
            idc = np.clip(idx, 0, len(ids_sorted) - 1)
            ok = (idx >= 0) & (idx < len(ids_sorted)) & (w6_sorted[idc] == q_w6)
            cy.append(np.where(ok, vals_sorted[idc], np.nan))
            cd.append(np.where(ok, np.abs(ids_sorted[idc] - q_ids).astype(float), np.inf))
    Y_raw = np.stack(cy, 1); D = np.stack(cd, 1)
    o = np.argsort(D, 1)
    return Y_raw, np.take_along_axis(Y_raw, o, 1), np.take_along_axis(D, o, 1)


class LocationTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=5, smoothing=20, random_state=SEED):
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def _fit_state(self, X, y):
        # label dipetakan ke 1..5 (cross_val_predict bisa mengirim label 0..4)
        y = np.asarray(y); y = np.searchsorted(np.unique(y), y) + 1; yf = y.astype(float)
        st = {'prior': np.bincount(y - 1, minlength=5) / len(y), 'gm': yf.mean(), 'dist': {}, 'res': {}, 'inter': {}}
        oh = pd.DataFrame(np.eye(5)[y - 1], columns=GRADES)
        for c in LOC_LEVELS:
            st['dist'][c] = oh.groupby(X[c].values).sum()
        # kerentanan struktural dan residu
        hm, hs = hash_keys(X.mat_found), hash_keys(X.skey)
        st['mat'] = _sum_count(hm, yf); st['skey'] = _sum_count(hs, yf)
        par, _ = _lookup_mean(st['mat'], hm, st['gm'], 30)
        vul, _ = _lookup_mean(st['skey'], hs, par, 30)
        r = yf - vul
        for c in LOC_LEVELS:
            st['res'][c] = _sum_count(X[c].values, r)
        for a, b in INTERACTIONS:
            st['inter'][(a, b)] = _sum_count(hash_keys(X[a].astype(str) + '#' + X[b].astype(str)), yf)
        o = np.argsort(X[ID_COL].values)
        st['ids'] = X[ID_COL].values[o]; st['w6'] = X.p6.values[o]
        st['y'] = yf[o]; st['r'] = r[o]
        return st

    def _apply(self, st, X):
        with warnings.catch_warnings():   # nanmean pada baris tanpa tetangga -> NaN (disengaja)
            warnings.simplefilter('ignore', RuntimeWarning)
            return self._apply_inner(st, X)

    def _apply_inner(self, st, X):
        out = {}; m = self.smoothing
        prior = np.tile(st['prior'], (len(X), 1))
        for c in LOC_LEVELS:
            G = np.nan_to_num(st['dist'][c].reindex(X[c].values).values)
            n = G.sum(1)
            p = (G + m * prior) / (n + m)[:, None]
            for k in range(5):
                out[f'te_{c}_g{k+1}'] = p[:, k]
            out[f'te_{c}_mean'] = p @ np.arange(1, 6)
            out[f'te_{c}_n'] = n
            prior = p
        q, qw = X[ID_COL].values, X.p6.values
        Yraw, Ys, Ds = _knn(st['ids'], st['w6'], st['y'], q, qw, 16)
        for k in (1, 2, 4, 8):
            out[f'nb{k}_mean'] = Ys[:, 0] if k == 1 else np.nanmean(Ys[:, :k], 1)
        kosong = np.isnan(Ys[:, :8]).all(1)
        for g in GRADES:
            out[f'nb8_frac{g}'] = np.where(kosong, np.nan, np.nanmean(Ys[:, :8] == g, 1))
        out['nb1_dist'] = np.where(np.isinf(Ds[:, 0]), np.nan, np.log1p(Ds[:, 0]))
        out['nb4_dist'] = np.where(np.isinf(Ds[:, 3]), np.nan, np.log1p(Ds[:, 3]))
        out['nb_prev'] = Yraw[:, 0]; out['nb_next'] = Yraw[:, 1]
        out['nb16_mean'] = np.nanmean(Ys, 1)
        # kerentanan, residu lokasi, residu tetangga
        par, _ = _lookup_mean(st['mat'], hash_keys(X.mat_found), st['gm'], 30)
        out['vuln'], _ = _lookup_mean(st['skey'], hash_keys(X.skey), par, 30)
        rp = np.zeros(len(X))
        for c in LOC_LEVELS:
            rp, _ = _lookup_mean(st['res'][c], X[c].values, rp, 5)
            out[f'res_{c}'] = rp
        _, Yr, Dr = _knn(st['ids'], st['w6'], st['r'], q, qw, 16)
        for k in (2, 4, 8, 16):
            out[f'resnb{k}'] = np.nanmean(Yr[:, :k], 1)
        w = np.where(np.isinf(Dr), 0, 1 / (1 + np.log1p(np.where(np.isinf(Dr), 0, Dr))))
        out['resnb_w'] = np.nansum(np.nan_to_num(Yr) * w, 1) / np.maximum(w.sum(1), 1e-9)
        out['expected'] = out['vuln'] + out['res_p10']
        for a, b in INTERACTIONS:
            v, n = _lookup_mean(st['inter'][(a, b)], hash_keys(X[a].astype(str) + '#' + X[b].astype(str)), st['gm'], 10)
            out[f'te_{a}_x_{b}'] = v; out[f'te_{a}_x_{b}_n'] = n
        return pd.DataFrame(out, index=X.index)

    def fit(self, X, y):
        self.state_ = self._fit_state(X, np.asarray(y))
        self.feature_names_ = list(self._apply(self.state_, X.head(2)).columns)
        return self

    def transform(self, X):
        X = X.reset_index(drop=True)
        return pd.concat([X, self._apply(self.state_, X)], axis=1)

    def fit_transform(self, X, y, **fit_params):
        X = X.reset_index(drop=True); y = np.asarray(y)
        parts = []
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for a, b in skf.split(X, y):
            parts.append(self._apply(self._fit_state(X.iloc[a], y[a]), X.iloc[b]))
        F = pd.concat(parts).loc[X.index]
        self.fit(X, y)
        return pd.concat([X, F], axis=1)
