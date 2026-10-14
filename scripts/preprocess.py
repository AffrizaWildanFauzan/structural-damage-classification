"""Memuat data lomba dan membangun fitur dasar (tanpa label).

Modul ini berisi langkah *data cleaning* dan fitur yang tidak memakai target,
sehingga aman dihitung di train dan test sekaligus.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEED = 42
GRADES = [1, 2, 3, 4, 5]
ID_COL = 'building_id'
TARGET = 'damage_grade'
CAT_COLS = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
            'other_floor_type', 'position', 'plan_configuration', 'superstructure']
NUM_RAW = ['count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq']
STRUCT = NUM_RAW + CAT_COLS
LOC_LEVELS = ['district_id', 'vdcmun_id', 'ward_id', 'p6', 'p8', 'p9', 'p10']  # kasar -> halus
WEAK_MATERIALS = ['mud_mortar_stone', 'adobe_mud', 'stone_flag', 'mud_mortar_brick', 'bamboo', 'timber']


def load_data(data_dir='data'):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    y = parse_target(train[TARGET])
    return train, test, y


def parse_target(series):
    return series.astype(str).str.replace('Grade ', '', regex=False).astype(int).values


class BuildingFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        s = d[ID_COL].astype(str)
        d['p6'] = s.str[:6].astype(np.int64)
        d['p8'] = s.str[:8].astype(np.int64)
        d['p9'] = s.str[:9].astype(np.int64)
        d['p10'] = s.str[:10].astype(np.int64)
        d['seq'] = s.str[6:10].astype(int)
        d['d11'] = s.str[10].astype(int)
        d['d12'] = s.str[11].astype(int)
        # cleaning: 999 adalah kode khusus, bukan umur sungguhan
        d['age_999'] = (d.age_building >= 999).astype(int)
        d['age_cap'] = d.age_building.where(d.age_building < 999, np.nan).clip(upper=200)
        d['area_per_floor'] = d.plinth_area_sq_ft / d.count_floors_pre_eq.clip(lower=1)
        d['height_per_floor'] = d.height_ft_pre_eq / d.count_floors_pre_eq.clip(lower=1)
        d['vol'] = d.plinth_area_sq_ft * d.height_ft_pre_eq
        d['weak_mat'] = d.superstructure.str.replace('has_superstructure_', '').isin(WEAK_MATERIALS).astype(int)
        d['skey'] = d[STRUCT].astype(str).agg('|'.join, axis=1)
        d['mat_found'] = d.superstructure + '|' + d.foundation_type
        d['mat_found_roof'] = d.mat_found + '|' + d.roof_type
        return d


def add_unsupervised_features(train, test):
    full = pd.concat([train.drop(columns=[TARGET], errors='ignore'), test], ignore_index=True)
    full = BuildingFeatures().fit_transform(full)
    for c in ['vdcmun_id', 'ward_id', 'p6', 'p9', 'p10', 'mat_found_roof']:
        full['cnt_' + c] = full[c].map(full[c].value_counts()).astype(float)
    for c in ['p10', 'p6']:
        g = full.groupby(c)
        full[f'{c}_weak'] = g.weak_mat.transform('mean')
        full[f'{c}_floors'] = g.count_floors_pre_eq.transform('mean')
        full[f'{c}_age'] = g.age_cap.transform('mean')
        full[f'{c}_area'] = g.plinth_area_sq_ft.transform('mean')
        full[f'{c}_floors_rel'] = full.count_floors_pre_eq - full[f'{c}_floors']
    n = len(train)
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True)
