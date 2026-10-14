"""Memuat pipeline terlatih dan membuat file submission.

Contoh pemakaian dari terminal::

    python -m scripts.predict --data-dir data --model models/pipeline_lgbm.pkl --out submissions/submission.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from scripts.preprocess import GRADES, ID_COL, TARGET, add_unsupervised_features, load_data


def make_submission(pipeline, train, test):
    _, X_test = add_unsupervised_features(train, test)
    pred = pipeline.predict(X_test).astype(int)
    sub = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: pred})
    validate_submission(sub, test)
    return sub


def validate_submission(sub, test):
    assert len(sub) == len(test), 'jumlah baris berbeda dengan test'
    assert sub[ID_COL].is_unique, 'building_id duplikat'
    assert (sub[ID_COL].values == test[ID_COL].values).all(), 'urutan building_id berubah'
    assert set(sub[TARGET].unique()) <= set(GRADES), 'grade di luar 1..5'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--model', default='models/pipeline_lgbm.pkl')
    p.add_argument('--out', default='submissions/submission.csv')
    a = p.parse_args()
    train, test, _ = load_data(a.data_dir)
    sub = make_submission(joblib.load(a.model), train, test)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(a.out, index=False)
    print(f'tersimpan: {a.out} {sub.shape}')
    print(sub[TARGET].value_counts(normalize=True).sort_index().round(4))


main()
