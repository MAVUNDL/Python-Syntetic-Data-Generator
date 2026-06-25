import pandas as pd
import numpy as np
#from __future__ import annotations
from typing import List
from schema.schema import AnomalySpecification

""" These methods define anomaly injection techniques """

def inject_anomalies(df: pd.DataFrame, anomalies: List[AnomalySpecification], rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for anomaly in anomalies:
        if rng.random() > anomaly.prob:
            continue
        name = anomaly.name
        if name == "missing_values":
            _missing_values(df, anomaly, rng)
        elif name == "duplicate_values":
            df = _duplicate_values(df, anomaly, rng)
        elif name == "invalid_category":
            _invalid_category(df, anomaly, rng)
        elif name == "invalid_date":
            _invalid_date(df, anomaly, rng)
        elif name == "outliers":
            _outliers(df, anomaly, rng)
    return df


def _target_columns(df, anomaly):
    if anomaly.columns == "any":
        return list(df.columns)
    return [c for c in anomaly.columns if c in df.columns]


def _missing_values(df, anomaly, rng):
    cols = _target_columns(df, anomaly)
    for col in cols:
        mask = rng.random(len(df)) < anomaly.rate
        df.loc[mask, col] = np.nan


def _duplicate_values(df, anomaly, rng):
    n = max(1, int(len(df) * anomaly.rate))
    idx = rng.choice(len(df), size=n, replace=False)
    dupes = df.iloc[idx].copy()
    return pd.concat([df, dupes], ignore_index=True)


def _invalid_category(df, anomaly, rng):
    cols = _target_columns(df, anomaly)
    for col in cols:
        mask = rng.random(len(df)) < anomaly.rate
        df.loc[mask, col] = "INVALID"


def _invalid_date(df, anomaly, rng):
    cols = _target_columns(df, anomaly)
    for col in cols:
        mask = rng.random(len(df)) < anomaly.rate
        df.loc[mask, col] = "9999-99-99"


def _outliers(df, anomaly, rng):
    if anomaly.distribution is None:
        return
    cols = _target_columns(df, anomaly)
    for col in cols:
        if col not in df.columns:
            continue
        mask = rng.random(len(df)) < anomaly.rate
        n = mask.sum()
        if n > 0:
            vals = anomaly.distribution.sample(n, rng)
            # Upcast column to float to safely accept outlier values
            try:
                df[col] = df[col].astype(float)
                df.loc[mask, col] = vals
            except Exception:
                pass