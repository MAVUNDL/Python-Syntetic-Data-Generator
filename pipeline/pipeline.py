from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Optional

from schema.schema import DatasetSchema, FieldSpecification
from distribution.distributions import Sequential, WeightedChoiceMapping
from anomalities.anomalities import inject_anomalies
from schema_drift.drift import apply_schema_drift
 
""" This class runs the whole pipeline generation of synthetic data """
class Pipeline:
    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    @classmethod
    def from_config(cls, path: str) -> "Pipeline":
        return cls(DatasetSchema.from_yaml(path))
    
    """ Internal generation """
 
    def _generate_groups(self, n_rows: int, rng: np.random.Generator, fixed_date: Optional[str] = None) -> pd.DataFrame:
        schema = self.schema
        rpk = schema.records_per_primary_key
 
        # Determine number of primary key groups
        if rpk is not None:
            # Sample group sizes until we reach ~n_rows
            groups = []
            total = 0
            g_idx = 0
            while total < n_rows:
                size = max(1, int(round(rpk.sample(1, rng)[0])))
                groups.append(size)
                total += size
                g_idx += 1
        else:
            groups = [1] * n_rows
 
        n_groups = len(groups)
        records = []
 
        for g_idx, g_size in enumerate(groups):
            row_data: dict = {}
 
            for fs in schema.fields:
                dist = fs.distribution
 
                # Handle WeightedChoiceMapping — emits multiple columns
                if isinstance(dist, WeightedChoiceMapping):
                    if fs.unique_per_id:
                        mapping = dist.sample(1, rng)
                        for col, arr in mapping.items():
                            row_data[col] = np.full(g_size, arr[0])
                    else:
                        mapping = dist.sample(g_size, rng)
                        for col, arr in mapping.items():
                            row_data[col] = arr
                    continue
 
                # Handle Sequential with date override
                if isinstance(dist, Sequential):
                    if fixed_date is not None and fs.temporal:
                        row_data[fs.name] = np.full(g_size, fixed_date, dtype=object)
                    else:
                        row_data[fs.name] = dist.sample_for_group(g_idx, g_size)
                    continue
 
                # unique_per_id: sample once per group
                if fs.unique_per_id:
                    val = dist.sample(1, rng)[0]
                    row_data[fs.name] = np.full(g_size, val)
                else:
                    row_data[fs.name] = dist.sample(g_size, rng)
 
            records.append(pd.DataFrame(row_data))
 
        df = pd.concat(records, ignore_index=True)
        return df
 
    def _cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for fs in self.schema.fields:
            # WeightedChoiceMapping expands to multiple cols — skip placeholder
            from distribution.distributions import WeightedChoiceMapping
            if isinstance(fs.distribution, WeightedChoiceMapping):
                for col in fs.distribution.columns:
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(object)
                        except Exception:
                            pass
                continue
            if fs.name not in df.columns:
                continue
            try:
                df[fs.name] = df[fs.name].astype(fs.dtype)
            except Exception:
                pass
        return df
 
    def _nullable_cols(self) -> list[str]:
        result = []
        for fs in self.schema.fields:
            from distribution.distributions import WeightedChoiceMapping
            if isinstance(fs.distribution, WeightedChoiceMapping):
                continue
            if fs.nullable:
                result.append(fs.name)
        return result
 
    def _finalize(self, df: pd.DataFrame, rng: np.random.Generator, seed: Optional[int] = None) -> tuple[pd.DataFrame, dict]:
        df = self._cast_dtypes(df)
        df = inject_anomalies(df, self.schema.anomalies, rng)
        drift_log = {}
        if self.schema.schema_drift is not None:
            df, drift_log = apply_schema_drift(df, self.schema.schema_drift, rng, self._nullable_cols())
        return df, drift_log
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def run(self, n_rows: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = self._generate_groups(n_rows, rng)
        df, _ = self._finalize(df, rng, seed)
        return df
 
    def run_for_date(self, date_str: str, n_rows: int = 500, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = self._generate_groups(n_rows, rng, fixed_date=date_str)
        df, _ = self._finalize(df, rng, seed)
        return df
 
    def run_date_range(self, start: str, end: str, rows_per_day: int = 500, seed: Optional[int] = None) -> pd.DataFrame:
        d0 = date.fromisoformat(start)
        d1 = date.fromisoformat(end)
        all_dfs = []
        day_offset = 0
        d = d0
        while d <= d1:
            day_seed = None if seed is None else seed + day_offset
            rng = np.random.default_rng(day_seed)
            df = self._generate_groups(rows_per_day, rng, fixed_date=str(d))
            df, _ = self._finalize(df, rng, day_seed)
            all_dfs.append(df)
            d += timedelta(days=1)
            day_offset += 1
        return pd.concat(all_dfs, ignore_index=True)
 
    def run_with_drift_log(self, n_rows: int = 1000, seed: Optional[int] = None) -> tuple[pd.DataFrame, dict]:
        """Like run() but also returns the drift log."""
        rng = np.random.default_rng(seed)
        df = self._generate_groups(n_rows, rng)
        df, drift_log = self._finalize(df, rng, seed)
        return df, drift_log