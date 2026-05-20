"""Schema drift injection for MessyData.

Schema drift simulates real-world upstream data source changes:
- Column renames (e.g., 'amount' → 'amount_v2', 'total_amt')
- Column drops (nullable columns silently vanish)
- Unexpected new columns appear
- Type coercions (numeric stored as string)
- Column reordering
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from schema.schema import SchemaDriftSpecification

# Plausible rename suffixes/prefixes for realistic-looking drift
_RENAME_SUFFIXES = ["_v2", "_new", "_updated", "_migrated", "_2", "_bak", "_alt"]
_RENAME_PREFIXES = ["new_", "upd_", "legacy_"]
_EXTRA_COL_NAMES = [
    "source_system", "etl_batch_id", "ingest_ts", "pipeline_version",
    "record_hash", "__deleted", "_fivetran_synced", "raw_payload",
    "region_code", "data_center", "tenant_id", "partition_key",
]


def _rename_col(col: str, rng: np.random.Generator) -> str:
    coin = rng.integers(0, 3)
    if coin == 0:
        suffix = rng.choice(_RENAME_SUFFIXES)
        return col + suffix
    elif coin == 1:
        prefix = rng.choice(_RENAME_PREFIXES)
        return prefix + col
    else:
        # Snake-case variant
        return col.replace("_", "") if "_" in col else col + "_field"


def apply_schema_drift(
    df: pd.DataFrame,
    spec: SchemaDriftSpecification,
    rng: np.random.Generator,
    nullable_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply schema drift to df. Returns (new_df, drift_log)."""
    if rng.random() > spec.prob:
        return df, {}

    df = df.copy()
    log = {}

    # 1. Rename columns
    if spec.rename_columns:
        cols = list(df.columns)
        n_rename = max(1, int(len(cols) * spec.rename_rate))
        to_rename = rng.choice(cols, size=min(n_rename, len(cols)), replace=False)
        rename_map = {c: _rename_col(c, rng) for c in to_rename}
        # Avoid collisions
        rename_map = {k: v for k, v in rename_map.items() if v not in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)
            log["renamed"] = rename_map

    # 2. Drop nullable columns
    if spec.drop_columns and nullable_cols:
        available = [c for c in nullable_cols if c in df.columns]
        n_drop = max(1, int(len(available) * spec.drop_rate))
        to_drop = rng.choice(available, size=min(n_drop, len(available)), replace=False)
        df = df.drop(columns=list(to_drop))
        log["dropped"] = list(to_drop)

    # 3. Add unexpected columns
    if spec.add_columns:
        used = set(df.columns)
        candidates = [c for c in _EXTRA_COL_NAMES if c not in used]
        n_add = min(spec.add_count, len(candidates))
        if n_add > 0:
            new_cols = rng.choice(candidates, size=n_add, replace=False)
            for col in new_cols:
                # Generate plausible junk values
                dtype_coin = rng.integers(0, 3)
                if dtype_coin == 0:
                    df[col] = [str(rng.integers(1000, 9999)) for _ in range(len(df))]
                elif dtype_coin == 1:
                    df[col] = rng.random(len(df)).round(4)
                else:
                    df[col] = None
            log["added"] = list(new_cols)

    # 4. Type coercion (numeric column stored as string)
    if spec.change_dtype:
        numeric_cols = [c for c in df.select_dtypes(include="number").columns]
        if numeric_cols:
            col = rng.choice(numeric_cols)
            df[col] = df[col].astype(str).replace("nan", "")
            log["type_changed"] = {col: "numeric→string"}

    # 5. Reorder columns
    if spec.reorder_columns:
        cols = list(df.columns)
        rng.shuffle(cols)
        df = df[cols]
        log["reordered"] = True

    return df, log