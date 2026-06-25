# Synthetic Data Generator

A declarative, reproducible dirty-data generator for testing data pipelines, quality frameworks, and observability tooling. No LLM. No API key. Pure Python.

Define your schema in YAML (or Python), run the pipeline, get a DataFrame or CSV that looks like production data went wrong — because it did.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Installation](#installation)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Schema reference](#schema-reference)
  - [DatasetSchema](#datasetschema)
  - [FieldSpec](#fieldspec)
  - [Distributions](#distributions)
  - [AnomalySpec](#anomalyspec)
  - [SchemaDriftSpec](#schemadriftspec)
- [Pipeline API](#pipeline-api)
- [Design notes](#design-notes)
- [Known limitations](#known-limitations)

---

## Why this exists

Most synthetic data generators either:

- produce clean, unrealistic data (useless for testing quality checks)
- require an LLM to generate values (slow, non-deterministic, expensive)
- only support simple random distributions with no schema-level structure

This tool generates **structurally realistic messy data** from a statistical spec: correlated columns, group-level primary keys with variable record counts, configurable anomaly injection, and probabilistic schema drift that simulates what actually happens when an upstream source quietly changes its schema.

The intended use case is regression testing data quality rules, alerting thresholds, and pipeline resilience — not generating training data for ML.

---

## Installation

**Requirements:** Python ≥ 3.9, `pandas`, `numpy`, `pyyaml`, `click`

```bash
git clone <repo>
cd messydata
pip install -e .
```

Or with uv:

```bash
uv pip install -e .
```

There are no optional extras. The package has no dependency on any LLM, cloud service, or external API.

---

## Architecture

```
messydata/
├── distributions.py   # Distribution base class + 11 concrete implementations
├── schema.py          # DatasetSchema, FieldSpec, AnomalySpec, SchemaDriftSpec
├── pipeline.py        # Pipeline — generation engine, public API entry point
├── anomalies.py       # Anomaly injection (missing, duplicates, invalid values, outliers)
├── drift.py           # Schema drift engine (rename, drop, add, type-change, reorder)
└── cli.py             # Click CLI: generate / validate / schema
examples/
└── retail_config.yaml # Full working example with drift enabled
```

**Data flow:**

```
YAML config
    │
    ▼
DatasetSchema.from_yaml()
    │  deserialises fields, anomalies, drift spec
    ▼
Pipeline._generate_groups()
    │  samples group sizes from records_per_primary_key
    │  iterates groups, samples each field per group
    │  handles unique_per_id, temporal overrides, mapping columns
    ▼
Pipeline._cast_dtypes()
    │  coerces columns to declared dtypes
    ▼
inject_anomalies()
    │  probabilistic injection: missing, duplicates, invalid values, outliers
    ▼
apply_schema_drift()       ← fires probabilistically, may be a no-op
    │  rename / drop / add columns, type-coerce, reorder
    ▼
pd.DataFrame  +  drift_log: dict
```

The RNG is a `numpy.random.Generator` (PCG64) seeded per call. Passing the same `seed` always produces identical output including identical drift decisions.

---

## Quick start

### From YAML

```python
from messydata import Pipeline

pipeline = Pipeline.from_config("examples/retail_config.yaml")

# Generate 1000 rows
df = pipeline.run(n_rows=1000, seed=42)

# Generate for a single date (temporal fields pinned to that date)
df = pipeline.run_for_date("2025-06-01", n_rows=500, seed=42)

# Generate a date range (one batch per day, seed incremented per day)
df = pipeline.run_date_range("2025-01-01", "2025-03-31", rows_per_day=500, seed=42)

# Generate and inspect what drift fired
df, drift_log = pipeline.run_with_drift_log(n_rows=1000, seed=42)
print(drift_log)
# {'renamed': {'store_id': 'storeid'}, 'dropped': ['payment_method'],
#  'added': ['raw_payload'], 'type_changed': {'storeid': 'numeric→string'},
#  'reordered': True}
```

### From Python (no YAML)

```python
from messydata import (
    Pipeline, DatasetSchema, FieldSpec, AnomalySpec, SchemaDriftSpec,
    Lognormal, Normal, WeightedChoice, WeightedChoiceMapping, Sequential,
)

schema = DatasetSchema(
    name="transactions",
    primary_key="txn_id",
    records_per_primary_key=Lognormal(mu=1.5, sigma=0.5),
    fields=[
        FieldSpec(
            name="txn_id",
            dtype="int32",
            distribution=Sequential(start=1000),
            unique_per_id=True,
            nullable=False,
        ),
        FieldSpec(
            name="amount",
            dtype="float32",
            distribution=Lognormal(mu=4.0, sigma=0.8),
            nullable=False,
        ),
        FieldSpec(
            name="channel",
            dtype="object",
            distribution=WeightedChoice(
                values=["web", "mobile", "branch"],
                weights=[0.6, 0.3, 0.1],
            ),
        ),
    ],
    anomalies=[
        AnomalySpec(name="missing_values", prob=1.0, rate=0.04),
        AnomalySpec(name="duplicate_values", prob=0.2, rate=0.01),
    ],
    schema_drift=SchemaDriftSpec(prob=0.25),
)

df, drift_log = Pipeline(schema).run_with_drift_log(n_rows=5000, seed=7)
```

---

## Schema reference

### DatasetSchema

Top-level schema object.

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | yes | Schema name (informational) |
| `primary_key` | `str` | no | Name of the PK field. Default: `"id"` |
| `records_per_primary_key` | `Distribution` | no | Distribution of group sizes. If omitted, each PK gets exactly 1 row |
| `fields` | `List[FieldSpec]` | yes | Ordered list of field definitions |
| `anomalies` | `List[AnomalySpec]` | no | Anomaly injection specs. Default: `[]` |
| `schema_drift` | `SchemaDriftSpec` | no | Drift config. If omitted, no drift is applied |

**YAML example:**

```yaml
name: orders
primary_key: order_id
records_per_primary_key:
  type: lognormal
  mu: 2.0
  sigma: 0.5
fields: [...]
anomalies: [...]
schema_drift:
  prob: 0.2
```

---

### FieldSpec

Defines a single column.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Column name |
| `distribution` | `Distribution` | — | How values are sampled |
| `dtype` | `str` | `"object"` | Pandas dtype to cast to after sampling: `int32`, `float32`, `float64`, `object`, etc. |
| `unique_per_id` | `bool` | `False` | If `True`, one value is sampled per primary key group and repeated across all rows in that group (e.g. customer city — same for all orders from the same customer) |
| `nullable` | `bool` | `True` | Marks this column as a candidate for schema drift drops. Does not add NULLs by itself — use `AnomalySpec(name="missing_values")` for that |
| `temporal` | `bool` | `False` | If `True` and the distribution is `sequential`, the column is pinned to the `fixed_date` when calling `run_for_date()` or `run_date_range()` |

> **`unique_per_id` vs per-row sampling:** `unique_per_id=True` is the correct setting for any field that is a property of the entity (customer, order, store) rather than the event (line item). Getting this wrong produces data where the same order has two different customers.

---

### Distributions

All distributions are available as Python classes and as YAML `type` strings.

#### Continuous

| Type | Parameters | Notes |
|---|---|---|
| `uniform` | `min`, `max` | Flat distribution over `[min, max)` |
| `normal` | `mean`, `std` | Gaussian. Can produce negatives — cast to `float32` and check downstream |
| `lognormal` | `mu`, `sigma` | Log of the variable is normally distributed. Good for prices, durations, counts |
| `weibull` | `a`, `scale=1.0` | Failure/reliability modelling |
| `exponential` | `scale=1.0` | Inter-arrival times |
| `beta` | `a`, `b` | Bounded `[0, 1]`. Good for rates, fractions |
| `gamma` | `shape`, `scale=1.0` | Generalises exponential |

#### Categorical

| Type | Parameters | Notes |
|---|---|---|
| `weighted_choice` | `values: list`, `weights: list[float]` | Weighted sampling with replacement. Weights are normalised automatically |
| `weighted_choice_mapping` | `columns: dict[str, list]`, `weights: list[float]` | Samples an index and maps it to multiple correlated columns simultaneously. The field `name` is a placeholder — it is not used as a column name; the column names come from `columns` |

`weighted_choice_mapping` is the right tool when two columns are structurally correlated — e.g. `product_id` and `product_name` must match. Sampling them independently would produce mismatched pairs.

```yaml
- name: product          # placeholder name, not used as a column
  distribution:
    type: weighted_choice_mapping
    columns:
      product_id:   [1001, 1002, 1003]
      product_name: [Widget, Gadget, Doohickey]
    weights: [0.5, 0.3, 0.2]
```

#### Special

| Type | Parameters | Notes |
|---|---|---|
| `sequential` | `start` (int or `"YYYY-MM-DD"`), `step=1` | Monotonically increasing. Integers count up per group; dates advance by `step` days per group |

#### Mixture

Composes multiple distributions with weighted selection. Useful for bimodal price distributions, multi-population age distributions, etc.

```yaml
distribution:
  type: mixture
  weights: [0.7, 0.3]
  components:
    - type: lognormal
      mu: 3.0
      sigma: 0.5
    - type: lognormal
      mu: 6.5
      sigma: 0.3
```

---

### AnomalySpec

Each anomaly fires independently on each `run()` call according to `prob`.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Anomaly type (see below) |
| `prob` | `float` | Probability this anomaly fires on a given run. `1.0` = always, `0.0` = never |
| `rate` | `float` | Fraction of affected rows/cells when the anomaly fires |
| `columns` | `"any"` or `list[str]` | Which columns to target. `"any"` targets all columns |
| `distribution` | `Distribution` | Only for `outliers` — the distribution used to generate outlier values |

**Anomaly types:**

| Name | Effect |
|---|---|
| `missing_values` | Sets cells to `NaN`. Applied per-cell at `rate` probability across target columns |
| `duplicate_values` | Appends `rate * n_rows` duplicate rows (randomly sampled from existing rows) |
| `invalid_category` | Replaces values in target columns with the string `"INVALID"` |
| `invalid_date` | Replaces values in target columns with the string `"9999-99-99"` |
| `outliers` | Replaces values in target columns with samples from the specified `distribution` |

**Example — layered anomalies:**

```yaml
anomalies:
  - name: missing_values
    prob: 1.0          # always inject some nulls
    rate: 0.04         # 4% of cells

  - name: duplicate_values
    prob: 0.3          # fires 30% of runs
    rate: 0.01

  - name: outliers
    prob: 0.15
    rate: 0.02
    columns: [unit_price]
    distribution:
      type: lognormal
      mu: 7.0
      sigma: 0.2
```

---

### SchemaDriftSpec

Simulates upstream source changes. When drift fires, the output DataFrame may have different columns than the schema declares — the point is to test whether your downstream pipeline handles this gracefully.

| Field | Type | Default | Description |
|---|---|---|---|
| `prob` | `float` | `0.1` | Probability drift fires on a given `run()` call |
| `rename_columns` | `bool` | `true` | Rename a fraction of columns (e.g. `amount` → `amount_v2`, `new_amount`, `amountfield`) |
| `drop_columns` | `bool` | `true` | Silently drop a fraction of nullable columns |
| `add_columns` | `bool` | `true` | Inject unexpected new columns with realistic ETL-style names (`etl_batch_id`, `raw_payload`, `_fivetran_synced`, etc.) |
| `change_dtype` | `bool` | `true` | Cast a numeric column to string — simulates a source system exporting numbers as text |
| `reorder_columns` | `bool` | `true` | Shuffle column order |
| `rename_rate` | `float` | `0.15` | Fraction of columns to rename when `rename_columns` is active |
| `drop_rate` | `float` | `0.10` | Fraction of nullable columns to drop when `drop_columns` is active |
| `add_count` | `int` | `1` | Number of unexpected columns to add when `add_columns` is active |

Only columns marked `nullable: true` are candidates for dropping. Primary key and required fields are safe.

**Inspecting drift:**

```python
df, drift_log = pipeline.run_with_drift_log(n_rows=1000, seed=42)

# drift_log is empty dict {} when drift did not fire
# When it does fire:
# {
#   'renamed':      {'store_id': 'storeid', 'amount': 'amount_v2'},
#   'dropped':      ['payment_method'],
#   'added':        ['etl_batch_id'],
#   'type_changed': {'storeid': 'numeric→string'},
#   'reordered':    True
# }
```

---

## Pipeline API

```python
class Pipeline:
    def __init__(self, schema: DatasetSchema): ...

    @classmethod
    def from_config(cls, path: str) -> Pipeline:
        """Load schema from a YAML file and construct a Pipeline."""

    def run(self, n_rows: int = 1000, seed: int | None = None) -> pd.DataFrame:
        """Generate n_rows. Drift log is silently discarded."""

    def run_with_drift_log(
        self, n_rows: int = 1000, seed: int | None = None
    ) -> tuple[pd.DataFrame, dict]:
        """Generate n_rows and return (df, drift_log)."""

    def run_for_date(
        self, date_str: str, n_rows: int = 500, seed: int | None = None
    ) -> pd.DataFrame:
        """Generate n_rows with all temporal fields pinned to date_str (YYYY-MM-DD)."""

    def run_date_range(
        self, start: str, end: str, rows_per_day: int = 500, seed: int | None = None
    ) -> pd.DataFrame:
        """
        Generate one batch per calendar day between start and end (inclusive).
        Each day gets seed + day_offset so output is reproducible but varies per day.
        Returns a single concatenated DataFrame.
        """
```

**Reproducibility:** Passing the same `seed` guarantees identical output including identical drift decisions. Omit `seed` (default `None`) for non-deterministic output.


## Design notes

**Why not use Faker or SDV?**
Faker generates plausible-looking string values (names, addresses) — useful for masking PII, not for testing statistical properties of a pipeline. SDV fits generative models to real data — useful for privacy-preserving synthesis, but requires real data as input and is overkill for pipeline testing. This tool generates data from an explicit statistical spec with no real data dependency.

**Why `records_per_primary_key` instead of just row count?**
Real transaction tables don't have one row per entity. A customer has many orders, an order has many line items. Generating flat independent rows produces a dataset where every distribution is i.i.d., which is statistically incorrect. `records_per_primary_key` models the grouping structure so aggregations (sum per customer, count per order) produce realistic distributions.

**Why `unique_per_id`?**
Without it, a field like `customer_email` would get a new random value on every row — the same customer would have a different email on each of their orders. `unique_per_id=True` samples once per group and repeats the value, which is the correct model for entity attributes.

**Why not just use Polars?**
The output type is `pd.DataFrame` because most data quality tools (Great Expectations, Soda, dbt tests) integrate with pandas natively. The internal generation uses numpy arrays and pandas only for final assembly, so performance is reasonable for the target scale (10k–1M rows).

**Seeding model:**
The pipeline uses `numpy.random.default_rng(seed)` which creates a PCG64 generator. Each `run()` call creates a fresh generator from the seed, so calling `run()` twice with the same seed is idempotent. `run_date_range()` increments the seed by the day offset so each day gets a different but reproducible stream.

---

## Known limitations

- **No Parquet output.** The CLI outputs CSV or JSON Lines. Parquet requires `pyarrow` which is not a declared dependency. Wrap the output yourself: `pipeline.run().to_parquet("out.parquet")`.
- **`WeightedChoiceMapping` field name is a placeholder.** The `name` field on a `WeightedChoiceMapping` field spec is not used as a column name. Column names come from the `columns` dict keys. This is a known API inconsistency.
- **`records_per_primary_key` is approximate.** Group sizes are sampled until the total exceeds `n_rows`, so the actual row count may be slightly over. The excess is not trimmed.
- **Schema drift does not persist across runs.** Each `run()` call makes an independent drift decision. There is no state that carries a rename from one run into the next. If you need persistent drift (simulate a source that was renamed on 2025-03-01 and stays renamed), apply the drift log manually.
- **No foreign key relationships between tables.** Each schema is independent. If you need referential integrity across multiple generated tables, generate the parent table first and sample its PKs for use as FK values in child tables.
