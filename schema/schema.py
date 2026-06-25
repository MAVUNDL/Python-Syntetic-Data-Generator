import numpy as np
from dataclasses import dataclass, field
from distribution.distributions import Distribution
from typing import Any, Optional,List

""" This class defines a column and the type of data it represents """

@dataclass
class FieldSpecification:
    name: str
    distribution: Distribution
    dtype: str = "object"
    unique_per_id: bool = False
    nullable: bool = True
    temporal: bool = False

""" This class defines the type on anomaly to be applied on the set of columns and how it will be applied"""

@dataclass
class AnomalySpecification:
    name: str
    prob: float
    rate: float
    columns: Any = "any"
    distribution: Optional[Distribution] = None

""" This class defines a schema drift to be applied to the dataset """

@dataclass
class SchemaDriftSpecification:
    prob: float = 0.1          # probability that drift fires on a given run/date
    # Which drift types to enable
    rename_columns: bool = True      # randomly rename a column (suffix _v2, etc.)
    drop_columns: bool = True        # randomly drop a nullable column
    add_columns: bool = True         # randomly add an unexpected column
    change_dtype: bool = True        # cast a numeric column to string (type change)
    reorder_columns: bool = True     # shuffle column order
    # Rates
    rename_rate: float = 0.15       # fraction of columns to rename
    drop_rate: float = 0.1          # fraction of nullable columns to drop
    add_count: int = 1              # number of extra columns to add


@dataclass
class DatasetSchema:
    name: str
    fields: List[FieldSpecification]
    primary_key: str = "id"
    records_per_primary_key: Distribution = None
    anomalies: List[AnomalySpecification] = field(default_factory=list)
    schema_drift: Optional[SchemaDriftSpecification] = None
 
    @classmethod
    def from_dict(cls, SchemaConfig: dict) -> "DatasetSchema":
        from distribution.distributions import distribution_factory
 
        fields = []
        for field in SchemaConfig["fields"]:
            dist = distribution_factory(field["distribution"])
            fields.append(FieldSpecification(
                name=field["name"],
                distribution=dist,
                dtype=field.get("dtype", "object"),
                unique_per_id=field.get("unique_per_id", False),
                nullable=field.get("nullable", True),
                temporal=field.get("temporal", False),
            ))
 
        anomalies = []
        for anomaly in SchemaConfig.get("anomalies", []):
            dist = None
            if "distribution" in anomaly:
                dist = distribution_factory(anomaly["distribution"])
            anomalies.append(AnomalySpecification(
                name=anomaly["name"],
                prob=anomaly["prob"],
                rate=anomaly["rate"],
                columns=anomaly.get("columns", "any"),
                distribution=dist,
            ))
 
        rpk = distribution_factory(SchemaConfig["records_per_primary_key"]) if "records_per_primary_key" in SchemaConfig else None
 
        drift = None
        if "schema_drift" in SchemaConfig:
            schemaDrift = SchemaConfig["schema_drift"]
            drift = SchemaDriftSpecification(**schemaDrift)
 
        return cls(
            name=SchemaConfig["name"],
            primary_key=SchemaConfig.get("primary_key", "id"),
            records_per_primary_key=rpk,
            fields=fields,
            anomalies=anomalies,
            schema_drift=drift,
        )
 
    @classmethod
    def from_yaml(cls, path: str) -> "DatasetSchema":
        import yaml
        with open(path) as f:
            SchemaConfig = yaml.safe_load(f)
        return cls.from_dict(SchemaConfig)