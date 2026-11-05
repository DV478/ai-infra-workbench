from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

_YAML_PATH = Path(__file__).parent / "data" / "gpu_specs.yaml"


Vendor = Literal["nvidia", "google", "aws"]


class GpuSpec(BaseModel):
    """One row in the catalog. Numeric fields are DENSE tensor-core figures.

    All fields that end in a unit (``_tflops_dense``, ``_bandwidth_gbs``,
    ``_watts``, ``_gb``) are the measured quantity in that unit.
    """

    id: str
    name: str
    vendor: Vendor
    architecture: str
    release_year: int
    form_factor: str

    fp16_tflops_dense: float
    fp8_tflops_dense: float | None = None
    int8_tops_dense: float | None = None

    memory_gb: int
    memory_type: str
    memory_bandwidth_gbs: float

    tdp_watts: float | None = None
    nvlink_bandwidth_gbs: float | None = None
    pcie_gen: int | None = None

    confidence: Literal["high", "medium", "low"]
    citation: str
    citation_date: date
    notes: str | None = None

    # Convenience properties -------------------------------------------------

    @property
    def peak_fp16_flops(self) -> float:
        """Peak dense FP16 FLOPs per second."""
        return self.fp16_tflops_dense * 1e12

    @property
    def peak_memory_bandwidth(self) -> float:
        """Peak HBM/GDDR bandwidth in bytes/sec."""
        return self.memory_bandwidth_gbs * 1e9


class Catalog(BaseModel):
    catalog_version: str
    schema_version: int
    gpus: list[GpuSpec]

    def by_id(self, gpu_id: str) -> GpuSpec:
        for g in self.gpus:
            if g.id == gpu_id:
                return g
        raise KeyError(f"No GPU with id={gpu_id!r}. Known: {[g.id for g in self.gpus]}")

    def ids(self) -> list[str]:
        return [g.id for g in self.gpus]

    def by_vendor(self, vendor: Vendor) -> list[GpuSpec]:
        return [g for g in self.gpus if g.vendor == vendor]


@lru_cache(maxsize=1)
def load_catalog(path: str | Path = _YAML_PATH) -> Catalog:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Catalog.model_validate(data)
