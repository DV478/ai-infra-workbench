"""Pretty display names for internal IDs.

No user-facing text in Build 1 should ever contain a raw id like
``h100_sxm5_80gb`` or ``gpu_price``. The IDs stay in tool-call JSON and
engine code only — the UI always gets the human name.
"""

from __future__ import annotations

from core.gpu_specs import GpuSpec, load_catalog
from core.gpu_specs.catalog import Catalog
from core.market_intel import FactKind


def gpu_display_name(gpu_id: str, catalog: Catalog | None = None) -> str:
    """Return the catalog's friendly ``name`` for a given GPU id.

    Falls back to the raw id if lookup fails, so a missing row never
    blanks the UI (and the id itself tells us what to fix).
    """
    cat = catalog if catalog is not None else load_catalog()
    try:
        return cat.by_id(gpu_id).name
    except KeyError:
        return gpu_id


def gpu_options(catalog: Catalog | None = None) -> dict[str, str]:
    """Mapping of ``{pretty_name: gpu_id}`` for ``st.selectbox``.

    Ordered by vendor, then release year, then name — so related GPUs
    cluster and newer rows are close to older ones in the same family.
    """
    cat = catalog if catalog is not None else load_catalog()
    sorted_gpus = sorted(cat.gpus, key=lambda g: (g.vendor, g.release_year, g.name))
    return {g.name: g.id for g in sorted_gpus}


def gpu_pretty_to_id(pretty_name: str, catalog: Catalog | None = None) -> str:
    """Inverse of :func:`gpu_display_name` for widget ↔ engine round-tripping."""
    cat = catalog if catalog is not None else load_catalog()
    for g in cat.gpus:
        if g.name == pretty_name:
            return g.id
    raise KeyError(f"No GPU with pretty name {pretty_name!r}")


_FACT_KIND_LABELS: dict[FactKind, str] = {
    FactKind.GPU_PRICE: "GPU price",
    FactKind.GPU_SHIPMENT: "GPU shipment",
    FactKind.CAPEX: "Capex",
    FactKind.CAPACITY: "Capacity",
    FactKind.CONTRACT: "Contract",
    FactKind.ROADMAP: "Roadmap",
}


def fact_kind_display(kind: FactKind | str) -> str:
    """``'gpu_price'`` or ``FactKind.GPU_PRICE`` → ``'GPU price'``."""
    if isinstance(kind, str):
        try:
            kind = FactKind(kind)
        except ValueError:
            return kind
    return _FACT_KIND_LABELS.get(kind, str(kind))


# Model IDs used by the Advisor picker (M6) and Cost tracker page (M7).
# Keep this list tight — two options, confirmed with owner 2026-11-08.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "gpt-4o-mini": "OpenAI gpt-4o-mini",
    "claude-haiku-5-5": "Claude Haiku 5.5",
}


def model_display_name(model_id: str) -> str:
    """Pretty label for a model id. Falls back to the raw id."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def model_backend(model_id: str) -> str:
    """Infer backend from model id prefix: ``'openai'`` or ``'anthropic'``."""
    if model_id.startswith("gpt") or model_id.startswith("o"):
        return "openai"
    if model_id.startswith("claude"):
        return "anthropic"
    raise ValueError(f"Unknown model id prefix for backend detection: {model_id!r}")


def _tolerant_gpu_lookup(catalog: Catalog, gpu_id: str) -> GpuSpec | None:
    """Silent lookup helper — returns None instead of raising. For use in
    rendering paths where missing rows should fall through, not crash."""
    try:
        return catalog.by_id(gpu_id)
    except KeyError:
        return None
