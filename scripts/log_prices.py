"""Weekly GPU price logger.

Appends rows to ``data/prices/prices.parquet``. The file is the project's
one piece of price history — git-committed by the GitHub Actions cron that
invokes this script every Monday at 09:00 UTC.

Policy (see memory/feedback_price_tracker.md):
- APIs only where they exist (Vast.ai public search).
- Manual entry for providers without APIs (Lambda, CoreWeave, Nebius, Nscale).
- No HTML scraping.

Usage:
    python scripts/log_prices.py                 # run all API-based collectors
    python scripts/log_prices.py --manual        # interactive manual entry
    python scripts/log_prices.py --dry-run       # print, don't write

The manual-entry flow prompts for each provider the user chose to track.
In CI the cron runs without ``--manual``; manual entry is for the owner's
Monday 5-minute log.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

PRICES_PATH = Path(__file__).resolve().parent.parent / "data" / "prices" / "prices.parquet"

# GPU model strings we track. Keep aligned with gpu_specs.yaml ids so joins
# downstream are easy.
TRACKED_GPUS = {
    "h100_sxm5_80gb",
    "h100_pcie_80gb",
    "h200_sxm_141gb",
    "b100",
    "b200",
    "gb200",
    "b300",
    "a100_sxm_80gb",
    "l40s",
}


@dataclass
class PricePoint:
    retrieved_at: str  # ISO UTC
    provider: str
    gpu_id: str
    gpu_name_raw: str  # as advertised by the provider
    hourly_usd: float
    region: str | None
    commitment: str  # 'on_demand' | 'reserved_1y' | 'reserved_3y' | 'spot' | 'contract'
    source_kind: str  # 'api' | 'manual' | 'published_page'
    source_url: str | None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Vast.ai — public search API
# ---------------------------------------------------------------------------

VAST_SEARCH_URL = "https://console.vast.ai/api/v0/bundles/"

# Vast describes GPUs by its own strings; map to our ids.
VAST_GPU_MAP = {
    "H100 SXM5": "h100_sxm5_80gb",
    "H100 PCIE": "h100_pcie_80gb",
    "H100 NVL": "h100_pcie_80gb",
    "H200 SXM": "h200_sxm_141gb",
    "H200 SXM5": "h200_sxm_141gb",
    "B200": "b200",
    "B100": "b100",
    "A100 SXM4": "a100_sxm_80gb",
    "A100 PCIE": "a100_sxm_80gb",
    "L40S": "l40s",
}


def fetch_vast_prices(timeout: float = 15.0) -> list[PricePoint]:
    """Query Vast.ai search for on-demand listings and keep the minimum
    hourly rate per GPU model. Vast marketplace prices move quickly — this
    is a best-current-offer snapshot, not a contract rate."""
    try:
        r = httpx.get(VAST_SEARCH_URL, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[vast] fetch failed: {e}", file=sys.stderr)
        return []

    now = datetime.now(UTC).isoformat()
    offers = data.get("offers", []) if isinstance(data, dict) else data
    best_per_gpu: dict[str, PricePoint] = {}

    for o in offers:
        gpu_raw = (o.get("gpu_name") or "").strip()
        gpu_id = VAST_GPU_MAP.get(gpu_raw)
        if not gpu_id:
            continue
        # Vast returns $/hr per host; divide by num_gpus for per-GPU rate.
        num = o.get("num_gpus") or 1
        host_hr = o.get("dph_total") or o.get("dph_base")
        if not host_hr or num == 0:
            continue
        per_gpu_hr = float(host_hr) / float(num)
        pp = PricePoint(
            retrieved_at=now,
            provider="vast_ai",
            gpu_id=gpu_id,
            gpu_name_raw=gpu_raw,
            hourly_usd=per_gpu_hr,
            region=o.get("geolocation") or None,
            commitment="on_demand",
            source_kind="api",
            source_url=VAST_SEARCH_URL,
            notes="min offer across marketplace at retrieval time",
        )
        if gpu_id not in best_per_gpu or pp.hourly_usd < best_per_gpu[gpu_id].hourly_usd:
            best_per_gpu[gpu_id] = pp

    return list(best_per_gpu.values())


# ---------------------------------------------------------------------------
# Manual entry — the 5-minute Monday log
# ---------------------------------------------------------------------------

MANUAL_PROVIDERS = ["lambda", "coreweave", "nebius", "nscale", "runpod"]


def collect_manual(providers: list[str]) -> list[PricePoint]:
    """Prompt-driven manual entry. Press enter to skip a provider or GPU."""
    now = datetime.now(UTC).isoformat()
    rows: list[PricePoint] = []
    print("\nManual price log. Enter per-GPU USD/hr list price. Blank skips.\n")
    for provider in providers:
        print(f"--- {provider} ---")
        source_url = input("  source URL (pricing page) [enter to skip provider]: ").strip()
        if not source_url:
            continue
        for gpu_id in sorted(TRACKED_GPUS):
            raw = input(f"  {gpu_id} on-demand $/hr [skip]: ").strip()
            if not raw:
                continue
            try:
                hourly = float(raw)
            except ValueError:
                print(f"    skipped {gpu_id}: invalid number")
                continue
            region = input("    region (optional): ").strip() or None
            note = input("    note (optional): ").strip() or None
            rows.append(
                PricePoint(
                    retrieved_at=now,
                    provider=provider,
                    gpu_id=gpu_id,
                    gpu_name_raw=gpu_id,
                    hourly_usd=hourly,
                    region=region,
                    commitment="on_demand",
                    source_kind="manual",
                    source_url=source_url,
                    notes=note,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def append_rows(rows: list[PricePoint], path: Path = PRICES_PATH) -> int:
    if not rows:
        return 0
    df_new = pd.DataFrame([asdict(r) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manual",
        action="store_true",
        help="run the manual-entry flow for non-API providers",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rows that would be written and exit",
    )
    args = ap.parse_args()

    collected: list[PricePoint] = []
    collected.extend(fetch_vast_prices())

    if args.manual:
        collected.extend(collect_manual(MANUAL_PROVIDERS))

    if not collected:
        print("No rows collected.", file=sys.stderr)
        return 0

    if args.dry_run:
        for r in collected:
            print(asdict(r))
        return 0

    n = append_rows(collected)
    print(f"appended {n} rows to {PRICES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
