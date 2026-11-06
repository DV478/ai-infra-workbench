# AI Compute Economics Workbench

Three demoable builds for AI infrastructure economics: inference cost + build-vs-rent, GPU cloud account health, and GPU capacity planning. Measured, modelled and simulated data are labelled on every output.

See [AI_INFRA_PROJECT.md](AI_INFRA_PROJECT.md) for the full spec.

---

## Status

**As of 2026-11-06 — Week 1-3 workbench + dual LLM backend + expanded corpus/gold-set all pushed to `main`.** Latest commit `ecd9cbf`. Deploy live at [ai-infra-workbench.streamlit.app](https://ai-infra-workbench.streamlit.app).

- **135/135 tests pass** (23 W1 + 39 W2 + 27 W3 agent + 4 memo + 24 market intel + 14 market-intel eval + 4 dual-backend)
- Ruff clean on `core/`, `scripts/`, `tests/`, `build1_inference_calculator/`
- Dual LLM backend: `ANTHROPIC_API_KEY` accepts either `sk-ant-*` (Anthropic/Haiku 4.5) or `sk-*` (OpenAI/gpt-4o-mini); backend auto-picked from key prefix. Works in Streamlit app, scripts, and MCP server.
- Market-intel demo corpus: 8 sample docs across 4 doc types (press release, earnings transcript, investor day, news) covering CoreWeave, Nscale, Nebius, Lambda, Crusoe, Nvidia, Together. All marked `[SYNTHETIC — demo corpus]` inline.
- Market-intel gold eval set: 12 cases (up from 8) covering every fact kind + guardrails (API-pricing-isn't-gpu-price, distractors, hedges, roadmap-only).
- `python scripts/run_market_intel.py --fake` extracts **40 facts across 8 docs** to `data/market_intel.duckdb` with zero API spend.
- `python scripts/run_market_intel_eval.py --fake` — baseline 4/12 pass (33%), quote-valid 100%. Rule-based fake can't handle contracts/roadmaps; real-LLM run (owner, pending) should jump substantially.
- Vast.ai live price API confirmed working (Week 1).

---

## Rules enforced in code

- Every number in the UI carries a `source_id` referencing the `sources` DuckDB table.
- Every source row is tagged `measured | modelled | simulated | assumption` with citation + retrieval date.
- Agents never calculate figures — they call tool-wrapped functions in `core/`, which return `(value, source_id)` pairs via `Figure`.
- `data/prices/` is append-only. History only exists if the weekly logger runs — it is the one data file we commit.

---

## Key decisions (locked in)

| Area | Decision | Rationale |
|---|---|---|
| Repo shape | **Single monorepo**, Streamlit multi-page, one hosted URL | Easier solo maintenance; shared `core/` package serves all three builds |
| Sequence | **1 → 3 → 2** as per spec | No current Nscale/Nebius live application to trigger a bring-forward |
| Build 1 demo angle | **Nscale/Nebius neocloud unit economics** | Most unique to this project; both London; strongest GTM/strategy differentiator |
| Build 1 demo scale | Keep **70B at 50 rps** headline, with error bars from T4 % error + MLPerf | Models what interviews will ask about; honest error disclosure is the defence |
| GPU validation | **Kaggle T4** + **vLLM**; HF Transformers as naive baseline for a delta chart | Longer sessions than Colab, 2× T4 free; vLLM is the serving reference |
| MLPerf cross-check | Pin one round (v5.0 or v4.1), closed datacenter, Llama-2-70B + GPT-J-6B on H100 SXM | Reproducibility |
| GPU catalog scope | T4 / L4 / L40S / A100 / H100 (SXM, PCIe) / H200 / **B100 / B200 / GB200 / B300 / R100** / TPU v5e/v5p/Trillium / Trn1 / Trn2 | Current-gen as of Nov 2026, Rubin flagged `confidence: low` |
| Price logger | **APIs only** (Vast.ai, RunPod) + 5-min manual Monday log. **No scraping.** | Cleaner interview story, no silent breakage |
| CUDA matmul notebook | Ship properly (2-3 days) | Strong interview talking point if done well |
| Open-model-on-T4 priced through Build 1 | **Skip** | Validation path already carries the technical story |
| MCP vs HTTP | MCP server per build for Claude Desktop; in-process tool calls for hosted Streamlit | No inter-service hops on the hosted demo |
| Hosted agent runtime | Direct Anthropic SDK + prompt caching, Haiku 4.5 | Fits £5-10/mo budget |
| AI Investment RAG reuse | **Not vendored** — sibling repo's scrapers conflict with price-tracker policy; extractor is text-in/facts-out | Keeps "no scraping" rule intact; owner pastes docs into `data/raw/market/` |

---

## Repository layout

```
core/                           shared domain — pure Python, no Streamlit
├── sources/                    ✅ schema + DuckDB store + seed script (148 rows from YAML)
├── gpu_specs/                  ✅ hand-curated YAML (Turing → Rubin, 17 GPUs)
├── cost_engine/                ✅ roofline + KV cache (W1); token_cost + TCO (W2)
├── mlperf/                     ✅ CSV ingestion + cross-check vs roofline
├── agents/                     ✅ compute advisor — Anthropic SDK, 10 tool wrappers (W3)
├── eval/                       ✅ 20-question eval set + grader (W3)
├── memo.py                     ✅ 2-page sizing memo generator (W3)
├── market_intel/               ✅ fact extractor: text-in, structured facts out, quote-verified (W3)
├── dcgm_schema/                (Week 5) Build 3
├── focus_schema/               (Week 5) Build 3
└── connectors/                 (Week 5) Build 3 swap point

build1_inference_calculator/    ✅ Streamlit multi-page (Spec, Validation, TCO, Advisor, Market Intel)
build3_account_health/          (Weeks 5-6)
build2_capacity_planner/        (Weeks 7-10)

data/
├── prices/                     ✅ append-only parquet; weekly GH Actions writes here
├── raw/mlperf/                 drop MLPerf CSVs here, named <round>__<label>.csv
├── raw/market/                 ✅ drop earnings/news text here; filename sets type + company + date
├── processed/                  Kaggle notebook outputs: t4_smoke_test.json, t4_validation.parquet
├── traces/                     cluster trace samples (Week 5)
├── sources.duckdb              ✅ generated by `python -m core.sources.seed`
└── market_intel.duckdb         ✅ generated by `python scripts/run_market_intel.py` (git-ignored)

notebooks/
├── 00_t4_smoke_test.ipynb      ✅ Kaggle-ready; vLLM + TinyLlama-1.1B on T4 (W1)
├── 01_roofline_t4_h100.ipynb   ✅ written-up roofline analysis (W2)
├── 02_cuda_matmul.ipynb        ✅ Kaggle-ready; naive vs tiled vs cuBLAS on T4 (W3)
└── 03_vllm_t4_validation.ipynb ✅ Kaggle-ready full sweep: 3 models × 4 batches × 3 in_lens (W2)

scripts/
├── log_prices.py               ✅ Vast.ai API + manual Monday log
├── run_eval.py                 ✅ live compute-advisor eval runner (Anthropic or OpenAI key)
├── run_market_intel.py         ✅ extract facts from data/raw/market/ (--fake for no-spend)
└── mcp_advisor_server.py       ✅ MCP server — run advisor in Claude Desktop, no API spend

docs/
└── interview_notes.md          ✅ per-build "how to tell this in an interview" with role mapping

.github/workflows/
├── price-log.yml               ✅ cron: Mondays 09:00 UTC
└── tests.yml                   ✅ lint + pytest on push/PR
```

---

## Running locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Seed the sources DuckDB from the GPU spec YAML (148 rows):
```powershell
python -m core.sources.seed
```

Launch the Build 1 Streamlit app (Spec / Validation / TCO / Advisor pages):
```powershell
streamlit run build1_inference_calculator/app.py
```

Dry-run the price logger against live Vast.ai:
```powershell
python scripts/log_prices.py --dry-run
```

Run the compute-advisor eval against a live LLM backend. `ANTHROPIC_API_KEY`
accepts **either** an Anthropic (`sk-ant-*`) or OpenAI (`sk-*`) key — the
backend is auto-picked from the prefix. Default models: Claude Haiku 4.5
for Anthropic, `gpt-4o-mini` for OpenAI.
```powershell
$env:ANTHROPIC_API_KEY = "<sk-ant-... or sk-...>"
python scripts/run_eval.py --limit 5     # smoke test
python scripts/run_eval.py               # full 20-question run → eval_report.md
```

Run the compute advisor from Claude Desktop with **no API spend** (MCP):
```powershell
pip install "mcp>=1.0"
# then add the config block from scripts/mcp_advisor_server.py docstring
# to claude_desktop_config.json and restart Claude Desktop
```

Extract facts from earnings transcripts / press releases / news:
```powershell
# drop .txt or .md files in data/raw/market/ named
#   <doc_type>__<company>__<YYYY-MM-DD>__<slug>.txt
python scripts/run_market_intel.py --fake --dry-run   # rule-based extractor, no API
python scripts/run_market_intel.py --fake             # write facts to data/market_intel.duckdb
$env:ANTHROPIC_API_KEY = "<sk-ant-... or sk-...>"     # Anthropic or OpenAI key — backend auto-picked
python scripts/run_market_intel.py                    # real extraction via Haiku 4.5 or gpt-4o-mini
```
Then visit the Market Intel page in the Streamlit app — it reads
`data/market_intel.duckdb` and renders every fact with its verbatim quote.

---

## Deploying to Streamlit Community Cloud

The repo is wired for a one-click deploy.

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **New app**.
3. Fill in:
   - **Repository**: `DV478/ai-infra-workbench`
   - **Branch**: `main`
   - **Main file path**: `build1_inference_calculator/app.py`
   - **App URL** (optional): a custom subdomain
4. Click **Advanced settings → Secrets** and paste one of:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."   # Anthropic key — uses Claude Haiku 4.5
   # or
   ANTHROPIC_API_KEY = "sk-..."       # OpenAI key — uses gpt-4o-mini
   ```
   The env var name stays `ANTHROPIC_API_KEY` for either provider; the backend is picked from the key prefix. Without a key set, those pages fall back to the no-spend UI (tool explorer / rule-based extractor).
5. Click **Deploy**.

The app installs from `requirements.txt` (the pyproject.toml is used only for local dev). Each page has a `sys.path` shim at the top that makes `from core.X import ...` resolve even if the editable install step is skipped.

**Memory budget sanity check:** the free tier is 1 GB. This app loads:
- 17-GPU YAML catalog (~50 KB)
- 148-row `sources.duckdb` (~100 KB)
- `prices.parquet` (grows ~5 KB/week — fine for years)
- Optional `market_intel.duckdb` (grows with the number of docs you process; ~1 KB/fact)

Total runtime usage under load (Streamlit + pandas + plotly + duckdb) is ~400-500 MB, well under 1 GB.

---

## Next steps — owner action items (blocked on you, not Claude)

Ordered by what unlocks most downstream work.

1. ~~**First git commit + GitHub push.**~~ ✅ Done 2026-11-05 — repo at https://github.com/DV478/ai-infra-workbench.

2. **Enable the price-log workflow** on GitHub. Visit `https://github.com/DV478/ai-infra-workbench/actions` and approve workflows if GitHub prompts. First log fires Mondays 09:00 UTC.

3. **Run `00_t4_smoke_test.ipynb` on Kaggle T4** → commit `data/processed/t4_smoke_test.json`. First measured number.

4. **Run `03_vllm_t4_validation.ipynb` on Kaggle T4** (~30 min). Commit `data/processed/t4_validation.parquet`. Calibrates `utilisation_frac`; the Validation page auto-renders predicted-vs-measured scatter once this file lands.

5. **Drop MLPerf CSVs** into `data/raw/mlperf/` named `<round>__<label>.csv` (e.g. `v5.0__datacenter.csv`) from https://mlcommons.org/en/inference-datacenter-50/. Validation page auto-reads them.

6. **Live advisor eval** (`ANTHROPIC_API_KEY` accepts either prefix):
   ```powershell
   $env:ANTHROPIC_API_KEY = "<sk-ant-... or sk-...>"
   python scripts/run_eval.py --limit 5    # smoke
   python scripts/run_eval.py              # full → eval_report.md
   ```
   Record pass rate. If <70%, tune `SYSTEM_PROMPT` in [core/agents/compute_advisor.py](core/agents/compute_advisor.py).

7. **Live market-intel eval** against the expanded 12-case gold set:
   ```powershell
   $env:ANTHROPIC_API_KEY = "<sk-ant-... or sk-...>"
   python scripts/run_market_intel_eval.py --limit 3   # smoke
   python scripts/run_market_intel_eval.py             # full → market_intel_eval_report.md
   ```
   Baseline on `--fake` is 4/12 (33%), quote-valid 100%. Real-LLM run should jump substantially.

8. **MCP server smoke test** in Claude Desktop (optional; zero-API-spend path):
   ```powershell
   pip install "mcp>=1.0"
   # add the config block from scripts/mcp_advisor_server.py docstring to claude_desktop_config.json
   # restart Claude Desktop; 10 tools should appear in the picker
   ```

### Flagged risks for next session

- If measured T4 TPS falls outside `test_tinyllama_decode_t4_sanity`'s 60-150 band ([tests/test_cost_engine.py](tests/test_cost_engine.py)), recalibrate `utilisation_frac` in [core/cost_engine/roofline.py](core/cost_engine/roofline.py). W2 notebooks assume 0.55 for T4; W1 catalog defaults to 0.65 (H100). Reconcile once Kaggle runs land.
- Blackwell is `confidence: medium`, Rubin is `low`. The sources seed script maps these to `MODELLED` (not `MEASURED`) so UI badges stay honest.
- MLPerf `_predict` map in [pages/2_Validation.py](build1_inference_calculator/pages/2_Validation.py) only knows Llama-2/3 70B. Extend for Mixtral and GPT-J-6B when their CSVs arrive.
- Live Streamlit's Market Intel page shows an empty-state because `data/market_intel.duckdb` is gitignored. Non-blocking (the page handles this gracefully with a clear message), but if the demo viewer should see the 40-fact table out-of-the-box, add a cold-start auto-populate hack that runs the `--fake` extractor over the committed `data/raw/market/` docs on first page load.

---

## Next Claude-doable work (pick up here next session)

Priority order (all three are independent — do in any order):
1. **Extend compute-advisor eval set** (currently 20 Qs) with adversarial / multi-tool / guardrail questions. High value if the owner runs the live eval first and surfaces specific weaknesses.
2. **Fix MLPerf `_predict`** in [pages/2_Validation.py](build1_inference_calculator/pages/2_Validation.py) for Mixtral + GPT-J-6B so the Validation page is useful when more CSVs land.
3. **Market-intel Streamlit cold-start auto-populate** — small QoL fix so the live demo page shows the 40-fact table out of the box.
3. **Hosted Streamlit Cloud deploy — owner click.** Follow the Deploy section above.

Done in Week 3 so far (do not redo):
- Compute advisor agent + 10 tool wrappers
- 20-question eval harness (grader returns per-component breakdown)
- `scripts/run_eval.py` CLI + `scripts/mcp_advisor_server.py` MCP wrapper
- 2-page sizing memo generator with TCO-page download button
- `notebooks/02_cuda_matmul.ipynb` (naive / tiled / cuBLAS)
- `docs/interview_notes.md` per-build audience-tuned talking tracks
- `core/market_intel/`: fact extractor with Anthropic SDK client + deterministic fake client for CI; quote-level citation enforced per fact
- `scripts/run_market_intel.py`: walks `data/raw/market/`, writes facts to DuckDB, supports `--fake` and `--dry-run`
- `build1_inference_calculator/pages/5_Market_Intel.py`: filters by kind/company/confidence, GPU-price and capex aggregates
- RAG-repo reuse decision revised: sibling RAG is **not** vendored — its scrapers conflict with the price-tracker policy. Extractor is text-in/facts-out; adapter deferred until a concrete use case.

---

## Reference memories

Decisions, source locations, and context are tracked in Claude's project memory at
`C:\Users\DV\.claude\projects\c--Users-DV-OneDrive-Desktop-Claude-AI-INFRA-PROJECT\memory\`.
Index: `MEMORY.md`. Status snapshot: `project_status.md`.
