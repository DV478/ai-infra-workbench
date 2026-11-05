# AI Compute Economics Workbench

Three demoable builds for AI infrastructure economics: inference cost + build-vs-rent, GPU cloud account health, and GPU capacity planning. Measured, modelled and simulated data are labelled on every output.

See [AI_INFRA_PROJECT.md](AI_INFRA_PROJECT.md) for the full spec.

---

## Status

**As of 2026-11-05 (Week 1 + Week 2 + Week 3 agent + market intel extractor built, before first commit):** scaffold, cost engine, MLPerf ingestion, sources seed, Streamlit multi-page app for Build 1 (5 pages), four Kaggle-ready notebooks, compute-advisor agent, eval harness, MCP server, memo generator, and market intelligence extractor are in place. Nothing is committed yet.

- **107/107 tests pass** (23 W1 + 39 W2 + 27 W3 agent + 4 memo + 14 market intel)
- Ruff clean on `core/`, `scripts/`, `tests/`, `build1_inference_calculator/`
- `python -m core.sources.seed` runs: 17 GPUs → 148 Source rows at `data/sources.duckdb`
- `python scripts/run_market_intel.py --fake --dry-run` extracts 11 facts from the 2 shipped sample docs with zero API spend
- Perfect-stub eval harness run: 100% pass (20/20); numeric-in-range 93.3%
- Vast.ai live price API confirmed working (Week 1)

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
├── run_eval.py                 ✅ live compute-advisor eval runner (needs ANTHROPIC_API_KEY)
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

Run the compute-advisor eval against the live Anthropic API:
```powershell
$env:ANTHROPIC_API_KEY = "<your key>"
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
$env:ANTHROPIC_API_KEY = "<key>"
python scripts/run_market_intel.py                    # real extraction via Claude Haiku 4.5
```
Then visit the Market Intel page in the Streamlit app — it reads
`data/market_intel.duckdb` and renders every fact with its verbatim quote.

---

## Next steps — owner action items (blocked on you, not Claude)

Ordered by criticality. Items 1 and 2 are the biggest exposures.

1. **First git commit + GitHub push.** Repo has zero commits. If the machine dies, all code is lost.
   ```powershell
   git add .
   git commit -m "Week 1-3 scaffold: cost engine, agent, Streamlit app"
   git remote add origin <your empty GitHub repo URL>
   git push -u origin main
   ```

2. **Enable the price-log workflow** on GitHub after push. First real log fires **Monday 2026-11-09 09:00 UTC**. Price history only exists from the week logging starts. If push slips past Monday, trigger manually via Actions → "Run workflow".

3. **Run `00_t4_smoke_test.ipynb` on Kaggle T4** → commit `data/processed/t4_smoke_test.json`. First measured number in the project.

4. **Run `03_vllm_t4_validation.ipynb` on Kaggle T4** (~30 min). Commit `data/processed/t4_validation.parquet`. The Streamlit Validation page auto-renders predicted-vs-measured scatter + mean absolute error once this file lands. Calibrates `utilisation_frac`.

5. **Drop MLPerf CSVs** into `data/raw/mlperf/` named `<round>__<label>.csv` (e.g. `v5.0__datacenter.csv`) from https://mlcommons.org/en/inference-datacenter-50/. Validation page auto-reads them.

6. **Live advisor eval** once you have an Anthropic API key:
   ```powershell
   $env:ANTHROPIC_API_KEY = "..."
   python scripts/run_eval.py --limit 5    # smoke
   python scripts/run_eval.py              # full → eval_report.md
   ```
   Record pass rate in README. If <70%, tune `SYSTEM_PROMPT` in `core/agents/compute_advisor.py`.

7. **MCP server smoke test** in Claude Desktop (optional; unlocks the zero-API-spend path):
   ```powershell
   pip install "mcp>=1.0"
   # add the config block from scripts/mcp_advisor_server.py docstring to claude_desktop_config.json
   # restart Claude Desktop; 10 tools should appear in the picker
   ```

### Flagged risks for next session

- If measured T4 TPS falls outside `test_tinyllama_decode_t4_sanity`'s 60-150 band ([tests/test_cost_engine.py](tests/test_cost_engine.py)), recalibrate `utilisation_frac` in [core/cost_engine/roofline.py](core/cost_engine/roofline.py). W2 notebooks assume 0.55 for T4; W1 catalog defaults to 0.65 (H100). Reconcile once Kaggle runs land.
- Blackwell is `confidence: medium`, Rubin is `low`. The sources seed script maps these to `MODELLED` (not `MEASURED`) so UI badges stay honest.
- MLPerf `_predict` map in [pages/2_Validation.py](build1_inference_calculator/pages/2_Validation.py) only knows Llama-2/3 70B. Extend for Mixtral and GPT-J-6B when their CSVs arrive.
- **No commits yet.** This is still the biggest single exposure.

---

## Week 3 remaining (next Claude session)

Priority order:
1. **Hosted Streamlit Community Cloud deploy.** `requirements.txt` or `pyproject.toml` compat check; secrets setup for `ANTHROPIC_API_KEY`; memory budget check for 148 source rows + parquet files + market_intel DuckDB.
2. **Extend eval set** once the live eval reveals weaknesses — expect `SYSTEM_PROMPT` tuning when real traces come back.
3. **Market intel live eval.** Drop 5-10 real earnings transcripts or press releases in `data/raw/market/`, run the extractor with a live `ANTHROPIC_API_KEY`, check that the quote-verification layer isn't silently dropping valid facts.

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
