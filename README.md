# AI Compute Economics Workbench

**Should you rent, buy, or API an H100 for Llama-3-70B?**
A calculator plus two AI agents that answer in £ per million tokens — and show every tool call behind the number.

🌐 **Live demo:** [ai-infra-workbench.streamlit.app](https://ai-infra-workbench.streamlit.app)
🧪 **Source:** [github.com/DV478/ai-infra-workbench](https://github.com/DV478/ai-infra-workbench)

Three demoable builds for AI infrastructure economics. Build 1 (this one) covers inference cost + build-vs-rent. Builds 2 and 3 are planned: GPU capacity planning and GPU cloud account health.

See [AI_INFRA_PROJECT.md](AI_INFRA_PROJECT.md) for the full spec; [docs/ux_review.md](docs/ux_review.md) and [docs/ux_rebuild_plan.md](docs/ux_rebuild_plan.md) for the UX rebuild notes.

---

## What a recruiter should notice

- **Six pages, one story.** Every page answers a question (not an engine component). Sidebar flows Advisor → Cost tracker → What does it cost → Will it fit → How accurate → Market prices.
- **Dual-backend AI agent.** The Advisor runs on your choice of **OpenAI gpt-4o-mini** (default) or **Claude Haiku 5.5**. Switch mid-chat. Same tools, same engine, different brain. A typical Q&A costs ~£0.0004 on gpt-4o-mini; ~£0.0002 on Haiku 5.5.
- **Live £ cost tracker.** Every Advisor turn is logged with input / output / cached tokens, latency, and £. The dedicated **Cost tracker** page renders aggregates by model — same questions, different brains, side by side.
- **Verbatim-quote extraction.** The **Market prices** page carries structured facts (GPU prices, capex, shipments) pulled from real press releases + transcripts — each row shows the exact source sentence. If the quote isn't literally there, the fact never gets stored.
- **Every number is tagged.** 🟢 measured, 🟡 modelled, 🟠 simulated, ⚪ assumption — visible on every hero metric, not buried in a footnote.
- **The memo.** The *What does it cost* page generates a 2-page sizing memo (markdown) you can hand to a deal desk.

---

## Status

**Build 1 UX rebuild complete** (M1 → M11, Nov 2026). Deployed at [ai-infra-workbench.streamlit.app](https://ai-infra-workbench.streamlit.app).

- **170/170 tests pass** — 135 engine + 29 shared-UI helper + 6 cost-tracker page tests.
- Ruff clean on `core/`, `scripts/`, `tests/`, `build1_inference_calculator/`.
- **Two-model Advisor picker** (owner-confirmed): OpenAI gpt-4o-mini default, Claude Haiku 5.5 switch. Both use the owner's existing credits.
- **Zero-spend demo** — if no API key is set, the Advisor falls back to a canned transcript, the Validation page renders bundled synthetic fixtures, and Market prices cold-starts from the committed 8-document demo corpus via a rule-based extractor.
- Pricing (Nov 2026 confirmed): gpt-4o-mini $0.15 / $0.60 per M · Claude Haiku 5.5 $0.10 / $0.50 per M. USD→GBP 0.78.
- Vast.ai live price API confirmed working; weekly GitHub Actions log kicked off.

Latest commit on `main`: see `git log --oneline -5`.

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
| Hosted agent runtime | Direct OpenAI/Anthropic SDK + prompt caching, picker between gpt-4o-mini + Haiku 5.5 | Owner holds both; ~£0.0002–0.0004 per Q&A fits any budget |
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

build1_inference_calculator/    ✅ Streamlit multi-page — six pages in sidebar narrative order:
│   ├── _ui/                    ✅ shared primitives (display_names, explain, presets, cost_counter, page_header, provenance, tool_card, sidebar) + demo fixtures (bundled synthetic T4/MLPerf/corpus/transcript)
│   ├── app.py                  ✅ landing: hero + worked example + four CTAs + methodology
│   └── pages/
│       ├── 1_Ask_the_Advisor.py           ✅ chat + 2-model picker + cost counter + named tool cards
│       ├── 2_Cost_tracker.py              ✅ observability — token economics per turn
│       ├── 3_What_does_it_cost.py         ✅ TCO headline + memo download
│       ├── 4_Will_it_fit.py               ✅ memory-fit check + catalog reference
│       ├── 5_How_accurate.py              ✅ roofline + T4 + MLPerf, synthetic fallbacks
│       └── 6_Market_prices.py             ✅ facts with verbatim quotes, cold-start auto-populate
build3_account_health/          (planned — not shipped)
build2_capacity_planner/        (planned — not shipped)

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

Launch the Build 1 Streamlit app (Advisor / Cost tracker / What does it cost / Will it fit / How accurate / Market prices):
```powershell
streamlit run build1_inference_calculator/app.py
```

Dry-run the price logger against live Vast.ai:
```powershell
python scripts/log_prices.py --dry-run
```

Run the compute-advisor eval against a live LLM backend. `ANTHROPIC_API_KEY`
accepts **either** an Anthropic (`sk-ant-*`) or OpenAI (`sk-*`) key — the
backend is auto-picked from the prefix. Default models: **Claude Haiku 5.5**
for Anthropic, **gpt-4o-mini** for OpenAI.
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
python scripts/run_market_intel.py                    # real extraction via Haiku 5.5 or gpt-4o-mini
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
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."

   # Optional — PIN for the Cost tracker "Clear log" admin button.
   # Without this set, the button appears unprotected on the public URL.
   COST_TRACKER_RESET_PIN = "pick-anything"
   ```
   Both keys are supported. You only need *one* for the Advisor to work, but setting both lets a visitor switch models in the UI live. Without any key, pages fall back to no-spend demo fixtures (canned Advisor transcript, synthetic T4 + MLPerf data, rule-based Market Intel extractor).
5. Click **Deploy**. Full step-by-step for first go-live is in [docs/m10_owner_actions.md](docs/m10_owner_actions.md).

The app installs from `requirements.txt` (the pyproject.toml is used only for local dev). Each page has a `sys.path` shim at the top that makes `from core.X import ...` resolve even if the editable install step is skipped.

**Memory budget sanity check:** the free tier is 1 GB. This app loads:
- 17-GPU YAML catalog (~50 KB)
- 148-row `sources.duckdb` (~100 KB)
- `prices.parquet` (grows ~5 KB/week — fine for years)
- Optional `market_intel.duckdb` (grows with the number of docs you process; ~1 KB/fact)

Total runtime usage under load (Streamlit + pandas + plotly + duckdb) is ~400-500 MB, well under 1 GB.

---

## Owner action items (blocked on live keys / hardware)

In rough order of what unlocks the most downstream work.

1. **Wire up live API keys** on Streamlit Cloud. Full step-by-step at [docs/m10_owner_actions.md](docs/m10_owner_actions.md). Without this, the Advisor runs on a canned transcript — the picker, cost counter, and Cost tracker page are all in place.

2. **Record real costs** after the first live Q&A. The README already claims ~£0.0004 on gpt-4o-mini and ~£0.0002 on Haiku 5.5 — confirm against the Cost tracker page and overwrite the placeholders if they drift.

3. **Enable the price-log workflow** on GitHub if GitHub prompts. Weekly logs fire Mondays 09:00 UTC into `data/prices/`.

4. **Run `00_t4_smoke_test.ipynb` on Kaggle T4** → commit `data/processed/t4_smoke_test.json`. The *How accurate* page is already wired to prefer real data and fall back to a synthetic fixture — this swap-in is zero-touch once the JSON lands.

5. **Run `03_vllm_t4_validation.ipynb` on Kaggle T4** (~30 min). Commit `data/processed/t4_validation.parquet` to calibrate `utilisation_frac`.

6. **Drop MLPerf CSVs** into `data/raw/mlperf/` named `<round>__<label>.csv`. *How accurate* auto-reads them and falls back to the bundled synthetic MLPerf fixture otherwise.

7. **Live advisor eval** against the 20-question set:
   ```powershell
   $env:ANTHROPIC_API_KEY = "<sk-ant-... or sk-...>"
   python scripts/run_eval.py --limit 5    # smoke
   python scripts/run_eval.py              # full → eval_report.md
   ```

8. **Live market-intel eval** against the 12-case gold set:
   ```powershell
   $env:ANTHROPIC_API_KEY = "<sk-ant-... or sk-...>"
   python scripts/run_market_intel_eval.py             # full → market_intel_eval_report.md
   ```

### Known gaps (not blocking)

- Blackwell catalog rows are `confidence: medium`, Rubin is `low` — sources seed maps these to `MODELLED` so provenance badges stay honest.
- If measured T4 TPS falls outside the 60-150 range enforced by `test_tinyllama_decode_t4_sanity`, recalibrate `utilisation_frac` in [core/cost_engine/roofline.py](core/cost_engine/roofline.py).
- Build 2 (capacity planner) and Build 3 (account health) are planned but not shipped. Build 1 alone is a complete artifact.

---

## Reference memories

Decisions, source locations, and context are tracked in Claude's project memory at
`C:\Users\DV\.claude\projects\c--Users-DV-OneDrive-Desktop-Claude-AI-INFRA-PROJECT\memory\`.
Index: `MEMORY.md`. Status snapshot: `project_status.md`.
