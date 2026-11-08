# UX rebuild execution plan — Build 1

> Executor-Claude: this file is a top-to-bottom work list. Grep for `### M` to find your milestone. Each milestone lands as its own commit. Do not skip the "Preserved strengths" and "Design principles" sections — they govern every call below.

## Plan at a glance

End state: six Streamlit pages that read as a product, not a dev readme. A non-technical reader (London AI-infra recruiter / GTM / strategy) lands, sees a one-sentence pitch + worked example, picks one of 2–3 "start here" CTAs, and within 30 seconds on any page can say what it does and try a scenario. No internal IDs in the UI. Every number has units + a plain-English one-liner. Advisor has a two-model picker (**gpt-4o-mini default, Claude Haiku 5.5 switch**) and a per-turn £ cost counter. A dedicated **Cost & token tracker** page logs every Advisor turn with input/output/cached tokens, latency, £ — so the owner (and any recruiter who clicks) can see exactly how LLM pricing works. OpenAI key goes in last — zero live LLM spend until M10.

Ordering: shared UI primitives → Landing → Spec → TCO (headline) → Validation → Advisor (picker UI stubbed) → Cost tracker (new) → Market Intel (cold-start auto-populate) → consistency pass → OpenAI key wiring + smoke test → README refresh. Rough total: ~31–47 h. One commit per milestone; tests green at every commit.

## Design principles (locked in)

- Explain-first, jargon-last. Each page's first screen answers: what is this / why care / where do I start / what do numbers mean.
- Internal IDs never leak (`h100_sxm5_80gb`, `gpu_price`, `FactKind.capex` → pretty names only). IDs stay in tool-call JSON.
- Every page has a "Try this scenario" preset chip row. Default = Nscale/Nebius angle (Llama-3-70B on H100 SXM5, 60% busy, $2.49/hr).
- Every number has units + a one-sentence plain-English interpretation (caption or `help=`).
- Hero metric above dataframes. No page opens with a raw `st.dataframe`.
- Sidebar in narrative order, not engine-build order.
- Dual-backend (Anthropic/OpenAI) is a feature — surface the picker.
- No engine rewrites. `core/*` frozen; add helpers under `build1_inference_calculator/_ui/`.
- One milestone = one commit.

## Preserved strengths (do not touch)

From `docs/ux_review.md` "What's working":
- Traffic-light confidence badges (`1_Spec.py:37-38`) — extend to TCO rows + Market Intel.
- Measured / modelled / simulated / assumption tagging (`app.py:36`) — make it louder via `_ui/provenance.py`.
- Tool-call expander on Advisor (`4_Advisor.py:89-92, 111-115`) — keep, rerender as named cards.
- Verbatim-quote column on Market Intel (`5_Market_Intel.py:92-103`) — keep widest.
- Memo download on TCO (`3_TCO.py:176-207`) — keep, promote next to headline.
- Dual-backend auto-detect in `ComputeAdvisor` (`core/agents/compute_advisor.py:602-609`) — keep as fallback under the picker.
- Demand-stress framing (`3_TCO.py:156-166`) — keep the copy; promote the chart.
- Modular engine (`core/*`) — never edited by this plan.

## Open questions resolved

1. **Headline scenario.** Hardcode Llama-3-70B + H100 SXM5 + Nscale neocloud as the primary preset. Secondary: Llama-3-8B + L40S. Tertiary: Qwen2.5-3B + L4.
2. **Market Intel's place.** Keep in Build 1 sidebar; reframe copy as "prices and capex that validate TCO assumptions". **CONFIRM WITH OWNER** before moving out of Build 1.
3. **Sidebar reorder.** Advisor → TCO → Spec → Validation → Market Intel. Advisor leads (strongest page).
4. **Collapse Spec into TCO.** No — keep separate (fit vs cost). **CONFIRM WITH OWNER** only if Spec can't carry its weight after M3.
5. **Bundle demo data for Validation.** Yes. Commit tiny synthetic `t4_smoke_test.json` + MLPerf CSV under `build1_inference_calculator/_ui/examples/`, tagged `[SYNTHETIC — demo]`. Real data in `data/processed/` + `data/raw/mlperf/` takes precedence; fallback shows a banner.
6. **Memo as PDF.** Markdown only. Caption: "Print to PDF from your editor or `pandoc memo.md -o memo.pdf`". No server-side pandoc.
7. **Copy-pass scope.** Full sweep via one central `_ui/explain.py` dict (one source for ~30 terms; consumed everywhere).
8. **Picker scope + default** (owner-confirmed 2026-11-08). Two options only: **gpt-4o-mini (default)** and **Claude Haiku 5.5 (switch)**. Dropped: `gpt-5-mini` (adds no demo value at sub-penny deltas), `gpt-5-nano` (cost savings are noise at demo volume), `gpt-4.1-mini` (deprecated per OpenAI), `claude-haiku-4-5` (superseded by 5.5 at ~10× lower price). Rationale: gpt-4o-mini is the model name a non-technical recruiter recognises; Haiku 5.5 uses the owner's existing Claude credits + demonstrates dual-backend. Easy to add a third model later if needed.
9. **Cost & token tracker page** (owner-requested 2026-11-08). New page: logs every Advisor turn with model, input/output/cached tokens, latency, and £ cost. Sits in the sidebar directly after Advisor. Aggregates cost per model, cache-hit rate, cumulative session cost. Educational copy explains what tokens are, what caching does, why pricing varies. **Primary audience: owner, learning token economics.** Secondary: recruiter credibility signal ("they instrumented their own LLM spend"). See M7.
10. **Favicon / OG image / custom domain** (owner-confirmed 2026-11-08). Skipped. Streamlit Cloud makes OG meta awkward; ROI low vs UX fundamentals. Revisit if the deployed link gets shared in Slack and looks bare.

Executor: if any of these feels wrong mid-build, SendMessage back before deciding. Do not silently re-scope.

## Shared components to build first

New package `build1_inference_calculator/_ui/`. Build these once; pages consume them.

- **`_ui/display_names.py`** — `gpu_display_name(gpu_id)` wraps `catalog.by_id(gpu_id).name`. `gpu_options(catalog)` returns `{pretty_name: id}` for `st.selectbox`. `fact_kind_display(kind)` (`"gpu_price"` → `"GPU price"`). No engine change — `name` is already in the catalog YAML. Consumed: Spec, Validation, TCO, Market Intel.

- **`_ui/explain.py`** — `GLOSSARY: dict[str, str]` keyed by term (`roofline`, `kv_cache`, `busy_fraction`, `pue`, `kernel_utilisation`, `break_even`, `stranded_capex`, `arithmetic_intensity`, `fp8`, `int4`, `fact_kind`, ~30 entries matching HIGH list at `ux_review.md:21`). Helpers: `tooltip(key)` (short `help=`), `caption(key)` (longer sentence), `break_even_sentence(be)` (dataclass → English).

- **`_ui/presets.py`** — `preset_chips(key_prefix, scenarios)`. `Scenario` dataclass (gpu_id, model_name, batch, busy, in_len, out_len, util). `SCENARIOS = [NSCALE_70B_H100 (default), MIDSIZE_8B_L40S, EDGE_3B_L4]`. Clicks write to `st.session_state`; pages read as widget defaults. Consumed: Spec, Validation, TCO.

- **`_ui/page_header.py`** — `page_header(what, why, start, numbers)` renders a bordered four-line container. Every page.

- **`_ui/provenance.py`** — `badge(kind)` returns emoji+label (🟢 measured, 🟡 modelled, 🟠 simulated, ⚪ assumption). `legend_expander()` renders the key.

- **`_ui/cost_counter.py`** — `PRICE_PER_MTOK` dict with two entries (Nov 2026 pricing):
    - `gpt-4o-mini`: input $0.15, cached $0.075, output $0.60
    - `claude-haiku-5-5`: input $0.10, cached $0.01, output $0.50 (exact model-id string confirmed in M10 against live API)

  `USD_TO_GBP = 0.78` with `# TODO: refresh quarterly`. `cost_for_turn(model, usage)` → `{gbp, usd, input_tok, output_tok, cached_tok, cache_hit_pct, model}`. `render_cost_badge(cost, latency_ms)` draws the two-line display from `ux_review.md:120-122`. **Also exposes `log_turn(model, usage, latency_ms, question, answer_md)`** that appends to `data/advisor_turns.duckdb` (schema: `ts TIMESTAMP, model TEXT, input_tok INT, output_tok INT, cached_tok INT, latency_ms REAL, usd REAL, gbp REAL, question TEXT, answer_md TEXT`). Consumed by Advisor (M6) and the Cost tracker page (M7).

- **`_ui/tool_card.py`** — `render_tool_call(tc)`. Dispatch dict maps tool name → emoji + title (`check_memory_fit` → "📏 Memory fit check"; `compare_tco` → "💷 TCO comparison") + one-line summary from the result. Raw JSON in nested `st.expander("Raw input/output JSON")`.

- **`_ui/sidebar.py`** — unified sidebar config, called from `app.py` in M8.

Copy tone: British English. Short, active sentences. £ on Advisor; USD on TCO (operator-facing).

## Milestones (ordered, with dependencies)

Each milestone is one commit. `[deps: Mx]` flags what must have landed first. Tests must stay green at the end of each milestone — if a test breaks, it's a real signal, not a nuisance.

### M1. Shared UI primitives + display-name layer  `[deps: none]`

Build the `_ui/` package end-to-end before touching any page.

Tasks:
- Create `build1_inference_calculator/_ui/` with the eight modules listed above. No page code changes yet.
- Add a small unit test file `tests/test_ui_helpers.py` covering `gpu_display_name` (round-trips an id to a known pretty name), `tooltip("roofline")` returns a non-empty string, `cost_for_turn` math for one known-usage dict. No Streamlit imports in tests — keep them pure.
- Run `pytest` and `ruff check build1_inference_calculator`. Both must pass.

Effort: **M (3–5 h)**.

Files touched: new `build1_inference_calculator/_ui/*.py`, new `tests/test_ui_helpers.py`.

### M2. Landing page rewrite  `[deps: M1]`

Rewrite `app.py` around three stacked blocks (`ux_review.md:51`).

Tasks:
- (1) One-sentence hero: "Should you rent, buy, or API an H100 for Llama-3-70B? This workbench answers in £/M tokens." (2) Worked-example card: small Plotly bar chart from a *static* precomputed TCO result (don't run `compare_options` on render — hard-code the 5-bar Nscale result as a dict). Beside it: "At 60% busy, neocloud wins by 3.4× vs on-demand." (3) Four-button CTA panel: "Ask the advisor", "See the TCO", "Check if it fits", "Validate the model".
- Collapse the current status table into `st.expander("Methodology & data status")`. No filesystem paths; say "Roofline model, calibrated against Kaggle T4" instead.
- Fix the broken link (`app.py:60-62`) → GitHub README URL.
- Call `page_header(...)` at top. Set `st.set_page_config` once here (keep `⚙️`).

Effort: **M (2–3 h)**.

Files touched: `build1_inference_calculator/app.py`.

### M3. Spec page rewrite  `[deps: M1]`

Flip `pages/1_Spec.py` so memory-fit leads and catalog is reference (`ux_review.md:63`).

Tasks:
- `page_header(what="Will your model fit on this GPU?", why="HBM is the bottleneck for 70B-class models.", start="Pick a preset, or choose a model and GPU.", numbers="Weights + KV cache + activations must sum below GPU HBM.")`.
- Preset chip row; clicks populate widget state via session_state.
- Pretty-name GPU dropdowns via `gpu_display_name` / `gpu_options`. Drop `id` from `view_cols`.
- Replace the four `st.metric` tiles with one stacked-bar Plotly chart (Weights + KV + Activation stacked, HBM ceiling as red horizontal line). Caption from `explain.caption("memory_fit")`.
- Add `help=` tooltips to every widget (precision explains FP16/FP8/INT8/INT4; batch + seqlen explain KV impact).
- Full catalog moves into `st.expander("Browse full catalog", expanded=False)`, keeping confidence badges + citations. Rename `conf` column to "Confidence".
- Remove `st.set_page_config` from this file — set once in `app.py` (M8).

Effort: **M (3–4 h)**.

Files touched: `build1_inference_calculator/pages/1_Spec.py`.

### M4. TCO page rewrite — the headline page  `[deps: M1]`

Most important page; most hostile today. Follow `ux_review.md:89`.

Tasks:
- `page_header(what="What does serving this model cost per million tokens?", why="Own vs reserved vs on-demand vs neocloud vs API — one ruler.", start="Pick a preset, then tune.", numbers="$ per million output tokens, 3-year blended.")`.
- Preset chip row; Nscale clicked by default. Presets write `st.session_state` keys that widgets read as defaults.
- **Headline recommendation card** above the table: bordered container with the winning option in large text ("Neocloud — $1.90/M output tokens, 3.4× cheaper than on-demand"). Computed from `min(scenario.results, key=usd_per_m_tokens)`.
- Two side-by-side charts below the headline: (1) existing `$/M output tokens` bar; (2) demand-stress bar (promote from bottom). Each with a caption via `explain.caption(...)`.
- Break-even bullets in English: replace `"{pivot_variable} = {pivot:.3f}"` with sentences via `_ui/explain.py::break_even_sentence(be)`. E.g. "Owning beats renting once you're decoding more than 47% of the hours you pay for."
- Move 20+ widgets into `st.expander("Tune the inputs", expanded=False)`.
- **Sensitivity bar for `busy_fraction`**: small chart showing $/M tokens at busy = 0.3 / 0.5 / 0.7 / 0.9 for the current preset. Caption: "A 60% → 80% busy swing can flip the recommendation."
- Memo download: move into right-column callout next to the headline. Keep audience field + preview expander.
- Keep `demand_stress(...)` and `compare_options(...)` calls unchanged. No engine edits.
- Hide `id` column everywhere; use `gpu_display_name`.

Effort: **L (5–7 h)**.

Files touched: `build1_inference_calculator/pages/3_TCO.py`, small helper additions to `_ui/explain.py`.

### M5. Validation page rewrite  `[deps: M1]`

Reframe as "how much to trust these numbers" (`ux_review.md:75`).

Tasks:
- `page_header(what="How accurate are the predictions behind this workbench?", why="A cost model is only useful if its tok/s numbers are close to reality.", start="See the headline error; the roofline chart below shows why.", numbers="Lower error % = closer to measured.")`.
- Commit synthetic fallback fixtures under `build1_inference_calculator/_ui/examples/t4_smoke_demo.json` and `.../mlperf_demo.csv`, tagged `[SYNTHETIC — demo data]`. Page prefers real data in `data/processed/` + `data/raw/mlperf/`; falls back with a visible banner.
- Hero metric: "Mean error vs measured T4: **X %**", pulled from whichever source is active.
- Roofline chart: add the **memory-bandwidth ceiling** line (`gpu.memory_bandwidth_gbs * util / intensity`) called out at `ux_review.md:69`. Overlay the operating point for the selected model + batch. Caption: "Points above the line are impossible; points on the line are fully compute- or bandwidth-bound."
- Pretty-name GPU dropdown.
- Tooltip on "kernel utilisation" slider via `explain.tooltip("kernel_utilisation")`.
- Push batch-sweep dataframe + twin aggregate/per-request charts into `st.expander("Batch sweep")`.
- Replace `st.json(smoke)` with a metric row + 3-column table; raw JSON in `st.expander("Raw smoke-test JSON")`.
- Fix MLPerf `_predict` for Mixtral + GPT-J-6B (noted in `project_status.md`). Return `None` gracefully on unknown arch.

Effort: **M (3–5 h)**.

Files touched: `build1_inference_calculator/pages/2_Validation.py`, new `build1_inference_calculator/_ui/examples/` with two tiny synthetic fixtures.

### M6. Advisor page rewrite + OpenAI picker UI (stubbed)  `[deps: M1]`

Strongest page — do no harm. Add picker + cost counter; key wiring stays stubbed until M9. See `ux_review.md:115-124`.

Tasks:
- `page_header(what="Ask a sizing or build-vs-rent question.", why="Every number in the answer is a tool call against the engine — the agent never calculates.", start="Click a starter question or type your own.", numbers="Transparent: see the exact tool inputs/outputs for every claim.")`.
- **Model picker** (`st.selectbox("Model")`) above the chat. **Two options only, owner-confirmed:** `"OpenAI gpt-4o-mini"` (default) and `"Claude Haiku 5.5"` (switch). Store in `st.session_state.advisor_model`; on change rebuild `ComputeAdvisor(api_key=..., model=..., backend=...)` with `backend="openai"` for `gpt-*`, `"anthropic"` for `claude-*`. Caption under the picker: "Switch backends mid-chat — same tools, same engine, different brain."
- Auto-detect fallback: if `st.session_state.advisor_model` unset, use `detect_backend(api_key)`.
- **Starter-question chips** (4, each prefills `st.chat_input`): "Serve 70B at 50 rps for 3 years — rent or buy?", "Does Llama-3-8B fit on an L4?", "At what utilisation does owning beat reserved?", "What's the implied margin of Together's API vs owning H100s?".
- **Cost counter**: after each turn, call `render_cost_badge(...)` for £ cost + cumulative session total. Latency via `time.perf_counter()` around `.ask()`. On Anthropic, show cache-hit %. **Also call `log_turn(...)`** on every completed turn to persist to `data/advisor_turns.duckdb` (M7 reads this).
- **Tool-call expander**: replace raw `st.json` with `render_tool_call(tc)` named cards; raw JSON in a nested expander.
- **No-key fallback**: replace the schema explorer with a canned transcript from `_ui/examples/advisor_transcript.json` — same UI, labelled "Demo transcript — set an API key to chat live."
- "Clear chat" button next to the chat input.
- API-margin guardrail copy: when `tc.name == "api_margin"`, card title = "Benchmark comparison (reasoned, not an accusation)".
- Do not modify `core/agents/compute_advisor.py`.

Effort: **L (4–6 h)**.

Files touched: `build1_inference_calculator/pages/4_Advisor.py`, add starter prompts + transcript fixture under `_ui/examples/`, flesh out `_ui/cost_counter.py` and `_ui/tool_card.py` from M1 stubs.

### M7. Cost & token tracker page — NEW  `[deps: M1, M6]`

A dedicated page that teaches token economics using the owner's own Advisor turns as the dataset. Primary audience: the owner (learning how LLM pricing works). Secondary: recruiters (credibility signal — "they instrumented their own LLM spend"). Owner-requested 2026-11-08.

Tasks:
- New file `build1_inference_calculator/pages/N_Cost_tracker.py` (numeric prefix set in M9 renumber).
- `page_header(what="How every chat message gets priced.", why="LLM pricing has three levers — input tokens, output tokens, cached tokens. Watch them move live.", start="Chat on the Advisor page; every turn lands here with a full breakdown.", numbers="£ per turn, tokens in/out/cached, latency in ms.")`.
- Load `data/advisor_turns.duckdb` (schema from `_ui/cost_counter.py::log_turn`). If missing/empty, show a friendly empty-state: "No turns logged yet — ask the Advisor a question to populate this page." Link back to Advisor.
- **Hero row** (`st.columns(5)`):
  - Total turns logged
  - Cumulative £ cost
  - Average £ per turn
  - Overall cache-hit % (`sum(cached_tok) / sum(input_tok)`)
  - Average latency (ms)
- **Per-model comparison** (if 2+ models used): small grouped bar chart — £ per turn by model, side-by-side with avg latency. Caption: "Same questions, different brains — here's how they price out."
- **Pricing-model explainer** (above the table): a short 4-tile grid (`st.columns(4)`) with a one-line definition + the live £ figure for each lever:
  - "Input tokens" — your question + system prompt + tool-call history
  - "Output tokens" — the model's reply (priced ~4× input)
  - "Cached tokens" — repeated prefix charged at ~10% (Anthropic prompt cache; OpenAI auto-cache for long contexts)
  - "Latency" — wall-clock time per turn
- **Turn-by-turn table**: `st.dataframe` with columns Time, Model (pretty name via `_ui/display_names.py`), Input tok, Output tok, Cached tok, Cache-hit %, Latency (ms), **£ cost**. Sort descending by time. Use `column_config.NumberColumn` for formatting (e.g. £ to 4 d.p., tokens as integers). Expandable row (or a modal/expander below) for the actual Q&A text.
- **Pricing-table footnote** (`st.expander("Current per-model price card")`): render `PRICE_PER_MTOK` as a small table — input/cached/output $ per M tokens, with a "last confirmed YYYY-MM-DD" date.
- **Reset button** ("Clear turn log") — deletes `data/advisor_turns.duckdb`. Require a second-click confirm via `st.session_state`. Do not expose on a shared deployment without a password gate (TODO for M10).
- **Do not touch `core/agents/compute_advisor.py`.** This page is pure Streamlit + DuckDB read.
- Add tests: `tests/test_cost_counter.py` covering (a) `cost_for_turn` math for both models with a known usage dict, (b) `log_turn` → read-back round trip against a tmp DuckDB path, (c) cache-hit % edge cases (0 input, all-cached).

Design-principle checks:
- Pretty model names in the UI; model IDs only in the price-card expander.
- Every number has a plain-English caption.
- Hero row sits above the dataframe.

Effort: **M (4–6 h)**.

Files touched: new `build1_inference_calculator/pages/N_Cost_tracker.py`, `_ui/cost_counter.py` (flesh out `log_turn` + aggregate helpers), new `tests/test_cost_counter.py`.

### M8. Market Intel rewrite + cold-start auto-populate  `[deps: M1]`

Follow `ux_review.md:109-113`. Fix the two SME nits at `ux_review.md:107`.

Tasks:
- `page_header(what="Live prices and capex that back the TCO assumptions.", why="Every fact carries the verbatim quote it came from; nothing is invented.", start="Browse by fact kind or company; click a row for the source quote.", numbers="One row = one extracted fact with a confidence badge.")`.
- **Cold-start auto-populate**: on page load, if `data/market_intel.duckdb` missing, inline the `FakeExtractorClient` path from `scripts/run_market_intel.py`: walk `data/raw/market/*.{txt,md}`, write to `data/market_intel.duckdb`. Zero API spend. Banner: "Showing 40 facts from synthetic demo corpus. In production, set `ANTHROPIC_API_KEY` and run `scripts/run_market_intel.py`."
- Pretty-name `FactKind` filter via `fact_kind_display(...)` — "GPU price" not `gpu_price`.
- Hero chart on top: one Plotly chart of GPU-price observations over time or by provider (whichever the corpus supports).
- Fix nit 1 (`5_Market_Intel.py:109-116`): rename `latest_usd_per_hour` to `min_usd_per_hour_observed` and add `latest_citation_date` (`max("citation_date")` by subject). Update column headers via `column_config`.
- Fix nit 2 (`5_Market_Intel.py:122-137`): add `period` filter above capex aggregate; disable the sum when multiple periods selected and show a caption warning. Default: most recent period only.
- Verbatim-quote column stays widest. Drop snake_case `fact_kind` display.
- Methodology expander at the bottom: "40 facts across 8 documents; 12-case gold eval; 100% quote-valid on fake and real-LLM runs".

Effort: **M (3–5 h)**.

Files touched: `build1_inference_calculator/pages/5_Market_Intel.py`, small helper in `_ui/display_names.py` for `FactKind`.

### M9. Consistency pass  `[deps: M2–M8]`

Tasks:
- Single `st.set_page_config(...)` in `app.py`. Remove from all `pages/*.py`.
- Rename page files (Streamlit uses the numeric prefix for order, rest-of-filename for sidebar label). New sidebar order:
  - `pages/4_Advisor.py` → `pages/1_Ask_the_advisor.py`
  - (new from M7) → `pages/2_Cost_tracker.py`
  - `pages/3_TCO.py` → `pages/3_What_does_it_cost.py`
  - `pages/1_Spec.py` → `pages/4_Will_it_fit.py`
  - `pages/2_Validation.py` → `pages/5_How_accurate.py`
  - `pages/5_Market_Intel.py` → `pages/6_Market_prices.py`
- Narrow-viewport spot check; any `st.columns(5)` that wraps becomes 3+2.
- Dark-mode spot check (current Plotly `template="plotly_white"` under dark theme).
- Favicon: keep `⚙️`. Open-graph meta skipped per owner (resolved question #10).
- Grep check — no visible internal IDs:

      git grep -n "h100_sxm5_80gb\|_v5e\|trn2\|FactKind\." build1_inference_calculator/

Effort: **S (1–2 h)**.

Files touched: page renames, `app.py`, `_ui/sidebar.py`.

### M10. OpenAI + Anthropic key wiring + live demo smoke test  `[deps: M6, M7, M9]`

First live LLM spend happens here, not earlier.

Tasks:
- Owner adds `OPENAI_API_KEY` **and** `ANTHROPIC_API_KEY` to Streamlit Cloud secrets (executor flags this; cannot do it alone). Owner has existing Claude credits to cover the latter.
- Confirm exact Haiku 5.5 model-ID string against the live Anthropic API (e.g. `claude-haiku-5-5-20260815` or similar — my knowledge cutoff predates its release). Update `PRICE_PER_MTOK` and the picker `value=` to match.
- Local smoke test with both keys set: ask one starter question on each of `gpt-4o-mini` and `claude-haiku-5-5`. Record £ cost + latency per turn. Verify `render_tool_call` renders each tool result cleanly. Verify the **Cost tracker page (M7)** shows both turns correctly with per-model aggregates.
- Confirm cache-hit badge fires on the second Anthropic turn (prompt cache).
- If model IDs or pricing changed, patch `_ui/cost_counter.py::PRICE_PER_MTOK` + the picker options.
- Add a lightweight password gate (env var `COST_TRACKER_RESET_PIN`) around the M7 "Clear turn log" button before deploying.
- Carry the recorded cost per Q&A into M11.

Effort: **S (1–2 h)**.

Files touched: `_ui/cost_counter.py`, `pages/1_Ask_the_advisor.py`.

### M11. README refresh + CV link copy  `[deps: M10]`

The recruiter's first click is the GitHub README.

Tasks:
- Rewrite top of `README.md`: one-sentence pitch, deployed-site link as first CTA, one screenshot of the Advisor mid-answer (with tool-call card) **and** one of the Cost tracker page (makes the "they instrumented it" point), the three preset scenarios.
- "What a recruiter should notice" section: dual-backend (gpt-4o-mini + Haiku 5.5), verbatim quotes, measured/modelled tagging, memo download, **live £ cost tracker per Advisor turn**.
- Record real costs from M10: "A typical Advisor Q&A costs ~£0.0004 on gpt-4o-mini; ~£0.0002 on Haiku 5.5 (with prompt cache)."
- Owner-action-items section stays, moves below the pitch.
- Fix any filesystem path references (`AI_INFRA_PROJECT.md` — the broken link from M2).

Effort: **S (1–2 h)**.

Files touched: `README.md`, one screenshot under `docs/img/`.

## Effort rollup

| Milestone | Scope | Effort | Cumulative |
|---|---|---|---|
| M1 | Shared UI primitives + display names | M (3–5 h) | 5 h |
| M2 | Landing rewrite | M (2–3 h) | 8 h |
| M3 | Spec rewrite | M (3–4 h) | 12 h |
| M4 | TCO rewrite (headline page) | L (5–7 h) | 19 h |
| M5 | Validation rewrite + demo data | M (3–5 h) | 24 h |
| M6 | Advisor rewrite + 2-model picker UI | L (4–6 h) | 30 h |
| **M7** | **Cost & token tracker page (NEW)** | **M (4–6 h)** | **36 h** |
| M8 | Market Intel + cold-start | M (3–5 h) | 41 h |
| M9 | Consistency pass + page renames | S (1–2 h) | 43 h |
| M10 | OpenAI + Anthropic key wiring + smoke test | S (1–2 h) | 45 h |
| M11 | README refresh | S (1–2 h) | 47 h |

Rough total: **31–47 hours**. If the owner gives Claude one 4-hour session, M1 lands and M2 is in flight. A 10-hour arc gets M1–M4 (shared primitives + Landing + Spec + TCO headline). A full arc through M11 is a long weekend's work.

## Risk + things that will tempt scope creep

1. **Do not redesign the engine.** `core/cost_engine`, `core/gpu_specs`, `core/agents`, `core/market_intel`, `core/memo` stay frozen. If a UX fix needs an engine change, stop and SendMessage back — the fix is almost certainly wrong.
2. **Do not add new eval cases during the UX rebuild.** The eval set is stable at 20 advisor Qs + 12 market-intel golds. Growing them during UI work muddies attribution of regressions.
3. **Do not introduce a new chart library.** Plotly is already in; `altair`, `bokeh`, `matplotlib` add weight without helping.
4. **Do not rewrite the system prompt** in `compute_advisor.py:526-542`. The review called it well-written. Leave it alone.
5. **Do not touch tests you did not write.** Current 135 pass. If your change breaks one, the test is right until proven otherwise.
6. **Do not spend time on favicon/OG unless M8 confirms.** Streamlit Cloud makes this awkward; the cost-benefit on a demo site is low.
7. **Do not block on real Kaggle T4 data for M5.** The point of synthetic demo fixtures is to unblock the Validation page without owner action.

## Definition of done

Run this checklist at the end of M10. Any fail → go back.

- [ ] Non-technical reader on any page can say within 30 seconds what it does and try something.
- [ ] No `h100_sxm5_80gb`, `tpu_v5e`, `trn2`, `gpu_price`, `FactKind.capex` visible in the deployed UI (`git grep` across `build1_inference_calculator/`).
- [ ] Every slider / number input / dropdown has a plain-English `help=` tooltip.
- [ ] Every hero metric and chart has a one-sentence plain-English caption.
- [ ] Every page starts with `page_header(what/why/where/numbers)`.
- [ ] Every page has at least one preset chip; default = Nscale/Nebius 70B scenario.
- [ ] Spec / TCO / Validation lead with a hero metric; dataframes are below or in expanders.
- [ ] Advisor picker shows **two** options: "OpenAI gpt-4o-mini" (default) and "Claude Haiku 5.5".
- [ ] Advisor shows per-turn £ cost + latency per reply; session total updates.
- [ ] Advisor persists every turn to `data/advisor_turns.duckdb` via `log_turn`.
- [ ] Advisor no-key fallback is a labelled canned transcript, not a schema explorer.
- [ ] Advisor tool-calls render as named cards; raw JSON in nested expander.
- [ ] **Cost tracker page** renders hero row (turns, £, avg £/turn, cache-hit %, avg latency), 4-tile pricing-levers explainer, turn-by-turn table, and per-model comparison chart when ≥2 models are logged.
- [ ] **Cost tracker page** shows a friendly empty-state with a link to Advisor when no turns logged.
- [ ] TCO memo download sits next to the headline, not at the bottom.
- [ ] Market Intel auto-populates on first load (40 facts, zero API spend).
- [ ] Validation shows a non-zero hero accuracy number on first load (demo-data banner if synthetic).
- [ ] Sidebar order Advisor → Cost tracker → TCO → Spec → Validation → Market Intel, with question-shaped labels.
- [ ] `pytest`: **135+ pass**. Any changed test count justified in the commit message.
- [ ] `ruff check build1_inference_calculator core scripts tests`: clean.
- [ ] README first viewport tells the story; deployed-site link is the first CTA.
- [ ] Deployed site renders on Streamlit Cloud with both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` secrets set.

Commit messages: imperative subject + body explaining *why*. One commit per milestone. Example: `UX: shared _ui/ primitives + display-name helpers (M1)`.
