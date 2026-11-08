# UX + SME review — Build 1 (AI Compute Economics Workbench)

## Executive summary

- **Landing page tells a recruiter nothing about why they should care.** `app.py:28-56` opens with "Inference cost & build-vs-rent calculator" then a jargon-dense bullet list (roofline, MLPerf, break-even utilisation, 40%-demand-miss stress) before explaining who this is for or showing one example output. A non-technical reader bounces inside 30 seconds.
- **Every page is a control panel without a story.** Spec / Validation / TCO follow an identical shape: title, one-line caption, dense widget grid, raw dataframe. No "try this scenario", no annotated worked example, no "what question this page answers". The viewer is handed the controls before being shown what the car does.
- **SME layer is credible, UX layer hides it.** The engine does roofline + KV-cache + 5-way TCO with measured/modelled tagging, break-even math, demand-stress, implied-API-margin, verbatim-quote extraction, and tool-call transparency. All of this is exactly what a Nscale/Nebius panel wants to see, and almost none of it is legible in a first glance.
- **Advisor is the strongest page by a wide margin.** Chat + transparent tool-call panel is immediately legible to a non-technical reader. It should be the front door, not the fourth tab.
- **Salvageable with layout fixes, not a rebuild.** Information model is sound. What's missing is (a) a landing narrative, (b) presets per page, (c) plain-English captions on every number, (d) suppression of internal IDs like `h100_sxm5_80gb`, (e) a consistent "question this page answers" header. Two to three days of UX work, zero engine changes.

## Cross-cutting issues

**BLOCKER — No orientation.** `app.py` opens with "Pages: Spec, Validation, TCO" and no "what / who-for / 30-second use". Needs a hero: the question this tool answers ("Rent, buy, or API an H100 for Llama-3-70B?"), a worked example, *then* the page list.

**BLOCKER — Raw internal IDs in every selectbox.** `h100_sxm5_80gb`, `tpu_v5e`, `trn2` appear in GPU pickers at `1_Spec.py:80`, `2_Validation.py:52`, `3_TCO.py:41`, and in `compute_advisor.py:329-335`. Show "H100 SXM5 (80 GB)"; keep the ID in tool-call JSON only.

**BLOCKER — Unit-less, caption-less numbers.** `1_Spec.py:108-111` shows four `st.metric` tiles ("Weights 32.0 GB", "KV cache 2.1 GB", "Activations (allowance) 2.0 GB", "GPU HBM 80 GB"). A non-technical reader doesn't know what a KV cache is or why these should sum against HBM. Every metric block needs a one-sentence plain-English caption underneath.

**BLOCKER — No preset scenarios.** The demo angle is "Llama-3-70B on H100, Nscale/Nebius neocloud" — that exact scenario should be one click. Every page opens on `Llama-3-8B` defaults (`2_Validation.py:54`, `3_TCO.py:43`). Add 3–4 preset chips per page.

**HIGH — Jargon without definitions.** Undefined on the page they appear: *roofline* (`2_Validation.py:48`), *arithmetic intensity* (line 83), *kernel utilisation of peak* (`2_Validation.py:56`, `3_TCO.py:55`), *KV cache* (`1_Spec.py:109`), *busy fraction* (`3_TCO.py:45`), *PUE* (line 66), *FP8/INT4* (`1_Spec.py:84`), *break-even* (`3_TCO.py:148`), *stranded capex* (line 164). All need `help=` tooltips and a one-liner on first use.

**HIGH — No visual hierarchy.** Every section is `st.header` + inline widgets. No "input panel" vs "answer" separation; the reader can't tell which numbers are *the* numbers.

**HIGH — No narrative throughline between pages.** Spec → Validation → TCO → Advisor → Market Intel is the engine-build order, not a user story. Page names don't signal "catalog → does my model fit → is the prediction trustworthy → what does it cost → what does the market charge".

**HIGH — Provenance tagging is dropped after the GPU catalog.** Traffic-light badges at `1_Spec.py:37-38` are the strongest trust signal in the app, and the pattern appears nowhere else. On `3_TCO.py` nothing in the headline table marks which row is modelled vs assumption. Promote the tag to a visible column on every table.

**MEDIUM — Dataframes everywhere.** `st.dataframe` with 10+ columns at `1_Spec.py:54`, `2_Validation.py:115,137,200`, `3_TCO.py:131,160`, `5_Market_Intel.py:95`. Poison for a 30-second scan. Each page needs a hero metric or annotated chart *above* the dataframe.

**MEDIUM — "Not populated" states read as half-built.** `2_Validation.py:126,158,169` all tell the user to "run the notebook" or "drop files". Either bundle demo data or rewrite as "This panel populates once T4 measurements commit — see methodology".

**MEDIUM — Inconsistent `st.set_page_config`.** Set in `app.py:22`, `4_Advisor.py:31`, `5_Market_Intel.py:30`; missing in Spec/Validation/TCO. Call once in `app.py`.

**LOW — Broken link.** `app.py:60-62` links to `AI_INFRA_PROJECT.md` which Streamlit Cloud won't serve. Use a GitHub URL.

**LOW — No favicon / OG image / meta description.** A link pasted into Slack renders bare.

## Per-page critique

### Page: Landing (`app.py`)

**First-30-second read.** Title, a paragraph about serving open-weight LLMs, a bullet list of three pages whose jobs the reader doesn't know, and a status table whose middle column reads `core/cost_engine/roofline.py + data/processed/t4_smoke_test.json` (`app.py:55`). Reads as a developer readme.

**SME concerns.** The measured/modelled/simulated/assumption posture at `app.py:36` is right but buried. Status table at lines 54-55 leaks build state ("pending first Kaggle run", "weekly cron") — an SME reader reads "unfinished" where a mature product would show "methodology".

**UX issues.** No hero, no sub-title. Page descriptions (lines 40-47) explain *what each page does* in internal terms, not *what question each answers*. Filesystem paths on a public surface. No sample output, no "try this first" link.

**What's missing.** One-sentence positioning. A worked-example callout. Author/context line. A "Start with the Advisor" CTA.

**Suggested direction.** Three stacked blocks: (1) one-sentence what-this-is; (2) one worked-example card with a mock TCO bar chart and a one-line verdict; (3) a four-button "Jump to" panel. Collapse the status table into `st.expander("Methodology & data status")`.

### Page: Spec (`1_Spec.py`)

**First-30-second read.** A 12-column GPU table with codes, TFLOPS and bandwidth; a five-dropdown form; a "fits / does not fit" verdict. Non-technical reader bounces.

**SME concerns.** Catalog is solid — dense tensor-core figures, confidence tagged, Rubin flagged low, citation dates on every row. Memory-fit uses a reasonable 2 GB activation allowance (line 98). One nit: `_dtype_bytes` is used uniformly for weights *and* KV, ignoring that serving stacks often run FP16 weights with INT8 KV. The caption at lines 113-117 flags single-GPU; it should also flag the KV-dtype assumption.

**UX issues.** `view_cols` (lines 40-53) leads with the internal `id` column — drop it. Raw snake_case column names show through. Five-widget row (lines 78-88) wraps badly on narrow viewports. Four isolated `st.metric` tiles (lines 107-111) communicate less than one stacked-bar of weights + KV + activation against the HBM ceiling. The actionable insight ("70B doesn't fit in FP16 on 80 GB") is buried in the bottom caption.

**What's missing.** Plain-English intro. Preset chips. Tooltips on FP16/FP8/INT8/INT4. "What this tells you" one-liner after the result.

**Suggested direction.** Flip the page: memory-fit calculator on top (the answer), catalog below (the reference). Lead with a plain-English question, preset chips, then one stacked-bar chart. Push the full catalog into `st.expander("Browse full catalog")`.

### Page: Validation (`2_Validation.py`)

**First-30-second read.** A log-log chart with no caption explaining what a roofline *is*, two tok/s charts with "aggregate" vs "per request" unexplained, then `st.warning`/`st.info` that files don't exist (lines 126-129, 158-161). Reads as half-built.

**SME concerns.** Roofline methodology is correct and the caveats are in the docstring. The chart (lines 64-88) plots achievable TFLOPS vs arithmetic intensity *without* overlaying the operating point of the selected model — a roofline is only useful when you can see where decode and prefill land on it. The compute-ceiling annotation (line 78) is drawn but there's no memory-bandwidth ceiling line, which defeats the point of a roofline.

**UX issues.** Caption (lines 34-38) uses "roofline"/"vLLM"/"MLPerf" undefined. "Kernel utilisation of peak" slider (line 56) is a term of art — no mental model for setting 65%. Two charts (lines 103-113) titled "aggregate" vs "per request" with no explanation. `st.json(smoke)` at line 124 dumps raw JSON on a public demo. MLPerf panel (lines 164-201) is the most SME-valuable piece in the app; if the CSV isn't bundled, it's dead weight.

**What's missing.** A top-of-page headline accuracy metric ("predictions within X% of measured T4; within Y% of MLPerf H100"). Operating points overlaid on the roofline. Bundled demo data. A "why this matters" one-liner.

**Suggested direction.** Lead with the headline accuracy metric (the one number this page exists for). Below it: the roofline with memory ceiling, compute ceiling, and operating points. Push batch-sweep tables into an expander. Bundle one T4 run and one MLPerf round so no panel is empty.

### Page: TCO (`3_TCO.py`)

**First-30-second read.** Six header sections, ~20 numeric inputs before any output, then two bar charts and three dataframes. Most important page in the app and the most hostile to a first viewer — reads as a bank model.

**SME concerns.** The SME-strongest page. Straight-line depreciation over 3 years, financing + maintenance + PUE + hosting folded into an hourly rate, five options on `$/M output tokens`, break-evens, demand-stress. The "owned/reserved/neocloud pay full capacity under a demand miss" framing (lines 162-166) is the single most defensible piece of the Nscale/Nebius angle.

Two nits: (1) `OwnedEconomics` financing is a `%` but the UI doesn't say whether it's applied to outstanding balance or full capex — memo should disclose. (2) `busy_fraction=0.60` (line 45) is reasonable *and* the most sensitive input in the model — a 60% → 80% swing can flip the recommendation. Sensitivity needs to be visible, not hidden.

**UX issues.** 20+ widgets before any output. Headline table (line 131) is a raw dataframe — a hiring panel should see one headline ("Neocloud at $1.90/M, 3.4× cheaper than on-demand"). Default hourly rates (lines 90-94) are unsourced in the UI. Break-even bullets (lines 150-153) print `be.pivot_variable = 0.472` — opaque; needs rendering as "Owning beats renting if you're decoding more than 47% of the time". Demand-stress slider (line 157) drops the user into stress mode without framing. Memo download (lines 176-207) is a strong feature buried at the bottom.

**What's missing.** A "what this answers" paragraph at the top. Preset chip row. A headline recommendation card above the dataframe. A sensitivity bar for `busy_fraction`.

**Suggested direction.** Invert the page. Preset + locked headline ("Neocloud at $1.90/M"). Two charts (cost bar, demand-stress bar). "Tune the inputs" as an expander with all 20 widgets. Memo download prominent. Turn a form into a story.

### Page: Advisor (`4_Advisor.py`)

**First-30-second read.** Clear title, sensible caption, and either (a) a warning + raw tool explorer if no key, or (b) a chat with a placeholder "Serve 70B at 50 rps for 3 years — rent, buy or API?". The chat pattern is the single most legible thing in the app. The tool-call expander (lines 89-92, 111-115) is the technical-credibility payload.

**SME concerns.** The system prompt (`compute_advisor.py:526-542`) is well-written: "You never calculate a figure yourself", explicit output format, labelling discipline. One concern: `api_margin` returns a `gross_margin_pct` that could read as an accusation. The "Not an accusation — a reasoned benchmark" caveat lives in JSON the reader never sees; the chat answer should surface it.

**UX issues.** Tool-call expander dumps raw `st.json` — a GTM lead wants named cards with JSON on request. The no-key path (lines 43-74) renders a developer-grade schema view with a `st.text_area` for JSON params — it's what a visitor sees if the key isn't set. Should be a canned-transcript demo. Usage line (lines 116-121) shows raw tokens, no £ translation. Placeholder question is hardcoded (line 94); no example-question chips. No "clear chat".

**What's missing.** Example-question chips. Per-turn cost in £. A one-line "what tools this agent has" summary near the top.

**Suggested direction.** Already the strongest page. Promote it in the sidebar and link prominently from landing. Named-card tool-call display, starter-question chips, per-turn cost, bundled transcript for no-key visitors.

### Page: Market Intel (`5_Market_Intel.py`)

**First-30-second read.** Metric tiles, sidebar filters, a filtered table with verbatim quotes, two aggregates. The verbatim-quote column is correctly front-and-centre.

**SME concerns.** "Quote never invented" is the SME-valuable part. Two nits: (1) "GPU price snapshots" (lines 109-116) uses `min(value_num)` as `latest_usd_per_hour` — that's the *cheapest* observation, not the *latest*. Mislabelled. (2) "Capex by company" (lines 122-137) sums capex across periods — $10B 2026 + $5B 2025 for the same company produces a number that doesn't mean anything. Needs period filtering or a visible caveat.

**UX issues.** Nine-column dataframe (lines 91-94) fights for width. Empty-store state (lines 40-45) shows nothing — bundle sample facts. `FactKind` enum values leak into the filter dropdown as `gpu_price`, `gpu_shipment` (lines 70-71). No chart on a corpus-of-numbers page.

**What's missing.** A top-of-page explanation. Bundled sample data. One chart. A methodology expander — the extractor has a 12-case gold eval no visitor would ever learn about.

**Suggested direction.** Lead with positioning ("40 facts from AI-infra public filings, 100% quote-valid"). One headline chart. Filterable table. Methodology in an expander.

## Advisor page — special note on OpenAI wiring

Dual-backend plumbing is already in place (`compute_advisor.py:569-795`, auto-detect via key prefix). The UI needs to catch up.

- **Model picker at top of page, not auto-detect alone.** A visible `st.selectbox("Model")` above the chat: "Claude Haiku 4.5", "gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini". Store in `st.session_state`, rebuild `ComputeAdvisor` on change. Auto-detection is a fine fallback; a visible picker is the demo-worthy affordance — a reader sees "works on Anthropic and OpenAI; owner has thought about tier vs cost".
- **Per-turn cost in £ (UK-based owner).** Replace raw token counts (`4_Advisor.py:116-121`) with a two-line display: "**This turn: £0.0032** (142 in / 318 out, 1,024 cached)" and "**Session total: £0.014**". A small per-model £/MTok dict is enough. For a cost-conscious non-technical reader, a cumulative £ meter is far more legible than token counts.
- **Latency per turn.** "Responded in 2.4 s" alongside cost. Readers notice latency; they don't read tokens.
- **Cache-hit badge on Anthropic.** `cache_read_input_tokens` and `cache_creation_input_tokens` are already in the usage dict. A "Cache hit: 92%" badge is a one-line interview talking point.
- **Picker location.** Top of the Advisor page, not in the sidebar — the picker belongs with the chat.
- **Scope caveat.** The picker only affects Advisor. Market Intel's extractor uses its own backend-detect. Don't imply the picker is global.

## What's working

Preserve these — a redesign must not throw them out:

- **Traffic-light confidence badges on the GPU catalog** (`1_Spec.py:37-38`). The strongest single trust signal in the app. Extend to TCO rows and Market Intel facts.
- **"Every number is tagged measured/modelled/simulated/assumption"** (`app.py:36`). The differentiator from a thousand other cost calculators. Make it louder.
- **Tool-call transparency on the Advisor** (`4_Advisor.py:89-92,111-115`). Most demo-worthy pattern in the project. Just make the JSON more readable.
- **Verbatim-quote extraction on Market Intel** (`5_Market_Intel.py:92-103`). Credibility gold for a grounded-LLM pipeline.
- **Memo download on TCO** (`3_TCO.py:176-207`). A downloadable 2-page sizing memo is exactly what a Nscale deal-desk reviewer would produce.
- **Dual-backend architecture.** Interview talking point on avoiding vendor lock-in. Surface in the UI.
- **Demand-stress framing** (`3_TCO.py:156-166`). Exactly the question a GTM/deal-desk reader asks. Promote to a headline chart.
- **Modular engine** (`core/cost_engine/`). Every page reuses the same functions; the agent calls the same tool wrappers. UX redesign only touches `build1_inference_calculator/`.

## Open questions for the owner

1. **Headline demo scenario?** README says Llama-3-70B + H100 + Nscale/Nebius neocloud. Hardcode it as a pre-run landing-page worked example?
2. **Is Market Intel part of Build 1 or Build-3-adjacent?** Currently an orphan with no narrative link to inference cost. Reframe as "prices and capex that validate TCO assumptions", or move it out?
3. **Reorder the sidebar?** Advisor → TCO → Spec → Validation → Market Intel would put the strongest page first.
4. **Collapse Spec into TCO?** The memory-fit widget is a precondition for TCO. Merging cuts 5 → 4 pages and tightens the story.
5. **Bundle demo data for Validation?** The two "run the notebook" placeholders make the page look half-built. Committable mock T4 + one MLPerf round?
6. **Memo as PDF?** Markdown-only works for sophisticated readers; deal-desk expects PDF. Ship `pandoc` server-side or relabel the button?
7. **Copy-pass scope.** ~30 jargon terms named here need tooltips. Full sweep now, or triage to the 5–10 most visible?
