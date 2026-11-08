# M10 owner action checklist — go-live

Everything that can't be done without live API credentials. The code side of M10 is complete; these are the remaining steps.

## 1. Set secrets on Streamlit Cloud

Go to https://share.streamlit.io/, open the `ai-infra-workbench` app, click **Settings → Secrets**, and paste:

```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."

# Optional — password gate for the Cost tracker "Clear log" button.
# If unset, the button appears unprotected (fine for local dev).
COST_TRACKER_RESET_PIN = "pick-anything"
```

Both API keys are supported. You only need *one* to make the Advisor usable, but setting both lets a visitor flip between gpt-4o-mini and Claude Haiku 5.5 live.

## 2. Smoke test (local)

With `.streamlit/secrets.toml` holding the same values:

```bash
streamlit run build1_inference_calculator/app.py
```

Then on the **Ask the Advisor** page:

1. Pick **OpenAI gpt-4o-mini** (default). Click the first starter question. Expect:
   - Chat reply in 2–4 s.
   - A "Tool calls" section with 1–4 named cards (e.g. 💷 TCO comparison).
   - A per-turn cost line: "This turn: £0.0003 · 2.4 s · 400 in / 300 out / 0 cached (0% cache hit)".
   - Session total updated.
2. Switch the picker to **Claude Haiku 5.5**. Ask the same question. Expect:
   - Similar reply, similar cost (~£0.0002), plus cache-hit % > 0 on the **second** turn thanks to the Anthropic prompt cache.
3. Open the **Cost tracker** sidebar page. Expect:
   - Hero row populated (2+ turns, cumulative £, cache-hit %).
   - If you ran both models, the "Same questions, different brains" bar chart appears.
   - Turn-by-turn table lists every Q&A with input / output / cached columns.

## 3. Verify the exact model IDs

- `gpt-4o-mini` — OpenAI's stable public ID. No surprises expected.
- `claude-haiku-5-5` — confirmed via Anthropic docs (Oct 2026 release). If the API ever returns a `invalid_model` error, check [Anthropic's model list](https://docs.anthropic.com/en/docs/about-claude/models) and update:
  - `build1_inference_calculator/_ui/display_names.py::MODEL_DISPLAY_NAMES`
  - `build1_inference_calculator/_ui/cost_counter.py::PRICE_PER_MTOK`
  - `core/agents/compute_advisor.py::DEFAULT_MODEL`
  - `core/market_intel/extractor.py::DEFAULT_MODEL`

## 4. Record costs for the README (feeds M11)

While smoke-testing, note:

- Typical Q&A cost on **gpt-4o-mini**: £_____
- Typical Q&A cost on **Claude Haiku 5.5**: £_____
- Cache-hit % on second Anthropic turn: _____%

Drop these into `README.md` under "What a recruiter should notice" — M11 wires up placeholders you can swap in.

## 5. Push the deployed site update

```bash
git push origin main
```

Streamlit Cloud auto-redeploys on push. The secrets you set in step 1 persist across deploys.

## Guardrails

- The Advisor's cost counter is **best-effort** — if the SDK returns a usage shape we don't recognise, the counter shows "(cost accounting skipped)" rather than crashing. The chat still works.
- The Cost tracker's "Clear turn log" button requires `COST_TRACKER_RESET_PIN` on deployed instances. Without the PIN set, the button appears unprotected — fine locally, risky on the public URL.
- Every Advisor turn writes a row to `data/advisor_turns.duckdb`. That file is `.gitignore`d so it never gets committed, but it DOES persist across Streamlit Cloud restarts within the same container. Visitors to the public URL share the same log — this is the intended demo behaviour (a public "here's what every Q&A on this site cost" record).
