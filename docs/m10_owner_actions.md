# M10 owner action checklist — private deploy + keys

Everything you need to do to go from "code is ready" to "I can test the live app myself with real API keys, and no-one else can reach it." Owner-blocked: nothing here can be done from Claude Code.

**Decision locked in (2026-11-08):** deploy **Private on Streamlit Cloud**, viewer whitelist = one email (yours). URL stays stable. Zero code changes required.

---

## Step 1 — Flip the app to Private on Streamlit Cloud (2 min)

1. Go to **https://share.streamlit.io/**
2. Sign in with the GitHub account that owns the repo (**DV478**). If you see a different account, sign out and back in — the app only appears under the owner account.
3. You should see a tile for **ai-infra-workbench**. Click it (not the ⋮ menu — click the app name).
4. Top-right corner, click **⋮ menu → Settings**. A side-panel opens.
5. Click the **Sharing** tab in the side-panel.
6. Under **Who can view this app**, change from **Public** (the default) to **Private**.
7. Under **Viewers**, click **+ Invite viewers** and paste your email:
   ```
   dereck.varghese@gmail.com
   ```
   Press Enter or the ✓ button to confirm. The chip should turn green.
8. Click **Save** at the bottom of the panel.

**You should now see "Private" with a lock icon** next to the app name in the Streamlit Cloud dashboard. If you don't, refresh the dashboard.

### Verify privacy worked

- Open **ai-infra-workbench.streamlit.app** in a **private browsing / incognito window** (so your logged-in session doesn't leak through).
- You should see a Streamlit sign-in screen, **not** the app.
- Close the incognito window.
- In your normal browser (where you're signed in to Google as dereck.varghese@gmail.com), reload **ai-infra-workbench.streamlit.app**. The app should load.

If the incognito window loads the app, privacy didn't save — go back to Step 1.6 and re-check.

---

## Step 2 — Set spending caps on both providers BEFORE wiring the keys (5 min)

This is belt-and-braces. Even though the app is private, bugs happen (a leaked key, a runaway loop). Setting a hard cap is cheap insurance.

### OpenAI

1. Sign in at **https://platform.openai.com**
2. **Settings** (top right) → **Billing** → **Limits**
3. Set **Monthly budget**: `£10` (equivalent ~$12.50 at current FX). This is a hard cap — above this, API calls return a 429 error instead of charging.
4. Optional: set **Soft limit** to `£5` so you get an email alert when half-spent.
5. Save.

Get the key:
6. **API Keys** (left sidebar) → **+ Create new secret key**
7. Give it a name like `ai-infra-workbench-cloud`
8. **Copy the key now** — OpenAI shows it once, then hashes it. It looks like `sk-proj-...` or `sk-...`
9. Paste it into a scratchpad temporarily. You'll move it to Streamlit secrets in Step 3.

### Anthropic

1. Sign in at **https://console.anthropic.com**
2. **Settings** (gear icon) → **Billing** → **Spending limits**
3. Set **Monthly spend limit**: `$12` (close to £10). Same hard-cap behaviour.
4. Set **Email alert threshold**: `$6` so you get warned early.
5. Save.

Get the key:
6. **API Keys** (left sidebar) → **Create Key**
7. Name: `ai-infra-workbench-cloud`
8. **Copy immediately**. Looks like `sk-ant-...`
9. Paste into the same scratchpad.

**⚠️ Important:** The scratchpad is only to carry the keys from the provider dashboards to Streamlit secrets within the next few minutes. Delete the scratchpad as soon as you've pasted into Streamlit.

---

## Step 3 — Paste both keys into Streamlit secrets (2 min)

1. Back to **https://share.streamlit.io/**
2. Click the **ai-infra-workbench** app
3. **⋮ menu → Settings**
4. Click the **Secrets** tab in the side-panel
5. In the editor, paste this exact structure, replacing the placeholders with your real keys:

```toml
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxx"
ANTHROPIC_API_KEY = "sk-ant-xxxxxxxxxxxxx"

# Optional — PIN for the Cost tracker "Clear log" admin button.
# Any string you want. Leave this line out if you don't care.
COST_TRACKER_RESET_PIN = "pick-anything-you-like"
```

6. Click **Save**.
7. Streamlit auto-redeploys in ~30 seconds. You'll see a "Rebooting…" banner, then the app comes back.

**Delete the scratchpad with the keys.** The Streamlit Secrets panel is now the only copy.

---

## Step 4 — Smoke test the live app (5 min)

1. Open **ai-infra-workbench.streamlit.app** (normal browser, signed-in Google account).
2. You should land on the new hero page: *"Should you rent, buy, or API an H100 for Llama-3-70B?"*
3. Click **🤖 Ask the Advisor** or use the sidebar.
4. The model picker should default to **OpenAI gpt-4o-mini**.
5. Click the first starter chip: *"Serve Llama-3-70B at 50 rps for 3 years — rent, buy, or API?"*
6. Expected within 2–4 seconds:
   - A markdown answer with a recommendation.
   - A **Tool calls** section with 1–4 named cards (📏 Memory fit check, 💷 TCO comparison, ⚖️ Break-even).
   - A cost line below: `This turn: £0.00XX · 2.4 s · 400 in / 300 out / 0 cached (0% cache hit)`.
7. Switch the picker to **Claude Haiku 5.5**. Ask the same question. Expected:
   - Similar answer, similar cost (~£0.0002).
   - On the **second turn** with Haiku, the cache-hit % will be > 0 — Anthropic's prompt cache fires on repeated system prompts.
8. Open the **Cost tracker** sidebar page. Expected:
   - Hero row showing 2+ turns, total £, cache-hit %.
   - "Same questions, different brains" bar chart comparing the two models.
   - Turn-by-turn table with your two Q&As.

### If step 6 fails

- **401 / Invalid API key**: the key got mangled in paste. Re-check the Streamlit Secrets panel — no quotes missing, no trailing whitespace.
- **"You picked gpt-4o-mini but there's no OPENAI_API_KEY in secrets"**: the secret name is wrong. It must be exactly `OPENAI_API_KEY` (all caps, underscores, not `openai_api_key` or `OpenAI`).
- **Model not found / 404**: the Claude Haiku 5.5 ID may have changed since Nov 2026. Go to [docs.anthropic.com/en/docs/about-claude/models](https://docs.anthropic.com/en/docs/about-claude/models), find the current ID, and update in four places:
  - `build1_inference_calculator/_ui/display_names.py::MODEL_DISPLAY_NAMES`
  - `build1_inference_calculator/_ui/cost_counter.py::PRICE_PER_MTOK`
  - `core/agents/compute_advisor.py::DEFAULT_MODEL`
  - `core/market_intel/extractor.py::DEFAULT_MODEL`
- **App stuck on "Rebooting…"**: wait 60 seconds. If still stuck, click **⋮ menu → Reboot app** manually.

---

## Step 5 — Record real costs for the README (3 min)

The README has placeholder numbers (~£0.0004 on gpt-4o-mini, ~£0.0002 on Haiku 5.5). After your smoke test:

1. Open the **Cost tracker** page.
2. Look at the "Avg per turn" hero metric for each model.
3. If the real number is more than ±50% off from the README placeholder, update:
   - `README.md` (under "What a recruiter should notice")
   - Any mention in the UX rebuild plan / status doc

Commit the update with a short message like: `Docs: record real per-turn £ cost from first live run`.

---

## Step 6 — Going public later (one-flip switch)

When you want recruiters to click the link:

1. **https://share.streamlit.io/** → app → Settings → Sharing
2. Change **Private → Public**. No viewer list needed.
3. Save.

That's it. The URL and all secrets stay the same. The spending caps you set in Step 2 become your real safety net at that point.

If you want a middle step — share with 1–2 specific recruiters before going fully public — add their emails as viewers while keeping Private.

---

## Guardrails worth remembering

- **Spending caps on both provider dashboards are the real safety net**, not obscurity. Keep them in place even after going public.
- **`data/advisor_turns.duckdb` is gitignored** but persists inside the Streamlit Cloud container. If a visitor logs an embarrassing Q&A, open the Cost tracker's Admin section (requires `COST_TRACKER_RESET_PIN` if set) and clear the log.
- **Secret rotation**: if a key ever leaks, delete it in the provider dashboard first (kills it everywhere), then paste the new one into Streamlit Secrets. No redeploy needed.
- **Viewer list ≤ 5 on free tier**: if you want to add more than 5 viewers, you'd need Streamlit's paid tier. For portfolio use, 5 is plenty.
