# AI Compute Economics Workbench

## Purpose

Three finished, demoable builds that show interest in AI infrastructure, a willingness to learn the technical fundamentals, and the ability to execute. The owner is a business-side operator (strategy, operations, GTM) building first-hand fluency in GPUs, CUDA, inference compute and capacity planning.

These builds sit under Selected Builds on a CV. They are evidence of learning and execution, not a claim of professional experience in the domain. Finished and demoable beats broad and half-built.

## Target roles

London and UK roles come first. Current postings in scope:

| Employer type | Examples seen (Nov 2026) | Builds that fit |
|---|---|---|
| Model labs, EMEA | Anthropic data centre capacity delivery and transaction roles (London) | 1, 2 |
| Neoclouds, London | Nscale GTM strategy and planning, customer experience, business operations. Nebius strategic initiatives and GTM insights | 2, 3 |
| Consulting, London | BCG Vantage data centres and AI infrastructure. Accenture AI infrastructure delivery | 1, 2 |
| AI-native companies, UK | AI strategy, operations and GTM roles where compute cost fluency is a differentiator | 1 |

Most dedicated capacity-planning roles (CoreWeave, Google, Lambda, Together AI, OpenAI compute) are US-based. Treat them as secondary. Re-check live postings before tuning a build to one employer.

## Hard constraints

- Budget: about 20 GBP per month for one coding tool, plus 5 to 10 GBP per month for APIs. No GPU rental.
- Open source tools and public data only.
- GPU work runs on free tiers (Kaggle, Colab T4). No measured H100 data exists in this project.
- Python, DuckDB, Streamlit. No Rust.
- Respect site terms and dataset licences. Prefer official APIs over scraping.

## Rules for every number

- Each figure has a source or is marked as an assumption, in a sources column.
- Measured, modelled and simulated data are labelled as such in every output and README.
- H100 figures are modelled, validated on a T4, and cross-checked against MLPerf. Show the error rate against measured data wherever results appear.
- Builds 2 and 3 run on simulated data. They are never presented as real customer, contract or telemetry data.

## Rules for every agent

- The agent never calculates a figure. Every number comes from a tool call to the build's own code.
- It states its assumptions and which inputs are measured, modelled or simulated.
- It declines questions outside the model's scope.
- It ships with an evaluation set of about 20 questions with known answers, and the README reports the pass rate.
- Anything it drafts for a customer or stakeholder goes through a human approval step.

## Build 1: Inference cost and build-vs-rent calculator (priority 1)

Models the cost of serving open-weight LLMs, from hardware specs to cost per million tokens to a rent, buy or API decision.

Components:
- Spec layer: weight size, KV cache per token, fit check, theoretical prefill and decode speed. NVIDIA GPUs first, then Google TPU and AWS Trainium from public specs.
- Validation: predicted vs measured tokens per second on a free T4 for 1B to 3B models, with percentage error, then a cross-check against MLPerf Inference results.
- Token cost engine: cost per million tokens by accelerator, model, quantisation and utilisation, plus the implied margin of API providers.
- Build vs rent vs API: 3-year TCO for owned, reserved cloud, on-demand, neocloud and per-token API, with break-even utilisation and a stress test where demand lands 40% below forecast.
- Price tracker: a weekly scheduled job logging H100, H200 and B200 hourly prices across providers. Start in week 1, since history only exists if logging begins early.
- Learning notebooks: roofline analysis for T4 and H100, and a basic CUDA exercise (vector add, tiled matmul compared against cuBLAS through PyTorch).

Agents:
- Compute advisor. Takes a sizing question ("serve a 70B model at 50 requests per second in the UK for 3 years: rent, buy or API?"), calls the calculator functions, and returns a recommendation with assumptions and sources.
- Market intelligence extractor. Pulls GPU prices, capex guidance and capacity announcements from earnings transcripts and news into structured tables that feed the price tracker. Reuse the AI Value Chain RAG pipeline where possible. Runs in batches to stay inside the API budget.

User-facing output:
- Feels like a deal desk or solutions engineer's pricing tool.
- Inputs: model, accelerator, quantisation, traffic (requests per second, tokens in and out), region, utilisation, financing assumptions.
- Main screen: cost per million tokens, accelerators needed, 3-year TCO table, and a break-even chart where rent, buy and API cost lines cross as utilisation changes.
- Validation tab: predicted vs measured T4 speed with error %, and the roofline chart.
- Advisor chat panel showing the tool calls behind each answer.
- Export: one-page sizing memo (PDF).

## Build 2: GPU capacity planning and allocation simulator (priority 3)

Forecasts demand, allocates constrained capacity across competing workloads, and models delivery timelines. Runs on simulated demand and supply, scaled using public cluster and data centre datasets.

Components:
- Demand forecast across training, batch inference and real-time inference, under three scenarios, with a forecast-vs-actuals tracker that reports forecast error.
- Supply model with GPU deliveries, power (MW), space and lead times as constraints.
- Allocation policies compared: strict priority, fair share and value-based. Reports utilisation, unmet demand and revenue per policy.
- Contract-to-tokens delivery model: power, hardware delivery, networking, burn-in, validation and onboarding, with Monte Carlo on delays.
- Neocloud unit economics: revenue per GPU-hour, power cost, depreciation, financing and contract mix, with cluster break-even utilisation.
- One launch scenario: introducing a new GPU generation, with migration economics and repricing of the previous generation.

Agent (optional, after the build is complete):
- Scenario assistant. Runs what-if simulations ("demand lands 40% low", "power arrives 3 months late") and explains the trade-offs.

User-facing output:
- Feels like a quarterly capacity review pack.
- Inputs: demand scenarios, supply sites, MW, deliveries, lead times, allocation policy.
- Main screen: 24-month supply vs demand chart with the shortfall shaded, and P50/P90 delivery dates per contract.
- Allocation tab: who gets capacity under each policy, unmet demand and revenue.
- Export: capacity plan table (CSV) and summary PDF.

## Build 3: GPU cloud account health view (priority 2)

Flags idle reserved capacity, commitment burn, reliability problems and expansion signals for GPU cloud customers. Adapted from the Customer Health Platform.

How it works without real systems:
- Schema modelled on real sources: GPU telemetry using NVIDIA DCGM metric names, job logs, contracts (reserved GPUs, term, committed spend), billing using FinOps FOCUS columns, and support tickets.
- Usage patterns driven by public anonymised cluster traces (candidates: Alibaba GPU cluster trace, Microsoft Philly trace, Azure LLM inference traces), mapped to a book of about 30 fictional accounts across archetypes (startup ramping, enterprise pilot, bursty research lab, idle reservation, churn risk).
- Planted scenarios with known ground truth, such as 3 weeks of idle reserved capacity or job failures rising after a driver upgrade.
- Scoring reports how many planted events the health score caught and how many false alerts it raised.
- A separate connector layer, so the simulator can be swapped for real telemetry, CRM and billing feeds.

Agent:
- Account review drafter. Drafts weekly notes per account on idle reserved capacity, risks and expansion signals, with a human approval step before anything is treated as final.

User-facing output:
- Feels like a customer success manager's Monday morning dashboard.
- Main screen: portfolio table with health score, trend, idle reserved capacity, commitment burn and open alerts.
- Account drill-down: utilisation over time, job failures, tickets, contract terms, and why each alert fired.
- Drafted review notes with an approve or edit step.
- Export: weekly account review (PDF).

## Agent runtime and budget

- Expose each build's functions as an MCP server so the agents can run from Claude Desktop or Claude Code inside the existing coding-tool plan, with no API spend.
- The hosted demo uses a small API model, kept within the 5 to 10 GBP monthly budget.
- Optional: serve a small open model on the free T4 with vLLM, run the compute advisor on it, and price that run with Build 1. Colab sessions are not persistent, so this is for validation, not the public demo.

## Deliverables per build

- Hosted Streamlit app (free on Streamlit Community Cloud) with a public link
- README stating what is measured, modelled and simulated
- A 2-page memo summarising the decision the build supports
- A 2-minute screen recording
- Agent evaluation results

## Sequence

1. Weeks 1 to 4: Build 1 and its agents. Start price logging in week 1.
2. Weeks 5 to 6: Build 3. Bring forward if an Nscale or Nebius application is live.
3. Weeks 7 to 10: Build 2. Add the scenario assistant only if time allows.

## Out of scope

- Rust, multi-GPU and interconnect hands-on work (paid hardware needed). Interconnect is covered through reading only.
- Cloud cost attribution and FinOps tooling.
- GTM pipeline forecasting, and any quota, territory or sales compensation modelling.
- Training models from scratch, and kernel optimisation beyond the matmul exercise.

## Done means

- Build 1 reports a validation error against measured T4 results and a comparison to MLPerf, and produces a break-even utilisation for each option with every input sourced.
- Build 2 reports forecast error and per-policy results, and produces a delivery date distribution, not a single date.
- Build 3 reports detection and false-alert rates against planted scenarios.
- Each agent has an evaluation pass rate in its README.
- Each app runs from a clean clone with one documented command.
- Each build has a short "how to tell this in an interview" note linked to a target role.

## Left to Claude Code

- Repo structure (one repo or three), libraries, and data pipeline design
- Which free GPU environment and inference engine give the cleanest validation
- How to structure assumptions and sources so they are easy to audit
- Which agent framework or tool-calling pattern to use within the budget
- Questions to the owner when a choice depends on the target role or employer
