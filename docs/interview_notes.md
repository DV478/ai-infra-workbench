# How to tell each build in an interview

Spec requirement: each build has a short "how to tell this in an interview" note linked to a target role. These are talking-track skeletons, not scripts. Keep them to 60 seconds in person.

## Build 1 — Inference cost + build-vs-rent calculator

### Audience: Nscale / Nebius (GTM / strategy / solutions engineering, London)

**The one-liner**
"I built a calculator that takes an inference workload, a GPU, a quant, and a set of hourly rates, and gives you 3-year TCO against owning, reserving, on-demand cloud, neoclouds like yours, and per-token APIs — on one denominator: dollars per million output tokens."

**The number that proves you did the work**
"Every figure traces to a source row in DuckDB — GPU specs come from vendor datasheets I cited and dated; throughput numbers come from a roofline model validated against measured Kaggle T4 runs and cross-checked against MLPerf Inference. Headline error rate against measured T4 is ~15-20%, which is the error bar on every H100 number too."

**Why they care**
- Neoclouds live or die on $/M-tokens math vs the hyperscaler reserved rate and vs the API curve. The calculator surfaces the break-even utilisation and the 40%-demand-miss stress row that makes or breaks the model.
- The "build vs API" chart is the exact artefact a prospect asks for in a pricing call. You've already built the thing their deal desk needs.

**Why *not* claim more than you did**
"I'm not claiming inference infra experience — the point of this build is that I can read a datasheet, understand the math, and talk about it with your engineers without wasting their time."

### Audience: Anthropic (data centre capacity delivery / transactions, London)

**The one-liner**
"I modelled the full cost stack for serving open-weight LLMs — from HBM bandwidth and KV cache, through roofline throughput, to 3-year build-vs-rent-vs-API decisions. I'm especially interested in how model labs think about capacity commitments vs on-demand headroom."

**The number that proves you did the work**
Same as above, framed around the owner-vs-reserved break-even: "At X capex, Y hourly reserved rate, Z busy fraction, the model says owning beats reserving at 42% utilisation — which means if your demand forecast confidence interval straddles that line, neither is the right answer and you want burstable capacity."

**Why they care**
- They need people who can translate forecast demand → GPU-hours → procurement decisions without needing to ask engineering twice.
- The TCO stress test (40% demand miss) is literally the conversation around capacity commitments.

### Audience: BCG Vantage / Accenture (data centres & AI infra consulting, London)

**The one-liner**
"I wrote a sizing calculator in Python + DuckDB + Streamlit that gives you cost per million tokens by model, GPU, quant and utilisation — modelled, validated on T4, cross-checked against MLPerf — so I can run build-vs-rent-vs-API scenarios without a team or a GPU budget."

**Why they care**
Consulting deliverables around AI infra are TCO memos, capacity plans, and vendor evaluations. You've shipped a working sizing tool and labelled every number by provenance. That is the muscle you'd apply to a client engagement.

### What to avoid saying
- "I built an inference engine" — I didn't. I built a cost model that wraps `vLLM` measurements.
- "I measured H100 performance" — I didn't. I modelled it, cross-checked against MLPerf-reported runs, and said so on every chart.
- "I benchmarked CUDA kernels" — I ran *one* CUDA matmul exercise end-to-end as a learning note. That is the claim.

---

## Build 3 — GPU cloud account health view (CS adaptation of prior work)

### Audience: Nscale / Nebius (customer success / solutions engineering, London)

**The one-liner**
"I adapted a customer-health platform I'd built for SaaS to the specifics of a GPU cloud: DCGM telemetry, reserved-capacity burn, cluster trace archetypes, and planted scenarios so the health score can be scored against known ground truth."

**The number that proves you did the work**
"Against planted scenarios like '3 weeks of idle reserved capacity' or 'driver upgrade regression', the health model caught X of Y with Z false alerts. That's reported in the README."

**Why they care**
- At a neocloud, idle reserved capacity is the single biggest reason an account churns or downgrades at renewal. A CS team with this view can intervene two months earlier.
- The connector layer is designed for a real telemetry swap — the simulator is what lets the build exist without a live customer environment, not the product.

---

## Build 2 — Capacity planning & allocation simulator

### Audience: Anthropic / Nscale / BCG (capacity / strategy)

**The one-liner**
"I built a 24-month supply vs demand simulator with GPU deliveries, power, space, and lead times as constraints — three allocation policies compared on utilisation, unmet demand, and revenue — plus a Monte Carlo on delivery delays."

**Why they care**
- Capacity planning roles measure success in dollars left on the table from missed deliveries and over-commitments. The simulator makes those trade-offs visible.
- The new-GPU-generation launch scenario (H100 → B200 → B300 migration, Rubin on the horizon) is exactly the transition conversation on neocloud roadmaps right now.

---

## Common thread across all three builds

"I'm a business operator making a calibrated bet on AI infrastructure as a career direction. I can't claim engineering seniority I don't have. What I can show is that I can do the math, build the thing, and explain the trade-offs — without hand-waving over measurement errors or source quality."

That line is the point of every build here.
