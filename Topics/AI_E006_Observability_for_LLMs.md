# E006 — Observability for LLMs (Langfuse, LangSmith & co.)

**Tier:** 🟡 SKIM in curriculum → taught as full lesson (interview priority: Forward Deployed Engineer, agentic digital-shelf company)
**Date taught:** 2026-11-21

---

## 1. Why Prometheus / Grafana / Loki are not enough
They answer "is the system healthy?" (rate, errors, latency, log lines). LLM systems fail in ways those can't see:
- **Silent failures** — 200 OK in 2 s, answer is wrong; error dashboards stay green.
- **Multi-step runs** — retrieve → rerank → LLM → tool → LLM must be seen as one **nested tree** with each step's input/output. Loki can store the text, but not the tree or quality scores tied to it.
- **Content is the data** — exact prompt, retrieved chunks, raw response.
- **Cost** — tokens and money per call / request / customer.
- **Versions** — quality per prompt/model version.

Production setup: Prometheus/Grafana for system health **plus** an LLM tool for quality and debugging.

## 2. Core data model (Langfuse and LangSmith are equivalent)

```
Session  (one conversation / one customer case)        session_id
 └── Trace  (one request end to end)                   user_id, tags, metadata
      ├── Span        "retrieve"     input: query       output: 20 chunk ids     12 ms
      ├── Span        "rerank"       input: 20 chunks   output: top 5 + scores   80 ms
      ├── Generation  "llm-answer"   model, prompt, response, tokens in/out, cost, latency
      ├── Span        "tool: get_shipment"  args → result
      └── Score       faithfulness = 0.42 (LLM judge),  user_feedback = 👎
```
- **Trace** = one end-to-end request. **Observation** = a step: **span** (any work), **generation** (an LLM call: model, prompt, completion, tokens, cost, latency), **event** (point marker).
- **Session** groups traces of one conversation/case.
- **Score** = numeric/categorical/boolean on a trace *or a single observation*. Sources: user feedback, human annotation queues, LLM-as-judge evaluators, your own code (rule checks).

Example — medical RAG pipeline: **session = case**, **trace = one question** (a case has many questions), spans = hybrid retrieve, rerank, rule engine; generation = the SLM call; scores = RAGAS faithfulness / answer correctness from the Qwen judge.

## 3. Langfuse
Open source, **self-hostable** (Docker/K8s; v3 = Postgres + ClickHouse + Redis + blob storage) or cloud. Framework-agnostic.
1. **Tracing** with token/cost per model.
2. **Scores & evals** — UI-configured LLM-as-judge evaluators on sampled production traces (*online evals*), annotation queues, SDK scores.
3. **Datasets & experiments** — dataset items (input + expected output = golden set); run the app over a dataset, each run linked to traces; compare runs side by side (*offline evals*).
4. **Prompt management** — versioned prompts with labels (`production`, `staging`), fetched at runtime with caching; generations record prompt version → compare versions, roll back by moving the label.
5. **Dashboards** — cost, latency, usage by model / user / tag / prompt version.

```python
from langfuse import observe, get_client
from langfuse.openai import openai            # drop-in: each OpenAI call → a generation

@observe()                                    # → span
def retrieve(question): ...

@observe()                                    # → trace (outermost)
def answer(question, case_id):
    get_client().update_current_trace(session_id=case_id, tags=["prior-auth"])
    chunks = retrieve(question)
    resp = openai.chat.completions.create(model="...", messages=[...])
    return resp.choices[0].message.content
```
LangChain/LangGraph: pass Langfuse's `CallbackHandler` in the run config → every node, LLM and tool call traced.

## 4. LangSmith
LangChain's hosted platform (closed source; self-host = enterprise only). Zero-setup tracing for LangChain/LangGraph (`LANGSMITH_TRACING=true` + API key), `@traceable` for other code. Datasets, experiments, evaluators, online evaluation, monitoring, annotation queues, prompt hub/playground, LangGraph Studio integration.

| | Langfuse | LangSmith |
|---|---|---|
| Licence / hosting | Open source; self-host or cloud | Proprietary SaaS; self-host enterprise-only |
| Best fit | Framework-agnostic, sensitive data, cost control | All-in on LangChain/LangGraph, want smoothest setup |
| LangGraph setup | Callback handler | 2 env vars |
| Data residency | Full control self-hosted | Their cloud by default |

**Decision rule:** (1) data sensitivity → self-hosted Langfuse; (2) framework fit → LangSmith if LangChain/LangGraph-heavy and data can go to SaaS. Langfuse is a fine default otherwise.

## 5. Other tools to recognise
- **Arize Phoenix** — open source, OpenTelemetry-based, strong RAG/embedding analysis.
- **Helicone** — proxy: change base URL, every LLM call logged; zero code, but blind to retrieval/tool steps.
- **Braintrust** — evals-first. **W&B Weave, MLflow tracing** — ML platforms adding LLM tracing.
- **OpenTelemetry GenAI conventions / OpenLLMetry** — standard span attributes (`gen_ai.request.model`, token usage) → vendor-neutral traces.

**Proxy vs SDK:** proxy = no code, sees only LLM calls. SDK = sees the whole pipeline; needed for agents/RAG because the bug is usually not in the LLM step.

## 6. The improvement loop
```
1. Trace everything in production
2. Find bad traces   ← low judge score, thumbs-down, rule violation, cost spike
3. Add them to a dataset with correct answers (regression tests) — keep GOOD cases too
4. Fix: prompt version / retrieval (top-k, rerank cutoff) / model / temperature
5. Experiment on the full dataset: old vs new on quality, cost, latency
6. Ship: move the `production` label (rollback = move it back)
7. Monitor online scores; repeat
```
Offline evals (5) gate releases; online evals (1–2) catch what the dataset missed.

## 7. What to measure
| Category | Metrics |
|---|---|
| Quality | Judge scores, user feedback rate, human approval rate, rule-check pass rate |
| Latency | p50/p95 per step, time to first token, total request time |
| Cost | Tokens in/out, cost per request / customer / feature |
| Agent behaviour | Tool calls per run, rounds per run, runs hitting the cap, tool error rate |
| Retrieval | Chunks retrieved vs kept after threshold, top rerank score, "no evidence" rate |

## 8. Applied to digital-shelf content generation (the target company)
```
Trace: generate content for SKU 12345 on Amazon
 ├── Span  fetch product data      (catalogue attributes)
 ├── Span  fetch retailer rules    (title ≤ 200 chars, no "best"/"#1", no prices)
 ├── Span  fetch search keywords
 ├── Generation  write title + bullets   (prompt v7, tokens, cost)
 ├── Span  rule checker            (deterministic)
 └── Scores: rules_passed, factual_score (judge: claims not in product data?),
             keyword_cover, brand_approved (as-is / edited / rejected)
```
Dashboard: **approval rate as-is**, **unsupported-claim rate**, **cost per product / per customer**. Later score: did it work (search rank, AI-assistant visibility, conversion).
AI-assistant visibility area: **trace = one shopper-style question sent to an assistant**; scores = brand mentioned?, position, description accurate?

## 9. Common mistakes
- Logging only the final answer (can't tell retrieval vs generation failure).
- Traces without scores.
- Trusting an LLM judge without validating against human labels.
- Sending sensitive data to SaaS without masking (Langfuse supports masking; or self-host).
- Tracing 100% at high volume — sample, but keep every failed/low-scored trace.
- Datasets made only of bad cases — fixes can silently break good ones.

## 10. Interview framing
"Prometheus and Grafana tell me the system is up; they can't tell me an answer is wrong. For that I'd use an LLM tracing tool like Langfuse: each request is a trace with spans for retrieval, reranking and tool calls and generations with full prompts, tokens and cost. Quality scores attach to traces from LLM judges, rule checks and user feedback; bad traces become dataset items, every prompt or model change runs as an experiment on that dataset before release, and labelled prompt versions let me compare and roll back. I'd self-host Langfuse when data can't leave our environment, and use LangSmith for a LangGraph-heavy stack with no data restrictions."

## Glossary
| Term | Definition |
|---|---|
| Trace | One end-to-end request as a tree of steps |
| Span / Generation | A step / an LLM-call step with model, prompt, tokens, cost |
| Session | Group of traces (conversation, case) |
| Score | Quality signal attached to a trace or step |
| Online vs offline eval | Scoring sampled production traffic vs running a fixed dataset before release |
| Prompt label | Pointer (e.g. `production`) to a prompt version; move it to deploy/roll back |

## Cheat sheet
1. Metrics/logs = health; LLM tracing = correctness + why.
2. Session ⊃ trace ⊃ spans/generations; scores on traces or steps.
3. Score sources: judge, rules, humans, users.
4. Langfuse: open source, self-host, framework-agnostic, prompts + datasets + evals.
5. LangSmith: smoothest with LangChain/LangGraph; SaaS by default.
6. Choose by data sensitivity first, framework fit second.
7. SDK instrumentation over proxy for agents/RAG.
8. Loop: trace → bad traces → dataset (+ good cases) → fix → experiment → label → monitor.
9. Track quality, latency (p95, TTFT), cost per customer, agent rounds, retrieval kept-rate.
10. Content-gen dashboard: approval-as-is, unsupported-claim rate, cost per product.
