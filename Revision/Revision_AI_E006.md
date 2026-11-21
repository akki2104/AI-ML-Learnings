# Revision — E006: Observability for LLMs

**Topic completed:** 2026-11-21 (passed after retest)
**Spaced repetition:** +1d (2026-11-22) · +3d (2026-11-24) · +7d (2026-11-28) · +15d (2026-12-06) · +30d (2026-12-21)

| Date | Reviewed? | Score |
|------|-----------|-------|
| 2026-11-22 | ⬜ | — |
| 2026-11-24 | ⬜ | — |
| 2026-11-28 | ⬜ | — |
| 2026-12-06 | ⬜ | — |
| 2026-12-21 | ⬜ | — |

**Known weak spots:** session vs trace granularity (case = session, question = trace); "generation" as its own type; say LangSmith (not LangChain); content-gen trace = one product × one retailer job, not a shopper query.

## Active Recall Questions
1. Dashboards green, answers worse — why can't Prometheus/Grafana/Loki find it, and what does LLM tracing add?
2. Trace, span, generation, session, score — using the RAG pipeline.
3. Langfuse vs LangSmith — the two deciding factors; which for medical RAG and why?
4. The improvement loop from a bad production answer to a safe shipped fix.
5. Content-generation product: what is one trace, which scores, which 3 dashboard metrics?

## Answer Key
1. Nothing errors; quality isn't a system metric. Loki can store text but not the nested per-request tree or quality scores. Tracing adds the step tree with full prompts/outputs, tokens, cost, prompt version, and scores (judge, rules, users).
2. Session = case; trace = one question's journey; spans = retrieve, rerank, rule engine; generation = the LLM call (model, prompt, response, tokens, cost); scores = RAGAS metrics from the judge, on the trace or a single step.
3. Data sensitivity → self-hosted open-source Langfuse; framework fit → LangSmith for LangChain/LangGraph stacks with data allowed in SaaS. Medical: Langfuse, PHI can't leave.
4. Trace → find bad traces → add to dataset with correct answers (keep good cases) → change prompt/retrieval/model → experiment on full dataset → compare → move production label (rollback = move back) → monitor online scores.
5. Trace = content job for one product on one retailer; scores = rules passed, factual/unsupported claims (judge), keyword coverage, brand approval; dashboard = approval-as-is rate, unsupported-claim rate, cost per product/customer.
