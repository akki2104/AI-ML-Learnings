# AI/ML Interview Prep — Schedule

> Self-paced track running alongside the system design curriculum.
> No hard daily deadlines — use this as a target, not a constraint.
> Read `AI_ML_TopicPriority.md` for the recommended session order.

**Started:** 2026-09-19  
**Target completion:** Late October 2026 (6–7 weeks)  
**Rate assumption:** 2–3 sessions/week, 1–1.5h each  
**Total work:** ~25 hours across 16 sessions  

---

## Phase 1 — Vocabulary & RAG Core (Sessions 1–5, Weeks 1–2)

*Goal: Be able to walk through a full RAG pipeline in an interview with full vocabulary.*

| Session | Topics | Est. Time | Focus |
|---------|--------|-----------|-------|
| 1 | A001 Transformers · A002 Embeddings | ~75 min | The vocabulary foundation |
| 2 | B001 RAG Overview · B002 Chunking · B003 Embedding Models | ~110 min | RAG indexing layer |
| 3 | B004 Vector DBs · B005 Dense vs Sparse | ~75 min | Retrieval decision matrix |
| 4 | B006 Reranking · B007 RAG Evaluation | ~80 min | Hands-on retrieval quality work |
| 5 | B008 Advanced RAG · B009 Hallucination Mitigation | ~70 min | Depth topics |

**Phase 1 gate:** You can whiteboard a complete RAG pipeline with justifications for every
tech choice (chunking, embedding model, vector DB, reranker) and name your evaluation metrics.

---

## Phase 2 — Model Hosting & Inference (Sessions 6–8, Week 3)

*Goal: Explain how a 7B-class LLM runs in production — llama.cpp vs vLLM, quantization, numbers.*

| Session | Topics | Est. Time | Focus |
|---------|--------|-----------|-------|
| 6 | A003 Tokens · A004 Providers · C001 Serving Basics | ~70 min | Context window + inference vocab |
| 7 | C002 llama.cpp · C005 Quantization | ~75 min | How the model actually runs |
| 8 | C003 vLLM · C006 Latency/Throughput | ~80 min | Production serving numbers |

**Phase 2 gate:** You can explain the difference between llama.cpp and vLLM, why GGUF
quantization exists, and give approximate throughput numbers for a 7B model.

---

## Phase 3 — Fine-tuning & App Architecture (Sessions 9–12, Weeks 4–5)

*Goal: Know when to fine-tune, how agents work, how to control cost and ensure safety.*

| Session | Topics | Est. Time | Focus |
|---------|--------|-----------|-------|
| 9 | C004 Other Frameworks · C007 Scaling · D001 Fine-tune vs RAG | ~85 min | Decision trees |
| 10 | D002 LoRA · D003 RLHF · D004 Training Data · D005 Evals | ~80 min | Fine-tuning module |
| 11 | E001 Prompting · E002 Tool Calling · E003 Agents | ~95 min | App architecture |
| 12 | E004 Guardrails · E005 Cost · E006 Observability | ~95 min | Production concerns |

**Phase 3 gate:** Given any AI system design prompt, you can classify the problem, apply
the RAG vs fine-tuning vs prompting decision tree, and design the application layer.

---

## Phase 4 — Case Studies (Sessions 13–16, Weeks 6–7)

*Goal: Run a full 45-minute AI system design interview on any of these 6 case studies.*

| Session | Topics | Est. Time | Focus |
|---------|--------|-----------|-------|
| 13 | F001 Document Q&A System (RAG at scale) | ~60 min | Your core hands-on RAG work |
| 14 | F002 LLM Inference Service | ~60 min | Model serving at scale |
| 15 | F003 Multi-Agent Code Review · F004 Recommendation System | ~100 min | Breadth case studies |
| 16 | F005 Content Moderation · F006 RAG + Eval Feedback Loop | ~90 min | Production maturity |

**Phase 4 gate (= interview ready):** You can conduct a 45-minute AI system design interview
on Document Q&A, LLM Inference, or Multi-Agent systems, hitting all 6 scoring dimensions.

---

## Slack Checker

At the end of each week, ask: *"What did the schedule say? What did I cover?"*

| Drift | Action |
|-------|--------|
| ≤ 1 session behind | Normal. Absorb in the next session. |
| 2–3 sessions behind | Combine adjacent 🟡 skim topics into one session. |
| 4+ sessions behind | Skip Module D entirely (D003–D005) and start case studies earlier. Case studies > fine-tuning theory. |
| Ahead | Add back E001 (prompt engineering) or D003 (RLHF) if skipped. |

**Rule:** If forced to choose between one more theory topic and one more case study,
**always take the case study.**

---

## Revision Cadence

Same as system design track: learner revises on his own schedule (weekends / when enough
accumulates). Mentor surfaces the backlog in each session briefing but does not block
new content. Spaced repetition intervals: +1, +3, +7, +15, +30 days from completion.
