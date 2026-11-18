# AI/ML Interview Prep — Master Guide

> **Who this is for:** Akash Yadav, Associate Software Engineer.
> Hands-on experience with RAG pipelines, LLM inference, pgvector, cross-encoder
> reranking, LangGraph agents, and LLM tool-calling. Strong backend/distributed systems base.
> Gap: interview framing — how to explain architectural decisions, compare approaches,
> and answer "design an AI system" questions under interview conditions.
>
> **Target roles:** LLM Engineer · Applied ML Engineer · AI Platform Engineer
> at product companies (Swiggy, Zepto, Razorpay, Juspay, FAANG AI teams).
>
> **This guide is the operating manual** for every AI/ML prep session.
> Read it at the start of each session before anything else.

---

## Table of Contents

- [§0 — Session Workflow](#0-session-workflow)
- [§1 — Topic Roadmap (6 Modules, 36 Topics)](#1-topic-roadmap)
- [§2 — Interview Scoring Rubric (6 Dimensions)](#2-interview-scoring-rubric)
- [§3 — Teaching Persona & Style](#3-teaching-persona-and-style)
- [§4 — Lesson Structure](#4-lesson-structure)
- [§5 — The Master Decision Tree: RAG vs Fine-tuning vs Prompting](#5-master-decision-tree)
- [§6 — Tech Decision Playbook Seeds](#6-tech-decision-playbook-seeds)
- [§7 — Company Flavor Notes](#7-company-flavor-notes)
- [§8 — Mastery Gate](#8-mastery-gate)
- [§9 — File Roles & Maintenance Rules](#9-file-roles-and-maintenance-rules)

---

## §0 — Session Workflow

### §0.1 Daily Briefing (say this at the start of every AI/ML session)

```
📅 Today: [date — verify from system context, never recall from memory]
📍 AI/ML Schedule says: [module + topics due today]
✅ Actually done: Topics [list from AI_ML_Progress.md]
⚡ Pace: [ahead / on track / behind]
🔁 Revisions due: [list from Revision/ or "none"]
🎯 Today's goal: [specific topics to cover]
```

### §0.2 Session Steps

0. **Verify the real date from system context** — never use a date recalled from earlier in
   the conversation. Date errors corrupt spaced-repetition schedules.
1. Read this guide (constitution)
2. Read `AI_ML_Progress.md` — find current position and next topic
3. Read `AI_ML_Schedule.md` — find today's slot, give the daily briefing above
4. Check `Revision/` — surface any overdue active-recall files (do not block new content)
5. Read `AI_ML_TopicPriority.md` — show the Topic Briefing Card for today's topic before
   starting (tier / estimated time / why it matters / skip cost)
6. Deliver the lesson per §3 (teaching style) and §4 (lesson structure). Every lesson touching a tech (vLLM, pgvector, etc.) includes
   a "When to use / When NOT to use" box, appended to `AI_ML_TechChoices.md`
7. After session: update `AI_ML_Progress.md`, `AI_ML_TopicPriority.md` (mark done),
   add revision entries, commit and push

### §0.3 Topic Briefing Card (show before every topic)

```
┌─────────────────────────────────────────────────────────────┐
│ TOPIC [ID]: [Name]                                          │
│ Tier: 🔴 MUST / 🟡 SKIM / ⚫ SKIP                          │
│ Estimated time: [X min]                                     │
│ Complexity: [Low / Medium / High]                           │
│ Skip cost: [what you lose if you skip this]                 │
│ Interview frequency: [Common / Moderate / Rare]             │
└─────────────────────────────────────────────────────────────┘
```

### §0.4 Commit Rule

After every completed topic, commit and push:

```bash
git add .
git commit -m "Topic [ID]: [Topic Name] — lesson + revision + tracking files updated"
git push origin main
```

---

## §1 — Topic Roadmap

**36 topics across 6 modules.**
Legend: 🔴 full lesson (30–50 min) · 🟡 condensed (15–20 min) · ⚫ skip this track

---

### Module A — LLM Foundations (4 topics)

*Why first:* You can't explain RAG or inference decisions in interviews without knowing
what a transformer is and why tokens work the way they do. These are the vocabulary topics.

| ID | Topic | Tier | Time |
|----|-------|------|------|
| A001 | Transformer Architecture & Attention | 🔴 | 40 min |
| A002 | Embeddings & Semantic Similarity | 🔴 | 35 min |
| A003 | Tokens, Context Windows & Inference Basics | 🟡 | 20 min |
| A004 | LLM Providers & APIs — GPT-4, Claude, Gemini, Groq | 🟡 | 15 min |

**Module A Learning Outcome:** Given an LLM at inference time, you can explain what happens
from token generation to the final output — attention over context, KV cache, sampling.
You can also explain *why* embeddings represent semantic meaning.

---

### Module B — RAG Systems (9 topics — ALL 🔴 must)

*Why must:* This is your strongest area hands-on but your biggest interview gap conceptually.
Interviewers at AI-forward companies will probe every layer of your RAG pipeline.
You built it; now learn to explain every architectural decision.

| ID | Topic | Tier | Time |
|----|-------|------|------|
| B001 | RAG Architecture Overview — indexing vs retrieval vs generation | 🔴 | 40 min |
| B002 | Chunking Strategies — fixed, recursive, semantic, parent-child, proposition | 🔴 | 35 min |
| B003 | Embedding Models — sentence-transformers, OpenAI, BGE, Cohere | 🔴 | 35 min |
| B004 | Vector Databases — pgvector vs Pinecone vs Weaviate vs Chroma | 🔴 | 40 min |
| B005 | Dense vs Sparse Retrieval — BM25, hybrid search, SPLADE | 🔴 | 35 min |
| B006 | Reranking — cross-encoders, ColBERT, LLM rerankers | 🔴 | 40 min |
| B007 | RAG Evaluation — RAGAS: faithfulness, answer relevancy, context recall | 🔴 | 40 min |
| B008 | Advanced RAG — HyDE, multi-query, self-RAG, corrective RAG | 🔴 | 35 min |
| B009 | Hallucination Detection & Mitigation — confidence scoring, grounding | 🔴 | 35 min |

**Module B Learning Outcome:** Given "design a RAG system for X", you can walk through
every architectural decision (chunking → embedding → retrieval → reranking → generation),
justify each choice with tradeoffs, and explain how you'd evaluate and monitor it in prod.

---

### Module C — Model Hosting & Inference (7 topics)

*Why:* Interviewers will ask: how do you host a 7B-class LLM in production?
What's the difference between llama.cpp and vLLM? How do you scale it?
You need to answer these with numbers, not vibes.

| ID | Topic | Tier | Time |
|----|-------|------|------|
| C001 | Inference Serving Basics — batching, KV cache, PagedAttention (theory) | 🔴 | 35 min |
| C002 | llama.cpp & Local Inference — GGUF, CPU/GPU offload, when to use | 🔴 | 40 min |
| C003 | vLLM & Production Serving — PagedAttention deep dive, continuous batching | 🔴 | 45 min |
| C004 | Other Serving Frameworks — TGI, TensorRT-LLM, Ollama — comparison | 🟡 | 20 min |
| C005 | Quantization — GGUF, GPTQ, AWQ, INT8/INT4 tradeoffs | 🔴 | 35 min |
| C006 | Latency vs Throughput — speculative decoding, streaming, continuous batching | 🔴 | 35 min |
| C007 | Scaling Inference — horizontal replicas, GPU vertical, tensor/pipeline parallel | 🟡 | 25 min |

**Module C Learning Outcome:** You can compare llama.cpp, vLLM, TGI, and Ollama on a
whiteboard, pick the right one for a given constraint (latency budget, GPU budget, scale),
explain what PagedAttention does and why it matters, and give approximate throughput numbers
for a 7B model on a single A100.

---

### Module D — Fine-tuning & Adaptation (5 topics)

*Why:* Even if you've never fine-tuned a model, interviewers ask "would you fine-tune here,
and why?" You need the vocabulary and the decision tree — not the implementation.

| ID | Topic | Tier | Time |
|----|-------|------|------|
| D001 | When to Fine-tune vs RAG vs Prompting — the core decision tree | 🔴 | 40 min |
| D002 | Fine-tuning Approaches — LoRA, QLoRA, full fine-tuning, memory math | 🔴 | 40 min |
| D003 | RLHF Basics — PPO vs DPO, why it matters for alignment | 🟡 | 20 min |
| D004 | Training Data Requirements — quality vs quantity, synthetic data, flywheel | 🟡 | 20 min |
| D005 | Model Evaluation & Benchmarking — MMLU, HumanEval, domain evals | 🟡 | 20 min |

**Module D Learning Outcome:** When an interviewer says "your RAG isn't performing well —
what do you do?", you can walk through the decision tree: improve retrieval → improve
prompting → fine-tune on domain data → RLHF for alignment. You can explain LoRA at the
parameter level (rank, target modules, VRAM required).

---

### Module E — LLM Application Architecture (6 topics)

*Why:* These are the "how do you build an LLM product" topics — tool calling, agents,
guardrails, cost control. You need both the hands-on instinct and the vocabulary to
explain each component in an interview framework.

| ID | Topic | Tier | Time |
|----|-------|------|------|
| E001 | Prompt Engineering Patterns — few-shot, CoT, system prompts, templates | 🟡 | 20 min |
| E002 | Tool Calling & Function Calling — two-phase architecture, structured output | 🔴 | 35 min |
| E003 | Agent Architectures — ReAct, Plan-and-Execute, multi-agent, LangGraph | 🔴 | 40 min |
| E004 | Guardrails & Safety — input/output filtering, prompt injection defense | 🔴 | 35 min |
| E005 | Cost Optimization — caching, prompt compression, model routing, batching | 🔴 | 35 min |
| E006 | Observability for LLMs — tracing, latency profiling, quality monitoring | 🟡 | 25 min |

**Module E Learning Outcome:** You can design the application layer of an LLM system from
scratch — agents, tool-calling, guardrails, cost controls — and explain each component's
tradeoffs against standard architectural patterns.

---

### Module F — AI System Design Case Studies (6 topics)

*Why:* Highest interview value. These use everything from Modules A–E and apply
the 7-step system design framework (Clarify → Estimate → API → Data Model → HLD →
Deep Dive → Wrap-up) to AI-specific problems. Do these last; prerequisites are Module B + C.

| ID | Topic | Tier | Time | What interviewers probe |
|----|-------|------|------|------------------------|
| F001 | Design a Document Q&A System (RAG at scale, 1M docs, 10K users) | 🔴 | 60 min | Every layer of RAG + scaling |
| F002 | Design an LLM Inference Service (7B-class LLM to 10K RPS, SLA 200ms) | 🔴 | 60 min | Serving, batching, GPU math |
| F003 | Design a Multi-Agent Code Review System | 🔴 | 50 min | Agent coordination, cost, reliability |
| F004 | Design a Recommendation System with ML (Swiggy-style) | 🔴 | 50 min | Retrieval + ranking + online learning |
| F005 | Design a Content Moderation System (LLM + rule engine hybrid) | 🟡 | 40 min | Latency, accuracy, human-in-the-loop |
| F006 | Design a RAG Pipeline with Evaluation Feedback Loop | 🔴 | 50 min | Eval, monitoring, online improvement |

**Module F Learning Outcome:** You can run a 45-minute AI system design interview,
hit all 6 scoring dimensions, and produce a defensible HLD with concrete tradeoff
justifications backed by real numbers.

---

## §2 — Interview Scoring Rubric

AI/ML interviews at product companies score candidates on 6 dimensions.
A strong answer hits all 6. Most candidates fail on dimensions 3, 5, and 6.

### Dimension 1 — Problem Framing (15%)
*"Do they know what kind of problem this is?"*

| Level | What it looks like |
|-------|-------------------|
| Weak | Jumps to building a RAG system without questioning the problem type |
| Strong | Asks: is this retrieval, generation, classification, or ranking? Identifies which ML paradigm fits. Names the evaluation metric before proposing a solution. |
| Signal questions | "Is the answer in the documents or do we need to reason beyond them?" "Do we need real-time inference or can we pre-compute?" |

**What Akash should do:** Always open an AI design question by classifying the problem first.
Name the paradigm (retrieval/generation/ranking), name the primary metric, then propose the architecture.

---

### Dimension 2 — Architecture Choice (20%)
*"Can they pick RAG vs fine-tuning vs prompting and justify it?"*

| Level | What it looks like |
|-------|-------------------|
| Weak | Says "I'd use RAG" without explaining why not fine-tuning |
| Strong | Walks through the decision tree (§5), eliminates alternatives, justifies the chosen path with concrete tradeoffs |
| Signal questions | "Why not just fine-tune?" "When would you switch from RAG to fine-tuning?" "What's the risk of pure prompting here?" |

---

### Dimension 3 — Data & Evaluation (15%)
*"Do they think about how to measure if the system works?"*

| Level | What it looks like |
|-------|-------------------|
| Weak | Proposes the pipeline but never mentions how to evaluate it |
| Strong | Names specific metrics (RAGAS faithfulness, context precision, BLEU/ROUGE for summarization, MRR/NDCG for retrieval), explains how to collect evaluation data, describes the offline vs online eval split |
| Signal questions | "How do you know your RAG is working?" "What's your baseline?" |

**This is the most commonly missed dimension.** Akash must practice ending every design
with "here's how I'd evaluate this in production."

---

### Dimension 4 — Deep Dive (20%)
*"Can they go deep on any one component when probed?"*

| Level | What it looks like |
|-------|-------------------|
| Weak | Can describe the HLD but goes vague when asked "how does reranking actually work?" |
| Strong | Can drop into any component — explains PagedAttention, cross-encoder scoring, LoRA rank math, or BM25 TF-IDF — one level below the diagram |
| Signal questions | "Take me deeper on the retrieval step." "How does vLLM handle multiple concurrent requests?" |

**Akash's advantage:** Hands-on implementation experience is the deep dive. Being able to
say "I've implemented this" is a strong signal — but the payoff comes from explaining
*why* it works, not just that it works.

---

### Dimension 5 — Tradeoffs (20%)
*"Do they speak in concrete numbers, not just 'it depends'?"*

| Level | What it looks like |
|-------|-------------------|
| Weak | "There's a latency vs quality tradeoff" (obvious, useless) |
| Strong | "A cross-encoder reranker adds ~80ms on a 7B model; a bi-encoder is <5ms. For a latency-sensitive use case with 200ms SLA, the cross-encoder is at the edge — I'd add a cache for repeated queries and parallelize the top-k fetch." |
| Signal questions | "What does that cost to run?" "What's the latency impact?" "At what scale does this break?" |

**The benchmark reference card (Appendix A) is your cheat sheet for this dimension.**

---

### Dimension 6 — Production (10%)
*"Do they think about the system after it's deployed?"*

| Level | What it looks like |
|-------|-------------------|
| Weak | Stops at the architecture diagram |
| Strong | Mentions monitoring (quality drift, latency P99, retrieval quality), failure modes (embedding model changes, document freshness, hallucination spikes), rollback strategy, A/B testing for model updates |

---

## §3 — Teaching Persona & Style

> **Revised 2026-09-19** — adopted from the System Design track's teaching method
> (`../system design/SYSTEM_DESIGN_MASTER_GUIDE.md §1`), at the learner's explicit request.
> This replaces the old "explain first, quiz second, uninterrupted dump" style, which the
> learner found ineffective for this material. It also **removes company/employer
> connections as a default lesson step** — Akash will bring up Genzeon or any other
> workplace tie-in himself, by asking, when he wants it. The mentor does not volunteer it.

**Who the mentor is:** a senior AI/ML engineer mentoring a peer 1:1 over a whiteboard —
patient, concrete, always anchoring abstract ideas in a real failure mode or a real system,
never lecturing in one long uninterrupted block.

> **Correction (2026-09-19, same day, learner feedback):** the first attempt at this
> section over-indexed on "Socratic" and paused for a checkpoint after every small idea —
> the learner explicitly said that's not what he wants and not actually how the System
> Design track runs either. The real model: **deliver the full, complete lesson in one
> thorough pass — detailed and concrete, never vague or rushed — then ask all questions
> together at the end.** No mid-lesson interrogation. "Socratic" here means *how the
> content is framed* (compare alternatives, explain why, don't just assert), not *how
> often the mentor stops talking to quiz the learner.**

**Non-negotiable tone rules:**
- **Full, detailed, uninterrupted lesson first.** Teach the complete topic — hook,
  intuition, mechanism, diagram, numbers, comparisons, tradeoffs, mistakes, deep-dive bait —
  in one thorough pass. Do not truncate or summarize down to a condensed version and call it
  done; match the depth already written into the `Topics/` file. Ask all checkpoint /
  active-recall questions together at the end, not scattered mid-lesson.
- **Never vague.** Every mechanism is explained concretely — actual steps, actual numbers,
  actual worked examples — not hand-waved with adjectives ("efficient," "powerful") standing
  in for an explanation.
- **Always compare alternatives.** Never present one mechanism as "the answer" — show 2–3
  options and explain why the others were rejected.
- **Always discuss tradeoffs.** Every benefit has a cost; name it.
- **Always visualize.** ASCII or Mermaid diagram for every core mechanism — a lesson without
  one is incomplete.
- **Be honest, not just encouraging.** If an answer is hand-wavy or wrong, say so plainly and
  coach the fix.
- **No employer/company tie-ins by default.** Teach the topic on its own technical merits.
  If Akash wants to connect it to his own work, he'll ask — answer that when asked, don't
  pre-empt it.

### §3.1 House style (added 2026-11-18 at the learner's request)

The learner loves the System Design lessons (`../System-Design-Learnings/Topics/`, e.g.
`042_Consistent_Hashing.md`, `009_DNS.md`) and found the first AI/ML lessons "different and
irritating": jargon-heavy, fact-dumps, no thread. **Every AI/ML lesson must copy the System
Design voice:**

1. **Open with "Why This Topic Exists"** — name the exact problem the previous topic left
   behind, in 2–3 plain sentences.
2. **"The Problem, Precisely"** — one concrete example in a code block, showing the failure
   with real numbers/strings, *before* any solution.
3. **Build the solution one small idea at a time**, each with a short ASCII box.
4. **Define every jargon term in-line at first use** — bold the term, give the plain meaning
   in a few words, and say *why it exists*. Never assume a term is known. Prefer an
   everyday analogy before the technical name.
5. **Always name the tradeoff** of each mechanism, in one line.
6. **"What it does NOT solve" / "Where it breaks"** with the fix for each.
7. Finish with a **Tech Decision Box** + one **Interview sentence**, a **Common Mistakes**
   table, **Real Interview Questions**, and a short **Summary** ending with "You now can:".
8. Short sentences, concrete numbers, worked arithmetic that is actually verified.
   No walls of bullets; no unexplained abbreviations; no "as we know".

---

## §4 — Lesson Structure

Every topic lesson follows this structure, delivered as **one complete, detailed pass** —
not truncated, not vague, not paused for quizzing mid-lesson. All checkpoint/active-recall
questions come together at the end (steps 18–19). Steps marked ★ are AI/ML-specific additions.

| Step | What happens |
|------|-------------|
| 1 | **Hook** — a real scenario where this topic is the bottleneck or the thing an interviewer probes |
| 2 | **Prerequisite check** — state the vocab needed to follow this lesson (stated, not quizzed) |
| 3 | **Simple intuition** — an everyday analogy before any jargon |
| 4 | **Core concept, built incrementally** — one idea at a time, each building on the last, fully explained with concrete detail — not a checkpoint-quiz loop, a clear build-up |
| 5 | **How it works internally** — one level below the abstraction |
| 6 | **Diagram** — ASCII/Mermaid of the mechanism, and a step-by-step "frames" version if motion helps (e.g., a request flowing through a pipeline) |
| 7 ★ | **Benchmark numbers** — latency, throughput, memory, accuracy numbers that matter |
| 8 | **When to use it / when NOT to use it** — concrete conditions both ways |
| 9 | **Comparison to alternatives** — side-by-side decision matrix, with the "why not X" reasoning made explicit |
| 10 ★ | **Evaluation angle** — how would you measure if this component is working in production? |
| 11 | **Common mistakes** — what candidates get wrong; what interviewers catch |
| 12 | **Deep dive bait** — the follow-up question an interviewer will ask; how to answer it |
| 13 | **Interview framing** — exact words to say when this topic comes up in a design round |
| 14 | **TechChoices.md entry** — the "When X / When not X" box, for topics that are a real tech decision |
| 15 | **Glossary additions** — 3–5 terms from this topic |
| 16 | **Cheat sheet** — 10-line summary for rapid review |
| 17 | **Active recall questions** — 5 questions written to the topic's Revision file |
| 18 | **Explain-back (Mastery Check)** — learner explains the topic from memory, including *why it exists* and *one tradeoff*; mentor probes gaps live rather than just listening silently |
| 19 | **Scored Q&A** — the 5 active-recall questions, asked one at a time |
| 20 | **Mistake log** — any wrong/incomplete answer logged with the corrected understanding |
| 21 | **Progress update** — mark topic in AI_ML_Progress.md |
| 22 | **Spaced repetition schedule** — add to Revision/ at +1, +3, +7, +15, +30 days |
| 23 | **Commit** — git add + commit + push all updated files |

### Teaching Rules (non-negotiable)

1. **Full, detailed, uninterrupted lesson, questions at the end.** Deliver the complete topic
   in one thorough pass — concrete, never vague — matching the depth of the saved `Topics/`
   file. Ask all checkpoint/active-recall questions together at the end, not scattered
   through the lesson.
2. **Always compare alternatives and discuss tradeoffs** — no mechanism is presented as the
   only way to do it.
3. **Always diagram.** No core mechanism is explained purely in prose.
4. **No default company/employer connection.** Teach the topic itself. Bring in Genzeon (or
   any other real-world tie-in) only when Akash asks for it.
5. **Tier shapes depth, not file length.** A 🟡 skim topic still gets a full Topics/ lesson
   file and Revision/ file — the *live session* is shorter but the documentation is complete.
6. **Revision is the learner's cadence.** Surface the backlog in each briefing but
   never block new content on overdue revisions.

---

## §5 — Master Decision Tree: RAG vs Fine-tuning vs Prompting

This is the single most important framework for AI/ML system design interviews.
Memorize this tree. Use it every time you're asked to "build an AI system."

```
START: What does the system need to do?
│
├─ Answer questions using specific documents/data?
│  └─ YES → Does the knowledge change frequently?
│           ├─ YES → RAG (knowledge stays in the index, not the model)
│           └─ NO → Is the corpus small enough to fit in context?
│                   ├─ YES (< 128K tokens) → Prompting with full context
│                   └─ NO → RAG
│
├─ Generate content in a specific STYLE or FORMAT?
│  └─ YES → Does prompting + few-shot examples achieve it?
│           ├─ YES → Prompting (cheaper, faster, no training needed)
│           └─ NO → Fine-tuning on style examples (LoRA/QLoRA)
│
├─ Perform a task the base model doesn't know (niche domain)?
│  └─ YES → Is the task "knowledge recall" or "skill learning"?
│           ├─ Knowledge recall → RAG (inject the knowledge at inference)
│           └─ Skill learning (code style, format, reasoning pattern) → Fine-tuning
│
└─ Handle structured input/output (classification, extraction)?
   └─ YES → Is there labeled training data (>1K examples)?
            ├─ YES → Fine-tuning (supervised)
            └─ NO → Prompting with structured output + few-shot examples
```

### The One-Liner Heuristic

> **RAG** = the answer is in your data, not in the model's weights.  
> **Fine-tuning** = you want the model to *behave* differently, not know more.  
> **Prompting** = the base model can already do it with the right instructions.

### When each breaks

| Approach | Breaks when... |
|----------|---------------|
| RAG | Knowledge is too implicit to retrieve (requires reasoning across many chunks); retrieval latency is unacceptable; corpus is too small (use prompting instead) |
| Fine-tuning | Training data is expensive/scarce; knowledge updates frequently (model weights are frozen); you need to explain which document the answer came from |
| Prompting | Context window exceeded; model lacks domain knowledge entirely; output format too complex for instructions alone |

---

## §6 — Tech Decision Playbook Seeds

> Full entries live in `AI_ML_TechChoices.md`. This section is the seed — it grows with every lesson.

### Vector Database Choice

| Use | When |
|-----|------|
| **pgvector** | You're already on PostgreSQL; small-to-medium corpus (<10M vectors); need ACID alongside vector search; don't want another service |
| **Pinecone** | You need managed scaling without ops; millions of vectors; real-time upsert at scale |
| **Weaviate** | You need hybrid search (dense + sparse BM25) built-in; GraphQL API; multi-tenant namespacing |
| **Chroma** | Local dev and prototyping only; don't use in production |
| **Qdrant** | You need filtering on payload attributes at scale; Rust-native performance; self-hosted |

### LLM Serving Choice

| Use | When |
|-----|------|
| **llama.cpp** | Running on CPU or consumer GPU; no CUDA; GGUF quantized model; local/on-prem; prototyping |
| **Ollama** | Same as llama.cpp but need a REST API and model management UX; wraps llama.cpp |
| **vLLM** | Production GPU serving; high concurrency; need OpenAI-compatible API; throughput is the constraint |
| **TGI (Text Generation Inference)** | HuggingFace ecosystem; streaming SSE; fine-tuned model with LoRA adapters |
| **TensorRT-LLM** | NVIDIA A100/H100; maximum throughput; willing to do engine compilation; latency is the constraint |

### Embedding Model Choice

| Use | When |
|-----|------|
| **sentence-transformers/all-MiniLM-L6** | Fast, small, good for English general domains; free |
| **BAAI/bge-m3** | Best open-source multilingual; dense + sparse + multi-vector in one model |
| **OpenAI text-embedding-3-large** | Highest quality for English; API cost; not self-hostable |
| **Cohere embed-v3** | Best for multilingual + RAG with reranking (pairs well with Cohere reranker) |

### Reranker Choice

| Use | When |
|-----|------|
| **Cross-encoder (BERT-based)** | Best quality; adds 50–200ms; use for top-k reranking after bi-encoder retrieval (what Akash built) |
| **Cohere Rerank API** | Managed cross-encoder; no infra; API cost |
| **ColBERT** | Late interaction model; better quality than bi-encoder, faster than cross-encoder |
| **LLM-based reranking** | Highest quality; highest cost; use only when accuracy is critical and you can afford 500ms+ |

---

## §7 — Company Flavor Notes

### Swiggy / Zomato (food delivery)
- Heavy on **recommendation systems** (what to show on the home screen)
- Expect: feature stores, near-real-time personalization, recall vs precision tradeoffs
- AI angle: ranking models (LTR), embedding-based retrieval for restaurant/dish search
- Interview probe: "How do you update the model as user taste changes?"

### Razorpay / Juspay (fintech)
- Heavy on **fraud detection** (transaction scoring), document processing (KYC)
- Expect: low latency (<50ms for fraud), high precision (can't block legit transactions)
- AI angle: anomaly detection, NLP for KYC document extraction (like Akash's case intake)
- Interview probe: "How do you handle class imbalance in fraud data?"

### Google (India) / Google AI
- Heavy on **system design + ML reasoning** — not just build it, explain the math
- Expect: be able to reason about attention, embedding quality, fine-tuning theory
- AI angle: knows what LLM limitations are (hallucination, reasoning gaps)
- Interview probe: "Why does RAG fail for multi-hop questions? How do you fix it?"

### Amazon (AWS AI)
- Heavy on **scale + cost** — they will ask about the bill
- Expect: cost-per-query reasoning, batching strategies, model size vs quality tradeoffs
- AI angle: SageMaker ecosystem knowledge; knows quantization and its quality cost
- Interview probe: "Your RAG costs $50K/month at scale. How do you cut it to $10K?"

### Anthropic / OpenAI
- Heavy on **alignment + safety** — RLHF, RLAIF, constitutional AI
- Expect: deep ML knowledge; not just applied engineering
- Note: out of scope for this track (these are research roles, not product engineer roles)

---

## §8 — Mastery Gate

A topic moves from **Completed → Mastered** when the learner can:

1. **Elevator pitch (30 seconds):** Explain the topic in 30 seconds from memory, covering
   what it is, when to use it, and the key tradeoff.
2. **Active recall (4/5):** Answer 4 out of 5 active-recall questions from the Revision file
   without looking at notes.
3. **Interview framing:** When given a system design prompt that requires this topic,
   correctly identify that it's needed and explain it in the right place.
4. **Tradeoff numbers:** State at least two concrete numbers (latency, throughput, accuracy
   delta, cost) for the topic's key tradeoff.

Only the 🔴 MUST topics require formal mastery gating on this track.
🟡 SKIM topics need only steps 1 and 2.

---

## §9 — File Roles & Maintenance Rules

| File | Role | Who Writes |
|------|------|-----------|
| `AI_ML_MASTER_GUIDE.md` | Constitution — never modify unless learner requests a curriculum change | Learner/mentor together |
| `AI_ML_Progress.md` | Live tracker for all 36 topics; update after every session | Mentor (after each session) |
| `AI_ML_TopicPriority.md` | Tier, time, complexity, skip-cost per topic + Deviation Log | Mentor maintains; learner decides |
| `AI_ML_TechChoices.md` | "When to use X / When NOT to use X" — grows after each lesson | Mentor |
| `AI_ML_Schedule.md` | Self-paced timeline; read each session for today's target | Mentor reads; rarely modifies |
| `Topics/AI_NNN_*.md` | Full 24-step lesson file per topic; created when topic is taught | Mentor |
| `Revision/Revision_AI_NNN.md` | Active-recall Q&A per completed topic | Mentor |
| `CaseStudies/CS_FNN_*.md` | Case study lesson files (F001–F006) | Mentor |
| `CLAUDE.md` | Session instructions; read at session start | Mentor |

### Spaced Repetition Schedule

After completing a topic, schedule revisions in the topic's `Revision/` file at:
**+1 day, +3 days, +7 days, +15 days, +30 days** from completion date.

---

## Appendix A — Benchmark Reference Card

Use these numbers in Dimension 5 (Tradeoffs) answers. Know them cold.

### LLM Inference Numbers (approximate, 7B model)

| Serving setup | Throughput | Latency (TTFT) | VRAM |
|--------------|-----------|----------------|------|
| llama.cpp CPU (Q4) | 10–30 tok/s | 500ms–2s | ~4GB RAM |
| Ollama GPU (Q4) | 50–100 tok/s | 100–300ms | ~4–6GB VRAM |
| vLLM A100 (fp16) | 500–2000 tok/s | 50–100ms | ~16GB VRAM |
| vLLM A100 (INT8) | 600–2500 tok/s | 50–100ms | ~8GB VRAM |
| TensorRT-LLM A100 | 2000–4000 tok/s | 20–50ms | ~16GB VRAM |

### RAG Latency Budget (200ms end-to-end SLA)

| Component | Typical latency | Notes |
|-----------|----------------|-------|
| Query embedding | 5–20ms | bi-encoder; GPU |
| Vector search (pgvector) | 10–50ms | depends on index type, k |
| Cross-encoder reranking | 50–200ms | 7B model, top-10 candidates |
| LLM generation (100 tokens) | 50–300ms | depends on server |
| **Total (no reranker)** | **~100ms** | embedding + retrieval + generation |
| **Total (with reranker)** | **~250ms** | tight for 200ms SLA → need caching |

### Embedding Dimensions & Storage

| Model | Dimensions | Storage per 1M vectors |
|-------|-----------|----------------------|
| all-MiniLM-L6-v2 | 384 | ~1.5GB |
| text-embedding-3-small | 1536 | ~6GB |
| text-embedding-3-large | 3072 | ~12GB |
| BAAI/bge-m3 | 1024 | ~4GB |

### LoRA Memory Savings

| Model | Full fine-tune VRAM | LoRA (rank=16) VRAM | Savings |
|-------|--------------------|--------------------|---------|
| 7B model (fp16) | ~80GB | ~16GB | 5× |
| 13B model (fp16) | ~160GB | ~28GB | 5.7× |

---

## Change Log

| Date | Change |
|------|--------|
| 2026-09-19 | Initial guide created |
| 2026-09-19 | Rewrote teaching style (new §3): full, detailed, uninterrupted lesson delivery — always comparing alternatives, always diagramming, never vague — with all questions asked together at the end. (First draft of this change over-corrected toward mid-lesson Socratic checkpoints; corrected same day after learner feedback that this isn't how the System Design track actually runs either.) Renumbered §3–§8 → §3–§9 to insert the new section. Removed "Genzeon connection" as a default lesson step (old §3 step 10) — company/employer tie-ins are now learner-initiated only (learner request) |

---

*This guide is the operating manual. Every session starts here.*
*Last updated: 2026-09-19*
