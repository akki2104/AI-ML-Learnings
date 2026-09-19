# AI/ML Topic Priority

> Read this before every session. Show the Topic Briefing Card for today's topic.
> Tiers: 🔴 MUST (full lesson, 30–50 min) · 🟡 SKIM (condensed, 15–25 min) · ⚫ SKIP this track

---

## Priority Table

| ID | Topic | Tier | Est. Time | Complexity | Genzeon Link | Skip Cost | Interview Freq |
|----|-------|------|-----------|-----------|--------------|-----------|----------------|
| A001 | Transformer Architecture & Attention | 🔴 | 40 min | Medium | Foundation for Mistral-7B understanding | High — can't explain LLM behavior without it | Common |
| A002 | Embeddings & Semantic Similarity | 🔴 | 35 min | Medium | Direct — you build embedding indexes | Very high — asked in every RAG question | Very Common |
| A003 | Tokens, Context Windows & Inference Basics | 🟡 | 20 min | Low | Context window constraints your chunking | Medium — needed for chunking decisions | Common |
| A004 | LLM Providers & APIs | 🟡 | 15 min | Low | You've used Gemini, Groq | Low — mostly vocabulary | Moderate |
| B001 | RAG Architecture Overview | 🔴 | 40 min | Medium | You built this end-to-end | Critical — the core interview design | Very Common |
| B002 | Chunking Strategies | 🔴 | 35 min | Medium | Your clinical doc pipeline chose one | High — "how did you chunk?" is always asked | Common |
| B003 | Embedding Models | 🔴 | 35 min | Medium | You used sentence-transformers | High — "which embedding model and why?" | Common |
| B004 | Vector Databases | 🔴 | 40 min | Medium | You chose pgvector; know all alternatives | Very high — decision matrix is interview gold | Very Common |
| B005 | Dense vs Sparse Retrieval | 🔴 | 35 min | Medium | Your pipeline may be pure dense | High — hybrid search is a common deep dive | Common |
| B006 | Reranking | 🔴 | 40 min | Medium-High | You built cross-encoder reranking | Very high — this is on your resume | Very Common |
| B007 | RAG Evaluation | 🔴 | 40 min | Medium | Did you have eval? Interviewers always ask | Critical — most candidates skip this | Very Common |
| B008 | Advanced RAG | 🔴 | 35 min | High | Above what you've built — vocabulary only | High — shows depth beyond basics | Common |
| B009 | Hallucination Detection & Mitigation | 🔴 | 35 min | Medium | You built confidence scoring | Very high — clinical context makes this critical | Common |
| C001 | Inference Serving Basics | 🔴 | 35 min | Medium | Foundation for C002 + C003 | High — can't explain vLLM without KV cache | Common |
| C002 | llama.cpp & Local Inference | 🔴 | 40 min | Medium | Genzeon likely uses this or Ollama | Very high — it's in your work context | Common |
| C003 | vLLM & Production Serving | 🔴 | 45 min | High | Production Mistral-7B serving | Very high — standard for production AI | Very Common |
| C004 | Other Serving Frameworks | 🟡 | 20 min | Low | Comparison context for C002+C003 | Medium — know names and one key tradeoff each | Moderate |
| C005 | Quantization | 🔴 | 35 min | Medium | GGUF explains llama.cpp performance | High — "how did you run 7B on your hardware?" | Common |
| C006 | Latency vs Throughput | 🔴 | 35 min | Medium | Speculative decoding, streaming | High — every serving question needs numbers | Common |
| C007 | Scaling Inference | 🟡 | 25 min | Medium | Horizontal scaling of serving layer | Medium — needed for design case F002 | Moderate |
| D001 | When to Fine-tune vs RAG vs Prompting | 🔴 | 40 min | Medium | The decision tree — most asked question | Critical — never skip this | Very Common |
| D002 | Fine-tuning Approaches (LoRA, QLoRA) | 🔴 | 40 min | High | LoRA for domain adaptation | High — "would you fine-tune here?" | Common |
| D003 | RLHF Basics | 🟡 | 20 min | Medium | Conceptual vocabulary only | Medium — interviewers ask "what is RLHF?" | Moderate |
| D004 | Training Data Requirements | 🟡 | 20 min | Low | Your pipeline generates labeled examples | Low-medium — mostly vocabulary | Moderate |
| D005 | Model Evaluation & Benchmarking | 🟡 | 20 min | Low | Domain-specific eval for clinical | Low — vocabulary for answering eval questions | Moderate |
| E001 | Prompt Engineering Patterns | 🟡 | 20 min | Low | You write prompts in production | Medium — formalize what you already do | Common |
| E002 | Tool Calling & Function Calling | 🔴 | 35 min | Medium | You built two-phase in Hemut | High — agent design always involves tools | Common |
| E003 | Agent Architectures (ReAct, LangGraph) | 🔴 | 40 min | High | You built AgentAudit 4-agent system | Very high — it's on your resume | Common |
| E004 | Guardrails & Safety | 🔴 | 35 min | Medium | You built anti-hallucination grounding | High — clinical context makes safety critical | Common |
| E005 | Cost Optimization | 🔴 | 35 min | Medium | Dual-axis caching you built in Hemut | High — every production AI question needs cost | Common |
| E006 | Observability for LLMs | 🟡 | 25 min | Medium | How do you know your RAG degrades? | Medium — needed for production credibility | Moderate |
| F001 | Design a Document Q&A System | 🔴 | 60 min | High | This is your Genzeon work, formalized | Critical — highest interview value | Very Common |
| F002 | Design an LLM Inference Service | 🔴 | 60 min | High | Mistral-7B at scale | Very high — C module prerequisites needed | Common |
| F003 | Design a Multi-Agent Code Review System | 🔴 | 50 min | High | AgentAudit formalized | Very high — it's on your resume | Common |
| F004 | Design a Recommendation System with ML | 🔴 | 50 min | High | Different from RAG — ranking paradigm | High — Swiggy-style companies ask this | Common |
| F005 | Design a Content Moderation System | 🟡 | 40 min | Medium | LLM + rule engine hybrid | Medium — shows breadth beyond RAG | Moderate |
| F006 | Design a RAG Pipeline with Evaluation Feedback Loop | 🔴 | 50 min | High | The full loop: build + eval + improve | High — shows production maturity | Common |

---

## Recommended Session Order

Respects prerequisites (A → B → C → D → E → F) but Genzeon-relevant topics first within each module.

**Session 1:** A001 + A002 (transformers + embeddings — vocabulary foundation)  
**Session 2:** B001 + B002 + B003 (RAG overview + chunking + embeddings)  
**Session 3:** B004 + B005 (vector DBs + retrieval — decision matrices)  
**Session 4:** B006 + B007 (reranking + evaluation — Akash's core work)  
**Session 5:** B008 + B009 (advanced RAG + hallucination mitigation)  
**Session 6:** A003 + A004 + C001 (tokens + providers + inference basics — skims)  
**Session 7:** C002 + C005 (llama.cpp + quantization — what Genzeon uses)  
**Session 8:** C003 + C006 (vLLM + latency/throughput — production serving)  
**Session 9:** C004 + C007 + D001 (other frameworks + scaling + the master decision tree)  
**Session 10:** D002 + D003 + D004 + D005 (fine-tuning module — mostly skims)  
**Session 11:** E001 + E002 + E003 (prompt engineering + tool calling + agents)  
**Session 12:** E004 + E005 + E006 (guardrails + cost + observability)  
**Session 13:** F001 (Document Q&A case study — full 60 min)  
**Session 14:** F002 (Inference service case study)  
**Session 15:** F003 + F004 (code review + recommendation)  
**Session 16:** F005 + F006 (content moderation + RAG feedback loop)  

**~16 sessions × 1–1.5h each = 20–25 hours total.**

---

## Deviation Log

*(Record here when the learner overrides the recommended order or tier.)*

| Date | Topic | Default Tier | Chosen Tier | Reason |
|------|-------|-------------|-------------|--------|
| — | — | — | — | — |
