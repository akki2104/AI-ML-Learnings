# Topic B001 — RAG Architecture Overview

**Module:** B — RAG Systems
**Tier:** 🔴 MUST (interview fast-track: priority 1)
**Status:** Completed 2026-11-19 (passed after retest)
**Difficulty:** Medium

---

## 1. Why This Topic Exists

A002 gave us one trick: find text by *meaning* using embeddings. B001 shows how that trick becomes a whole system. **RAG (Retrieval-Augmented Generation)** is the standard design for "make an LLM answer questions about *our* documents", and it is the single most common AI system design interview question. Every later topic (chunking, vector DBs, reranking, evaluation, hallucination) is one box inside the diagram you learn here.

---

## 2. The Problem, Precisely

```
User asks the company chatbot:  "What is the refund window for annual plans?"

An LLM (Large Language Model — the text-generating AI, like Claude/GPT) alone:
  1. NEVER SAW the answer. It was trained on public internet text up to a
     KNOWLEDGE CUTOFF date; your private policy docs weren't in it.
  2. So it either says "I don't know" or, worse, HALLUCINATES — confidently
     invents a plausible answer ("30 days") that is simply wrong.
  3. Can't say WHERE the answer came from → no CITATION a user can verify.

"Fine, paste all the docs into the prompt."
  Company has 50,000 pages ≈ 25,000,000 tokens (token = word piece, A001/A003).
  A model's CONTEXT WINDOW (the max text it can read per request) ≈ 128,000
  tokens → fits ~0.5% of the docs. And even if it fit, you'd pay for 25M tokens
  on every single question.
```
So we need: **fetch only the 3–5 relevant paragraphs, put just those in the prompt.**

---

## 3. Core Concepts

### The idea: an open-book exam
A closed-book exam = LLM answering from memory (hallucination risk). An **open-book exam** = before answering, look up the relevant pages, then answer *from those pages*. RAG = give the LLM an open book, automatically, per question.

### The three stages
```
 ┌────────────────────────── 1. INDEXING (OFFLINE) ──────────────────────────┐
 │ documents → PARSE → CLEAN → CHUNK → EMBED → STORE (vector DB + metadata)  │
 └────────────────────────────────────────────────────────────────────────────┘
                                                      │ built ahead of time
 ┌────────────────────── 2. RETRIEVAL (ONLINE, per question) ────────────────┐
 │ question → EMBED (same model) → SEARCH top-k → (FILTER) → (RERANK) → chunks│
 └────────────────────────────────────────────────────────────────────────────┘
                                                      │ 3–5 best chunks
 ┌────────────────────── 3. GENERATION (ONLINE) ─────────────────────────────┐
 │ PROMPT = instructions + chunks + question → LLM → answer + citations       │
 └────────────────────────────────────────────────────────────────────────────┘
```
**OFFLINE** = done ahead of time, no user waiting. **ONLINE** = done live while a user waits (so latency matters).

### Stage 1 — Indexing (build the library)
- **Parse:** turn PDFs/HTML/Word into clean text. Harder than it sounds: tables, scanned pages (need OCR), headers/footers. Bad parsing = garbage in, garbage out.
- **Chunk:** split into pieces of roughly 256–512 tokens (why: A002 — model input limits, and one vector per page blurs topics). Detail in B002.
- **Embed:** one vector per chunk using an embedding model (A002).
- **Store with metadata:** keep the vector *plus* **metadata** (extra labels: source file, page, date, product, who may read it). Why: for citations, for **filtering** ("only 2025 docs"), and for **access control** (user A must not retrieve HR documents).
- **Tradeoff:** indexing is cheap per question but must be **refreshed** when docs change, or answers go stale (**freshness** problem).

### Stage 2 — Retrieval (find the right pages)
```
question ─► embed ─► vector DB returns top-k nearest chunks (ANN, A002)
                       │
           optional:   ├─ metadata FILTER  (date >= 2025, team = billing)
                       ├─ HYBRID: also BM25 keyword search, merge lists (B005)
                       └─ RERANK: cross-encoder re-orders top ~50 → keep best 3–5 (B006)
```
- **top-k** = how many chunks you fetch. **Tradeoff:** large k → more likely the answer is *in* there (higher recall) but more noise, higher cost, slower. Small k → cheap, but risks missing it.
- Typical recipe: retrieve ~20–50 candidates cheaply, **rerank** to the best 3–5.

### Stage 3 — Generation (write the answer from the pages)
The **prompt** (the full text sent to the LLM) is assembled from a **prompt template**:
```
SYSTEM:  You are a support assistant. Answer ONLY using the context below.
         If the answer is not in the context, say "I don't know."
         Cite the source id for each claim.
CONTEXT: [1] (refund_policy.pdf p.3) "Annual plans may be refunded within 14 days..."
         [2] (billing_faq.md)        "Monthly plans: no refund after renewal..."
QUESTION: What is the refund window for annual plans?
```
- **Grounding** = forcing the answer to be based on the supplied text, not model memory. This is what reduces hallucination (B009).
- **Temperature** = randomness setting; use low (0–0.2) for factual answers.
- **Citations** come for free because every chunk carries its source in metadata.
- **Tradeoff:** a stricter "only from context" prompt = fewer hallucinations but more "I don't know" refusals.

### Walking one question through (frames)
```
Frame 1  "refund window for annual plans?"  → embed → [0.12, -0.4, ...]
Frame 2  vector DB: top-30 chunks by similarity (incl. refund_policy p.3, p.7, FAQ…)
Frame 3  rerank 30 → keep 4; refund_policy p.3 now rank #1
Frame 4  build prompt (instructions + 4 chunks + question) ≈ 1,800 tokens
Frame 5  LLM: "Annual plans can be refunded within 14 days of purchase [1]."
```

---

## 4. Why RAG and Not the Alternatives

| Option | What it is | Strength | Weakness |
|---|---|---|---|
| **Prompting only** | Ask the LLM directly | Zero setup | Doesn't know your data; hallucinates |
| **Long context ("stuff everything")** | Paste all docs into a huge window | No retrieval system to build | Cost per question, slow, hard limit, accuracy drops in the middle ("lost in the middle": models attend less to the middle of long prompts) |
| **Fine-tuning** | Retrain the model on your data | Changes behaviour/style | Poor at injecting *facts*, goes stale, no citations, costly (D001 covers the decision) |
| **RAG** | Retrieve, then generate | Fresh (update the index, not the model), citations, access control, cheap per question | Quality depends on retrieval; more moving parts |

**Rule of thumb:** facts that change or must be cited → RAG. Behaviour/style → fine-tune. Small corpus (< ~20 docs, fits the window) → just put it all in the prompt.

---

## 5. Where RAG Breaks (debug by stage)

```
Symptom                              Likely stage    Fix
Right chunk never retrieved          Indexing/Retrieval  better chunking, hybrid search, bigger k, query rewriting
Right chunk retrieved but ranked low Retrieval           add a reranker
Too many/noisy chunks, answer off    Retrieval→Prompt    smaller k after reranking, dedupe
Right chunk IN the prompt, wrong ans Generation          stricter grounding prompt, better LLM, less context
Outdated answer                      Indexing            re-index on doc change (freshness)
User sees a doc they shouldn't       Metadata/Retrieval  filter by permissions BEFORE retrieval
```
**The first debugging question:** *"Was the correct chunk in the prompt?"* If no → retrieval problem. If yes → generation problem. (This is the answer to the A001 interview question "retrieval is right but the LLM is wrong".)

---

## 6. Numbers (typical, order of magnitude)

| Item | Typical |
|---|---|
| Chunk size | 256–512 tokens, 10–20% overlap |
| Retrieve → keep | ~20–50 candidates → 3–5 after rerank |
| Embed the question | 20–100 ms |
| Vector search | 10–50 ms |
| Cross-encoder rerank of ~50 | 100–300 ms |
| LLM generation | 1–5 s (the biggest part; first token in ~0.3–1 s if streaming) |
| End-to-end budget | ~2–6 s; **stream** the answer so users see text immediately |
| Prompt size | ~1,500–4,000 tokens (you pay per token) |

---

## 7. How to Know It's Working (preview of B007)
Measure the two halves **separately**, otherwise you can't tell which half is broken:
- **Retrieval:** recall@k, MRR (A002) on a labelled question→chunk set.
- **Generation:** **faithfulness** (is the answer supported by the retrieved text?) and **answer relevancy** (does it actually answer the question?).
- In production also track: "I don't know" rate, user thumbs-up/down, latency, cost per query.

---

## Tech Decision Box: RAG vs Alternatives

```
Use RAG when:        answers live in documents; knowledge changes; users need
                     citations; access control per user; corpus too big for context
Use LONG CONTEXT when: corpus is small (fits the window) and traffic is low
Use FINE-TUNING when:  you need a style/behaviour/format change, not new facts
Do NOT use RAG when:   questions need reasoning across MANY docs at once with no
                       single retrievable passage (multi-hop) — consider agents
                       (E003), a larger context, or pre-computed summaries
```
**Interview sentence:** "I'd build RAG in three stages. Offline, I parse, chunk and embed documents into a vector store with metadata for citations and access control. Online, I embed the question, retrieve about 30 candidates with hybrid search, rerank to the top 4, and build a grounded prompt that tells the model to answer only from that context and cite sources. I'd evaluate retrieval and generation separately — recall@k for retrieval, faithfulness for generation — so when something's wrong I know which half to fix."

---

## Common Mistakes

| Mistake | Correction |
|---|---|
| "RAG = vector DB + LLM" and stopping there | Name all stages: parse, chunk, embed, retrieve, rerank, prompt, generate, evaluate |
| Blaming the LLM first | Check retrieval first — usually the right chunk never reached the prompt |
| Stuffing a huge k into the prompt | More chunks ≠ better: noise, cost, "lost in the middle"; retrieve wide, rerank narrow |
| No metadata | You lose citations, filtering and per-user access control |
| Ignoring freshness | Plan re-indexing when documents change |
| One metric for the whole system | Evaluate retrieval and generation separately |
| Filtering permissions *after* generation | Filter before/during retrieval so restricted text never reaches the LLM |
| Saying RAG removes hallucination | It reduces it; the model can still ignore or misread context |

---

## Real Interview Questions

1. "Design a Q&A system over our company documents." (the full pipeline)
2. "Why RAG instead of fine-tuning or a long context window?" (alternatives matrix)
3. "The right document exists but the bot answers wrongly — how do you debug?" (stage-by-stage)
4. "How do you stop users seeing documents they shouldn't?" (metadata + filter before retrieval)
5. "How do you keep the index fresh when documents change daily?" (re-index/incremental updates)
6. "What would your latency budget look like?" (numbers + streaming)

---

## 8. Revision Questions
See `Revision/Revision_AI_B001.md`.

## 9. Summary
- An LLM alone doesn't know your private data, can hallucinate, can't cite, and can't read 25M tokens per question.
- RAG = open-book exam: retrieve the few relevant chunks, put them in the prompt, make the LLM answer only from them.
- Three stages: **Indexing** (offline: parse → chunk → embed → store + metadata), **Retrieval** (online: embed → top-k → filter → rerank), **Generation** (online: grounded prompt → answer + citations).
- Tradeoffs: bigger k = more recall but more noise/cost; stricter grounding = fewer hallucinations but more "I don't know".
- Debug by stage; first ask "was the right chunk in the prompt?"
- Evaluate retrieval and generation separately.

> **You now can:** draw the RAG pipeline from memory, explain why it beats prompting, long context and fine-tuning for document Q&A, debug a bad answer stage by stage, give a latency budget, and state how you'd evaluate it.
