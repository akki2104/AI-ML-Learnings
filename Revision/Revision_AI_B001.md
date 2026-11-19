# Revision — B001: RAG Architecture Overview

**Topic taught:** 2026-11-18 · **Completed (passed after retest):** 2026-11-19
**Spaced repetition schedule (from 2026-11-19):** +1d (2026-11-20) · +3d (2026-11-22) ·
+7d (2026-11-26) · +15d (2026-12-04) · +30d (2026-12-19)

| Date | Reviewed? | Score |
|------|-----------|-------|
| 2026-11-20 | ⬜ | — |
| 2026-11-22 | ⬜ | — |
| 2026-11-26 | ⬜ | — |
| 2026-12-04 | ⬜ | — |
| 2026-12-19 | ⬜ | — |

**Known weak spots (from InterviewMistakes.md):** first debug question = "was the right chunk in the prompt?"; fine-tuning is poor at facts/no citations; only documents are chunked; take the team id from the auth token, never from the client.

---

## Active Recall Questions

1. **Why can't you just ask an LLM about your company documents (or paste them all in the
   prompt)? Give three reasons.**

2. **Draw/describe the three RAG stages and what happens in each. Which are offline and which
   online, and why does that matter?**

3. **Compare RAG, long context, and fine-tuning for a document Q&A product. When would you
   pick each?**

4. **A user gets a wrong answer although the correct document exists. How do you debug it?
   What is your first question?**

5. **Interview: "Design a Q&A system over 50,000 internal documents with per-team access
   control." Walk through your design in ~2 minutes, including one tradeoff and how you'd
   evaluate it.**

---

## Answer Key

1. (a) Knowledge cutoff / never trained on private data → says "I don't know" or hallucinates.
   (b) No citations → users can't verify. (c) Context window too small (25M tokens vs ~128k)
   and paying for all of it on every question is slow and expensive; accuracy also drops in
   long prompts ("lost in the middle").

2. Indexing (offline): parse → clean → chunk → embed → store with metadata. Retrieval (online):
   embed query → top-k search → filter → rerank. Generation (online): grounded prompt +
   chunks + question → LLM → answer with citations. Offline work can be slow/batch; online
   work has a latency budget (a user is waiting), so it must be fast.

3. RAG: facts in documents that change, citations, access control, large corpus. Long
   context: small corpus that fits the window and low traffic. Fine-tuning: change
   behaviour/style/format, not inject facts. (Fine-tuning goes stale, no citations.)

4. First question: "Was the correct chunk in the prompt?" No → retrieval/indexing problem
   (chunking, hybrid search, bigger k, reranker, query rewriting). Yes but wrong answer →
   generation problem (stricter grounding prompt, better LLM, less noise). Also check
   freshness and permissions filtering.

5. Offline: parse/chunk/embed with metadata (source, page, date, team ACL) into a vector
   store. Online: embed question, hybrid retrieve ~30 filtered by the user's team
   permissions *before* retrieval, rerank to top 4, grounded prompt with citations, stream
   the answer. Tradeoff: larger k = more recall vs more noise/cost; stricter grounding =
   fewer hallucinations vs more refusals. Evaluate retrieval (recall@k, MRR) and generation
   (faithfulness, answer relevancy) separately; monitor latency, cost, refusal rate, thumbs.
