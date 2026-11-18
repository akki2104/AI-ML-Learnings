# Revision — A002: Embeddings & Semantic Similarity

**Topic taught:** 2026-10-31 · **Completed (passed after retest):** 2026-11-18
**Spaced repetition schedule (from 2026-11-18):** +1d (2026-11-19) · +3d (2026-11-21) ·
+7d (2026-11-25) · +15d (2026-12-03) · +30d (2026-12-18)

| Date | Reviewed? | Score |
|------|-----------|-------|
| 2026-11-19 | ⬜ | — |
| 2026-11-21 | ⬜ | — |
| 2026-11-25 | ⬜ | — |
| 2026-12-03 | ⬜ | — |
| 2026-12-18 | ⬜ | — |

**Known weak spots (from InterviewMistakes.md):** say the tradeoff of embeddings (one vector per chunk = fast but loses detail); "length 1 → three rulers, one ranking"; "new model = new map = re-embed everything" incl. storage maths (N × dims × 4 B).

---

## Active Recall Questions

1. **What is an embedding, and how does an embedding model produce one from a sentence?
   (Name the architecture type, the pooling step, and how it's trained.)**

2. **Cosine similarity, dot product and L2 distance: when do they give the same ranking, and
   what do you do at index time to exploit that?**

3. **Bi-encoder vs cross-encoder: how does each work, why can't you use a cross-encoder for
   first-stage retrieval over 10M chunks, and how do they combine in a pipeline?**

4. **Name three failure modes of dense embeddings and the mitigation for each.**

5. **You must change from model X to model Y. Why can't you just embed new documents with Y,
   and what is your migration and evaluation plan? Also: 1M chunks at 1024 dims float32 —
   how much raw vector storage, and two ways to shrink it?**

---

## Answer Key (self-check after attempting from memory)

1. A fixed-length vector (e.g. 768 floats). Produced by a bidirectional transformer encoder
   (no causal mask) giving one vector per token, pooled (mean or CLS) into one vector, then
   L2-normalized. Trained contrastively (InfoNCE) on (query, positive) pairs with in-batch
   and hard negatives so relevant pairs are pulled together.

2. When vectors are L2-normalized: dot = cosine, and ‖a−b‖² = 2 − 2cos, so ranking is
   identical. Normalize at index (and query) time and use inner product, the cheapest op.

3. Bi-encoder encodes query and doc separately; doc vectors precomputed; query-time = one
   encode + ANN lookup. Cross-encoder encodes "query + doc" jointly with full attention →
   better accuracy but needs one forward pass per pair (10M passes per query is infeasible).
   Pipeline: bi-encoder (+BM25) retrieves top 50–100 → cross-encoder reranks → top 3–5 to LLM.

4. (a) Exact IDs/rare tokens → hybrid with BM25. (b) Negation/fine distinctions → reranker /
   LLM verification. (c) Numbers/dates/structured constraints → metadata filters / SQL.
   (also: domain drift → fine-tune or choose domain model; silent truncation → chunking.)

5. Vector spaces from different models are incomparable, so new-model queries can't search
   old-model vectors. Plan: new versioned index, re-embed whole corpus, dual-write new docs,
   evaluate recall@k/MRR/nDCG on the labelled set, shift traffic (blue/green), delete old.
   Storage: 1M × 1024 × 4 B ≈ 4.1 GB. Shrink via int8 (4×) / binary (32×) with rescoring,
   Matryoshka truncation, or a smaller dimension model.
