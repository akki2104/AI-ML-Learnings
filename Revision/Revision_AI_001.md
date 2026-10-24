# Revision — A001: Transformer Architecture & Attention

**Topic completed:** 2026-10-24
**Spaced repetition schedule:** +1d (2026-10-25) · +3d (2026-10-27) · +7d (2026-10-31) ·
+15d (2026-11-08) · +30d (2026-11-23)

| Date | Reviewed? | Score |
|------|-----------|-------|
| 2026-10-25 | ⬜ | — |
| 2026-10-27 | ⬜ | — |
| 2026-10-31 | ⬜ | — |
| 2026-11-08 | ⬜ | — |
| 2026-11-23 | ⬜ | — |

**Known weak spots (from InterviewMistakes.md):** KV cache is linear in length (compute is
quadratic); GQA shares K/V across query heads, never across tokens.

---

## Active Recall Questions

1. **Walk through the attention mechanism step by step: what are Query, Key, and Value, and
   how do they combine to produce a token's new representation?**

2. **Why is standard attention O(n²) in sequence length, and what does Mistral-7B do
   differently to manage long-context cost?**

3. **What is the KV cache, why does it exist, and how does grouped-query attention (GQA)
   reduce its memory footprint?**

4. **What's the difference between causal (masked) attention and full (bidirectional)
   attention — and give one real example of each from your own RAG pipeline.**

5. **An interviewer asks: "Your RAG system retrieves the correct chunk but the LLM still
   answers incorrectly — how does attention help you reason about what went wrong?" How do
   you answer?**

---

## Answer Key (for self-check after attempting from memory)

1. Each token's embedding is projected through three learned weight matrices into Query
   (what it's looking for), Key (how it's labeled/found), Value (what content it offers).
   Attention score = scaled dot product of one token's Query with every other token's Key,
   passed through softmax to get weights summing to 1. The new representation is the
   weighted sum of all tokens' Value vectors using those weights.

2. Because every token computes a score against every other token — n tokens × n tokens =
   n² score computations, and this repeats per layer/head. Mistral-7B uses sliding-window
   attention (4096-token window) so each token only attends to a fixed local window instead
   of the full sequence, plus grouped-query attention to shrink the KV cache.

3. The KV cache stores the Key and Value vectors computed for every previously generated
   token, so they don't need to be recomputed at every new generation step (which would make
   generation O(n³) overall). GQA reduces its memory by having multiple query heads (e.g. 32)
   share a smaller number of KV head groups (e.g. 8) instead of every head having its own K/V,
   cutting cache size ~4x with minimal quality loss.

4. Causal attention masks future tokens so a token can only attend to itself and the past —
   required for autoregressive generation (Mistral-7B generating your RAG answer). Full
   bidirectional attention lets every token see every other token in both directions — used
   by the cross-encoder reranker (B006), which jointly attends over the full query+candidate
   pair to score relevance.

5. Frame it as: retrieval got the right evidence into the context window, but the *generation*
   step is where attention decides how much weight to give that evidence vs. other tokens
   (system prompt, question phrasing, other retrieved chunks). Possible causes: the answer is
   positioned in a "lost in the middle" spot of a long context where attention is less
   reliable; too many competing chunks dilute attention; or the sliding window means very
   early context isn't fully available to later tokens. Mitigation: reorder retrieved chunks
   (most relevant near the start/end), reduce chunk count via reranking (B006), or shorten
   the context.
