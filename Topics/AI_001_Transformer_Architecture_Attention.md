# A001 — Transformer Architecture & Attention

**Tier:** 🔴 MUST | **Est. time:** 40 min
**Date taught:** 2026-09-19

---

## 1. Hook

You're in an interview and they ask: *"Your RAG system retrieves the right chunk, but the
LLM still gives a wrong answer sometimes — why?"* If you don't know what happens inside the
transformer between "chunk goes in" and "answer comes out," you can't diagnose that. Attention
is the mechanism that decides what the model *actually looks at* when generating each word —
and it's the single most-asked "explain this from first principles" question in AI interviews.

## 2. Prerequisite check

You need: basic linear algebra (vectors, dot products, matrix multiplication) and the idea
that words become numbers (vectors) before a model can process them. That's it — no calculus
needed for the interview-level explanation.

## 3. Concept introduction (no jargon)

A transformer is a next-word-prediction machine. Given a sequence of words, it predicts what
word comes next, one at a time. The breakthrough — **attention** — is how it decides which
earlier words matter most when predicting the next one.

Before transformers (RNNs/LSTMs), models read text left to right, one word at a time, carrying
a single "memory" vector forward. By the time you reached word 50, the model had mostly
forgotten word 2. Attention fixes this: **every word can directly look at every other word**,
regardless of distance, and learn how much to weight each one.

Analogy: imagine reading the sentence *"The trophy didn't fit in the suitcase because **it**
was too big."* To know what "it" refers to, you don't re-read sequentially — your eyes jump
straight to "trophy" and "suitcase" and weigh which one makes sense. That direct, weighted
jump is attention.

## 4. How it works internally

**Step-by-step, at the level an interviewer expects:**

1. **Tokenization + embedding.** Input text is split into tokens, each mapped to a vector
   (e.g., 4096 dimensions for a 7B model).
2. **Positional encoding.** Since attention has no inherent sense of order (unlike an RNN),
   position information must be injected. The original Transformer *added* a sinusoidal
   position vector to each token embedding. Modern models (Mistral, Llama) instead use
   **RoPE (rotary position embeddings)**, which does *not* touch the embedding: it rotates
   the Query and Key vectors by an angle proportional to token position, inside every
   attention layer, so the Q·K dot product naturally depends on the *relative* distance
   between two tokens.
3. **Q, K, V projections.** Each token's vector is projected through three learned weight
   matrices into three new vectors: **Query** (what am I looking for?), **Key** (what do I
   contain, as a label?), **Value** (what do I actually offer, as content?).
4. **Attention scores.** For each token, compute the dot product of its Query with every
   other token's Key. High dot product = high relevance. Scale by √(head dimension) to keep
   gradients stable, then softmax to turn scores into weights that sum to 1.
5. **Weighted sum.** Multiply each token's Value vector by its attention weight and sum —
   this produces the new representation for the current token, now infused with context from
   every relevant token in the sequence.
6. **Multi-head attention.** Steps 3–5 run in parallel across multiple "heads" (e.g., 32
   heads in a 7B model), each learning to attend to different kinds of relationships (one
   head might track subject-verb agreement, another might track coreference like "it" →
   "trophy"). Outputs are concatenated and projected back down.
7. **Feed-forward layer.** The attention output passes through a position-wise MLP (usually
   4x the hidden dimension, then back down) — this is where most of the model's "knowledge"
   is thought to live, as opposed to attention which handles "relationships."
8. **Residual connections + layer norm.** Each sub-layer's output is added back to its input
   (residual) and normalized — this is what makes it possible to stack 32+ layers without
   gradients vanishing.
9. **Stack and repeat.** A 7B model stacks ~32 of these transformer blocks. Each layer
   refines the representation further.
10. **Output projection.** The final layer's output is projected to vocabulary size (~32K–128K
    tokens) and softmax gives a probability distribution — the next-token prediction.

**Causal masking (decoder-only, GPT-style — what Mistral/Llama/GPT use):** during generation,
a token can only attend to itself and earlier tokens, never future ones (enforced by masking
future positions with -∞ before softmax). This is what makes autoregressive generation
consistent — the model never "cheats" by looking ahead.

## 5. Diagram

```
Input tokens: "The" "trophy" "didn't" "fit" "in" "the" "suitcase" "because" "it"
                                                                              │
                                                                      [predicting "it"]
                                                                              │
                              ┌───────────────────────────────────────────────┐
                              │  Query("it") · Key(every prior token)          │
                              │                                                 │
                              │  "The"      → score 0.01                       │
                              │  "trophy"   → score 0.52  ← high attention     │
                              │  "didn't"   → score 0.02                       │
                              │  "fit"      → score 0.08                       │
                              │  "in"       → score 0.01                       │
                              │  "the"      → score 0.01                       │
                              │  "suitcase" → score 0.30  ← high attention     │
                              │  "because"  → score 0.05                       │
                              └───────────────────────────────────────────────┘
                                                │
                              softmax-weighted sum of Value vectors
                                                │
                              new contextual representation of "it"
                              (now encodes: "refers to trophy, in
                               relation to suitcase, re: size/fit")
```

Multi-head version: run 32 of these score tables in parallel, each with its own learned
Q/K/V projections, then concatenate the 32 outputs and project back to the model dimension.

## 6. ★ Benchmark numbers

| Metric | Typical value (7B model, e.g. Mistral-7B) |
|--------|-------------------------------------------|
| Hidden dimension | 4096 |
| Number of layers | 32 |
| Attention heads | 32 (Mistral uses **grouped-query attention**: 32 query heads, 8 KV heads) |
| Parameters | ~7 billion |
| Context window | 8K (base Mistral-7B) – 32K (Mistral-7B-v0.2+) |
| Attention complexity | O(n²) in sequence length n — doubling context ≈ 4x the attention compute |
| Sliding window attention (Mistral) | 4096-token window — each token attends to at most the last 4096 tokens, not the full context, to keep compute manageable |

**Why this matters for interviews:** the O(n²) cost of standard attention is *the* reason
long-context models are expensive and why techniques like sliding-window attention (Mistral),
PagedAttention (vLLM, covered in C001/C003), and FlashAttention (fused, memory-efficient
attention kernel) exist.

## 7. When to use it

Trick framing — you don't "choose" to use a transformer the way you choose a vector DB. The
real interview-relevant "when" is about **attention variants**:

- **Full (bidirectional) attention** — encoder models (BERT-style), good for
  embeddings/classification where you can see the whole input at once (no generation).
- **Causal (masked) attention** — decoder-only models (GPT, Mistral, Llama), required for
  autoregressive generation.
- **Sliding-window attention** — when you need long-context efficiency and most relevant
  context is local (Mistral's approach).
- **Grouped-query attention (GQA)** — when you want to reduce KV-cache memory at inference
  time with minimal quality loss (used in Mistral, Llama 2/3).

## 8. When NOT to use it / failure modes

- **Full O(n²) attention at very long context** (100K+ tokens) — becomes prohibitively
  expensive without approximations (sliding window, sparse attention, or architectural
  alternatives like state-space models/Mamba).
- **Multi-head attention with too many heads relative to model size** — diminishing returns
  and can hurt training stability; head count is tuned per model size, not arbitrarily large.
- **Standard multi-head attention (MHA) at inference with long context and high concurrency**
  — the KV cache (one Key+Value pair per token per head, stored for every generated token)
  explodes memory; this is why GQA/MQA exist and why C001 (KV cache) is a direct follow-on
  topic.

## 9. Comparison to alternatives

| Architecture | How it handles sequence context | Strength | Weakness |
|---|---|---|---|
| RNN/LSTM | Sequential hidden state, one token at a time | Low memory, naturally streaming | Forgets long-range dependencies; can't parallelize training |
| Transformer (full attention) | Every token attends to every token directly | Best long-range dependency modeling; fully parallelizable in training | O(n²) compute/memory in sequence length |
| Transformer (sliding-window, Mistral) | Each token attends to a fixed local window | Much cheaper at long context | Can miss very long-range dependencies beyond the window (mitigated by stacking layers — info can still propagate across windows over multiple layers) |
| State-space models (Mamba) | Recurrent-style state update, but parallelizable via convolution | O(n) not O(n²); efficient at very long context | Newer, less battle-tested; weaker at some in-context recall tasks vs. attention |

## 10. ★ Evaluation angle

You don't "evaluate attention" directly in production — but you evaluate its *symptoms*:
- **Attention/context utilization**: does the model actually use the middle of a long context,
  or does it exhibit "lost in the middle" behavior (documented phenomenon where models attend
  more reliably to the start/end of context than the middle)? Test by placing the answer at
  different positions in a long prompt and measuring accuracy — this is a real production
  eval technique ("needle in a haystack" test).
- **Attention visualization** (research/debug tool, rarely production): plotting attention
  weights between query and retrieved-chunk tokens to sanity-check the model is "looking at"
  the right evidence — useful when debugging why RAG gives a wrong answer despite correct
  retrieval.

## 11. Interview framing (exact words to say)

*"A transformer processes a sequence by letting every token attend to every other token
through learned Query/Key/Value projections — the attention score between two tokens is a
scaled dot product of Query and Key, softmaxed into weights, used to combine Value vectors.
Multi-head attention runs this in parallel across multiple subspaces so different heads can
specialize in different relationships. In decoder-only models (GPT, Mistral, Llama), attention
is causally masked so each token only sees the past. Techniques like sliding-window attention
and grouped-query attention exist specifically to keep the KV cache and compute manageable at
longer context lengths."*

## 12. Common mistakes

- Saying "attention lets the model see the whole input" without explaining *how* (Q/K/V, dot
  product, softmax) — interviewers will probe one level deeper.
- Confusing **self-attention** (tokens attending to other tokens in the *same* sequence) with
  **cross-attention** (decoder tokens attending to a *separate* encoder output — used in
  encoder-decoder models like T5/original Transformer for translation; NOT used in
  decoder-only models like Mistral/Llama/GPT).
- Thinking more attention heads always = better — it's a tuned hyperparameter, not "more is
  more."
- Forgetting that attention is O(n²) — this is the single most common "why is long context
  expensive" gap.
- Not knowing the difference between MHA, GQA, and MQA when asked about KV-cache/inference
  efficiency (see the comparison table above and the TechChoices entry).

## 13. Deep dive bait — the follow-up question

**"How does the KV cache relate to attention, and why does it matter for serving?"**

Answer: during generation, instead of recomputing Key and Value vectors for all previous
tokens at every new step (which would be O(n²) *per token generated*, making generation
O(n³) overall), the model caches the K and V vectors for every previous token. Each new
token only needs to compute its own Q, K, V once, then attend against the cached K/V of all
prior tokens. This cache grows linearly with sequence length and is the dominant memory cost
during inference at long context — directly setting up **C001 (Inference Serving Basics)**
where PagedAttention (vLLM) is explained as a way to manage this cache efficiently across many
concurrent requests. GQA/MQA reduce this cache size by sharing K/V across multiple query
heads (as Mistral does: 8 KV head-groups instead of 32).

---

## 14. TechChoices.md entry

See `AI_ML_TechChoices.md` → **"Attention Mechanism Variants"** section (added this session).

## 15. Glossary additions

| Term | Definition |
|---|---|
| **Attention** | Mechanism where each token computes a weighted combination of other tokens' Value vectors, weights derived from Query-Key similarity |
| **Query / Key / Value (Q/K/V)** | Three learned projections of a token vector: Query = what I'm looking for, Key = how I'm labeled/findable, Value = what content I contribute |
| **Multi-head attention (MHA)** | Running attention in parallel across N independent Q/K/V subspaces ("heads"), each learning different relationship types |
| **Grouped-query attention (GQA)** | Multiple query heads share a smaller number of Key/Value head groups, cutting KV-cache memory vs. full MHA (used in Mistral-7B) |
| **Causal masking** | Preventing a token from attending to future tokens, required for autoregressive (left-to-right) generation |
| **KV cache** | Stored Key/Value vectors from previously generated tokens, reused at each new generation step instead of recomputed — dominant memory cost at inference |

## 16. Cheat sheet (10-line summary)

1. Transformer = repeated blocks of [attention → feed-forward], stacked ~32x for a 7B model.
2. Attention lets every token directly weigh every other token, unlike RNNs' sequential memory.
3. Mechanism: Query·Key → scaled dot product → softmax → weighted sum of Values.
4. Multi-head = run this in parallel across subspaces (e.g., 32 heads) for different relationship types.
5. Decoder-only models (Mistral, Llama, GPT) use causal masking — no looking at future tokens.
6. Attention cost is O(n²) in sequence length — the core reason long context is expensive.
7. Mistral-7B uses grouped-query attention (32 query heads, 8 KV groups) + 4096-token sliding window.
8. KV cache stores past Key/Value vectors so generation doesn't recompute attention from scratch each step.
9. GQA/MQA exist specifically to shrink the KV cache for cheaper, faster inference at scale.
10. Cross-encoder rerankers use full joint attention over query+candidate — why they're accurate but slow vs. bi-encoders.

---

*Lesson file complete. See `Revision/Revision_AI_001.md` for active recall questions and
spaced repetition schedule.*
