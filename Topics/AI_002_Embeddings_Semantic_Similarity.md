# Topic A002 — Embeddings & Semantic Similarity

**Module:** A — LLM Foundations
**Tier:** 🔴 MUST
**Status:** Taught 2026-10-31 (recall Q&A in progress)
**Difficulty:** Medium

---

## 1. Why This Topic Exists

A001 showed how a transformer turns text into vectors (lists of numbers) while it reads. Embeddings reuse exactly that trick for a different job: turn a whole chunk of text into **one** vector so a computer can compare *meaning* with simple maths. Every RAG design ("how do you find the right documents?") sits on this, so it's asked in nearly every AI system design round.

---

## 2. The Problem, Precisely

```
Help page title:   "Refund policy"
Customer types:    "how do I get my money back"

Keyword search (BM25 — scores docs by how many of the query's exact words
they contain): shared words = 0  →  no result.
```
Humans know these mean the same thing. A computer only compares numbers, so the real question is:

**How do we turn text into numbers such that similar meaning = similar numbers?**

---

## 3. Core Concepts

### Text as a point on a map
```
A VECTOR is a list of numbers.        [0.2, -0.7, 0.5]
Each number is one DIMENSION.         3 numbers = a point in a room
                                      768 numbers = a point in a 768-D "room"
                                      (can't be pictured, maths works the same)

An EMBEDDING = the vector a model produces for a piece of text.
Same length for "hi" and for a paragraph — so any two texts can be compared the same way.
```
Analogy: a city map where pizza shops cluster in one area and hospitals in another. An embedding map does that for sentences: "refund policy" and "get my money back" land in the same neighbourhood, "reset password" far away. **Search = drop a pin for the query, read off what's nearest.** The individual numbers mean nothing alone — only distances between points do.

### Who draws the map: the embedding model
```
"get my money back"
   │ TOKENIZER      chops text into small pieces (tokens), each = an ID number.
   │                Why: networks only eat numbers; a fixed set of pieces handles any word.
   ▼
 tokens ──► TRANSFORMER ENCODER   (A001's attention, but reading in BOTH directions)
   │         one vector per token, each already mixed with its neighbours' context
   ▼
 POOLING        merges the per-token vectors into ONE vector for the whole text
   │            (mean pooling = average them; [CLS] pooling = use a special summary token)
   │            Why: one fixed-size vector per text is what makes storing/comparing easy.
   ▼
 NORMALIZE      rescale the vector to length 1 (why: next section)
   ▼
 embedding  [768 numbers]
```
**Encoder vs decoder:** a decoder (ChatGPT-style) writes text one token at a time and wears a **causal mask** (a rule: "no peeking at tokens to your right", otherwise it would cheat when predicting the next word). An encoder only *reads*, so the mask is removed and every token sees both sides — **bidirectional**. That's why "bank" resolves correctly in "river bank" vs "bank account".

### How the model learns what "similar" means: contrastive training
A plain transformer's vectors are bad for similarity; it must be trained on examples.
```
CONTRASTIVE LEARNING = teach by contrast: pull matching pairs together, push mismatches apart.

 training pair:  (question, passage that answers it)  → POSITIVE: pull together
 other passages: NEGATIVES                            → push away
   • in-batch negatives: the other pairs' passages in the same BATCH (group of examples
     processed together) come for free — bigger batch = more negatives = sharper map
   • hard negatives: look relevant but aren't (e.g. "refund policy for hotels")
     — these teach fine distinctions
 LOSS (InfoNCE) = one score for "how wrong is the model now"; training lowers it.
```
**Consequence:** "similar" means *whatever the training pairs said*. A model trained on web Q&A can be mediocre on legal text or code — it is **domain dependent**.

### Measuring "close": three rulers
```
a = (3,4)    b = (6,8)    c = (4,-3)

                      a vs b     a vs c
Dot product (Σ a·b)      50          0     multiply matching numbers, add up
Cosine similarity       1.0          0     ANGLE only: 1 = same direction, 0 = unrelated
L2 distance (straight line) 5      7.07    smaller = closer
```
**Normalizing** = divide a vector by its length so length becomes 1: `(3,4)` has length 5 → `(0.6,0.8)`.

**The shortcut to remember:** once every vector has length 1, dot product = cosine, and `distance² = 2 − 2·cosine` (check: (1,0) vs (0,1): distance² = 2, cosine = 0 ✓). So **all three rulers give the same ranking** (ordered result list). In practice: normalize once when storing, then use dot product forever — it's the fastest operation on hardware. (Cosine *distance* = 1 − cosine *similarity*; don't mix them up.)

### Using it: build once, search many times
```
OFFLINE (ahead of time)                          ONLINE (per user request)
docs → CHUNK → embed → normalize → VECTOR DB     query → embed (SAME model) → top-k nearest
                                                          → (rerank) → LLM → answer
```
- **Chunk** = a piece of a few hundred words. Why: models have a max input (often 512 tokens) and cut the rest **silently**; also one vector for 10 pages blurs every topic into one average. This is the whole reason chunking exists (B002).
- **Vector database** = a store built to answer "which saved vectors are nearest this one?".
- **top-k** = the k best matches (e.g. 5).
- **ANN (Approximate Nearest Neighbour)** = an index that finds *almost* the best matches while checking a tiny fraction of vectors — checking all 10M per query would be far too slow. Tradeoff: tiny accuracy loss for huge speed (B004).

### Fast vs accurate: bi-encoder and cross-encoder
```
BI-ENCODER     query ─► vec     chunk ─► vec  (separately, chunk vecs prepared in advance)
               fast: millions searched in ms.  Cost: each text is squashed into ONE
               vector before seeing the other → fine detail lost.

CROSS-ENCODER  [query + chunk] together in one transformer ─► relevance score
               accurate (every query word attends to every chunk word).
               Cost: nothing can be precomputed — 10M chunks = 10M model runs per query.

RERANKING      bi-encoder takes top ~50–100 ─► cross-encoder re-orders just those (B006)
```
**The tradeoff of embedding at all:** compressing a whole chunk into one fixed-size vector is what makes search fast and cheap, *and* what loses detail (exact IDs, negation, numbers).

---

## 4. Where Embeddings Break

```
Exact tokens (ERR_4021, SKUs)  → blurred        FIX: hybrid search = BM25 + embeddings,
                                                 merged with RRF (Reciprocal Rank Fusion:
                                                 score items by their rank in each list)
Negation ("allowed" vs "not")  → look almost identical   FIX: reranker / LLM check
Numbers & dates                → no real arithmetic      FIX: metadata filters / SQL
Domain mismatch                → poor on jargon          FIX: domain model or fine-tune
Missing prefix                 → quiet recall drop       FIX: models like e5/bge expect
                                 "query: …"/"passage: …" — follow the model card
```

---

## 5. Size, Changing the Model, Measuring

**Size:** one 768-dim vector as **float32** (4 bytes/number) = 3,072 B ≈ 3 KB.
```
Storage = N × dimensions × 4 bytes
1M × 768  ≈ 3 GB    1M × 1024 ≈ 4 GB    1M × 1536 ≈ 6 GB    1M × 3072 ≈ 12 GB
```
Shrink it: **quantization** (fewer bits per number: int8 = 4× smaller, binary = 32× smaller, usually followed by a **rescore** — re-compare the top candidates with full-size vectors); **Matryoshka embeddings** (trained so the first 256 numbers are already a decent embedding — truncate 3072 → 256); or a smaller-dimension model.

**Changing the model:** every model draws its own map with its own coordinates. Comparing a Y-vector to X-vectors is like comparing GPS coordinates from two different systems — meaningless. So you must **re-embed every chunk** into a new versioned index, test, then switch traffic (**blue/green**: run old and new side by side, then flip). Doing this by accident is **embedding drift**.

**Measuring:** build a **labelled set** of 100–300 real queries each mapped to the chunk(s) that answer it.
- **Recall@k** — % of queries with a correct chunk in the top-k. The key metric: the LLM can only use what retrieval returned.
- **MRR** — how near the top the first correct chunk sits (rank 1 → 1, rank 2 → ½, …, averaged).
- **nDCG@k** — like MRR, but rewards putting *more* relevant chunks higher.
- **MTEB** — public leaderboard; shortlist only, then test on your own data.

Rough numbers: embed a query 5–20 ms (small model, CPU) or 50–200 ms via API; related-text cosine ≈ 0.7–0.9, unrelated ≈ 0–0.3, but **values differ per model**, so never hard-code "0.8 = relevant" without calibrating.

---

## Tech Decision Box: Retrieval Approach & Similarity Metric

```
Use BM25 (keyword) when:        IDs, jargon, a baseline, tiny corpus
Use DENSE embeddings when:      users paraphrase; natural-language queries; multilingual
Use HYBRID (BM25 + dense) when: production default — covers each one's blind spot
Use a CROSS-ENCODER when:       reranking the top 50–100 only
Use ColBERT (many vectors per doc, token-level match) when:
                                quality-critical and you can afford a big index

Metric: normalized vectors → dot product (same ranking as cosine and L2, cheapest).
Raw un-normalized dot product only when length carries signal (e.g. popularity in recommenders).
Dimensions: 384–768 for latency/memory/10M+ docs; 1024–3072 (or Matryoshka-truncated)
for quality-critical, < ~5M docs.
```
**Interview sentence:** "I'd use a bi-encoder — a transformer encoder trained contrastively so queries and their relevant passages land close together. I embed chunks offline, normalize them, store them in an ANN index, and at query time take the top-k by inner product, which equals cosine on normalized vectors. Because dense retrieval is weak on exact IDs and negation, I'd add BM25 and rerank with a cross-encoder, pick the model by recall@k on our own labelled queries, and version the index so a model change is a re-embed and cutover."

---

## Common Mistakes

| Mistake | Correction |
|---|---|
| Comparing vectors from different models | Different maps — re-embed everything into a new index when the model changes |
| Forgetting to normalize / mismatched metric | Normalize once at index time, use dot product |
| Skipping `query:`/`passage:` prefixes | Silent recall drop; follow the model card |
| Ignoring the max token limit | Text beyond it is silently cut → chunk under the limit |
| "Cosine 0.8 means relevant" | Values are model-specific — calibrate on a labelled set |
| "Embeddings understand text" | They reflect the training data; weak on negation, rare tokens, numbers |
| Using only embeddings | Add BM25 (exact tokens), filters (numbers), reranker (negation) |
| Saying distance and similarity interchangeably | Distance = 1 − similarity |

---

## Real Interview Questions

1. "How does semantic search find 'refund policy' for the query 'get my money back'?" (the core mechanism)
2. "Cosine vs dot product vs L2 — which do you use and why?" (normalization identity)
3. "Why not use a cross-encoder for first-stage retrieval?" (bi vs cross)
4. "Where does dense retrieval fail and how do you compensate?" (hybrid, reranker, filters)
5. "You want to swap the embedding model in production — walk me through it." (re-index, blue/green, evaluation)
6. "How much memory does a 10M-chunk, 768-dim index need, and how would you cut it?" (storage maths + quantization)

---

## 6. Revision Questions
See `Revision/Revision_AI_002.md`.

## 7. Summary
- An embedding is a fixed-length vector for a chunk of text; nearby vectors mean similar meaning.
- Made by a bidirectional transformer encoder → pooling → normalize; trained contrastively on (question, answer) pairs with hard negatives, so "similar" is learned and domain-dependent.
- Normalized vectors make dot product = cosine = same ranking as L2 — normalize once, use dot product.
- Bi-encoder = fast retrieval over precomputed vectors; cross-encoder = accurate but per-pair, so it reranks only the top 50–100.
- The one-vector-per-chunk squeeze is the core tradeoff: fast and cheap, but weak on exact IDs, negation, numbers → hybrid + reranker + filters.
- Storage = N × dims × 4 bytes; shrink with int8/binary/Matryoshka. Changing the model means re-embedding everything.

> **You now can:** explain what an embedding is and how it's made, compare the similarity metrics and why normalizing makes them agree, contrast bi- vs cross-encoders, list where dense retrieval fails with a fix for each, size an index, and plan a model migration.
