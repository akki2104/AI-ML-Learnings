# A002 — Embeddings & Semantic Similarity

**Tier:** 🔴 MUST | **Est. time:** 35 min
**Date taught:** 2026-10-31 (rewritten same day as a beginner-friendly "story" version at the
learner's request: every jargon term is defined where it first appears, with *why it exists*)

> **How to read this:** it is one story with 9 chapters. Each chapter exists because the
> previous one left a problem. Every new term is written in **bold** with a plain meaning and
> a "why do we need it".

---

## The story in one line

A support chatbot can't find answers because customers don't use the same words as the
documents. The fix: turn text into numbers that capture *meaning*, so "nearby numbers" means
"similar meaning". Everything in this topic is how to do that, measure "nearby", do it fast,
and know where it breaks.

---

## Chapter 1 — The problem: keyword search is literal

An online store has a help page titled **"Refund policy"**. A customer types:
**"how do I get my money back"**.

Keyword search (matching the exact words) finds nothing, because the two texts share zero
words. Humans know they mean the same thing; the computer doesn't.

- **Keyword search / BM25:** an old, reliable method that scores documents by how many of the
  query's exact words they contain (rare words count more). *Why it exists:* it is fast and
  needs no AI. *Its limit:* it can't see that "money back" ≈ "refund".

We need a way for a computer to compare **meaning**. Computers only compare numbers, so:
**turn each text into numbers in such a way that similar meanings give similar numbers.**

## Chapter 2 — The core idea: text as a point on a map

- **Vector:** just a list of numbers, e.g. `[0.2, -0.7, 0.5]`. You can think of it as a point
  on a map (2 numbers = a point on paper, 3 numbers = a point in a room).
- **Dimension:** how many numbers are in the list. A 768-dimension vector has 768 numbers —
  a point in a 768-D "room". We can't picture it, but the maths works exactly like 2-D.
- **Embedding:** the vector you get for a piece of text. Same length (e.g. 768 numbers)
  whether the text is "hi" or a paragraph. *Why a fixed length?* So any two texts can be
  compared with the same maths.

Analogy: a city map where similar places are placed near each other — all pizza shops in one
area, all hospitals in another. An embedding places *sentences* on such a map. "Refund
policy" and "get my money back" land in the same neighbourhood; "reset password" lands far
away. **Search = place the query on the map and look at what's nearest.**

The individual numbers mean nothing by themselves (no number means "is about money"). Only
the *distances between points* carry information.

## Chapter 3 — Who draws the map? The embedding model

An **embedding model** is a small neural network that converts text → vector. It is built
from pieces you met in A001:

1. **Tokenizer** — chops text into **tokens** (word pieces like "refund", "##s"), each mapped
   to an ID number. *Why:* neural networks need numbers, and a fixed vocabulary of pieces
   handles any word.
2. **Transformer encoder** — a stack of transformer layers (A001) that turns every token into
   a vector that has *absorbed context from the other tokens* using **attention** (the
   mechanism letting each token look at the others and weigh how relevant each is).
   - **Encoder vs decoder:** a *decoder* (like ChatGPT) generates text one token at a time and
     uses a **causal mask** (a rule: "you may not look at tokens to your right/future",
     otherwise it would cheat while predicting the next word). An *encoder* doesn't generate,
     it only understands, so it removes that rule.
   - **Bidirectional:** because there's no mask, each token sees words on both its left and
     right. *Why it helps:* "bank" in "river bank" vs "bank account" is decided by words on
     either side.
3. **Pooling** — after step 2 you have one vector *per token*, but you want *one vector per
   text*. Pooling merges them. **Mean pooling** = average all token vectors (most common).
   **[CLS] pooling** = use the vector of a special summary token added at the start.
   *Why:* a single fixed-size vector per text is what makes comparison and storage simple.
4. **Normalize** — rescale the vector so its length is exactly 1 (see Chapter 4 for why).

```
"get my money back" → tokenizer → tokens → transformer ENCODER (sees both sides)
                   → one vector per token → POOLING → one vector → normalize → embedding
```

## Chapter 4 — How does the model learn what's "similar"? Contrastive training

Out of the box, a transformer's vectors are *not* good for similarity. It has to be trained.

- **Training pairs:** examples like (question, passage that answers it) collected from the web,
  support logs, etc.
- **Contrastive learning:** a training method that teaches by contrast: *pull matching pairs
  closer together, push non-matching pairs apart.* Like teaching someone to recognise a dog by
  showing "this is a dog" and "this is not".
  - **Positive:** the passage that truly answers the question.
  - **Negative:** a passage that doesn't. The model learns to place the question *away* from it.
  - **Batch / in-batch negatives:** training processes many pairs at once (a **batch**). For
    each question, the *other* questions' passages in that batch are free negatives. Bigger
    batch = more negatives = a sharper map. That's why embedding models train with huge
    batches.
  - **Hard negative:** a negative that *looks* relevant but isn't (e.g. a page about
    "refund policy for hotels" for a store-refund question). Training on these teaches fine
    distinctions.
  - **Loss (InfoNCE):** a single score of "how wrong is the model right now" that training
    tries to reduce. InfoNCE is the specific loss used here: it rewards a high similarity for
    the positive relative to all the negatives.

**Why this matters to you:** "similar" is *whatever the training pairs said*. A model trained
on web Q&A can be mediocre on legal text or source code. That is **domain dependence** (a
**domain** = the kind of text: legal, medical, code, e-commerce…).

## Chapter 5 — How do we measure "close"? Three rulers

We have two embeddings, `a` and `b`. How close are they?

Tiny 2-D example: `a = (3, 4)`, `b = (6, 8)` (same direction, b is twice as long),
`c = (4, −3)` (at a right angle to a).

| Measure | Plain meaning | a vs b | a vs c |
|---|---|---|---|
| **Dot product** `a·b = a₁b₁ + a₂b₂` | multiply matching numbers, add up. Big when vectors point the same way *and* are long | 3·6+4·8 = **50** | 3·4+4·(−3) = **0** |
| **Cosine similarity** | the *angle* between the arrows; ignores length. 1 = same direction, 0 = unrelated, −1 = opposite | **1.0** | **0** |
| **L2 (Euclidean) distance** | straight-line distance between the two points; smaller = closer | **5** | **√50 ≈ 7.07** |

- **Cosine = dot product ÷ (length of a × length of b).** *Why cosine for text:* the length
  of an embedding mostly reflects things like text length, not meaning, so we want to compare
  direction only.
- **Normalizing (L2-normalizing)** = divide a vector by its length so the length becomes 1.
  `a = (3,4)` has length 5, so normalized `a = (0.6, 0.8)`.
- **The shortcut that matters in interviews:** once *all* vectors have length 1:
  - dot product = cosine (the length division does nothing, it's dividing by 1),
  - and `distance² = 2 − 2 × cosine` (check: (1,0) vs (0,1): distance² = 2, cosine = 0 ✓),
    so smaller distance ⇔ higher cosine.
  
  So all three rulers produce the **same ranking** (**ranking** = the ordered list of results).
  *Practical use:* normalize once when you store each vector, then use the cheapest ruler
  (dot product) forever. It is the fastest operation on hardware.
- **Distance vs similarity:** similarity is "higher = closer"; distance is "lower = closer".
  Cosine *distance* = 1 − cosine similarity. Don't mix them up.

## Chapter 6 — Using it: build the map once, search it many times

Two phases (**offline** = done ahead of time, not while a user waits; **online** = done live
per user request):

**Offline (indexing):**
1. Split documents into **chunks** — pieces of a few hundred words. *Why:* models have a
   **maximum input length** (often 512 tokens; extra text is **silently truncated**, i.e.
   dropped with no error), and one vector for a whole 10-page doc would blur every topic into
   one average. This is the entire reason chunking exists (topic B002).
2. Embed each chunk, normalize, and store the vectors in a **vector database** — a database
   built to answer "which stored vectors are nearest to this one?" quickly (B004).

**Online (query):**
1. Embed the user's query with the **same model**.
2. Ask the vector database for the **top-k** nearest vectors (**top-k** = the k best matches,
   e.g. k = 5).
3. Optionally rerank (Chapter 7), then give those chunks to the LLM to write the answer.

Why not compare against every stored vector? With 10 million vectors, that's 10 million
comparisons per query. Vector databases use an **ANN index** (**Approximate Nearest
Neighbour**): a clever structure (like a shortcut road network) that finds *almost* the best
matches while checking only a tiny fraction of vectors. *Tradeoff:* tiny accuracy loss for a
huge speed gain. (Details in B004; today just know the term.)

```
OFFLINE:  docs → chunk → embed → normalize → [vector DB / ANN index]
ONLINE:   query → embed (same model) → top-k nearest from DB → (rerank) → LLM → answer
```

## Chapter 7 — Fast vs accurate: bi-encoder and cross-encoder

- **Bi-encoder** = what we've described: query and each chunk are encoded **separately**
  into vectors ("bi" = two independent towers). *Pro:* chunk vectors are prepared in advance,
  so searching millions is fast. *Con:* each text is squeezed into one vector *before* seeing
  the other, so fine detail is lost.
- **Cross-encoder** = feed **query + chunk together** into one transformer so every query
  word can attend to every chunk word; it outputs one relevance score. *Pro:* much more
  accurate. *Con:* nothing can be prepared in advance; scoring 10M chunks means 10M model
  runs per query. Impossible live.
- **Reranking** = a second stage: the fast bi-encoder picks the top ~50–100 candidates, then
  the slow, accurate cross-encoder re-orders just those. You get speed *and* quality. (This is
  topic B006.)

## Chapter 8 — Where embeddings break (and the fix for each)

| Failure | Example | Fix |
|---|---|---|
| **Exact tokens** (IDs, codes) get blurred | search `ERR_4021` finds other errors | **Hybrid search**: run BM25 *and* embeddings, merge the two ranked lists (e.g. with **RRF**, Reciprocal Rank Fusion: score items by their rank in each list). B005 |
| **Negation** | "refund allowed" ≈ "refund not allowed" | reranker / LLM check |
| **Numbers & dates** | "orders above ₹5000 last month" | **metadata filters** (structured conditions like `amount > 5000`) or SQL |
| **Domain mismatch** | legal jargon | pick/fine-tune a domain model |
| **Silent truncation** | chunk tail never embedded | keep chunks under the model's limit |
| **Missing prefix** | recall drops quietly | some models (e5, bge) expect text like `"query: ..."` / `"passage: ..."` because queries (short questions) and chunks (long statements) are different; follow the model card |

## Chapter 9 — Practical housekeeping: size, change, and measurement

**Size.** One embedding of 768 numbers stored as **float32** (a number stored in 4 bytes)
= 768 × 4 = 3,072 bytes ≈ 3 KB. So **1M chunks ≈ 3 GB** (1536 dims ≈ 6 GB; 3072 dims ≈ 12 GB;
1024 dims ≈ 4 GB). Formula: `N × dimensions × 4 bytes`.

Ways to shrink:
- **Quantization** = store each number with fewer bits. **int8** (1 byte) = 4× smaller;
  **binary** (1 bit) = 32× smaller. *Cost:* some accuracy loss, usually repaired by a
  **rescore** step (re-compare the top candidates using the full-size vectors).
- **Matryoshka embeddings** = models trained (like Russian nesting dolls) so that the first
  256 numbers already form a decent embedding, letting you **truncate** 3072 → 256 and lose
  little quality.
- Pick a smaller model (fewer dimensions).

**Changing the model.** Vectors from model X and model Y live on *different maps*; distances
between them are meaningless. If you switch, **re-embed every chunk** into a new, versioned
index (**re-index**), test it, then switch traffic over (**blue/green** = run old and new
side by side, then flip). Mixing models silently ruins search; this is also called
**embedding drift** when it happens by accident.

**Measuring quality.** You can't eyeball it. Make a **labelled set**: 100–300 real queries
each paired with the chunk(s) that correctly answer them. Then compute:
- **Recall@k** — in what % of queries is a correct chunk inside the top-k? The most important
  one, because the LLM can only use what retrieval returned.
- **MRR (Mean Reciprocal Rank)** — how near the top the first correct chunk is (rank 1 → 1,
  rank 2 → ½, rank 5 → ⅕, averaged).
- **nDCG@k** — like MRR but rewards putting *more* relevant chunks higher when relevance has
  degrees.
- **MTEB** — a public leaderboard comparing embedding models. Use it to shortlist, then
  always test on *your own* data.

Rough numbers: embedding a query ≈ 5–20 ms (small model, CPU) or 50–200 ms via an API;
related-text cosine ≈ 0.7–0.9, unrelated ≈ 0–0.3, but these values **differ per model**, so
never hard-code "0.8 means relevant" without calibrating on your labelled set.

---

## When to use / when not

**Use embeddings when:** people phrase things differently from the documents, queries are
natural language, you need multilingual search, or you want deduplication/clustering by meaning.
**Don't rely on embeddings alone when:** exact identifiers matter, negation matters, numbers
or dates matter, or the corpus is tiny (BM25 or just paste everything into the prompt).

## Alternatives at a glance

| Approach | Choose when |
|---|---|
| BM25 (keyword) | IDs, jargon, a baseline |
| Dense bi-encoder (embeddings) | default for natural-language search |
| Hybrid (BM25 + embeddings) | production default |
| SPLADE (learned sparse: AI-expanded keywords) | want semantic matching + keyword-style index |
| ColBERT (many vectors per doc, token-level match) | quality-critical, can afford big index |
| Cross-encoder | reranking top-k only |

## Common mistakes
1. Mixing vectors from different models. 2. Forgetting to normalize or using a mismatched
metric. 3. Skipping `query:`/`passage:` prefixes. 4. Ignoring the token limit. 5. Treating
cosine 0.8 as universal. 6. Saying embeddings "understand" text (they reflect training
data). 7. Mixing up distance and similarity.

## Deep dive bait
- **Why cosine, or why dot product?** Normalized → identical ranking; cosine removes length
  (not meaningful for text); dot product is the cheapest way to compute it.
- **Model changed — now what?** Re-embed all into a versioned index, evaluate on labelled
  set, blue/green cutover, delete old.
- **Why can't one vector represent a long document?** Fixed-size bottleneck: topics average
  into a vague blob, and the tail gets truncated → chunk, or use multi-vector (ColBERT).

## Interview framing (exact words)
> "For retrieval I'd use a bi-encoder embedding model — a transformer encoder trained
> contrastively so queries and their relevant passages land close together. I embed chunks
> offline, normalize them, and store them in an ANN index; at query time I embed the query
> with the same model and take the top-k by inner product, which equals cosine on normalized
> vectors. Because dense retrieval is weak on exact identifiers and negation, I'd pair it with
> BM25 in a hybrid setup and rerank the top candidates with a cross-encoder. I'd choose the
> model by recall@k on a labelled sample of our own queries, not just MTEB, and I'd version
> the index so a model change is a re-embed-and-cutover."

## TechChoices.md entry
See `AI_ML_TechChoices.md` → **"Similarity Metric & Embedding Dimensionality"**.

## Glossary additions
| Term | Definition |
|---|---|
| **Embedding** | Fixed-length list of numbers from a model; closeness ≈ similar meaning |
| **Vector / dimension** | A list of numbers / how many numbers are in it |
| **Bi-encoder** | Encodes query and doc separately; precomputable; used for retrieval |
| **Cross-encoder** | Reads query+doc together; accurate but slow; used for reranking |
| **Cosine similarity** | Angle-based closeness (−1..1), ignores length |
| **Contrastive loss (InfoNCE)** | Training that pulls matching pairs together, pushes mismatches apart |
| **Matryoshka embeddings** | First k numbers form a usable smaller embedding |
| **ANN** | Approximate Nearest Neighbour: fast, almost-exact vector search |
| **Recall@k** | % of queries whose correct chunk appears in the top-k |

## Cheat sheet (10 lines)
1. Embedding = text → fixed-length vector; nearby = similar meaning.
2. Made by a bidirectional transformer encoder + pooling, trained contrastively.
3. "Similar" is learned from training pairs, so it depends on the domain.
4. Normalize vectors → dot = cosine → same ranking as L2.
5. Bi-encoder = fast retrieval; cross-encoder = accurate reranking of top 50–100.
6. Max input is often 512 tokens; extra is silently cut → chunk.
7. Use model-required `query:`/`passage:` prefixes.
8. Storage = N × dims × 4 bytes; shrink with int8/binary/Matryoshka.
9. Weak on exact IDs, negation, numbers → hybrid + filters + reranker.
10. Model change = re-embed all; measure with recall@k/MRR/nDCG on your own labelled set.
