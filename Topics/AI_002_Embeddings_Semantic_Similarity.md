# A002 — Embeddings & Semantic Similarity

**Tier:** 🔴 MUST | **Est. time:** 35 min
**Date taught:** 2026-10-31

---

## 1. Hook

Interviewer: *"A user searches 'how do I get my money back' and your docs only say 'refund
policy'. Keyword search returns nothing. How does your system still find it?"* The answer is
embeddings: a model that maps text to a point in space so that *meaning*, not spelling,
determines closeness. Every RAG design question ("how do you retrieve?") sits on top of this.
If you can't say what an embedding is, how similarity is computed, and why cosine vs dot
product vs L2 doesn't matter much once vectors are normalized, the rest of RAG is shaky.

## 2. Prerequisite check

From A001: tokens become vectors; attention mixes information across tokens; a transformer
outputs one vector per token. You also need: dot product, vector length (norm), and the idea
that a vector is a list of N numbers = a point/arrow in N-dimensional space.

## 3. Simple intuition

Think of a map of a city where **similar things are placed near each other**: all pizza
places in one neighbourhood, all hospitals in another. Now the "map" has 768 dimensions
instead of 2, and instead of shops it places sentences. "Refund policy" and "get my money
back" land in the same neighbourhood even though they share no words. Searching = drop a pin
for the query, look at what's nearest.

## 4. Core concept, built incrementally

**4.1 What an embedding is.** A fixed-length vector (e.g. 768 floats) produced by an
*embedding model* from a variable-length piece of text. Same length for "hi" and for a
paragraph. The numbers are individually meaningless; only *relative positions* matter.

**4.2 Where it comes from.** An embedding model is a transformer **encoder** (BERT-style,
bidirectional attention — every token sees every other, no causal mask, since we're
understanding, not generating). It outputs one vector per token; we **pool** them into one
vector: mean-pooling (average of token vectors; most common, e.g. bge/e5/MiniLM) or the
[CLS] token (some BERT-derived models). Newer LLM-based embedders (e.g. e5-mistral) use a
decoder with last-token pooling, trading cost for quality.

**4.3 How it learns "meaning": contrastive training.** Raw BERT embeddings are poor for
similarity (anisotropic — everything points in a similar direction). Embedding models are
fine-tuned on (query, relevant passage) pairs with a contrastive loss (InfoNCE):

- Pull the query vector toward its positive passage.
- Push it away from **negatives**: other passages in the same batch (in-batch negatives) and
  mined **hard negatives** (look relevant but aren't).
- Larger batches = more negatives = better embeddings (why training uses huge batches).

So "similar" means "what the training pairs said belongs together" — which is why a model
trained on web Q&A can be mediocre on legal or code text. Similarity is *learned and
domain-dependent*, not a law of nature.

**4.4 Similarity metrics.** Given vectors a and b:

| Metric | Formula | Looks at | Notes |
|---|---|---|---|
| **Dot product** | Σ aᵢbᵢ | angle **and** length | Fast; length acts as a "confidence/popularity" signal |
| **Cosine** | a·b / (‖a‖‖b‖) | angle only | Range −1..1; default for text |
| **L2 (Euclidean)** | √Σ(aᵢ−bᵢ)² | straight-line distance | Lower = closer |

**Key identity:** if vectors are L2-normalized (‖a‖=‖b‖=1), then
dot = cosine, and ‖a−b‖² = 2 − 2·cos(a,b). So all three produce the **same ranking**.
Practical consequence: normalize at index time, then use the cheapest op (dot product /
inner product) in the vector DB. The mistake is mixing a model that expects cosine with an
index configured for L2 on un-normalized vectors.

**4.5 Bi-encoder vs cross-encoder (ties to reranking, B006).**

- **Bi-encoder** (embedding model): encode query and each doc **independently**. Doc vectors
  are precomputed offline; query-time cost = 1 encode + a nearest-neighbour lookup. Scales
  to millions of docs. Loses fine-grained query↔doc token interactions because everything is
  squeezed into one vector.
- **Cross-encoder**: feed "query [SEP] doc" together; full attention across both; outputs a
  relevance score. More accurate, but must run per (query, doc) pair — can't precompute. So:
  bi-encoder retrieves top-50/100, cross-encoder reranks.

**4.6 Asymmetric search.** Queries are short questions, documents are long passages.
Many models expect prefixes: e5 uses `"query: "` / `"passage: "`; bge uses an instruction
for queries only. Forgetting the prefix silently degrades recall — a classic production bug.

**4.7 Dimensionality and compression.**

- 384 (MiniLM) → 768 (bge-base, e5-base) → 1024 (bge-large) → 1536/3072 (OpenAI
  text-embedding-3-small/large).
- Storage = N × d × 4 bytes (float32). 1M docs × 768 = ~3 GB; × 1536 = ~6 GB; × 3072 = ~12 GB.
  Plus index overhead (HNSW adds roughly 1.5–2× for graph links).
- **Matryoshka embeddings** (OpenAI v3, nomic, etc.): trained so the first k dims are a
  usable embedding; truncate 3072 → 256 with modest quality loss.
- **Quantization**: int8 = 4× smaller, binary = 32× smaller (Hamming distance), usually with
  a rescore step using full vectors on the top candidates.

## 5. How it works internally (one level down)

```
text ──► tokenizer ──► token ids ──► transformer ENCODER (bidirectional attn, ~12-24 layers)
                                            │
                              one vector per token  [t1][t2]...[tn]   (each 768-d)
                                            │
                                     POOLING (mean / CLS)
                                            │
                                      one 768-d vector
                                            │
                                  L2-normalize (length = 1)
                                            │
                                     stored in vector DB
```

Max input length matters: many encoders cap at **512 tokens** (bge-m3 / OpenAI v3 allow 8k).
Text beyond the limit is **silently truncated** — this is the real reason chunking exists
(B002), alongside the fact that a single vector for a 10-page doc blurs all topics together.

## 6. Diagram

```
          INDEX TIME (offline, once)                    QUERY TIME (online, per request)

  docs ─► chunk ─► embed model ─► [vec] ─┐         query ─► embed model ─► [qvec]
                                         ▼                                    │
                                 ┌──────────────┐        top-k nearest        │
                                 │  vector DB   │ ◄───────────────────────────┘
                                 │ (ANN index)  │ ───► k chunks ─► (rerank) ─► LLM
                                 └──────────────┘

   Embedding space (2-D cartoon):

         "refund policy" •  • "get my money back"
                              • "return an item"           (cluster: refunds)

                                              • "reset password"
                                              • "forgot login"   (cluster: auth)

   cos("get my money back","refund policy") ≈ 0.75   cos("get my money back","reset password") ≈ 0.15
```

Frames of a query: (1) "get my money back" → encode → 768-d vector; (2) ANN index
compares against millions of stored vectors (approximately — see B004); (3) returns the 5
highest dot products; (4) those chunks go into the prompt.

## 7. ★ Benchmark numbers

| Item | Typical value |
|---|---|
| Embed one short query, small model on CPU | 5–20 ms |
| Embed one query via API (OpenAI) | 50–200 ms incl. network |
| Throughput, bge-base on one GPU (batch, 256 tok) | ~1–3k chunks/s |
| Index 1M chunks (avg 256 tok), one GPU | ~10–20 min |
| Storage, 1M × 768 float32 | ~3 GB (int8 ≈ 0.75 GB, binary ≈ 96 MB) |
| Storage, 1M × 1536 float32 | ~6 GB |
| Cosine for strongly related text pair | ~0.7–0.9 (model-dependent!) |
| Cosine for unrelated text pair | ~0.0–0.3 (some models never go below ~0.5) |
| OpenAI text-embedding-3-small price | ≈ $0.02 / 1M tokens |

**Warning:** absolute cosine values are not comparable across models. A threshold of 0.8
means different things for different models — calibrate on labelled data.

## 8. When to use embeddings / when NOT to

**Use semantic (embedding) search when:**
- Users phrase things differently from the documents (synonyms, paraphrase, questions vs statements).
- Natural-language queries over unstructured text; multilingual retrieval.
- Recommendation / dedup / clustering by meaning.

**Don't rely on embeddings alone when:**
- Queries contain exact tokens that must match: IDs, error codes, SKUs, names, `ERR_4021`
  (embeddings blur rare tokens; BM25 nails them) → hybrid search (B005).
- Negation and fine distinctions matter ("refund allowed" vs "refund not allowed" embed very close).
- Numbers/units/dates ("revenue above 5M in Q3") → use structured filters / SQL.
- Corpus is tiny (a few hundred docs): BM25 or just stuff it in context.

## 9. Comparison to alternatives

| Approach | Strength | Weakness | Choose when |
|---|---|---|---|
| **BM25 / TF-IDF (sparse)** | Exact terms, no model, cheap, explainable | No synonyms/paraphrase | IDs, jargon, baseline |
| **Dense bi-encoder embeddings** | Semantic match, fast with ANN | Rare-token/negation blindness, domain drift, needs reindex on model change | Default for NL search |
| **Learned sparse (SPLADE)** | Semantic expansion + inverted index | Heavier to build | Want interpretability + semantics |
| **Late interaction (ColBERT)** | Token-level matching, better accuracy | Many vectors per doc → big index | Quality-critical, can pay storage |
| **Cross-encoder** | Best accuracy | O(candidates) per query, no precompute | Reranking top-k only |
| **Hybrid (dense + BM25, RRF)** | Covers both failure modes | Two indexes, fusion tuning | Production default |

Metric choice: normalized vectors → dot product (fastest, identical ranking to cosine & L2).
Un-normalized and magnitude is meaningful (some recommenders) → dot. Model card says
"cosine" → normalize and use inner product.

## 10. ★ Evaluation angle

You cannot judge embeddings by eyeballing. Build a **labelled retrieval set**: 100–300 real
queries each mapped to the chunk(s) that answer them. Measure:
- **Recall@k** (is the right chunk in top-k?) — the main retrieval metric, since the LLM and
  reranker can only use what's retrieved.
- **MRR** (how high is the first right chunk) and **nDCG@k** (graded relevance).
- Slice by query type (keyword-ish vs paraphrased), and watch **embedding drift**: if you
  swap models/versions or query-prefix handling, vectors from different models are *not
  comparable* — you must re-embed the entire corpus (a re-index migration, usually blue/green).
- Public benchmark: MTEB. Treat it as a shortlist, then test on your own data — leaderboard
  winners often lose on domain text.

## 11. Common mistakes

1. Comparing vectors from **different models** (or after a model upgrade) — meaningless.
2. Forgetting to normalize, or mismatching index metric with model's training metric.
3. Skipping query/passage prefixes for models that require them.
4. Ignoring the max token limit → silent truncation, chunks whose tails are never embedded.
5. Believing "cosine 0.8 = relevant" universally; thresholds are model-specific.
6. Claiming embeddings "understand" text — they capture training-distribution similarity,
   fail on negation, rare tokens, and numbers.
7. Using the LLM's own hidden states / a generative LLM as an embedder without a
   contrastive-trained embedding head.
8. Saying cosine distance and similarity interchangeably (distance = 1 − similarity).

## 12. Deep dive bait

**Q: "Why not just use dot product, or why cosine?"** Answer: with normalized vectors they're
equivalent; cosine removes the effect of vector length, which for text mostly reflects length
or token frequency, not meaning. Dot product is then used purely as an implementation of
cosine because it's cheapest in hardware (single fused multiply-add) and supported by every ANN index.

**Q: "Embedding model changed — what happens?"** All stored vectors become incomparable with
new query vectors. Plan: version the index, embed corpus with the new model into a new
index, A/B on the labelled set, flip traffic, delete old. Cost = corpus tokens × price + a
dual-write period for new docs.

**Q: "How big can an embedding be before it's not worth it?"** Quality gains flatten past
~768–1024 dims on most tasks; storage and ANN latency grow linearly. Matryoshka truncation or
int8/binary + rescore recovers most of the quality at a fraction of the memory.

**Q: "Why can't one vector represent a long document?"** It's a fixed-size bottleneck
(compression); multiple topics average into a vague centroid and the tail gets truncated →
chunk, or use multi-vector (ColBERT).

## 13. Interview framing (exact words)

> "For retrieval I'd use a bi-encoder embedding model — a bidirectional transformer trained
> contrastively so that queries and their relevant passages land close together. I embed
> chunks offline, normalize them, and store them in an ANN index; at query time I embed the
> query with the same model and take the top-k by inner product, which equals cosine on
> normalized vectors. Because dense retrieval is weak on exact identifiers and negation, I'd
> pair it with BM25 in a hybrid setup and rerank the top candidates with a cross-encoder. I'd
> pick the model by measuring recall@k on a labelled sample of our own queries, not just MTEB,
> and I'd version the index so a model change is a re-embed-and-cutover rather than an
> in-place edit."

## 14. TechChoices.md entry

See `AI_ML_TechChoices.md` → **"Similarity Metric & Embedding Dimensionality"** (added this
session). The existing "Embedding Model" section is extended in B003.

## 15. Glossary additions

| Term | Definition |
|---|---|
| **Embedding** | Fixed-length vector from a model such that semantically similar inputs are close in space |
| **Bi-encoder** | Encodes query and document independently into vectors; doc vectors precomputable; basis of dense retrieval |
| **Cross-encoder** | Scores a (query, doc) pair jointly with full attention; accurate but not precomputable; used for reranking |
| **Cosine similarity** | a·b/(‖a‖‖b‖); angle between vectors, ignoring length; range −1..1 |
| **Contrastive loss / InfoNCE** | Training objective pulling positive pairs together and pushing negatives apart; in-batch and hard negatives |
| **Matryoshka embeddings** | Embeddings trained so a prefix of the dimensions is itself a usable, smaller embedding |
| **Pooling** | Collapsing per-token vectors into one vector (mean or CLS) |

## 16. Cheat sheet (10-line summary)

1. Embedding = fixed-length vector from a model; closeness ≈ semantic similarity.
2. Embedders are bidirectional **encoders** + pooling, trained **contrastively** with hard negatives.
3. Similarity is learned and domain-dependent; cosine values aren't comparable across models.
4. Normalize vectors → dot = cosine; L2 gives the same ranking (‖a−b‖² = 2−2cos).
5. Bi-encoder = fast, precomputed, retrieval; cross-encoder = accurate, per-pair, reranking.
6. Max input (often 512 tokens) truncates silently → chunking is mandatory.
7. Use query/passage prefixes when the model requires them.
8. Storage = N × d × 4 B (1M × 768 ≈ 3 GB); int8 4×, binary 32×, Matryoshka truncation.
9. Weak on exact IDs, negation, numbers → hybrid with BM25 + structured filters.
10. Changing the model = re-embed everything; evaluate with recall@k/MRR/nDCG on your own data.

---

*Lesson file complete. See `Revision/Revision_AI_002.md` for active recall questions and
spaced repetition schedule.*
