# Interview Mistakes Log

Format: mistake → why it's wrong → correct understanding → hook → recurrence count.

### 2026-10-24 — [A001] KV cache growth
- Mistake: said the KV cache "grows exponentially" at long context.
- Why it's wrong: cache size is linear in sequence length (2 × layers × KV heads × head_dim × tokens × bytes). The *compute* is quadratic; the *cache* is linear per sequence. It dominates memory because it is multiplied by concurrent requests and long contexts.
- Correct: compute O(n²), cache O(n) per sequence, total cache memory ≈ O(n × concurrent users).
- Hook: "Quadratic compute, linear cache, multiplied by users."
- Recurs? 1

### 2026-10-24 — [A001] What GQA shares
- Mistake: said GQA shares the KV cache "for a few tokens."
- Why it's wrong: every token still gets its own cached K/V. GQA shares K/V across *query heads* within a group.
- Correct: 32 query heads → 8 K/V heads; each K/V head serves 4 query heads, so the cache is 4x smaller. MQA = 1 K/V head for all query heads. MHA = one K/V per query head.
- Hook: "GQA shares across heads, never across tokens."
- Recurs? 1

### 2026-10-24 — [A001] Weighted-sum wording / positional encoding / block tail
- Mistake: said the attention weight is multiplied with "the current Value vector"; said positional encoding is added to the embedding; muddled order of head-concat, residual, norm, FFN.
- Correct: each *other* token's Value × *its own* weight, then summed. RoPE rotates Q and K inside attention (embedding untouched). Block = attention → concat heads + output projection → residual + LayerNorm → FFN → residual + LayerNorm; repeated ~32x.
- Recurs? 1

### 2026-11-18 — [A002] Similarity metrics (no answer)
- Mistake: could not say when cosine, dot product and L2 agree.
- Correct: if every vector is normalized to length 1, dot = cosine and distance² = 2 − 2·cosine, so all three rank identically. Normalize once at index time, then use dot product.
- Hook: "Length 1 → three rulers, one ranking."
- Recurs? 1

### 2026-11-18 — [A002] Changing the embedding model (no answer)
- Mistake: no answer on why a new model can't just embed new docs, and no storage maths.
- Correct: each model has its own coordinate system; X-vectors and Y-vectors aren't comparable → re-embed ALL chunks into a new versioned index, evaluate recall@k, blue/green cutover. 1M × 1024 × 4 B ≈ 4 GB; shrink with int8 (÷4), binary (÷32), Matryoshka truncation.
- Hook: "New model = new map = re-embed everything."
- Recurs? 1

### 2026-11-18 — [A002] Embedding tradeoff missing
- Mistake: explained how an embedding is made but gave no tradeoff.
- Correct: one fixed-size vector per chunk is what makes search fast and cheap, and also what loses detail (exact IDs, negation, numbers).
- Recurs? 1


### 2026-11-19 — [B001] Debugging: missing the first question
- Mistake: jumped to metrics and tuning chunk size/temperature; never asked "was the correct chunk in the prompt?".
- Correct: first split the problem by stage — correct chunk NOT in the prompt → retrieval/indexing (chunking, hybrid search, bigger k, reranker, query rewriting); IN the prompt but answer wrong → generation (stricter grounding prompt, better model, less noise, then temperature).
- Hook: "Was the right chunk in the prompt?"
- Recurs? 1

### 2026-11-19 — [B001] Why not fine-tune for document facts
- Mistake: said fine-tuning changes behaviour but argued only that embeddings are cheaper to refresh.
- Correct: fine-tuning is poor at injecting facts (can still hallucinate), goes stale, gives no citations, cannot enforce per-user access, and is slow/costly to redo per document; use it for style/format/behaviour.
- Recurs? 1

### 2026-11-19 — [B001] Pipeline stages incomplete
- Mistake: said the query is turned into chunks; left out parse/metadata, filtering, reranking and the grounded prompt.
- Correct: only the documents are chunked; the query is just embedded. Full chain: parse → chunk → embed → store+metadata | embed query → hybrid search → filter → rerank → grounded prompt → LLM + citations.
- Recurs? 1

### 2026-11-19 — [B001] Access control & numbers wording
- Mistake: said the client sends the team id with the query; called vectors "int32".
- Correct: derive the team id from the authenticated token on the server (a client-sent id can be spoofed); embeddings are float32 (4 bytes/number — hence the 3 GB). Query endpoint should be POST (body carries the question).
- Hook: "Identity comes from the token, never from the request."
- Recurs? 1
