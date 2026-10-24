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
