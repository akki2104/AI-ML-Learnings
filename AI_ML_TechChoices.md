# AI/ML Tech Choices — Decision Playbook

> Grows after every lesson. Each entry answers: "When do I choose X? When do I NOT choose X?"
> Use this in interviews when explaining tech decisions (Dimension 5: Tradeoffs).

---

## RAG vs Fine-tuning vs Prompting

**Use RAG when:**
- The answer exists in specific documents (knowledge is retrievable, not implicit)
- Knowledge changes frequently (you can't retrain the model every time docs update)
- You need source attribution / citations (users must see which document was used)
- Domain data is too large to fit in context (> context window limit)

**Use Fine-tuning when:**
- You need the model to adopt a specific *behavior* or *style*, not just know more facts
- You have 1K+ labeled examples of the desired behavior
- Inference latency budget allows a larger custom model
- The base model consistently fails on your task even with good prompts

**Use Prompting when:**
- The base model can already do the task — it just needs the right instructions
- You need to move fast (no training data collection required)
- The input fits in the context window
- You want to A/B test quickly without retraining

**Do NOT use RAG when:**
- The reasoning required spans many documents implicitly (multi-hop reasoning) — retrieval
  can't surface the right context; use fine-tuning or a larger context model instead
- Your corpus is small (<20 documents) — just put them all in the prompt
- Retrieval latency makes the SLA impossible and caching won't help

---

## Vector Database

**Use pgvector when:**
- You're already on PostgreSQL — avoid another service to operate
- Corpus is small-to-medium (< 5M vectors)
- You need ACID transactions alongside vector search (e.g., update a doc = delete + re-insert atomically)
- You want SQL JOINs on metadata + vector similarity in one query

**Do NOT use pgvector when:**
- Corpus > 10M vectors — pgvector's HNSW index starts to degrade; use Qdrant or Pinecone
- You need real-time upserts at high write QPS — pgvector locks during index updates
- You need built-in hybrid search (dense + sparse) — Weaviate or Qdrant do this better

**Use Pinecone when:**
- Managed scaling is the priority (no ops team)
- Millions of vectors with low-latency reads
- Metadata filtering at scale

**Use Weaviate when:**
- You need hybrid search (dense + BM25 sparse) built into one query
- Multi-tenant namespacing (one index per customer)
- GraphQL API is preferred

**Use Chroma when:**
- Local development and prototyping only — never in production

---

## LLM Serving Framework

**Use llama.cpp when:**
- Running on CPU (no CUDA) or consumer GPU (RTX 3080/4090)
- GGUF-quantized model (Q4, Q5, Q8)
- On-premises / air-gapped / no cloud GPU budget
- Latency > throughput (single-user or low-concurrency)

**Do NOT use llama.cpp when:**
- You need to serve 10+ concurrent users — it does not handle batching
- You need production throughput > 100 tok/s on a single model

**Use Ollama when:**
- Same use case as llama.cpp, but you want a REST API + model management without custom code
- Local dev environment serving multiple models

**Use vLLM when:**
- Production serving on A100/H100 GPUs
- High concurrency (10+ simultaneous requests)
- OpenAI-compatible `/v1/completions` API needed
- Throughput is the constraint (continuous batching squeezes max tok/s)

**Do NOT use vLLM when:**
- You're on CPU or a consumer GPU — llama.cpp is better
- You need to load LoRA adapters dynamically per request — use TGI instead

**Use TGI (Text Generation Inference) when:**
- HuggingFace ecosystem (model from HF Hub)
- SSE streaming is a first-class requirement
- You need multi-LoRA serving (different adapters per request)

**Use TensorRT-LLM when:**
- NVIDIA A100/H100 and latency is the constraint (not throughput)
- Willing to pay compilation cost per model version
- Need the absolute lowest TTFT (time-to-first-token)

---

## Embedding Model

**Use sentence-transformers/all-MiniLM-L6-v2 when:**
- English-only domain
- Speed and size matter (fast, small, 384 dims)
- Budget is zero (open-source, self-hostable)

**Use BAAI/bge-m3 when:**
- Multilingual content
- Need dense + sparse + multi-vector in one model (no separate BM25 system)
- Best open-source quality for most tasks

**Use OpenAI text-embedding-3-large when:**
- Maximum English quality
- OK with API cost and latency
- Don't need self-hosting

**Do NOT use text-embedding-3-large when:**
- Multilingual content (use bge-m3)
- Budget/latency constraints (use MiniLM or bge-small)

---

## Reranker

**Use cross-encoder (BERT-based) when:**
- Best quality is needed; 50–200ms reranking budget available
- You already retrieved a candidate set (top-k from bi-encoder) — reranker scores pairs
- Clinical / legal / high-stakes domains where retrieval precision matters

**Do NOT use cross-encoder when:**
- First-stage retrieval (too slow — it scores every doc-query pair)
- SLA < 100ms and no caching possible

**Use ColBERT when:**
- You want better quality than bi-encoder but faster than cross-encoder
- Memory budget allows storing per-token embeddings (30x more than bi-encoder)

**Use LLM-based reranking (GPT-4, Claude) when:**
- Highest possible quality; cost is not a constraint
- You can afford 500ms+ for the reranking step

---

## Chunking Strategy

**Use fixed-size chunking (512 tokens, 20% overlap) when:**
- Prototyping — simplest to implement
- Documents are homogeneous in structure (all similar length paragraphs)

**Use recursive text splitting when:**
- Documents have mixed structure (headers, lists, paragraphs)
- Default for most RAG pipelines

**Use semantic chunking when:**
- You want chunks that respect semantic boundaries (sentence similarity threshold)
- Willing to pay extra embedding cost at index time

**Use parent-child chunking when:**
- Retrieval needs small, precise chunks but generation needs more context
- Store small child chunks for retrieval; expand to parent chunk for LLM context

**Use proposition chunking when:**
- Corpus is dense (academic papers, legal docs) with many facts per sentence
- Each chunk should contain exactly one retrievable fact

---

*(This file grows. Add a new entry after every lesson that covers a tech choice.)*
