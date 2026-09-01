# Chunk Re-Ranking for Precision

## Overview

Re-ranking is a second-pass scoring step that improves the relevance of retrieved chunks before they are sent to the LLM. Instead of relying solely on vector similarity scores, re-ranking uses more careful evaluation (typically via an LLM) to bubble the most relevant chunks to the top of the final context.

## Why Re-Ranking?

Vector similarity is fast and good at finding *related* chunks, but the initial ordering often mixes high-quality and marginal matches. Re-ranking solves this by:

1. **Improving precision**: Chunks that most directly answer the query move to the top
2. **Reducing noise**: Marginally related chunks drop down
3. **Better context for LLM**: The model receives the highest-quality information first

The trade-off: re-ranking adds latency and cost because each candidate requires extra scoring.

## Architecture

```
User Query
    ↓
[1. Initial Retrieval]
    - Vector similarity search
    - Returns k=10 candidates
    - Fast, broad coverage
    ↓
[2. Re-Ranking]
    - LLM scores each candidate
    - Re-sorts by relevance
    - Selects top k=3
    - Careful, slow, expensive
    ↓
[3. Final Context]
    - Top 3 most relevant chunks
    - Sent to LLM for answer
```

## Key Functions

### `rerank_score_with_llm(client, model, query, chunk)`

Scores a single chunk's relevance to a query using an LLM.

**How it works:**
- Sends the query and chunk to the LLM
- Asks for a 0-10 relevance score
- Returns the score, clamped to [0.0, 10.0]

**Example:**
```python
from openai import OpenAI
from src.reranking import rerank_score_with_llm

client = OpenAI(api_key="...")
chunk = {
    "text": "Clinical trial showed adverse events...",
    "metadata": {"source": "trial.txt"}
}
query = "What adverse events were reported?"

score = rerank_score_with_llm(client, "gpt-4", query, chunk)
print(f"Relevance score: {score}")  # e.g., 9.2
```

### `rerank_candidates(query, candidates, client, model, final_k)`

Re-ranks a set of candidates and returns the top-k results.

**Process:**
1. Scores each candidate using `rerank_score_with_llm()`
2. Sorts by re-rank score (descending)
3. Returns top `final_k` results with `rerank_score` field added

**Example:**
```python
from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates

# Step 1: Initial retrieval (k=10)
query_embedding = embed_query(client, "text-embedding-3-large", query)
candidates = retrieve_top_k(query, query_embedding, chunk_records, k=10)

# Step 2: Re-rank to get top-3
final_context = rerank_candidates(
    query, 
    candidates, 
    client, 
    "gpt-4",
    final_k=3
)

# Results are sorted by rerank_score
for i, chunk in enumerate(final_context, start=1):
    print(f"{i}. Score: {chunk['rerank_score']:.1f}")
    print(f"   Text: {chunk['text'][:100]}...")
```

### `rerank_and_compare(query, candidates, client, model, final_k)`

Re-ranks and returns a comparison structure showing before/after ordering.

**Returns:** Dictionary with:
- `before`: Initial retrieval order (top `final_k`)
- `after`: Re-ranked order (top `final_k`)
- `query`: The user query
- `candidate_count`: Total candidates re-ranked
- `final_k`: Number of results shown

### `display_comparison(comparison)`

Formats a comparison for human-readable output.

**Example output:**
```
================================================================================
BEFORE RE-RANKING (initial vector retrieval order)
================================================================================

Rank: 1
  Vector Score: 0.8523
  Source: trial.txt
  Text: Clinical trial showed adverse events including headache...

Rank: 2
  Vector Score: 0.7812
  Source: protocol.txt
  Text: The study protocol outlines methodology...

================================================================================
AFTER RE-RANKING (LLM-scored order)
================================================================================

Rank: 1
  Vector Score: 0.8523
  Rerank Score: 9.50
  Source: trial.txt
  Text: Clinical trial showed adverse events including headache...

Rank: 2
  Vector Score: 0.7812
  Rerank Score: 6.80
  Source: protocol.txt
  Text: The study protocol outlines methodology...
```

## Common Patterns

### Pattern 1: Retrieve, Re-rank, Use

```python
from src.config import load_settings
from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates
from openai import OpenAI

settings = load_settings()
client = OpenAI(base_url=settings["openai_base_url"], api_key=settings["openai_api_key"])

# Retrieve larger candidate set
query = "What adverse events were reported?"
query_embedding = embed_query(client, settings["embed_model"], query)
candidates = retrieve_top_k(query, query_embedding, chunks, k=10)

# Re-rank to final size
final_context = rerank_candidates(query, candidates, client, settings["chat_model"], final_k=3)

# Use top-3 chunks as context for LLM
context_text = "\n\n".join([chunk["text"] for chunk in final_context])

# Send to LLM...
```

### Pattern 2: Compare Before/After

```python
from src.reranking import rerank_and_compare, display_comparison

comparison = rerank_and_compare(
    query,
    candidates,
    client,
    settings["chat_model"],
    final_k=3
)

print(display_comparison(comparison))
```

### Pattern 3: Selective Re-Ranking

Re-rank only if precision matters, not every query:

```python
# For questions where accuracy is critical
if query_type == "CRITICAL":
    final_context = rerank_candidates(query, candidates, client, model, final_k=3)
else:
    # Fast path: use initial retrieval only
    final_context = candidates[:3]
```

## Cost & Latency Analysis

### Retrieval Phase
- **Latency**: ~50-200ms (depends on vector DB)
- **Cost**: Minimal (vector DB query)
- **Result**: k=10 candidates (mixed quality)

### Re-Ranking Phase
- **Latency**: ~500ms - 2s (depends on LLM and network)
- **Cost**: k=10 LLM calls × model price (e.g., 10 × GPT-4 calls)
- **Result**: Top-3 high-precision chunks

### Total Impact
```
Without re-ranking: 50ms retrieval
With re-ranking:    50ms retrieval + 1500ms LLM scoring = 1550ms total

Cost increase: 10× LLM calls beyond initial retrieval
Speed decrease: ~10-15× slower end-to-end (but higher quality)
```

## When to Use Re-Ranking

**✓ Use re-ranking when:**
- Precision/quality of context is critical
- The cost of wrong answers is high (medical, legal, financial)
- User queries are complex and need careful matching
- You have time budget (e.g., not real-time APIs)

**✗ Skip re-ranking when:**
- Speed matters more than precision (real-time chat)
- Candidate vectors are already very high quality
- Your vector model is already well-tuned for your domain
- Cost is constrained

## Testing & Validation

Run the test suite:
```bash
python -m pytest src/test_reranking.py -v
```

Run the demo:
```bash
python -m src.reranking_demo
```

The demo outputs:
- Before/after ranking comparison
- Timing breakdown (retrieval vs. re-ranking)
- Cost analysis
- JSON results to `outputs/reranking_demo_results.json`

## Integration with RAG Pipeline

To integrate re-ranking into your full RAG system:

```python
# In your RAG query handler:
from src.reranking import rerank_candidates

def answer_question(query: str, chunks: list[dict]) -> str:
    # 1. Embed query
    embedding = embed_query(client, embed_model, query)
    
    # 2. Retrieve candidates (larger k)
    candidates = retrieve_top_k(query, embedding, chunks, k=10)
    
    # 3. Re-rank for precision (optional, based on query type)
    context_chunks = rerank_candidates(
        query, 
        candidates, 
        client, 
        chat_model,
        final_k=3
    )
    
    # 4. Build context and query LLM
    context = "\n\n".join([c["text"] for c in context_chunks])
    response = client.chat.completions.create(...)
    
    return response.choices[0].message.content
```

## Advanced Topics

### Custom Re-Ranking Strategies

The current implementation uses LLM scoring. Alternative approaches:

1. **Cross-encoder models** (e.g., `ms-marco-MiniLM-L-12-v2`)
   - Faster than LLM, specialized for ranking
   - ~100ms per candidate instead of 500ms+

2. **Hybrid scoring**
   - Combine vector score + keyword overlap + LLM score
   - Balance speed and quality

3. **Domain-specific scoring**
   - Use custom logic for pharmaceutical/medical text
   - Check for specific entity types, citations, evidence quality

4. **Question-aware re-ranking**
   - Different scoring for different question types
   - Medical Q: emphasize trials, safety data
   - Mechanism Q: emphasize molecular details

### Optimizations

**Batch scoring**: Score multiple candidates in parallel
```python
def rerank_candidates_batch(query, candidates, client, model, final_k=3):
    # Score in batches of 5 to reduce latency
    # (e.g., using parallel API calls or multi-turn prompts)
    ...
```

**Caching**: Cache re-rank scores for common queries
```python
cache = {}
query_hash = hash(query)
if query_hash in cache:
    return cache[query_hash][:final_k]
```

**Early stopping**: Stop scoring if a candidate is clearly irrelevant
```python
if rerank_score < 2.0:
    # Very low score, skip further analysis
    continue
```

## See Also

- [Retrieval basics](../docs/indexing_embeddings.md)
- [LLM integration](../docs/prompt_design.md)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)
- [Pinecone Rerankers guide](https://www.pinecone.io/learn/series/rag/rerankers/)
- [LlamaIndex Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
