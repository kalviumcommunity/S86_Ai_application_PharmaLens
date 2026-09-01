# Re-Ranking Quick Start

Add precision-focused retrieval to your RAG pipeline in 3 steps.

## Step 1: Import the Module

```python
from src.reranking import rerank_candidates
from src.retrieval import retrieve_top_k, embed_query
```

## Step 2: Retrieve Larger Candidate Set

Instead of retrieving exactly the chunks you need, get more:

```python
# Old: retrieve exactly k=3
candidates = retrieve_top_k(query, embedding, chunks, k=3)

# New: retrieve more candidates for re-ranking
candidates = retrieve_top_k(query, embedding, chunks, k=10)
```

## Step 3: Re-Rank and Select Top-k

```python
# Re-rank all candidates and get top-3 by relevance
final_context = rerank_candidates(
    query,
    candidates,
    client,
    "gpt-4",  # Use your chat model
    final_k=3
)

# Use final_context as before
context_text = "\n\n".join([c["text"] for c in final_context])
```

## Complete Example

```python
from openai import OpenAI
from src.config import load_settings
from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates

# Load config
settings = load_settings()
client = OpenAI(
    base_url=settings["openai_base_url"],
    api_key=settings["openai_api_key"]
)

# Your query and chunks
query = "What adverse events were reported?"
chunk_records = [...]  # Your chunks

# Step 1: Embed query
query_embedding = embed_query(client, settings["embed_model"], query)

# Step 2: Retrieve 10 candidates
candidates = retrieve_top_k(query, query_embedding, chunk_records, k=10)

# Step 3: Re-rank to top-3
final_context = rerank_candidates(
    query,
    candidates,
    client,
    settings["chat_model"],
    final_k=3
)

# Step 4: Build prompt
context = "\n\n".join([c["text"] for c in final_context])
prompt = f"""Using this context:

{context}

Answer the question: {query}"""

# Step 5: Get answer
response = client.chat.completions.create(
    model=settings["chat_model"],
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

## Optional: Show Comparison

To see how re-ranking changed the order:

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

Output:
```
================================================================================
BEFORE RE-RANKING (initial vector retrieval order)
================================================================================

Rank: 1
  Vector Score: 0.8523
  Source: trial.txt
  Text: Clinical trial showed adverse events...

...

================================================================================
AFTER RE-RANKING (LLM-scored order)
================================================================================

Rank: 1
  Vector Score: 0.8523
  Rerank Score: 9.50
  Source: trial.txt
  Text: Clinical trial showed adverse events...
```

## Configuration Variations

### Fast Path (No Re-Ranking)
```python
# Just use vector retrieval
final_context = retrieve_top_k(query, embedding, chunks, k=3)
```

### Balanced (Re-Rank Smaller Set)
```python
# Get 5 candidates, re-rank to 3
candidates = retrieve_top_k(query, embedding, chunks, k=5)
final_context = rerank_candidates(query, candidates, client, model, final_k=3)
```

### Precision-Focused (Re-Rank Larger Set)
```python
# Get 20 candidates, re-rank to 5
candidates = retrieve_top_k(query, embedding, chunks, k=20)
final_context = rerank_candidates(query, candidates, client, model, final_k=5)
```

## Conditional Re-Ranking

Only re-rank when it matters:

```python
# Fast path for simple queries
if is_simple_query(query):
    final_context = retrieve_top_k(query, embedding, chunks, k=3)
else:
    # Precision path for complex queries
    candidates = retrieve_top_k(query, embedding, chunks, k=10)
    final_context = rerank_candidates(query, candidates, client, model, final_k=3)
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `candidate_k` | 10 | Initial retrieval size (retrieve this many) |
| `final_k` | 3 | Final output size (keep this many after re-rank) |
| `model` | "gpt-4" | LLM used for scoring (your chat model) |

**Rule of thumb**: candidate_k should be 2-4× final_k for re-ranking to be effective.

## Cost Impact

Retrieving k=10 and re-ranking to top-3:
- **Retrieval cost**: Minimal (vector DB lookup)
- **Re-ranking cost**: 10 LLM calls
- **Total additional cost**: ~$0.0003 per query (GPT-4)
- **Latency**: ~1.5 seconds additional

Worth it when:
- Answer quality improvement > latency/cost trade-off
- Users accept wait time for better answers
- Precision is worth more than speed

## Troubleshooting

**Q: Re-ranking is too slow**
- A: Reduce candidate_k (e.g., 5 instead of 10)
- A: Use a faster model for re-ranking (e.g., GPT-3.5)
- A: Consider cross-encoder models instead

**Q: Re-ranking costs too much**
- A: Use it selectively (only complex queries)
- A: Reduce candidate_k
- A: Use a cheaper model for re-ranking

**Q: Re-ranking isn't improving answers**
- A: Try increasing candidate_k (e.g., 15-20)
- A: Check if your queries are already well-matched
- A: Verify the re-rank model understands your domain

**Q: I'm getting errors during scoring**
- A: Check that chunks have "text" field
- A: Verify API keys are valid
- A: Check network connectivity to LLM API
- A: See logs for specific error messages

## Next Steps

1. **Measure baseline**: Try with/without re-ranking on real queries
2. **Tune parameters**: Adjust candidate_k and final_k for your use case
3. **Profile performance**: Time retrieval vs re-ranking vs LLM
4. **A/B test**: Compare answer quality with users
5. **Optimize**: Consider cross-encoders or cached scores

## See Also

- [Full Documentation](./reranking.md)
- [Implementation Details](./RERANKING_IMPLEMENTATION.md)
- [Examples](../src/reranking_examples.py)
- [Test Suite](../src/test_reranking.py)
