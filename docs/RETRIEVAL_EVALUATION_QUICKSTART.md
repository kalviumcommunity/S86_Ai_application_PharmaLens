# Retrieval Evaluation Quick Start

Add measurement to your retrieval pipeline in 4 steps.

## Step 1: Build Labelled Queries

Create a list of queries with known relevant chunks:

```python
labelled_queries = [
    {
        "query": "What adverse events were reported?",
        "relevant_chunk_ids": {"trial.txt:0", "safety.txt:1"}
    },
    {
        "query": "Who is eligible?",
        "relevant_chunk_ids": {"eligibility.txt:1"}
    },
]
```

**Chunk ID format**: `"source:chunk_index"`
- `source`: Document name (e.g., "trial.txt")
- `chunk_index`: Index within document (e.g., 0, 1, 2...)

## Step 2: Create Retrieval Function

Wrap your retrieval in a function that takes (query, k):

```python
from src.retrieval import retrieve_top_k, embed_query
from openai import OpenAI

client = OpenAI(api_key="...")

def retrieve_fn(query, k=5):
    embedding = embed_query(client, "text-embedding-3-large", query)
    return retrieve_top_k(query, embedding, chunks, k=k)
```

## Step 3: Evaluate

```python
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

results = evaluate_queries(labelled_queries, retrieve_fn, k=5)
metrics = aggregate_metrics(results)

print(f"Recall:    {metrics['avg_recall']:.1%}")
print(f"Precision: {metrics['avg_precision']:.1%}")
```

## Step 4: Inspect Failures

```python
from src.retrieval_evaluation import find_failures

failures = find_failures(results)

for failure in failures:
    print(f"\nQuery: {failure['query']}")
    print(f"Expected: {failure['relevant_ids']}")
    print(f"Retrieved: {failure['retrieved_ids']}")
    print(f"Recall: {failure['recall']:.1%}")
```

## Complete Example

```python
from src.retrieval import retrieve_top_k, embed_query
from src.retrieval_evaluation import (
    evaluate_queries,
    aggregate_metrics,
    find_failures,
    detailed_report,
)
from openai import OpenAI

# Setup
client = OpenAI(api_key="...")
chunks = [...]  # Your chunks

# Labelled queries
queries = [
    {"query": "What adverse events?", "relevant_chunk_ids": {"trial.txt:0"}},
    {"query": "Who is eligible?", "relevant_chunk_ids": {"eligibility.txt:1"}},
]

# Retriever
def retrieve_fn(query, k=5):
    embedding = embed_query(client, model, query)
    return retrieve_top_k(query, embedding, chunks, k=k)

# Evaluate
results = evaluate_queries(queries, retrieve_fn, k=5)
metrics = aggregate_metrics(results)

# Report
print(f"Recall:  {metrics['avg_recall']:.1%}")
print(f"Precision: {metrics['avg_precision']:.1%}")

failures = find_failures(results)
if failures:
    print(f"\n{len(failures)} queries failed:")
    for f in failures:
        print(f"  - {f['query']}: recall={f['recall']:.1%}")
```

## Measuring k Values

```python
for k in [3, 5, 10, 20]:
    results = evaluate_queries(queries, retrieve_fn, k=k)
    metrics = aggregate_metrics(results)
    print(f"k={k:2d}: recall={metrics['avg_recall']:.1%}, "
          f"precision={metrics['avg_precision']:.1%}")
```

Output:
```
k=3: recall=60.0%, precision=100.0%
k=5: recall=80.0%, precision=90.0%
k=10: recall=95.0%, precision=75.0%
k=20: recall=100.0%, precision=60.0%
```

## API Quick Reference

| Function | Purpose |
|----------|---------|
| `evaluate_retrieval()` | Evaluate single query |
| `evaluate_queries()` | Evaluate multiple queries |
| `aggregate_metrics()` | Average recall/precision |
| `find_failures()` | Find queries below threshold |
| `detailed_report()` | Generate formatted report |

## Improving Recall

Try these in order:

1. **Increase k**
   ```python
   results = evaluate_queries(queries, retrieve_fn, k=10)  # Try larger k
   ```

2. **Better chunking**
   - Smaller chunks (e.g., 256 instead of 512 tokens)
   - Better segmentation (split at section boundaries)

3. **Hybrid search**
   ```python
   # Add keyword matching to vector search
   results = hybrid_search(query, embedding, chunks, keyword_terms=["adverse", "events"])
   ```

4. **Re-ranking** (if precision matters)
   ```python
   from src.reranking import rerank_candidates
   candidates = retrieve_top_k(query, embedding, chunks, k=10)
   final = rerank_candidates(query, candidates, client, model, final_k=5)
   ```

## Key Metrics

- **Recall@k**: Did the right chunk appear in top-k?
  - Target: 95%+ for RAG
  - If low: Increase k, improve chunking, use hybrid search

- **Precision@k**: Were retrieved chunks actually relevant?
  - Target: 70%+ for RAG
  - If low: Use re-ranking, better embeddings, metadata filters

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Low recall | Can't find relevant chunks | Increase k, better chunking |
| Low precision | Too much noise | Use re-ranking, metadata filters |
| All failures | Broken retrieval | Check embeddings, verify labels |

## See Also

- [Full Documentation](./retrieval_evaluation.md)
- [Re-Ranking for Precision](./reranking.md)
- [Examples](../src/retrieval_evaluation_demo.py)
- [Tests](../src/test_retrieval_evaluation.py)
