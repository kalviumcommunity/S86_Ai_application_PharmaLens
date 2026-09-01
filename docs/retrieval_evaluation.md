# Retrieval Evaluation & Recall Testing

## Overview

Retrieval quality should be measured, not guessed. A labelled query set maps each test query to the chunks it should retrieve. By evaluating on this set, you measure **recall** (did we get the relevant chunk?) and **precision** (how many retrieved chunks were actually relevant?).

This guide shows how to:
1. Build labelled query sets with known relevant chunks
2. Evaluate retrieval quality with recall and precision
3. Aggregate metrics across queries
4. Inspect failures to identify improvement opportunities

## Why Measure Retrieval?

Vector search finds *similar* chunks, but not always the *right* chunks. Without measurement:
- You don't know if retrieval is working
- You can't track improvements
- You don't see failure patterns

With measurement:
- You get clear metrics (recall, precision)
- You can A/B test changes
- You find systematic failures to fix

## Building a Labelled Query Set

A labelled query set is a list of (query, relevant_chunk_ids) pairs:

```python
labelled_queries = [
    {
        "query": "How can a learner reset their password?",
        "relevant_chunk_ids": {"account-guide.md:0", "account-guide.md:1"}
    },
    {
        "query": "What evidence is required for project submission?",
        "relevant_chunk_ids": {"submission-rubric.md:2"}
    },
    {
        "query": "Tell me about clinical trials",
        "relevant_chunk_ids": {"trial.txt:0", "eligibility.txt:1", "protocol.txt:2"}
    }
]
```

### Chunk ID Format
Chunk IDs uniquely identify a chunk: `"source:chunk_index"`
- `source`: The source document/file name
- `chunk_index`: The index within that document

Example: `"clinical_trial_overview.txt:2"` refers to chunk #2 of the clinical trial overview.

### Where Labels Come From
- **Manual review**: Read documents and identify which chunks answer each query
- **FAQ answers**: Use existing Q&A as ground truth
- **Support tickets**: Map common questions to answers
- **Domain experts**: Ask subject matter experts which chunks are relevant

**Start small**: 5-10 labelled queries are enough to detect major issues.

## Measuring Recall and Precision

### Recall: Did We Find It?
**Recall** = (# relevant chunks retrieved) / (# relevant chunks that exist)

- Answers: "Of the chunks that should be here, how many did we find?"
- Range: 0.0 to 1.0
- Ideal: 1.0 (found all relevant chunks)
- For RAG: This is the most critical metric. If the relevant chunk isn't retrieved, the LLM can't use it.

### Precision: Were They Right?
**Precision** = (# relevant chunks retrieved) / (# chunks retrieved)

- Answers: "Of the chunks we returned, how many were actually relevant?"
- Range: 0.0 to 1.0
- Ideal: 1.0 (all retrieved chunks were relevant)
- For RAG: High precision means less noise/irrelevant context for the LLM

### Example

Query: "What adverse events were reported?"

Expected (relevant): `{"trial.txt:0", "safety.txt:1"}`
Retrieved (top-3): `{"trial.txt:0", "eligibility.txt:2", "protocol.txt:3"}`

- Hits (both retrieved AND relevant): `{"trial.txt:0"}`
- Recall = 1/2 = 0.5 (found 1 of 2 relevant)
- Precision = 1/3 = 0.33 (1 of 3 retrieved were relevant)

## Code Pattern

### Step 1: Define Evaluation Function

```python
from src.retrieval_evaluation import evaluate_retrieval

def evaluate_query(query, retrieved_chunks, relevant_chunk_ids):
    """Evaluate a single query."""
    result = evaluate_retrieval(query, retrieved_chunks, relevant_chunk_ids)
    return result
```

### Step 2: Evaluate Multiple Queries

```python
from src.retrieval_evaluation import evaluate_queries

labelled_queries = [
    {"query": "...", "relevant_chunk_ids": {...}},
    {"query": "...", "relevant_chunk_ids": {...}},
]

def retrieve_fn(query, k=5):
    """Your retrieval function."""
    embedding = embed_query(query)
    return retrieve_top_k(query, embedding, chunks, k=k)

results = evaluate_queries(labelled_queries, retrieve_fn, k=5)
```

### Step 3: Aggregate Metrics

```python
from src.retrieval_evaluation import aggregate_metrics

metrics = aggregate_metrics(results)
print(f"Avg Recall:    {metrics['avg_recall']:.1%}")
print(f"Avg Precision: {metrics['avg_precision']:.1%}")
```

### Step 4: Inspect Failures

```python
from src.retrieval_evaluation import find_failures, detailed_report

failures = find_failures(results, recall_threshold=1.0)

for failure in failures:
    print(f"Query: {failure['query']}")
    print(f"Expected: {failure['relevant_ids']}")
    print(f"Got:      {failure['retrieved_ids']}")
    print(f"Recall: {failure['recall']:.1%}\n")

# Or generate full report
report = detailed_report(results)
print(report)
```

## API Reference

### `evaluate_retrieval(query, retrieved_chunks, relevant_chunk_ids)`

Evaluate a single query's retrieval results.

**Args:**
- `query` (str): The query string
- `retrieved_chunks` (list): List of retrieved chunk dicts with "metadata" field
- `relevant_chunk_ids` (set): Set of chunk IDs that should have been retrieved

**Returns:** Dict with:
- `recall`: Fraction of relevant chunks retrieved (0-1)
- `precision`: Fraction of retrieved chunks that are relevant (0-1)
- `hits`: List of chunk IDs that were both retrieved and relevant
- `num_relevant`, `num_retrieved`, `num_hits`: Counts

### `evaluate_queries(labelled_queries, retrieve_fn, k=5)`

Evaluate retrieval on multiple labelled queries.

**Args:**
- `labelled_queries`: List of dicts with "query" and "relevant_chunk_ids"
- `retrieve_fn`: Callable(query, k) that returns list of retrieved chunks
- `k`: Number of results to retrieve per query

**Returns:** List of evaluation results

### `aggregate_metrics(results)`

Aggregate recall/precision across all results.

**Returns:** Dict with:
- `avg_recall`, `avg_precision`: Average across all queries
- `min_recall`, `max_recall`: Min/max across all queries
- `recall_by_query`, `precision_by_query`: Per-query breakdown

### `find_failures(results, recall_threshold=1.0, precision_threshold=0.0)`

Find queries where recall or precision fell below thresholds.

**Args:**
- `recall_threshold`: Queries with recall < this are failures (default 1.0 = perfect)
- `precision_threshold`: Queries with precision < this are failures

**Returns:** List of failed results

### `detailed_report(results, include_all=False)`

Generate a formatted report showing metrics and failures.

**Args:**
- `include_all`: If True, show all queries; if False, show only failures

**Returns:** Formatted report string

## Common Patterns

### Pattern 1: Quick Evaluation

```python
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

results = evaluate_queries(labelled_queries, retrieve_fn, k=5)
metrics = aggregate_metrics(results)

print(f"Recall@5: {metrics['avg_recall']:.1%}")
```

### Pattern 2: Find and Fix Failures

```python
from src.retrieval_evaluation import find_failures

failures = find_failures(results)

for failure in failures:
    print(f"Query: {failure['query']}")
    print(f"Expected: {failure['relevant_ids']}")
    print(f"Retrieved: {failure['retrieved_ids']}")
```

### Pattern 3: Measure k=5 vs k=10

```python
for k in [5, 10, 20]:
    results = evaluate_queries(labelled_queries, retrieve_fn, k=k)
    metrics = aggregate_metrics(results)
    print(f"k={k}: recall={metrics['avg_recall']:.1%}")
```

Output:
```
k=5: recall=0.850
k=10: recall=0.950
k=20: recall=0.980
```

This shows how increasing k improves recall.

### Pattern 4: Track Over Time

```python
# Run evaluation regularly
results = evaluate_queries(labelled_queries, retrieve_fn, k=5)
metrics = aggregate_metrics(results)

# Log metrics
log_metrics({
    "timestamp": time.time(),
    "recall": metrics['avg_recall'],
    "precision": metrics['avg_precision'],
})
```

## Improving Recall

If recall is low, try these improvements (one at a time to measure impact):

1. **Increase k**: Try k=10 or k=20 instead of k=5
2. **Better chunking**: Smaller or better-segmented chunks
3. **Hybrid search**: Add keyword matching to vector search
4. **Better embeddings**: Use a domain-tuned model
5. **Re-ranking**: Apply re-ranking to improve ordering
6. **Query rewriting**: Expand query to include synonyms
7. **Metadata filters**: Filter by document type, date, etc.

### Example: Testing k Values

```python
best_k = None
best_recall = 0.0

for k in [3, 5, 10, 15, 20]:
    results = evaluate_queries(labelled_queries, retrieve_fn, k=k)
    metrics = aggregate_metrics(results)
    
    if metrics['avg_recall'] > best_recall:
        best_recall = metrics['avg_recall']
        best_k = k
    
    print(f"k={k:2d}: recall={metrics['avg_recall']:.1%}, "
          f"precision={metrics['avg_precision']:.1%}")

print(f"\nBest: k={best_k} with recall={best_recall:.1%}")
```

## Common Failure Patterns

### Pattern: Low Recall, High Precision
- Most retrieved chunks are relevant, but missing some
- **Causes**: k too small, weak chunking, missing variations
- **Fix**: Increase k, improve chunking, add query rewriting

### Pattern: High Recall, Low Precision
- Found all relevant chunks, but also many irrelevant ones
- **Causes**: Broad retrieval, noisy embeddings
- **Fix**: Better embeddings, hybrid search (keyword filter), re-ranking

### Pattern: Low Both
- Not finding relevant chunks, and those retrieved aren't good
- **Causes**: Very broken retrieval (wrong embeddings, bad chunks)
- **Fix**: Debug chunking, verify embeddings, check labels

## Example Report

```
================================================================================
RETRIEVAL EVALUATION REPORT
================================================================================

Summary Metrics:
  Queries evaluated: 5
  Avg Recall:        85.0%
  Avg Precision:     90.0%
  Min Recall:        50.0%
  Max Recall:        100.0%

================================================================================
RETRIEVAL FAILURES (2 queries)
================================================================================

1. Query: What adverse events were reported?
   Recall: 50.0% | Precision: 100.0%
   Expected (2): ['trial.txt:0', 'safety.txt:1']
   Retrieved (2): ['trial.txt:0', 'eligibility.txt:2']
   ✓ Hits (1): ['trial.txt:0']

2. Query: Who is eligible?
   Recall: 0.0% | Precision: 0.0%
   Expected (1): ['eligibility.txt:1']
   Retrieved (2): ['trial.txt:0', 'protocol.txt:2']
   ✗ No hits (missed all relevant chunks)
```

## Testing & Validation

Run the test suite:
```bash
python -m unittest src.test_retrieval_evaluation -v
```

Run the demo:
```bash
python -m src.retrieval_evaluation_demo
```

## See Also

- [Retrieval Basics](../docs/indexing_embeddings.md)
- [Re-Ranking for Precision](../docs/reranking.md)
- [LangSmith RAG Evaluation](https://docs.smith.langchain.com/evaluation/tutorials/rag)
- [Pinecone Retrieval Evaluation](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)
