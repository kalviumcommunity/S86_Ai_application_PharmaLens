# Retrieval Evaluation: Measure What Matters

Measuring retrieval quality is hard. Guessing is easy. This module makes it easy to measure.

## The Problem

Vector search finds *similar* chunks, but doesn't guarantee *relevant* chunks. Without measurement:
- You don't know if retrieval is working
- You can't track improvements
- You can't see failure patterns
- You optimize blindly

## The Solution

A labelled query set + evaluation system gives you:
- **Recall**: Did we find the relevant chunks?
- **Precision**: Were the retrieved chunks actually relevant?
- **Failure analysis**: Which queries fail and why?
- **Improvement tracking**: Measure the impact of changes

## Quick Example

```python
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

# Define what you expect
queries = [
    {"query": "What adverse events?", 
     "relevant_chunk_ids": {"trial.txt:0"}},
    {"query": "Who is eligible?", 
     "relevant_chunk_ids": {"eligibility.txt:1"}},
]

# Evaluate your retriever
results = evaluate_queries(queries, retrieve_fn, k=5)

# Get metrics
metrics = aggregate_metrics(results)
print(f"Recall:    {metrics['avg_recall']:.1%}")  # 85.0%
print(f"Precision: {metrics['avg_precision']:.1%}")  # 90.0%
```

## What You Get

### ✓ Recall Measurement
```python
# Query: "What adverse events were reported?"
# Expected: {"trial.txt:0", "safety.txt:1"}
# Retrieved: {"trial.txt:0", "eligibility.txt:2"}

# Recall = 1/2 = 50% (found 1 of 2 relevant chunks)
```

### ✓ Precision Measurement
```python
# Precision = 1/3 = 33% (1 of 3 retrieved were relevant)
```

### ✓ Failure Analysis
```python
failures = find_failures(results)
for f in failures:
    print(f"Query: {f['query']}")
    print(f"Expected: {f['relevant_ids']}")
    print(f"Retrieved: {f['retrieved_ids']}")
```

### ✓ Improvement Tracking
```python
# Before improvement
before = evaluate_queries(queries, old_retrieve, k=5)
before_metrics = aggregate_metrics(before)  # recall=80%

# After improvement (e.g., hybrid search)
after = evaluate_queries(queries, new_retrieve, k=5)
after_metrics = aggregate_metrics(after)   # recall=90%

print(f"Improvement: {(after_metrics['avg_recall'] - before_metrics['avg_recall']):.0%}")
```

## Key Metrics

| Metric | Formula | Range | Ideal | RAG Priority |
|--------|---------|-------|-------|--------------|
| Recall | (relevant found) / (relevant total) | 0-1 | 1.0 | **CRITICAL** |
| Precision | (relevant found) / (total retrieved) | 0-1 | 1.0 | High |

**For RAG**: Recall is most important - if the right chunk isn't retrieved, the LLM can't use it.

## Common Use Cases

### Use Case 1: Validate New Embedding Model
```python
# Before: Using model A
before = evaluate_queries(test_queries, retrieve_with_model_a, k=5)

# After: Using model B (supposedly better)
after = evaluate_queries(test_queries, retrieve_with_model_b, k=5)

# Compare
print(f"Model A recall: {aggregate_metrics(before)['avg_recall']:.1%}")
print(f"Model B recall: {aggregate_metrics(after)['avg_recall']:.1%}")
```

### Use Case 2: Find Optimal k Value
```python
for k in [3, 5, 10, 20]:
    results = evaluate_queries(queries, retrieve_fn, k=k)
    metrics = aggregate_metrics(results)
    print(f"k={k}: recall={metrics['avg_recall']:.1%}, "
          f"precision={metrics['avg_precision']:.1%}")
```

Output:
```
k=3:  recall=60.0%, precision=100.0%  ← High precision, low coverage
k=5:  recall=80.0%, precision=90.0%   ← Balanced
k=10: recall=95.0%, precision=75.0%   ← High coverage, more noise
k=20: recall=100.0%, precision=60.0%  ← Maximum coverage, lots of noise
```

### Use Case 3: Debug Retrieval Failures
```python
failures = find_failures(results)
print(f"{len(failures)} queries failed")

for failure in failures:
    print(f"\nQuery: {failure['query']}")
    print(f"Expected: {failure['relevant_ids']}")
    print(f"Retrieved: {failure['retrieved_ids']}")
    print(f"Recall: {failure['recall']:.0%}")
```

### Use Case 4: Measure Re-Ranking Impact
```python
# Retrieval only (k=5)
no_rerank = evaluate_queries(queries, retrieve_fn, k=5)

# Retrieval + re-ranking (k=5 after re-ranking from k=10)
with_rerank = evaluate_queries(queries, reranked_retrieve_fn, k=5)

# Compare precision (re-ranking improves ordering)
metrics_before = aggregate_metrics(no_rerank)
metrics_after = aggregate_metrics(with_rerank)

print(f"Precision without re-ranking: {metrics_before['avg_precision']:.1%}")
print(f"Precision with re-ranking:    {metrics_after['avg_precision']:.1%}")
```

## Getting Started

### 1. Read the Quick Start (5 min)
[docs/RETRIEVAL_EVALUATION_QUICKSTART.md](./RETRIEVAL_EVALUATION_QUICKSTART.md)

### 2. Run the Demo (2 min)
```bash
python -m src.retrieval_evaluation_demo
```

### 3. Run the Examples (5 min)
```bash
python -m src.retrieval_evaluation_examples
```

### 4. Build Your Labelled Queries
Start with 5-10 test queries with known relevant chunks

### 5. Evaluate Your Retrieval
```python
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

results = evaluate_queries(your_queries, your_retrieve_fn, k=5)
metrics = aggregate_metrics(results)
print(f"Recall: {metrics['avg_recall']:.1%}")
```

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/retrieval_evaluation.py` | Core evaluation module | 400 |
| `src/retrieval_evaluation_demo.py` | Full demonstration | 150 |
| `src/retrieval_evaluation_examples.py` | 5 runnable examples | 350 |
| `src/test_retrieval_evaluation.py` | 21 unit tests | 450 |
| `docs/retrieval_evaluation.md` | Complete guide | 600 |
| `docs/RETRIEVAL_EVALUATION_QUICKSTART.md` | 4-step quick start | 150 |
| `RETRIEVAL_EVALUATION_GUIDE.md` | User guide & API ref | 500 |

## Test Status

```
Ran 21 tests in 0.006s
✓ All PASSED
```

Coverage:
- ✓ Recall/precision calculations
- ✓ Chunk ID generation
- ✓ Failure detection
- ✓ Metric aggregation
- ✓ Report generation
- ✓ Edge cases
- ✓ Error handling

## API Overview

### Main Functions

```python
# Single query evaluation
evaluate_retrieval(query, retrieved_chunks, relevant_chunk_ids)

# Multiple queries
evaluate_queries(labelled_queries, retrieve_fn, k=5)

# Aggregate metrics
aggregate_metrics(results)

# Find failures
find_failures(results, recall_threshold=1.0)

# Generate report
detailed_report(results)

# Recall at different k values
recall_at_k_series(results)
```

### Chunk ID Format

Chunks are identified by: `"source:chunk_index"`

```python
# Example: "clinical_trial_overview.txt:2"
# Refers to chunk #2 of the clinical trial overview file
```

### Labelled Query Format

```python
{
    "query": "What adverse events were reported?",
    "relevant_chunk_ids": {"trial.txt:0", "safety.txt:1"}
}
```

## Integration with Existing Code

### Works With
- ✓ `src/retrieval.py` - Vector search results
- ✓ `src/filtered_retrieval.py` - Hybrid search results
- ✓ `src/reranking.py` - Ranked candidate results
- ✓ Any retrieval function returning list of chunks

### Example Integration
```python
from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

# Your RAG pipeline with evaluation
def retrieve_with_evaluation(query, k=5):
    # Step 1: Vector search
    embedding = embed_query(query)
    candidates = retrieve_top_k(query, embedding, chunks, k=10)
    
    # Step 2: Re-rank (optional)
    final = rerank_candidates(query, candidates, client, model, final_k=k)
    
    return final

# Evaluate it
test_queries = load_labelled_queries()
results = evaluate_queries(test_queries, retrieve_with_evaluation, k=5)

metrics = aggregate_metrics(results)
print(f"Recall: {metrics['avg_recall']:.1%}")
```

## Improving Recall (Step by Step)

If your recall is low, try these improvements **in order**:

1. **Increase k** (k=5 → k=10)
   - Simplest change
   - More chunks = more chances to find relevant one

2. **Better chunking** (if recall still < 80%)
   - Smaller chunks (256 tokens instead of 512)
   - Split at section boundaries

3. **Hybrid search** (if recall still < 85%)
   - Combine vector search with keyword matching
   - Add metadata filters

4. **Better embeddings** (if recall still < 90%)
   - Use domain-tuned model
   - Fine-tune on your data

5. **Query rewriting** (if specific queries fail)
   - Expand query with synonyms
   - Add context terms

6. **Re-ranking** (to improve precision)
   - LLM re-scores top-k candidates
   - Improves relevance ordering

## Performance

```
Typical performance for 100 labelled queries, k=5:
- Evaluation time: ~0.5-1 second (depends on retrieve_fn speed)
- Memory: ~5 MB for results
- Output size: ~2 MB JSON

Scales linearly with number of queries.
```

## Status

✓ Production ready  
✓ 21 tests passing  
✓ Comprehensive documentation  
✓ 5 runnable examples  
✓ Full demo  

## Next Steps

1. [Quick Start Guide](./RETRIEVAL_EVALUATION_QUICKSTART.md) (5 min)
2. [Run Demo](../src/retrieval_evaluation_demo.py) (2 min)
3. [Build Labelled Queries](./retrieval_evaluation.md#building-a-labelled-query-set) (10 min)
4. [Evaluate Your System](./retrieval_evaluation.md#code-pattern) (5 min)
5. [Improve Incrementally](./retrieval_evaluation.md#improving-recall) (ongoing)

---

**Questions?** See [retrieval_evaluation.md](./retrieval_evaluation.md) for detailed documentation.
