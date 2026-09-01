# ✓ Retrieval Evaluation & Recall Testing - Complete

## By the End, You Can Now:

### ✓ Build a Labelled Query Set with Known Relevant Chunks
```python
labelled_queries = [
    {
        "query": "What adverse events were reported?",
        "relevant_chunk_ids": {"trial.txt:0", "safety.txt:1"}
    },
    {
        "query": "Who is eligible?",
        "relevant_chunk_ids": {"eligibility.txt:1"}
    }
]
```

### ✓ Measure Recall at Top-k
```python
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

results = evaluate_queries(labelled_queries, retrieve_fn, k=5)
metrics = aggregate_metrics(results)

print(f"Recall@5: {metrics['avg_recall']:.1%}")  # e.g., 85.0%
```

### ✓ Report Precision or Other Quality Signals
```python
print(f"Precision@5: {metrics['avg_precision']:.1%}")  # e.g., 90.0%

# Per-query details
for result in results:
    print(f"{result['query']}: recall={result['recall']:.0%}, "
          f"precision={result['precision']:.0%}")
```

### ✓ Inspect Failures and Identify Likely Causes
```python
from src.retrieval_evaluation import find_failures

failures = find_failures(results)

for failure in failures:
    print(f"Query: {failure['query']}")
    print(f"Expected: {failure['relevant_ids']}")
    print(f"Retrieved: {failure['retrieved_ids']}")
    print(f"Cause: Low recall (can't find relevant chunks)")
```

---

## What's Included

### Core Implementation (3 Files)

| File | Purpose |
|------|---------|
| [src/retrieval_evaluation.py](../src/retrieval_evaluation.py) | Core evaluation functions |
| [src/retrieval_evaluation_demo.py](../src/retrieval_evaluation_demo.py) | Full demonstration |
| [src/test_retrieval_evaluation.py](../src/test_retrieval_evaluation.py) | 21 unit tests (all passing) |

### Documentation (2 Files)

| File | Purpose |
|------|---------|
| [docs/retrieval_evaluation.md](../docs/retrieval_evaluation.md) | Complete guide & API reference |
| [docs/RETRIEVAL_EVALUATION_QUICKSTART.md](../docs/RETRIEVAL_EVALUATION_QUICKSTART.md) | 4-step quick start |

---

## Key Concepts

### Recall: Did We Find It?
**Recall** = (# relevant chunks retrieved) / (# relevant chunks that exist)

- Answers: "Of the chunks that should be here, how many did we find?"
- Range: 0.0 to 1.0
- Ideal: 1.0 (found all relevant chunks)
- For RAG: **Most critical metric** - if the right chunk isn't retrieved, the LLM can't use it

### Precision: Were They Right?
**Precision** = (# relevant chunks retrieved) / (# chunks retrieved)

- Answers: "Of the chunks we returned, how many were actually relevant?"
- Range: 0.0 to 1.0
- Ideal: 1.0 (all retrieved chunks were relevant)
- For RAG: High precision = less noise/irrelevant context

### Example

Query: "What adverse events occurred?"
Expected (relevant): `{"trial.txt:0", "safety.txt:1"}`
Retrieved (top-3): `{"trial.txt:0", "eligibility.txt:2", "protocol.txt:3"}`

- Hits (both retrieved AND relevant): `{"trial.txt:0"}`
- **Recall** = 1/2 = **50%** (found 1 of 2 relevant chunks)
- **Precision** = 1/3 = **33%** (1 of 3 retrieved were relevant)

---

## Quick Start (4 Steps)

### Step 1: Build Labelled Queries
```python
queries = [
    {"query": "What adverse events?", 
     "relevant_chunk_ids": {"trial.txt:0"}},
    {"query": "Who is eligible?", 
     "relevant_chunk_ids": {"eligibility.txt:1"}},
]
```

### Step 2: Create Retrieval Function
```python
def retrieve_fn(query, k=5):
    embedding = embed_query(client, model, query)
    return retrieve_top_k(query, embedding, chunks, k=k)
```

### Step 3: Evaluate
```python
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

results = evaluate_queries(queries, retrieve_fn, k=5)
metrics = aggregate_metrics(results)

print(f"Recall:    {metrics['avg_recall']:.1%}")
print(f"Precision: {metrics['avg_precision']:.1%}")
```

### Step 4: Inspect Failures
```python
from src.retrieval_evaluation import find_failures

for failure in find_failures(results):
    print(f"Query: {failure['query']}")
    print(f"Expected: {failure['relevant_ids']}")
    print(f"Retrieved: {failure['retrieved_ids']}")
```

---

## API Quick Reference

### Main Functions

| Function | Purpose |
|----------|---------|
| `evaluate_retrieval(query, retrieved_chunks, relevant_ids)` | Evaluate single query → recall/precision |
| `evaluate_queries(labelled_queries, retrieve_fn, k)` | Evaluate multiple queries → list of results |
| `aggregate_metrics(results)` | Average recall/precision across results |
| `find_failures(results, recall_threshold)` | Find queries below threshold |
| `detailed_report(results)` | Generate formatted report for inspection |

### Example Usage

```python
# Evaluate single query
result = evaluate_retrieval(
    "What adverse events?",
    retrieved_chunks,
    {"trial.txt:0", "safety.txt:1"}
)
print(f"Recall: {result['recall']:.1%}")  # e.g., 50.0%

# Evaluate multiple
results = evaluate_queries(queries, retrieve_fn, k=5)

# Get metrics
metrics = aggregate_metrics(results)
print(f"Avg Recall: {metrics['avg_recall']:.1%}")  # e.g., 85.0%

# Find failures
failures = find_failures(results, recall_threshold=1.0)
print(f"Failed: {len(failures)}/{len(results)}")

# Full report
report = detailed_report(results)
print(report)
```

---

## Common Patterns

### Pattern 1: Measure k Values
```python
for k in [3, 5, 10, 20]:
    results = evaluate_queries(queries, retrieve_fn, k=k)
    metrics = aggregate_metrics(results)
    print(f"k={k:2d}: recall={metrics['avg_recall']:.1%}, "
          f"precision={metrics['avg_precision']:.1%}")
```

Output:
```
k= 3: recall=60.0%, precision=100.0%
k= 5: recall=80.0%, precision=90.0%
k=10: recall=95.0%, precision=75.0%
k=20: recall=100.0%, precision=60.0%
```

### Pattern 2: A/B Test Retrieval Changes
```python
# Test with current retriever
baseline = evaluate_queries(queries, current_retrieve, k=5)
baseline_metrics = aggregate_metrics(baseline)

# Test with new retriever
improved = evaluate_queries(queries, new_retrieve, k=5)
improved_metrics = aggregate_metrics(improved)

# Compare
print(f"Baseline:  recall={baseline_metrics['avg_recall']:.1%}")
print(f"Improved:  recall={improved_metrics['avg_recall']:.1%}")
print(f"Gain:      {(improved_metrics['avg_recall'] - baseline_metrics['avg_recall']):.1%}")
```

### Pattern 3: Find Systematic Failures
```python
failures = find_failures(results)

# Group by recall
low_recall = [f for f in failures if f['recall'] == 0.0]
print(f"Zero recall: {len(low_recall)} queries")

partial_recall = [f for f in failures if 0 < f['recall'] < 1.0]
print(f"Partial recall: {len(partial_recall)} queries")

# Inspect patterns
for f in low_recall[:3]:
    print(f"\n{f['query']}")
    print(f"Expected: {f['relevant_ids']}")
    print(f"Got:      {f['retrieved_ids']}")
```

---

## Improving Recall

**Try these in order** (measure impact of each change):

1. **Increase k** (simplest)
   ```python
   results = evaluate_queries(queries, retrieve_fn, k=10)  # Try k=10 vs k=5
   ```
   → Larger k gets more chunks, improves recall

2. **Better chunking** (structural)
   - Smaller chunks (256 instead of 512 tokens)
   - Split at section boundaries, not random positions
   - Re-index corpus with better chunking

3. **Hybrid search** (algorithm)
   ```python
   from src.filtered_retrieval import hybrid_search
   
   def hybrid_retrieve(query, k=5):
       embedding = embed_query(query)
       return hybrid_search(query, embedding, chunks, 
                           keyword_terms=["adverse", "events"], k=k)
   
   results = evaluate_queries(queries, hybrid_retrieve, k=5)
   ```
   → Adds keyword matching to boost relevant chunks

4. **Better embeddings** (model)
   - Try domain-specific embedding model
   - Fine-tune embeddings on your data
   - Use a larger/better model

5. **Re-ranking** (scoring)
   ```python
   from src.reranking import rerank_candidates
   
   candidates = retrieve_top_k(query, embedding, chunks, k=10)
   final = rerank_candidates(query, candidates, client, model, final_k=5)
   ```
   → Score candidates more carefully before final selection

6. **Query rewriting** (preprocessing)
   ```python
   # Expand query with synonyms
   expanded_query = query + " adverse events complications side-effects"
   embedding = embed_query(expanded_query)
   ```
   → Help retriever find variations

---

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
   Retrieved (2): ['trial.txt:0', 'protocol.txt:2']
   ✓ Hits (1): ['trial.txt:0']

2. Query: Who is eligible?
   Recall: 0.0% | Precision: 0.0%
   Expected (1): ['eligibility.txt:1']
   Retrieved (3): ['trial.txt:0', 'protocol.txt:2', 'other.txt:3']
   ✗ No hits (missed all relevant chunks)
```

**Insights:**
- Query 1: Found 1/2 relevant (50% recall). Missing "safety.txt:1"
  - **Cause**: Weak embeddings or "adverse events" semantic distance
  - **Fix**: Try larger k, hybrid search, or better embeddings

- Query 2: Found 0/1 relevant (0% recall)
  - **Cause**: Query about eligibility retrieved wrong topic chunks
  - **Fix**: Check if eligibility chunks are chunked well, try hybrid search with "eligible" keyword

---

## Test Results

```
Ran 21 tests in 0.006s
✓ All tests PASSED
```

Tests cover:
- ✓ Recall and precision calculations
- ✓ Chunk ID generation
- ✓ Failure detection
- ✓ Metric aggregation
- ✓ Report generation
- ✓ Edge cases (duplicates, special chars)
- ✓ Error handling

---

## Files Overview

### Source Code
```
src/retrieval_evaluation.py         (400 lines)
  - build_chunk_id()
  - evaluate_retrieval()
  - evaluate_queries()
  - aggregate_metrics()
  - find_failures()
  - detailed_report()
  - recall_at_k_series()

src/retrieval_evaluation_demo.py    (150 lines)
  - build_labelled_queries()
  - demo_with_demo_data()
  - main()

src/test_retrieval_evaluation.py    (450 lines)
  - TestChunkID (3 tests)
  - TestEvaluateRetrieval (6 tests)
  - TestEvaluateQueries (2 tests)
  - TestAggregateMetrics (2 tests)
  - TestFindFailures (3 tests)
  - TestReports (3 tests)
  - TestEdgeCases (2 tests)
```

### Documentation
```
docs/retrieval_evaluation.md                     (Complete guide)
docs/RETRIEVAL_EVALUATION_QUICKSTART.md          (4-step quick start)
RETRIEVAL_EVALUATION_GUIDE.md                    (User guide - this file)
```

---

## Integration with Existing Code

### Works With
- ✓ `src/retrieval.py` - `retrieve_top_k()` output
- ✓ `src/filtered_retrieval.py` - `hybrid_search()` output
- ✓ `src/reranking.py` - Can evaluate before/after re-ranking
- ✓ Any retrieval function that returns list of chunks

### Example Integration
```python
# In your RAG pipeline
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

# Periodically evaluate
def evaluate_retrieval_quality(labelled_queries):
    results = evaluate_queries(labelled_queries, retrieve_fn, k=5)
    metrics = aggregate_metrics(results)
    
    # Log metrics
    logger.info(f"Recall@5: {metrics['avg_recall']:.1%}")
    
    # Alert if recall drops
    if metrics['avg_recall'] < 0.80:
        alert("Retrieval recall below threshold!")

# Run evaluation
evaluate_retrieval_quality(test_queries)
```

---

## Next Steps

1. **Read the Quick Start**
   → [docs/RETRIEVAL_EVALUATION_QUICKSTART.md](../docs/RETRIEVAL_EVALUATION_QUICKSTART.md)

2. **Run the Demo**
   ```bash
   python -m src.retrieval_evaluation_demo
   ```

3. **Build Test Queries**
   - Start with 5-10 labelled queries
   - Map to your actual chunks

4. **Evaluate Current Retrieval**
   - Measure baseline recall/precision
   - Identify failure patterns

5. **Improve Incrementally**
   - Try one change at a time
   - Measure impact with evaluation
   - Keep the changes that help

---

## Summary

✓ Complete evaluation system for measuring retrieval quality
✓ 21 unit tests (all passing)
✓ Demo showing system in action
✓ Comprehensive documentation
✓ 4-step quick start
✓ API reference
✓ Improvement strategies

**Status: PRODUCTION READY**

Start with [docs/RETRIEVAL_EVALUATION_QUICKSTART.md](../docs/RETRIEVAL_EVALUATION_QUICKSTART.md) - 5 minute read.
