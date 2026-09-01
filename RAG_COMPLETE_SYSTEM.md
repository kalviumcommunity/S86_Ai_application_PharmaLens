# Complete RAG System: Re-Ranking + Evaluation

This guide shows how Phase 1 (Re-Ranking) and Phase 2 (Evaluation) work together to build a complete, measurable RAG system.

---

## The Big Picture

```
User Query
    ↓
Vector Search (retrieve k=10 candidates)
    ↓
[PHASE 1] Re-Rank (LLM scores, keep k=3)
    ↓
[PHASE 2] Evaluate (measure recall/precision)
    ↓
LLM Generation (with top-3 relevant chunks)
    ↓
Answer to User
```

---

## Architecture

### Without Measurement (Old Way)
```
Retrieval → LLM → Answer
(How good? Unknown!)
```

### With Measurement (New Way - Complete System)
```
Retrieval → [Evaluate] → Metrics
               ↓
    Is recall 85%+? YES → [Re-Rank] → [Evaluate] → Better metrics
                     NO → Try improvements → [Re-Rank] → [Evaluate] → ...
               ↓
            LLM → Answer
```

---

## Complete Workflow

### Step 1: Build Your Baseline
```python
from src.retrieval import retrieve_top_k, embed_query
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

# Your test queries
test_queries = [
    {"query": "What adverse events?", "relevant_chunk_ids": {"trial.txt:0"}},
    {"query": "Who is eligible?", "relevant_chunk_ids": {"eligibility.txt:1"}},
]

def baseline_retrieve(query, k=5):
    embedding = embed_query(client, model, query)
    return retrieve_top_k(query, embedding, chunks, k=k)

# Measure baseline
results = evaluate_queries(test_queries, baseline_retrieve, k=5)
baseline_metrics = aggregate_metrics(results)

print(f"Baseline Recall:    {baseline_metrics['avg_recall']:.1%}")      # e.g., 75%
print(f"Baseline Precision: {baseline_metrics['avg_precision']:.1%}")   # e.g., 80%
```

### Step 2: Improve with Re-Ranking
```python
from src.reranking import rerank_candidates

def improved_retrieve(query, k=5):
    # Step 1: Get more candidates with vector search
    embedding = embed_query(client, model, query)
    candidates = retrieve_top_k(query, embedding, chunks, k=10)
    
    # Step 2: Re-rank and keep top k
    final = rerank_candidates(query, candidates, client, model, final_k=k)
    
    return final

# Measure improved retrieval
results = evaluate_queries(test_queries, improved_retrieve, k=5)
improved_metrics = aggregate_metrics(results)

print(f"Improved Recall:    {improved_metrics['avg_recall']:.1%}")      # e.g., 85%
print(f"Improved Precision: {improved_metrics['avg_precision']:.1%}")   # e.g., 90%
```

### Step 3: Compare and Decide
```python
recall_gain = improved_metrics['avg_recall'] - baseline_metrics['avg_recall']
precision_gain = improved_metrics['avg_precision'] - baseline_metrics['avg_precision']

print(f"Recall Improvement:    {recall_gain:+.1%}")        # +10%
print(f"Precision Improvement: {precision_gain:+.1%}")    # +10%

if improved_metrics['avg_recall'] > 0.85:
    print("✓ Deploy improved retriever (re-ranking enabled)")
else:
    print("✗ Try other improvements (larger k, better embeddings, etc.)")
```

---

## Integration Example: Full RAG Pipeline

```python
#!/usr/bin/env python3
"""
Complete RAG system with re-ranking and evaluation.

This shows how to:
1. Measure baseline retrieval
2. Improve with re-ranking
3. Verify improvements
4. Deploy with confidence
"""

from openai import OpenAI
from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates
from src.retrieval_evaluation import (
    evaluate_queries,
    aggregate_metrics,
    find_failures,
    detailed_report,
)

client = OpenAI(api_key="...")
model = "gpt-4"
chunks = load_your_chunks()

# Test queries with ground truth
TEST_QUERIES = [
    {"query": "What adverse events?", "relevant_chunk_ids": {"trial.txt:0"}},
    {"query": "Who is eligible?", "relevant_chunk_ids": {"eligibility.txt:1"}},
    {"query": "Tell me about the trial", "relevant_chunk_ids": {"trial.txt:0", "eligibility.txt:1"}},
]


def evaluate_retrieval_system(name, retrieve_fn, k=5):
    """Evaluate a retrieval system."""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {name}")
    print(f"{'='*80}")
    
    results = evaluate_queries(TEST_QUERIES, retrieve_fn, k=k)
    metrics = aggregate_metrics(results)
    failures = find_failures(results)
    
    print(f"Queries evaluated: {metrics['num_queries']}")
    print(f"Recall:    {metrics['avg_recall']:.1%}")
    print(f"Precision: {metrics['avg_precision']:.1%}")
    print(f"Failures:  {len(failures)}")
    
    if failures:
        print("\nFailed queries:")
        for f in failures:
            print(f"  - {f['query']}: recall={f['recall']:.0%}")
    
    return metrics, failures, results


def baseline_retrieve(query, k=5):
    """Vector search only."""
    embedding = embed_query(client, model, query)
    return retrieve_top_k(query, embedding, chunks, k=k)


def improved_retrieve(query, k=5):
    """Vector search + re-ranking."""
    # Get more candidates
    embedding = embed_query(client, model, query)
    candidates = retrieve_top_k(query, embedding, chunks, k=10)
    
    # Re-rank to final k
    final = rerank_candidates(query, candidates, client, model, final_k=k)
    
    return final


# MAIN WORKFLOW
if __name__ == "__main__":
    print("RAG SYSTEM EVALUATION")
    print("=" * 80)
    
    # Phase 1: Baseline
    baseline_metrics, baseline_failures, baseline_results = evaluate_retrieval_system(
        "BASELINE (Vector Search Only)", baseline_retrieve, k=5
    )
    
    # Phase 2: With Re-Ranking
    improved_metrics, improved_failures, improved_results = evaluate_retrieval_system(
        "IMPROVED (With Re-Ranking)", improved_retrieve, k=5
    )
    
    # Phase 3: Compare
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    recall_gain = improved_metrics['avg_recall'] - baseline_metrics['avg_recall']
    precision_gain = improved_metrics['avg_precision'] - baseline_metrics['avg_precision']
    
    print(f"Recall Improvement:    {recall_gain:+.1%}")
    print(f"Precision Improvement: {precision_gain:+.1%}")
    
    if recall_gain >= 0.05:  # At least 5% improvement
        print("\n✓ Re-ranking provides meaningful improvement")
        print("✓ Recommended: Deploy improved retriever")
    else:
        print("\n✗ Re-ranking provides minimal improvement")
        print("✗ Try other strategies (larger k, better embeddings)")
    
    # Phase 4: Deploy Decision
    print(f"\n{'='*80}")
    print("DEPLOYMENT DECISION")
    print(f"{'='*80}")
    
    if improved_metrics['avg_recall'] >= 0.85 and improved_metrics['avg_precision'] >= 0.70:
        print("✓ Deploy improved retriever to production")
    else:
        print("✗ Not ready for production yet")
        print(f"  - Target recall: 85%, Current: {improved_metrics['avg_recall']:.1%}")
        print(f"  - Target precision: 70%, Current: {improved_metrics['avg_precision']:.1%}")
```

---

## Common Scenarios

### Scenario 1: Baseline Too Low (< 70% Recall)

**Problem**: Even baseline retrieval is not working well

**Diagnosis**:
```python
failures = find_failures(baseline_results)
for f in failures:
    print(f"Query: {f['query']}")
    print(f"Expected: {f['relevant_ids']}")
    print(f"Retrieved: {f['retrieved_ids']}")
```

**Solutions** (try in order):
1. Increase k (k=5 → k=10)
2. Check embeddings (verify they're working)
3. Check chunking (are chunks well-formed?)
4. Try hybrid search (add keywords)

### Scenario 2: Baseline Good, Re-Ranking Helps (Scenario: Good)

**Example**:
- Baseline: recall=80%, precision=75%
- With re-ranking: recall=82%, precision=87%

**Decision**: Deploy re-ranking!
```python
if improved_metrics['avg_precision'] > baseline_metrics['avg_precision']:
    print("✓ Re-ranking improves precision, deploy it")
```

### Scenario 3: Re-Ranking Doesn't Help

**Example**:
- Baseline: recall=85%, precision=85%
- With re-ranking: recall=85%, precision=84%

**Decision**: Skip re-ranking (adds cost with no benefit)
```python
if recall_gain < 0.03 and precision_gain < 0.03:
    print("✗ Re-ranking doesn't help, keep baseline")
```

---

## Measuring Cost & Benefit

### Cost of Re-Ranking

```python
import time

# Measure retrieval time
start = time.time()
results_baseline = baseline_retrieve(query, k=5)
baseline_time = time.time() - start

# Measure retrieval + re-ranking time
start = time.time()
results_improved = improved_retrieve(query, k=5)
improved_time = time.time() - start

latency_multiplier = improved_time / baseline_time
print(f"Latency multiplier: {latency_multiplier:.1f}x")  # e.g., 1.5x

# API cost
# - Re-ranking calls LLM once per query
# - Cost ≈ 1.3x baseline (retrieval is cheap, LLM is expensive)
```

### Benefit of Re-Ranking

```python
precision_gain = improved_metrics['avg_precision'] - baseline_metrics['avg_precision']

print(f"Precision improvement: {precision_gain:.1%}")

# Is it worth the cost?
if precision_gain > 0.05:  # More than 5% precision gain
    print("✓ Worth the cost")
else:
    print("✗ Not worth the cost")
```

---

## Deployment Checklist

Before deploying to production, verify:

- [ ] Recall ≥ 85% (or your target)
- [ ] Precision ≥ 70% (or your target)
- [ ] Failures analyzed and understood
- [ ] Cost/benefit trade-off acceptable
- [ ] Error handling in place (LLM timeouts, API errors)
- [ ] Monitoring set up (log metrics over time)

---

## Monitoring Over Time

```python
import json
from datetime import datetime

def log_evaluation_metrics(metrics, retriever_type):
    """Log metrics for trend analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "retriever_type": retriever_type,  # "baseline" or "improved"
        "avg_recall": metrics['avg_recall'],
        "avg_precision": metrics['avg_precision'],
        "min_recall": metrics['min_recall'],
    }
    
    with open("evaluation_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Log after each evaluation
log_evaluation_metrics(baseline_metrics, "baseline")
log_evaluation_metrics(improved_metrics, "improved")
```

Then analyze trends:
```
# Day 1: baseline recall=80%
# Day 2: baseline recall=78%  ← Degraded! Investigate
# Day 3: baseline recall=76%  ← Keep investigating

# Possible causes:
# - New data with different distribution
# - Embedding model changed
# - Index corruption
```

---

## Integration with Existing Code

### With existing retrieval.py
```python
from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics

# Works with all retrieval functions
results = evaluate_queries(queries, retrieve_top_k, k=5)
```

### With existing config.py
```python
from src.config import load_config

config = load_config()
model = config["llm"]["model"]
client = OpenAI(api_key=config["llm"]["api_key"])

# Configuration is centralized
```

### With existing prompts
```python
from src.prompt_templates import load_template

# Re-ranking and evaluation use same LLM client
# Prompts can be customized if needed
```

---

## Full Example: Production Pipeline

```python
#!/usr/bin/env python3
"""Production RAG pipeline with re-ranking and evaluation."""

from src.retrieval import retrieve_top_k, embed_query
from src.reranking import rerank_candidates
from src.retrieval_evaluation import evaluate_queries, aggregate_metrics
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config):
        self.client = OpenAI(api_key=config["api_key"])
        self.model = config["model"]
        self.chunks = load_chunks(config["corpus_path"])
        self.enable_reranking = config.get("enable_reranking", False)
    
    def retrieve(self, query, k=5):
        """Retrieve chunks for a query."""
        embedding = embed_query(self.client, self.model, query)
        candidates = retrieve_top_k(query, embedding, self.chunks, k=10)
        
        if self.enable_reranking:
            candidates = rerank_candidates(
                query, candidates, self.client, self.model, final_k=k
            )
        
        return candidates[:k]
    
    def evaluate(self, test_queries):
        """Evaluate retrieval quality."""
        results = evaluate_queries(test_queries, self.retrieve, k=5)
        metrics = aggregate_metrics(results)
        
        logger.info(f"Recall:    {metrics['avg_recall']:.1%}")
        logger.info(f"Precision: {metrics['avg_precision']:.1%}")
        
        return metrics
    
    def answer(self, query):
        """Answer a user query."""
        chunks = self.retrieve(query, k=5)
        context = "\n".join(c["text"] for c in chunks)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer based on this context:\n" + context},
                {"role": "user", "content": query},
            ],
        )
        
        return response.choices[0].message.content

# Usage
if __name__ == "__main__":
    config = {
        "api_key": "...",
        "model": "gpt-4",
        "corpus_path": "data/sample_corpus",
        "enable_reranking": True,  # Enable re-ranking
    }
    
    pipeline = RAGPipeline(config)
    
    # Evaluate
    test_queries = [...]
    metrics = pipeline.evaluate(test_queries)
    
    # Answer
    answer = pipeline.answer("What adverse events were reported?")
    print(answer)
```

---

## Summary

**Phase 1 (Re-Ranking)**: Improve precision by re-ranking candidates
**Phase 2 (Evaluation)**: Measure improvement with recall and precision

Together, they enable:
1. ✓ Baseline measurement
2. ✓ Targeted improvements
3. ✓ Confidence in deployment
4. ✓ Ongoing monitoring

**Next Steps**:
1. Read: [PHASE_2_SUMMARY.md](PHASE_2_SUMMARY.md)
2. Read: [RERANKING_GUIDE.md](RERANKING_GUIDE.md)
3. Build test queries
4. Measure baseline
5. Try improvements
6. Deploy with confidence

---

**You now have everything needed to build a production-ready RAG system!** 🚀
