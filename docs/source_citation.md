# Source Citation & Attribution

## Overview

PharmaLens adds citations to generated answers so that users can
trace factual claims back to the retrieved clinical research chunks.

## Citation Flow

```text
User Question
      ↓
Query Embedding
      ↓
Qdrant Retrieval
      ↓
Retrieved Chunks
      ↓
Citation Mapping
      ↓
Grounded Gemini Generation
      ↓
Cited Answer
      ↓
Citation Verification
      ↓
Answer + Sources