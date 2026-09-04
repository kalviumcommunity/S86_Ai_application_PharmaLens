# RAG Evaluation & Answer Quality Scoring

## Overview

PharmaLens evaluates the complete RAG system using a small test set.

Each question contains:

- Expected answer points
- Expected source documents

The RAG pipeline generates an answer with citations, which is then
evaluated for correctness, grounding, and citation accuracy.

## Evaluation Flow

```text
Test Question
     ↓
Query Embedding
     ↓
Qdrant Retrieval
     ↓
Context Assembly
     ↓
Grounded Answer + Citations
     ↓
Correctness Scoring
     ↓
Grounding Scoring
     ↓
Citation Accuracy Scoring
     ↓
Evaluation Summary