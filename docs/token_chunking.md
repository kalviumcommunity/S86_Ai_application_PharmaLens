# Token-Aware Chunk Sizing & Overlap

## Purpose

PharmaLens uses Retrieval-Augmented Generation, so documents must be
split into chunks before embeddings and vector search.

Character-based chunking is unreliable because the model processes
tokens rather than characters.

This implementation therefore sizes chunks using `tiktoken`.

## Pipeline

```text
Document
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Token-Aware Chunking
    ↓
Overlap
    ↓
Embeddings
    ↓
Vector Database