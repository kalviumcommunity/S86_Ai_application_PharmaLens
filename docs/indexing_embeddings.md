# Indexing Embeddings & Metadata Storage

## Overview

PharmaLens now indexes the generated corpus embeddings into the Qdrant
vector database.

Each indexed record contains:

- Embedding vector
- Source text
- Source document metadata
- Study ID
- Chunk index

The existing Qdrant collection is:

- Collection: `rag_chunks`
- Vector dimension: `3072`
- Distance metric: `COSINE`

## Data Flow

```text
Corpus chunks
     |
     v
Batch embeddings
     |
     v
Embedding + source text + metadata
     |
     v
Qdrant collection
     |
     v
Count validation + spot check