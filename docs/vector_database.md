# PharmaLens Vector Database Setup

## Overview

PharmaLens uses Qdrant as the vector database for storing embedding
vectors together with their source text and metadata.

The vector database allows the application to later perform semantic
similarity searches over document chunks.

## Vector Database

- Vector database: Qdrant
- Deployment: Local Docker container
- URL: `http://localhost:6333`
- Collection: `rag_chunks`
- Distance metric: Cosine
- Embedding model: `gemini-embedding-001`
- Vector dimension: `3072`

## Collection Design

The `rag_chunks` collection stores vectors using the same dimension
returned by the embedding model.

Each stored record contains:

- `id` - unique Qdrant point identifier
- `vector` - embedding vector
- `text` - original document chunk
- `metadata.source` - source document
- `metadata.chunk_index` - chunk position
- `metadata.section` - document section
- `metadata.page` - page number

## Example Record

```text
ID:
f6b433df-4e04-4013-961a-ab05ebefad21

Vector length:
3072

Text:
Clinical trials evaluate medical treatments for safety and effectiveness.

Metadata:
source = clinical-trial-demo.txt
chunk_index = 0
section = Introduction
page = 1