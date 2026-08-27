# Corpus Preparation & Ingestion Validation

## Purpose

PharmaLens needs to process the complete document corpus through the
same preparation pipeline before documents are embedded and stored.

The ingestion pipeline performs:

```text
Load
  ↓
Clean
  ↓
Token-aware Chunk
  ↓
Metadata
  ↓
Validation