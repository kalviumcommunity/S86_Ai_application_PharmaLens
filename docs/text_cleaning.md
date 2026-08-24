# PharmaLens Text Extraction & Cleaning Pipeline

## Purpose

The PharmaLens corpus loader converts supported documents into plain
text. This assignment adds a cleaning layer so that extracted text
is suitable for later chunking, embedding, and retrieval.

## Pipeline

```text
Document
   ↓
Text Extraction
   ↓
Raw Text
   ↓
Unicode Normalization
   ↓
Boilerplate Removal
   ↓
Whitespace Normalization
   ↓
Clean Text
   ↓
Future Chunking