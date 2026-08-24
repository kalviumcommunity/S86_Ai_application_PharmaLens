# RAG Application

A clean, reproducible, and secure workspace for building an AI application using Retrieval-Augmented Generation (RAG).

This project follows a structured development environment that keeps dependencies isolated, secrets protected, and project files organized.

## Project Structure

```text
rag-app/
├── data/          # Source documents
├── src/           # Ingestion, embeddings, retrieval, and application code
├── prompts/       # Prompt templates
├── outputs/       # Logs, generated answers, and evaluation results
├── .env           # Real secrets, never committed
├── .env.example   # Required environment variables without real values
├── .gitignore     # Files and folders excluded from Git
├── requirements.txt
└── README.md
```

## Prerequisites

Make sure Python is installed on your machine.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

After activation, the terminal should show:

```text
(.venv)
```

This ensures that the project's dependencies are isolated from other Python projects on the machine.

## 4. Install Dependencies

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project uses packages including:

- OpenAI
- ChromaDB
- python-dotenv

The exact versions are recorded in `requirements.txt` to make the environment reproducible.

## 5. Configure Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

### Important

The `.env` file contains secrets and **must never be committed to Git**.

The repository includes `.env.example` as a safe template:

```env
OPENAI_BASE_URL=
OPENAI_API_KEY=
CHAT_MODEL=
EMBED_MODEL=
```

Copy `.env.example` to `.env` and provide your actual values.

## 6. Loading Secrets

Secrets should be loaded at runtime rather than hard-coded into the application.

Example:

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

The API key should never be written directly into source code.

## Reproducibility Test

A new developer should be able to set up the project using the following process:

```bash
git clone <repository-url>
cd rag-app

python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
copy .env.example .env
```

Then add the required API keys and configuration values to `.env`.

If the project can be set up successfully on a fresh machine using these steps, the development environment is reproducible.

## Security

This project follows basic security practices for AI application development:

- API keys are stored in `.env`.
- `.env` is excluded from Git.
- `.env.example` contains only variable names and no real secrets.
- Dependencies are isolated inside a virtual environment.
- Dependencies are recorded in `requirements.txt`.
- Source documents are kept separately from application code.

## Dependency Management

Dependencies are recorded using:

```bash
pip freeze > requirements.txt
```

This captures the installed package versions so that another developer can recreate the same environment.

## Development Principles

This project follows four core principles:

1. **Isolation**
   Project dependencies are installed inside a virtual environment.

2. **Organization**
   Documents, source code, prompts, and outputs are separated into dedicated directories.

3. **Security**
   API keys and other secrets are stored outside the source code.

4. **Reproducibility**
   Dependencies and required environment variables are documented so the project can be recreated on another machine.

## Structured JSON Output Demo

This repository includes a structured-output example script for RAG responses:

```bash
python -m src.structured_output_demo
```

What it demonstrates:

- Prompting the model for a fixed JSON shape (`answer`, `source`) using JSON response-format mode.
- Parsing model output into a Python dictionary.
- Handling malformed JSON safely with clear error reporting and best-effort recovery.
- Validating required fields before downstream use.

Sample parsed results are written to:

```text
outputs/structured_output_samples.json
```

The sample output includes a malformed-then-recovered case and a missing-field rejection case.

## Chunk Metadata & Source Tracking

Chunking preserves a consistent metadata dictionary beside every chunk. Each entry includes:

- `source`: the source document identifier used for citation.
- `chunk_index`: the chunk's one-based position in that document.
- `char_start` and `char_end`: the range in the normalized text used by the chunker.
- `section` and `page`: reserved fields populated when the source format provides them.

Run the demonstration and regenerate the committed sample chunks with:

```bash
python -m src.chunking
```

The generated report at `outputs/chunking_comparison.md` shows text plus metadata for both chunking strategies and traces a retrieved chunk to a clickable source document and character range.

## Reusable Prompt Templates

Prompt templates are separated from business logic in the `prompts/` folder and rendered at runtime via `src/prompt_templates.py`.

- `prompts/rag_system.txt` defines shared system behavior.
- `prompts/rag_user.txt` defines a reusable user template with named placeholders:
  - `{context}`
  - `{question}`
  - `{output_instructions}`

Two features reuse the same template structure:

- `src/prompt_demo.py` (chat-style request flow)
- `src/structured_output_demo.py` (structured JSON response flow)

To generate example rendered prompts for both chat and batch/CLI paths:

```bash
python -m src.template_render_demo
```

Rendered examples are saved to:

````text
outputs/prompt_template_renders.txt



## Multi-Format Corpus Loader Demo

This repository includes a corpus loader that converts mixed input documents into a common plain-text representation while preserving source identifiers.

Run it with:

```bash
python -m src.corpus_loader_demo
````

What it demonstrates:

- Loads multiple formats into plain text (`.txt`, `.md`, `.html`).
- Survives bad input by skipping missing or unsupported files with clear messages.
- Retains each document source (`source_id`) for future citation.
- Prints each loaded document's text length and short sample snippet.

Sample corpus files are in:

```text
data/sample_corpus/
```

Sample intake output is saved to:

```text
outputs/corpus_loader_intake.log
```

## Chunking Strategy Comparison

Run the chunking comparison on the cleaned clinical report:

```bash
python -m src.chunking
```

The command compares paragraph-aware chunks with fixed-size chunks using overlap, reports chunk counts and average character sizes, and writes inspectable samples to:

```text
outputs/chunking_comparison.md
```
