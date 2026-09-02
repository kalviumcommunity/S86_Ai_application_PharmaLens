                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Query Embedding │
                    │ Gemini Embedding│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Qdrant       │
                    │ Vector Retrieval│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Context Assembly│
                    │ + Source Metadata│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Gemini Generator│
                    │ Grounded Prompt │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Answer + Sources│
                    └─────────────────┘