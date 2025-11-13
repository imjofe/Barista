# Barista Backend

FastAPI service powering the Barista AI agent. The service exposes chat and tool endpoints, orchestrates the LangGraph-based workflow, and serves as the bridge between the frontend and external APIs.

## Requirements

- Python 3.11+
- Poetry 1.8+

## Getting Started

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload --port 8080
```

## Environment Variables

Copy `env.example` to `.env` and provide values:

### Required Azure OpenAI Configuration

**Chat Completions (AI Foundry format):**
- `AZURE_OPENAI_ENDPOINT` – Chat endpoint base URL (default: `https://dehub.services.ai.azure.com`)
- `AZURE_OPENAI_API_KEY` – Azure OpenAI API key (required)
- `AZURE_OPENAI_DEPLOYMENT_NAME` – Deployment name for chat completions (e.g., your DeepSeek deployment)
- `AZURE_OPENAI_API_VERSION` – API version for chat (default: `2024-05-01-preview`)

**Embeddings (Cognitive Services format):**
- `AZURE_OPENAI_EMBEDDING_ENDPOINT` – Embeddings endpoint base URL (default: `https://dehub.cognitiveservices.azure.com`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` – Deployment name for embeddings (default: `text-embedding-ada-002`)
- `AZURE_OPENAI_EMBEDDING_API_VERSION` – API version for embeddings (default: `2023-05-15`)

### Optional Configuration

- `STABILITY_API_KEY` – Optional image generation provider key
- `REDIS_URL` – Connection string for session memory (default: `redis://redis:6379/0`)
- `CHROMA_PERSIST_PATH` – Filesystem path for Chroma vector store persistence
- `ALLOW_ORIGINS` – Comma-separated list of allowed CORS origins

