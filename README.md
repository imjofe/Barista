# Barista AI Agent

An intelligent virtual assistant for coffee shops that replaces static menu boards with an interactive, AI-powered conversational interface.

## Features

- **Menu Q&A**: Answer questions about drinks, prices, and ingredients using RAG (Retrieval-Augmented Generation)
- **Availability Checking**: Time-based drink availability with intelligent suggestions
- **Promotions**: Daily specials and deals via mock API integration
- **Image Generation**: Visual drink representations using FLUX 1.1 on Azure Foundry (optional)
- **Voice Input/Output**: Azure Speech Services integration for speech-to-text and text-to-speech
- **Session Management**: Conversation continuity across interactions using LangGraph's MemorySaver

## Architecture

### Components

- **Frontend**: Next.js 14 with React, Tailwind CSS, and Web Speech API
- **Backend**: FastAPI with LangGraph agent orchestration
- **Knowledge Base**: ChromaDB vector store with OpenAI embeddings
- **Tools**: Availability checker, promotions API, image generation
- **Session Memory**: LangGraph checkpointing for conversation state

### Tech Stack

- **UI**: Next.js, React, Tailwind CSS, shadcn/ui components
- **Backend**: FastAPI, LangChain, LangGraph, Azure OpenAI
- **Vector Store**: ChromaDB
- **Embeddings**: Azure OpenAI text-embedding-ada-002
- **LLM**: Azure OpenAI (DeepSeek via Azure endpoint)
- **Image Generation**: FLUX 1.1 on Azure Foundry
- **Speech Services**: Azure Speech Services (STT/TTS)
- **Memory**: LangGraph MemorySaver for conversation state persistence
- **Deployment**: Docker containers managed by Coolify

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Azure OpenAI API key and endpoint access
- (Optional) Azure Speech Services API key for voice input/output

### Local Development

1. **Start Docker daemon** (if not running):
   ```bash
   sudo systemctl start docker
   # Or on some systems:
   sudo service docker start
   ```

2. **Fix Docker permissions** (if you get permission denied errors):
   ```bash
   # Add your user to the docker group
   sudo usermod -aG docker $USER
   
   # Log out and log back in, or run:
   newgrp docker
   
   # Verify you can access Docker:
   docker ps
   ```

3. **Set up environment variables**:
   
   For docker-compose, create a `.env` file in the project root:
   ```bash
   # Create .env file with your Azure OpenAI credentials
   cat > .env << EOF
   AZURE_OPENAI_ENDPOINT=https://dehub.services.ai.azure.com
   AZURE_OPENAI_API_KEY=your-azure-api-key
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deepseek-deployment-name
   AZURE_OPENAI_API_VERSION=2024-05-01-preview
   AZURE_OPENAI_EMBEDDING_ENDPOINT=https://dehub.cognitiveservices.azure.com
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
   AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15
   AZURE_FLUX_DEPLOYMENT_NAME=FLUX-1.1-pro
   AZURE_FLUX_API_VERSION=2025-04-01-preview
   # Optional: Azure Speech Services for voice features
   # AZURE_SPEECH_KEY=your-azure-speech-key
   # AZURE_SPEECH_REGION=eastus
   EOF
   ```
   
   For local development without Docker, also set up `backend/.env`:
   ```bash
   cp backend/env.example backend/.env
   # Edit backend/.env with your API keys
   ```

4. **Start services**:
   ```bash
   docker-compose up --build
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8080
   - API Docs: http://localhost:8080/docs

### Development Mode

**Backend**:
```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload --port 8080
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

### Backend (`backend/.env`)

**Required Azure OpenAI Configuration:**

**Chat Completions:**
- `AZURE_OPENAI_ENDPOINT` – Chat endpoint (default: `https://dehub.services.ai.azure.com`)
- `AZURE_OPENAI_API_KEY` – Azure OpenAI API key (required)
- `AZURE_OPENAI_DEPLOYMENT_NAME` – Chat deployment name (e.g., your DeepSeek deployment)
- `AZURE_OPENAI_API_VERSION` – Chat API version (default: `2024-05-01-preview`)

**Embeddings:**
- `AZURE_OPENAI_EMBEDDING_ENDPOINT` – Embeddings endpoint (default: `https://dehub.cognitiveservices.azure.com`)
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` – Embedding deployment name (default: `text-embedding-ada-002`)
- `AZURE_OPENAI_EMBEDDING_API_VERSION` – Embeddings API version (default: `2023-05-15`)

**Optional:**
- `STABILITY_API_KEY` – Optional, for image generation
- `REDIS_URL` – Redis connection (default: `redis://redis:6379/0`)
- `CHROMA_PERSIST_PATH` – Vector store persistence path
- `ALLOW_ORIGINS` – CORS allowed origins (comma-separated)

### Frontend

- `NEXT_PUBLIC_API_BASE_URL` - Backend API URL (default: `http://localhost:8080`)

## Deployment with Coolify

1. **Prepare Docker images**:
   - Backend: Build from `backend/Dockerfile`
   - Frontend: Build from `frontend/Dockerfile`

2. **Configure in Coolify**:
   - Create applications for backend and frontend
   - Set environment variables as specified above
   - Mount volumes for ChromaDB persistence (`/data/chroma`)
   - Configure Redis service (or use external Redis)

3. **Service dependencies**:
   - Frontend depends on backend
   - Backend depends on Redis

## Project Structure

```
Barista/
├── backend/              # FastAPI backend service
│   ├── app/
│   │   ├── knowledge/    # RAG and ingestion
│   │   ├── tools/        # Agent tools
│   │   ├── memory/       # Session management
│   │   ├── routes/       # API endpoints
│   │   └── config.py     # Settings
│   ├── agent/
│   │   └── graph.py      # LangGraph agent
│   ├── menu.md           # Menu knowledge base
│   └── Dockerfile
├── frontend/             # Next.js frontend
│   ├── app/
│   │   ├── components/   # React components
│   │   └── page.tsx      # Main page
│   └── Dockerfile
├── context/
│   ├── agents.md         # Agent prompts documentation
│   └── barista_architecture.md
└── docker-compose.yml     # Local development setup
```

## API Endpoints

- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `POST /api/chat/` - Chat with the Barista agent

See `/docs` for interactive API documentation.

## Testing

Run backend tests:
```bash
cd backend
poetry run pytest
```

## Documentation

- **Agent Prompts**: See `context/agents.md` for detailed prompt configurations
- **Architecture**: See `context/barista_architecture.md` for system design

## License

MIT

