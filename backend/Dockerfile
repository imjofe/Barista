FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.0

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual env (we're in Docker)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Copy application code
COPY . .

# Ensure menu.md is accessible and verify it exists
RUN if [ -f menu.md ]; then \
      chmod 644 menu.md && \
      echo "menu.md found and permissions set"; \
    else \
      echo "WARNING: menu.md not found in image"; \
    fi

# Create data directory for ChromaDB
RUN mkdir -p /data/chroma

# Expose port
EXPOSE 8080

# Run the application
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

