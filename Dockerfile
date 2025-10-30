# Multi-stage build for efficient Docker image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Build tree-sitter grammars
RUN mkdir -p /root/.local/lib/python3.11/site-packages/tree_sitter_languages

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include local Python packages
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Create volume mount points
RUN mkdir -p /workspace /index /cache

# Environment variables with defaults
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV EMBEDDING_MODEL=nomic-embed-text
ENV INDEX_PATH=/index
ENV CACHE_PATH=/cache
ENV LOG_LEVEL=INFO

# Expose port for potential HTTP-based MCP (optional)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; from qdrant_client import QdrantClient; client = QdrantClient(host='${QDRANT_HOST}', port=${QDRANT_PORT}); client.get_collections()" || exit 1

# Entry point
CMD ["python", "-m", "src.server"]
