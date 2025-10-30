# Quick Start Guide

## 5-Minute Setup

### 1. Prerequisites

Install these first:
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **Ollama**: https://ollama.ai

```bash
# Pull the embedding model
ollama pull nomic-embed-text
```

### 2. Run Setup Script

```bash
cd codebase-contextifier-9000
./scripts/setup.sh
```

This will:
- Check prerequisites
- Create `.env` file
- Build Docker image
- Start services
- Configure Claude Desktop (optional)

### 3. Configure Your Codebase

Edit `.env` and set:
```bash
CODEBASE_PATH=/path/to/your/project
```

Then restart:
```bash
docker-compose restart
```

### 4. Use in Claude Desktop

In Claude Desktop, try these prompts:

**Index your code:**
```
Index the repository at /workspace
```

**Search for code:**
```
Search for "authentication logic" in the codebase
```

**Get symbols:**
```
Show me all functions in /workspace/src/auth.ts
```

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit .env with your codebase path
nano .env

# 3. Build and start
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Configure Claude Desktop
# Add config from claude_desktop_config.example.json
# to ~/Library/Application Support/Claude/claude_desktop_config.json
```

## Common Issues

### "Ollama connection failed"

```bash
# Check Ollama is running
ollama serve

# Test connection from Docker
docker exec codebase-mcp-server curl http://host.docker.internal:11434
```

### "No files indexed"

Check your `.env` file:
```bash
# Make sure path is absolute and correct
CODEBASE_PATH=/absolute/path/to/your/project

# Then restart
docker-compose restart
```

### "Qdrant connection failed"

```bash
# Check Qdrant container
docker-compose ps qdrant

# Restart Qdrant
docker-compose restart qdrant
```

## Testing the Setup

```bash
# Check all services are running
docker-compose ps

# Check MCP server logs
docker-compose logs mcp-server

# Check Qdrant logs
docker-compose logs qdrant

# Health check
docker exec -it codebase-mcp-server python -c "
from src.vector_db.qdrant_client import CodeVectorDB
db = CodeVectorDB(host='qdrant')
print('✓ Qdrant connected:', db.health_check())
"
```

## Next Steps

1. Read the full [README.md](README.md) for details
2. Check the [research document](AST_codeChunking.md) for background
3. Experiment with different search queries
4. Try different embedding models (edit `EMBEDDING_MODEL` in `.env`)

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Restart everything
docker-compose restart

# Stop everything
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Clear all data (fresh start)
docker-compose down -v
```
