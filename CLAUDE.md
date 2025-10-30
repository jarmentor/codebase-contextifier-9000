# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Codebase Contextifier 9000 is a Docker-based Model Context Protocol (MCP) server for semantic code search. It uses AST-aware chunking with tree-sitter, local LLM embeddings via Ollama, and Qdrant vector database for fast semantic search across codebases.

## Development Commands

### Docker Operations

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f mcp-server
docker-compose logs -f qdrant

# Restart after code changes
docker-compose restart mcp-server

# Rebuild after dependency changes
docker-compose up -d --build

# Stop all services
docker-compose down

# Clean start (removes volumes)
docker-compose down -v
```

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant locally
docker run -p 6333:6333 qdrant/qdrant

# Set environment variables
export QDRANT_HOST=localhost
export OLLAMA_HOST=http://localhost:11434
export INDEX_PATH=./index
export CACHE_PATH=./cache
export WORKSPACE_PATH=/path/to/codebase

# Run MCP server
python -m src.server
```

### Testing and Code Quality

```bash
# Run tests
pytest

# Format code
black src/

# Lint code
ruff src/
```

## Architecture Overview

### Core Components

The system has 5 main modules:

1. **MCP Server** (`src/server.py`): FastMCP server exposing tools via stdio protocol
2. **AST Chunker** (`src/indexer/ast_chunker.py`): Tree-sitter-based semantic code parsing
3. **Merkle Tree Indexer** (`src/indexer/merkle_tree.py`): Blake3-based incremental change detection
4. **Embeddings** (`src/indexer/embeddings.py`): Ollama integration with content-addressable caching
5. **Vector DB** (`src/vector_db/qdrant_client.py`): Qdrant client for semantic search

### Data Flow

```
File → AST Chunker → Semantic Chunks → Embeddings → Qdrant
         ↓                                              ↓
    Merkle Tree ← IncrementalIndexingSession ← SearchTool
```

1. **Indexing**: Files are parsed with tree-sitter to extract semantic chunks (functions, classes, methods)
2. **Hashing**: Each file gets a Blake3 content hash stored in merkle tree for change detection
3. **Embedding**: Chunks are sent to Ollama for embedding generation (cached by content hash)
4. **Storage**: Embeddings + metadata stored in Qdrant vector database
5. **Search**: Query embeddings are compared against stored vectors using cosine similarity

### AST-Aware Chunking

The chunker (`ast_chunker.py`) respects semantic boundaries:
- Functions, methods, classes, interfaces are extracted as complete units
- Each chunk includes context (e.g., parent class name for methods)
- Avoids splitting functions mid-code (30% better accuracy than fixed-size chunking)
- Supports 10+ languages via tree-sitter grammars

### Incremental Indexing

Uses Merkle tree pattern for efficiency:
- **First indexing**: All files are parsed, embedded, and stored
- **Re-indexing**: Only changed files are re-processed (detected via Blake3 hash comparison)
- **Cache hit rate**: Typically 80-95% on subsequent runs
- Session pattern: `IncrementalIndexingSession` context manager handles batch updates

### File Watcher

Real-time monitoring (`src/indexer/file_watcher.py`):
- Uses `watchdog` library to monitor filesystem events
- Debounces changes (default 2 seconds) to avoid re-indexing during rapid edits
- Automatically re-indexes modified files and removes deleted files from vector DB
- Runs async in background task started at server initialization

### Content-Addressable Caching

Embeddings are cached using content hashing:
```python
cache_key = blake3(model_name + chunk_content).hexdigest()
```
Benefits:
- Team members can share cache across machines
- Re-indexing after git operations is fast (unchanged files use cache)
- Deterministic: same content always produces same cache key

## MCP Tools Reference

The server exposes 8 MCP tools:

- `index_repository(repo_path, exclude_patterns, incremental)` - Index codebase
- `search_code(query, limit, language, file_path_filter, chunk_type)` - Semantic search
- `get_symbols(file_path, symbol_type)` - Extract AST symbols from file
- `get_indexing_status()` - Get index statistics
- `clear_index()` - Clear all indexed data
- `get_watcher_status()` - Check file watcher status
- `health_check()` - Check component health
- Auto-exposed by FastMCP via `@mcp.tool` decorator

## Configuration

Key environment variables (set in `.env` or `docker-compose.yml`):

- `CODEBASE_PATH` - Path to codebase to mount (Docker host path)
- `WORKSPACE_PATH` - Path inside container where codebase is mounted (default: `/workspace`)
- `OLLAMA_HOST` - Ollama API endpoint (default: `http://host.docker.internal:11434`)
- `EMBEDDING_MODEL` - Ollama model name (default: `nomic-embed-text`)
- `QDRANT_HOST` / `QDRANT_PORT` - Qdrant connection settings
- `INDEX_PATH` - Directory for merkle tree state (default: `/index`)
- `CACHE_PATH` - Directory for embedding cache (default: `/cache`)
- `MAX_CHUNK_SIZE` - Maximum characters per chunk (default: 2048)
- `BATCH_SIZE` - Embedding batch size (default: 32)
- `MAX_CONCURRENT_EMBEDDINGS` - Concurrent requests to Ollama (default: 4)
- `ENABLE_FILE_WATCHER` - Enable real-time file monitoring (default: `true`)
- `WATCHER_DEBOUNCE_SECONDS` - Delay before processing changes (default: 2.0)

## Adding New Languages

To support a new language:

1. Add tree-sitter grammar to `requirements.txt`:
   ```
   tree-sitter-kotlin>=0.20.0
   ```

2. Add language config to `config/languages.json`:
   ```json
   {
     "kotlin": {
       "name": "kotlin",
       "extensions": [".kt", ".kts"],
       "tree_sitter_language": "kotlin",
       "chunkable_nodes": ["function_declaration", "class_declaration"],
       "name_fields": {
         "function_declaration": "simple_identifier",
         "class_declaration": "type_identifier"
       }
     }
   }
   ```

3. Import module in `ast_chunker.py`:
   ```python
   import tree_sitter_kotlin as tskotlin
   ```

4. Add to `LANGUAGE_MODULES` dict:
   ```python
   "kotlin": tskotlin,
   ```

## Common Development Tasks

### Debugging Indexing Issues

```bash
# Check what files are detected
docker exec -it codebase-mcp-server python -c "
from src.indexer.grammars import get_language_registry
registry = get_language_registry()
print('Supported extensions:', registry.get_supported_extensions())
"

# Check merkle tree state
docker exec -it codebase-mcp-server python -c "
from pathlib import Path
from src.indexer.merkle_tree import MerkleTreeIndexer
indexer = MerkleTreeIndexer(Path('/index'))
print('Stats:', indexer.get_stats())
"

# View cache contents
docker exec -it codebase-mcp-server ls -lh /cache
```

### Testing Search Quality

Search quality depends on:
- **Embedding model**: `nomic-embed-text` (balanced), `mxbai-embed-large` (better accuracy)
- **Chunk size**: Smaller chunks = more precise but need more context
- **Query phrasing**: Natural language works better than keywords

### Performance Tuning

Indexing performance:
- Increase `MAX_CONCURRENT_EMBEDDINGS` if CPU allows
- Increase `BATCH_SIZE` if RAM allows (embeddings processed in batches)
- Use SSD for `/cache` and `/index` volumes

Search performance:
- Qdrant is already optimized (sub-10ms typically)
- Reduce `limit` parameter if only top results needed

## Important Implementation Notes

### Session Pattern for Index Updates

Always use `IncrementalIndexingSession` context manager when updating the index:

```python
with IncrementalIndexingSession(merkle_indexer) as session:
    # Plan updates
    files_to_index, files_to_remove = session.plan_incremental_update(all_files)

    # Process changes
    for file_path in files_to_remove:
        chunk_ids = session.remove_file(file_path)
        vector_db.delete_by_file_path(file_path)

    for file_path in files_to_index:
        chunks = chunker.chunk_file(file_path)
        embeddings_list = await embeddings.generate_embeddings_batched(...)
        vector_db.upsert_chunks(chunks, embeddings_list)
        session.update_file(file_path, chunk_ids)

    # Automatically commits on __exit__
```

This ensures merkle tree state is only saved on success.

### Async/Sync Boundaries

- FastMCP tools can be `async def` or regular `def`
- Embedding generation is async (uses httpx for Ollama API)
- Vector DB operations are sync (qdrant-client is sync)
- File watcher runs in background asyncio task

### Error Handling Philosophy

- Log errors but continue processing remaining files
- Invalid files are skipped (unsupported language, parse errors)
- Health checks warn but don't block startup (Ollama may be temporarily down)
- File watcher exceptions are caught to prevent crashes
