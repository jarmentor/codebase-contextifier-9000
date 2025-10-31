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

The system has 8 main modules:

1. **MCP Server** (`src/server.py`): FastMCP server exposing tools via stdio protocol
2. **AST Chunker** (`src/indexer/ast_chunker.py`): Tree-sitter-based semantic code parsing
3. **Merkle Tree Indexer** (`src/indexer/merkle_tree.py`): Blake3-based incremental change detection
4. **Embeddings** (`src/indexer/embeddings.py`): Ollama integration with content-addressable caching
5. **Vector DB** (`src/vector_db/qdrant_client.py`): Qdrant client for semantic search (code collections)
6. **Knowledge DB** (`src/vector_db/knowledge_db.py`): Separate Qdrant collection for dependency knowledge
7. **Job Manager** (`src/indexer/job_manager.py`): Background job orchestration and Docker container spawning
8. **Dependency Detector** (`src/indexer/dependency_detector.py`): WordPress, Composer, npm dependency detection

### Container-Based Indexing Architecture

The system uses on-demand Docker containers for indexing any repository on the host machine:

```
MCP Client → index_repository(host_path="/Users/you/projects/my-app")
                 ↓
          JobManager (src/server.py)
                 ↓
    Spawns ephemeral indexer container via Docker API
                 ↓
          Mounts host path → /workspace
                 ↓
    Indexer script (scripts/indexer.py) runs inside container
                 ↓
    AST Chunker → Embeddings → Shared Qdrant/Volumes
                 ↓
    Container auto-removes on completion
                 ↓
    Job status available via get_job_status()
```

**Key features:**
- **On-demand spawning**: No need to pre-configure docker-compose volumes
- **Shared backend**: All repositories write to same Qdrant instance
- **Background jobs**: Track progress with job_id, supports cancellation
- **Auto-cleanup**: Containers are removed after indexing completes

### Data Flow

```
File → AST Chunker → Semantic Chunks → Embeddings → Qdrant (code_chunks collection)
         ↓                                              ↓
    Merkle Tree ← IncrementalIndexingSession ← SearchTool
         ↓
    Per-repo state stored in /index/{repo_name}/
```

1. **Indexing**: Files are parsed with tree-sitter to extract semantic chunks (functions, classes, methods)
2. **Hashing**: Each file gets a Blake3 content hash stored in merkle tree for change detection (per-repo)
3. **Embedding**: Chunks are sent to Ollama for embedding generation (cached by content hash in shared /cache)
4. **Storage**: Embeddings + metadata stored in Qdrant vector database with repo_name metadata
5. **Search**: Query embeddings are compared against stored vectors using cosine similarity (filter by repo_name if needed)

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

The server exposes 14 MCP tools:

**Indexing:**
- `index_repository(host_path, repo_name, incremental, exclude_patterns)` - Spawn container to index any repository on host
- `get_job_status(job_id)` - Get progress of background indexing job
- `list_indexing_jobs()` - List all indexing jobs (past and present)
- `cancel_indexing_job(job_id)` - Cancel a running indexing job

**Search:**
- `search_code(query, limit, repo_name, language, file_path_filter, chunk_type)` - Semantic search across all indexed repos

**Symbols:**
- `get_symbols(file_path, symbol_type)` - Extract AST symbols from file

**Status:**
- `get_indexing_status()` - Get index statistics (code_db, knowledge_db, cache)
- `clear_index()` - Clear all indexed data
- `get_watcher_status()` - Check file watcher status
- `health_check()` - Check component health

**Dependencies (WordPress, Composer, npm):**
- `detect_dependencies(workspace_path)` - Detect available dependencies in workspace
- `list_indexed_dependencies()` - List dependencies already indexed in knowledge base
- `index_dependencies(dependency_names, workspace_id, workspace_path)` - Index specific dependencies

All tools are auto-exposed by FastMCP via `@mcp.tool` decorator.

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

## Multi-Repository Workflow

The system supports two deployment patterns:

### Pattern A: Centralized Server (Recommended)
1. Start backend once: `docker-compose up -d`
2. Connect Claude Desktop/Code to MCP server container
3. Index any repository: `index_repository(host_path="/Users/you/projects/my-app")`
4. Search across all indexed repos: `search_code(query="auth logic")`

### Pattern B: Per-Project Setup
1. Start shared backend: `docker-compose up -d`
2. Copy `.mcp.json` to each project directory
3. Open project in Claude Code to trigger indexing
4. Switch between projects as needed

**Key insight:** All projects share the same Qdrant/cache/index volumes, enabling:
- Cross-repository search
- Shared embedding cache (faster indexing)
- Centralized query server

See `MULTI_PROJECT_SETUP.md` for details.

## Background Jobs

Indexing large codebases runs in background jobs:

```python
# Start indexing
job = await index_repository(host_path="/Users/you/projects/large-app")
# → {"job_id": "abc123", "status": "queued"}

# Check progress
await get_job_status(job_id="abc123")
# → {
#     "status": "running",
#     "progress": {
#       "current_file": 45,
#       "total_files": 100,
#       "progress_pct": 45.0,
#       "chunks_indexed": 234,
#       "cache_hit_rate": "82.50%"
#     }
#   }

# List all jobs
await list_indexing_jobs()

# Cancel if needed
await cancel_indexing_job(job_id="abc123")
```

See `BACKGROUND_JOBS.md` for detailed workflow patterns.

## Dependency Knowledge Base

For WordPress/PHP projects with many dependencies:

```python
# Detect available dependencies
await detect_dependencies()
# → Lists WordPress plugins, themes, Composer packages, npm packages

# Index specific dependencies into knowledge base
await index_dependencies(
  dependency_names=["woocommerce", "acf"],
  workspace_id="my-site"
)

# Check what's indexed
await list_indexed_dependencies()
```

**Architecture:**
- Code chunks → `code_chunks` collection (searchable via `search_code`)
- Dependencies → `dependency_knowledge` collection (separate namespace)
- Deduplication: Same dependency version indexed once, linked to multiple workspaces

## Common Development Tasks

### Debugging Container-Based Indexing

```bash
# List indexer containers (ephemeral, auto-removed)
docker ps -a | grep indexer-

# View logs from specific job container
docker logs indexer-abc123

# Check job status via MCP tools
# Use get_job_status(job_id="abc123") in Claude

# View shared volumes
docker volume ls | grep codebase-contextifier
docker volume inspect codebase-contextifier-9000_index_data
```

### Debugging Indexing Issues

```bash
# Check what files are detected
docker exec -it codebase-mcp-server python -c "
from src.indexer.grammars import get_language_registry
registry = get_language_registry()
print('Supported extensions:', registry.get_supported_extensions())
"

# Check merkle tree state (per-repo)
docker exec -it codebase-mcp-server ls -la /index/
# Each subdirectory is a repo_name

# View cache contents (shared across all repos)
docker exec -it codebase-mcp-server ls -lh /cache

# Check Qdrant collections
docker exec -it codebase-qdrant wget -qO- http://localhost:6333/collections
```

### Testing Search Quality

Search quality depends on:
- **Embedding model**: `nomic-embed-text` (balanced), `mxbai-embed-large` (better accuracy)
- **Chunk size**: Smaller chunks = more precise but need more context
- **Query phrasing**: Natural language works better than keywords
- **Multi-repo search**: Filter by `repo_name` parameter if searching specific project

### Performance Tuning

Indexing performance:
- Increase `MAX_CONCURRENT_EMBEDDINGS` if CPU allows
- Increase `BATCH_SIZE` if RAM allows (embeddings processed in batches)
- Use SSD for `/cache` and `/index` volumes
- Container-based indexing: Spawn multiple indexer containers in parallel (each indexes different repo)

Search performance:
- Qdrant is already optimized (sub-10ms typically)
- Reduce `limit` parameter if only top results needed
- Use `repo_name` filter to limit search scope

## Important Implementation Notes

### Container Spawning Pattern

The `index_repository` tool spawns ephemeral Docker containers via `JobManager`:

```python
# In IndexingTool (src/tools/index_tool.py)
job = self.job_manager.create_job(repo_name, host_path)

success = await self.job_manager.spawn_indexer_container(
    job_id=job.job_id,
    host_path=host_path,  # Host machine path
    repo_name=repo_name,
    qdrant_host="codebase-qdrant",  # Container network name
    incremental=incremental,
    exclude_patterns=exclude_patterns,
)

# JobManager spawns container with:
# - Volume mount: host_path -> /workspace (read-only)
# - Shared volumes: index_data, cache_data (read-write)
# - Network: codebase-contextifier-9000_default
# - Auto-remove: Container deletes after completion
# - Monitor task: Polls container status, updates job progress
```

**Key considerations:**
- MCP server needs Docker socket access (`/var/run/docker.sock`)
- Indexer image must be pre-built (`codebase-contextifier-9000-indexer`)
- Container runs `scripts/indexer.py` as entrypoint
- Job status tracked in-memory (lost on MCP server restart, but indexed data persists)

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
- File watcher uses dual-threading model (watchdog observer + asyncio debounce processor)

### File Watcher Threading Model

The file watcher (`src/indexer/file_watcher.py`) uses a complex threading model:

```python
# In src/server.py main block:
asyncio.run(initialize_components())  # Initialize file_watcher object

start_file_watcher_sync()  # Start watchdog observer thread

# Start debounce processor in separate thread (runs its own asyncio loop)
watcher_thread = threading.Thread(
    target=run_watcher_debounce_in_thread,
    daemon=True
)
watcher_thread.start()

# Then run MCP server (FastMCP has its own event loop)
mcp.run()
```

**Why this architecture?**
- `watchdog` observer runs in background thread (synchronous)
- Debounce logic needs async/await for delays and callback
- FastMCP server runs in main thread with its own event loop
- Can't share event loops across threads, so debounce processor gets its own thread + loop

**Alternative considered:** Run debounce processor in FastMCP's event loop, but FastMCP initialization happens after `if __name__ == "__main__"`, making it hard to hook into.

### Error Handling Philosophy

- Log errors but continue processing remaining files
- Invalid files are skipped (unsupported language, parse errors)
- Health checks warn but don't block startup (Ollama may be temporarily down)
- File watcher exceptions are caught to prevent crashes
- Container spawn failures mark job as failed but don't crash MCP server
