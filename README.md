# Codebase Contextifier 9000

A Docker-based Model Context Protocol (MCP) server for semantic code search with AST-aware chunking, relationship tracking via Neo4j graph database, local LLM support, and incremental indexing.

## Documentation

- 📚 **[Quick Start Guide](QUICK_START.md)** - Get running in 5 minutes
- 🔧 **[Multi-Project Setup](MULTI_PROJECT_SETUP.md)** - Index multiple projects with shared backend
- ⚙️ **[Background Jobs](BACKGROUND_JOBS.md)** - Job-based indexing for large codebases
- 👁️ **[File Watcher Guide](docs/FILE_WATCHER.md)** - Real-time monitoring and auto-indexing
- 🔬 **[Research & Methodology](docs/AST_codeChunking.md)** - Deep dive into semantic code search
- 📖 **[Full Documentation](docs/)** - Complete docs directory

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
  - [Key Architectural Features](#key-architectural-features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Two Deployment Options](#two-deployment-options)
  - [Claude Desktop Configuration](#claude-desktop-configuration)
  - [Usage](#usage)
- [MCP Tools](#mcp-tools)
  - [Indexing Tools](#indexing-tools): `index_repository`, `get_job_status`, `list_indexing_jobs`, `cancel_indexing_job`
  - [Search Tools](#search-tools): `search_code`, `get_symbols`
  - [Graph Query Tools](#graph-query-tools): `find_usages`, `find_dependencies`, `query_graph`
  - [Dependency Tools](#dependency-tools): `detect_dependencies`, `index_dependencies`, `list_indexed_dependencies`
  - [Status Tools](#status-tools): `get_indexing_status`, `clear_index`, `get_watcher_status`, `health_check`
- [Supported Languages](#supported-languages)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Recommended Embedding Models](#recommended-embedding-models)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Architecture Details](#architecture-details)
  - [AST-Aware Chunking](#ast-aware-chunking)
  - [Incremental Indexing](#incremental-indexing)
  - [Content-Addressable Storage](#content-addressable-storage)
- [Roadmap](#roadmap)
- [Research & References](#research--references)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Features

- **AST-Aware Chunking**: Uses tree-sitter to respect function and class boundaries, maintaining semantic integrity
- **Relationship Tracking**: Neo4j graph database tracks function calls, imports, inheritance, and dependencies across your codebase
- **External Dependency Mapping**: Automatically creates placeholder nodes for external functions (WordPress, npm packages, etc.)
- **Job-Based Indexing**: Background indexing with progress tracking for large codebases
- **On-Demand Container Spawning**: Index any repository on your system without manual mounting
- **Multi-Repository Search**: Index and search across multiple projects with a shared backend
- **Real-Time Updates**: File system watcher automatically re-indexes changed files (optional)
- **Local-First**: All processing happens locally using Ollama for embeddings (no data leaves your machine)
- **Polyglot Support**: Supports 10+ programming languages including TypeScript, Python, PHP, Go, Rust, Java, C++, and more
- **Incremental Indexing**: Merkle tree-based change detection with 80%+ cache hit rates
- **Production-Grade**: Uses Qdrant vector database for sub-10ms search latency and Neo4j for relationship queries
- **Dependency Knowledge Base**: Special collection for indexing WordPress plugins, Composer packages, and npm modules
- **Flexible Deployment**: Per-project or centralized server deployment options
- **MCP Integration**: Works with Claude Desktop, Cursor, VS Code, and other MCP-compatible tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  MCP Client (Claude Code, Claude Desktop, Cursor, etc.)         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ MCP Protocol (stdio)
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  MCP Server Container (codebase-mcp-server)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  FastMCP Server - Exposes MCP Tools:                      │  │
│  │  • index_repository (spawns indexer containers)           │  │
│  │  • search_code (semantic search across all repos)         │  │
│  │  • find_usages, find_dependencies (graph queries)         │  │
│  │  • detect_dependencies, index_dependencies                │  │
│  │  • get_job_status, list_indexing_jobs, cancel_job        │  │
│  │  • get_symbols, get_indexing_status, health_check        │  │
│  └──────────────┬───────────────────────────────────────────┘  │
│                 │                                                │
│                 │ Spawns via Docker Socket                       │
│                 ▼                                                │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  On-Demand Indexer Containers (ephemeral)           │       │
│  │  • Mounts any host directory                        │       │
│  │  • AST-aware chunking with tree-sitter              │       │
│  │  • Extracts relationships (CALLS, IMPORTS, etc.)    │       │
│  │  • Generates embeddings via Ollama                  │       │
│  │  • Updates shared Qdrant & Neo4j databases          │       │
│  │  • Reports progress back to MCP server              │       │
│  └──────────────────────┬──────────────────────────────┘       │
└─────────────────────────┼──────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────────┐
          │                                   │
   ┌──────▼──────┐              ┌────────────▼────────┐
   │   Qdrant    │              │      Neo4j          │
   │  Container  │              │    Container        │
   │  (Vectors)  │              │  (Relationships)    │
   └──────┬──────┘              └──────┬──────────────┘
          │                            │
   ┌──────▼────────────────────────────▼───────────┐
   │  Persistent Docker Volumes:                   │
   │  • qdrant_data (vector DB)                    │
   │  • neo4j_data (graph DB)                      │
   │  • index_data (merkle trees)                  │
   │  • cache_data (embeddings cache)              │
   └───────────────────────────────────────────────┘

   ┌────────────────────────┐
   │  Ollama (Host)         │
   │  Embedding Model       │
   └────────────────────────┘
```

### Key Architectural Features

- **Dual Database Architecture**: Qdrant for semantic vector search, Neo4j for relationship graph queries
- **Container Orchestration**: MCP server spawns lightweight indexer containers on-demand via Docker socket
- **Multi-Repository Support**: Each repository gets its own merkle tree state, but shares the vector & graph databases
- **Shared Backend**: All projects use the same Qdrant & Neo4j instances, enabling cross-repository search and relationship tracking
- **Job-Based Processing**: Background jobs with progress tracking for large codebases
- **Content-Addressable Caching**: Embeddings are cached by content hash, shared across all repositories
- **Relationship Extraction**: AST-based extraction of CALLS, IMPORTS, EXTENDS, and IMPLEMENTS relationships
- **External Dependency Tracking**: Automatic creation of placeholder nodes for unresolved function calls

## Quick Start

**See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.**

### Prerequisites

1. **Docker Desktop** (or Docker + Docker Compose)
2. **Ollama** running locally with an embedding model:
   ```bash
   # Install Ollama: https://ollama.ai

   # Recommended: Google's Gemma embedding model (best quality)
   ollama pull embeddinggemma:latest

   # Alternative: Nomic Embed (faster, smaller)
   ollama pull nomic-embed-text
   ```

### Two Deployment Options

#### Option A: Centralized Server (Recommended)

Best for: Indexing from the MCP server, querying across all repositories

```bash
# 1. Start the backend
cd codebase-contextifier-9000
docker-compose up -d

# 2. Configure Claude Desktop (see below)

# 3. Index any repository
# In Claude: "Index the repository at /Users/me/projects/my-app"
```

#### Option B: Per-Project Setup

Best for: Each project manages its own indexing

```bash
# 1. Start shared backend (once)
cd codebase-contextifier-9000
docker-compose up -d

# 2. Copy .mcp.json to each project
cp .mcp.json.template ~/projects/my-app/.mcp.json

# 3. Open project in Claude Code
cd ~/projects/my-app
claude-code .
```

See [MULTI_PROJECT_SETUP.md](MULTI_PROJECT_SETUP.md) for details.

### Claude Desktop Configuration

**For Centralized Server (Option A):**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "codebase-contextifier": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "codebase-mcp-server",
        "python",
        "-m",
        "src.server"
      ]
    }
  }
}
```

**For Per-Project Setup (Option B):**

Just copy `.mcp.json.template` to your project directory - no manual configuration needed!

### Usage

Once configured, you can use these tools in Claude Desktop or Claude Code:

**Index any repository on your system:**
```
Claude, index the repository at /Users/me/projects/my-app
```

The system spawns a container, indexes the repository in the background, and reports progress.

**Monitor indexing progress:**
```
Claude, show me the status of job abc123
```

**Search for code across all indexed repositories:**
```
Claude, search for "authentication logic" in the codebase
```

**Search with filters:**
```
Claude, search for "error handling" filtering by language=python and repo_name=my-api
```

**Extract symbols from a file:**
```
Claude, get all functions from /workspace/src/utils.py
```

**Find all usages of a function (graph query):**
```
Claude, find all places where authenticate_user is called
```

**Find dependencies of a function (graph query):**
```
Claude, show me all functions that processPayment depends on
```

**Detect and index external dependencies:**
```
Claude, detect available WordPress plugins in this project
Claude, index the woocommerce plugin into the knowledge base
```

**Check system status:**
```
Claude, show me the indexing status and list all jobs
```

## MCP Tools

### Indexing Tools

#### `index_repository`

Index a repository from any directory on your host machine by spawning a lightweight indexer container.

**Parameters:**
- `host_path` (string, required): Absolute path on host machine to repository (e.g., `/Users/me/projects/my-app`)
- `repo_name` (string, optional): Unique identifier for this repository (defaults to directory name)
- `incremental` (bool): Use incremental indexing to only re-index changed files (default: `true`)
- `exclude_patterns` (string, optional): Comma-separated glob patterns to exclude (e.g., `"node_modules/*,dist/*"`)

**Returns:**
```json
{
  "success": true,
  "job_id": "abc123def456",
  "repo_name": "my-app",
  "status": "queued",
  "message": "Background indexing started for 'my-app'"
}
```

**Example:**
```python
# Index a WordPress site, excluding plugins and uploads
await index_repository(
    host_path="/Users/me/sites/my-wordpress",
    repo_name="my-wordpress",
    exclude_patterns="wp-content/plugins/*,wp-content/uploads/*,wp-includes/*"
)
```

#### `get_job_status`

Get the status and progress of an indexing job.

**Parameters:**
- `job_id` (string, required): Job identifier returned from `index_repository`

**Returns:**
```json
{
  "success": true,
  "job_id": "abc123def456",
  "repo_name": "my-app",
  "repo_path": "/Users/me/projects/my-app",
  "status": "running",
  "created_at": 1698765432.123,
  "started_at": 1698765433.456,
  "elapsed_seconds": 45.2,
  "progress": {
    "current_file": 45,
    "total_files": 100,
    "progress_pct": 45.0,
    "current_file_path": "/workspace/src/api/auth.py",
    "chunks_indexed": 234,
    "failed_files_count": 2,
    "cache_hit_rate": "35.50%"
  }
}
```

**Status values:** `"queued"`, `"running"`, `"completed"`, `"failed"`, `"cancelled"`

#### `list_indexing_jobs`

List all indexing jobs (past and present).

**Returns:**
```json
{
  "success": true,
  "total_jobs": 3,
  "jobs": [
    {
      "job_id": "abc123",
      "repo_name": "my-api",
      "status": "completed",
      "progress": { "progress_pct": 100.0, ... }
    },
    {
      "job_id": "def456",
      "repo_name": "frontend",
      "status": "running",
      "progress": { "progress_pct": 67.5, ... }
    }
  ]
}
```

#### `cancel_indexing_job`

Cancel a running indexing job.

**Parameters:**
- `job_id` (string, required): Job identifier to cancel

**Returns:**
```json
{
  "success": true,
  "message": "Job abc123 cancelled successfully"
}
```

### Search Tools

#### `search_code`

Search code using natural language queries with semantic understanding across all indexed repositories.

**Parameters:**
- `query` (string, required): Natural language search query (e.g., "authentication logic", "error handling")
- `limit` (int): Maximum number of results to return (default: 10)
- `repo_name` (string, optional): Filter by repository name (searches all repos if not specified)
- `language` (string, optional): Filter by programming language (e.g., "python", "typescript", "php")
- `file_path_filter` (string, optional): Filter by file path pattern (e.g., "src/components")
- `chunk_type` (string, optional): Filter by chunk type (e.g., "function", "class", "method")

**Returns:**
```json
{
  "success": true,
  "query": "authentication logic",
  "total_results": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.8234,
      "repo_name": "backend-api",
      "file": "/workspace/src/auth/login.ts",
      "lines": "42-68",
      "language": "typescript",
      "type": "function",
      "context": "class:AuthService",
      "code": "async function authenticateUser(username, password) { ... }"
    }
  ]
}
```

#### `get_symbols`

Extract symbols from a file using AST parsing.

**Parameters:**
- `file_path` (string): Path to source file
- `symbol_type` (string, optional): Filter by type (e.g., `"function"`, `"class"`)

**Returns:**
```json
{
  "success": true,
  "file_path": "/workspace/src/utils.py",
  "total_symbols": 15,
  "symbols": [
    {
      "name": "format_date",
      "type": "function_definition",
      "start_line": 42,
      "end_line": 58,
      "context": "N/A",
      "language": "python"
    }
  ]
}
```

### Graph Query Tools

#### `find_usages`

Find all places where a function, class, or symbol is used across the codebase using the graph database.

**Parameters:**
- `symbol_name` (string, required): Name of the function/class to find usages for
- `repo_name` (string, optional): Filter by repository name

**Returns:**
```json
{
  "success": true,
  "symbol_name": "authenticate_user",
  "total_usages": 12,
  "usages": [
    {
      "caller": "LoginController.handleLogin",
      "caller_file": "/workspace/src/controllers/login.ts",
      "line_number": 42,
      "relationship_type": "CALLS"
    }
  ]
}
```

#### `find_dependencies`

Find all functions, classes, or imports that a symbol depends on using the graph database.

**Parameters:**
- `symbol_name` (string, required): Name of the function/class to analyze
- `repo_name` (string, optional): Filter by repository name

**Returns:**
```json
{
  "success": true,
  "symbol_name": "processPayment",
  "total_dependencies": 8,
  "dependencies": [
    {
      "target": "validateCard",
      "target_file": "/workspace/src/utils/validation.ts",
      "relationship_type": "CALLS",
      "is_external": false
    },
    {
      "target": "stripe.charges.create",
      "relationship_type": "CALLS",
      "is_external": true
    }
  ]
}
```

#### `query_graph`

Execute custom Cypher queries against the Neo4j graph database for advanced relationship analysis.

**Parameters:**
- `cypher_query` (string, required): Cypher query to execute
- `limit` (int, optional): Maximum number of results (default: 100)

**Returns:**
```json
{
  "success": true,
  "query": "MATCH (f:Function)-[:CALLS]->(ext:ExternalFunction) WHERE ext.name =~ 'wp_.*' RETURN f.name, ext.name",
  "results": [
    {"f.name": "enqueue_scripts", "ext.name": "wp_enqueue_script"},
    {"f.name": "setup_theme", "ext.name": "wp_register_nav_menu"}
  ],
  "total_results": 2
}
```

### Dependency Tools

#### `detect_dependencies`

Detect available dependencies in the workspace (WordPress plugins/themes, Composer packages, npm modules).

**Parameters:**
- `workspace_path` (string, optional): Path to workspace (defaults to current workspace)

**Returns:**
```json
{
  "success": true,
  "dependencies": {
    "wordpress_plugins": ["woocommerce", "advanced-custom-fields"],
    "wordpress_themes": ["twentytwentyfour"],
    "composer_packages": ["symfony/console", "guzzlehttp/guzzle"],
    "npm_packages": ["react", "typescript"]
  },
  "total_dependencies": 6
}
```

#### `index_dependencies`

Index specific dependencies into the knowledge base for better understanding of external APIs.

**Parameters:**
- `dependency_names` (array, required): List of dependency names to index (e.g., `["woocommerce", "react"]`)
- `workspace_id` (string, required): Unique identifier for the workspace/project
- `workspace_path` (string, optional): Path to workspace

**Returns:**
```json
{
  "success": true,
  "indexed_dependencies": ["woocommerce"],
  "total_chunks": 1247,
  "message": "Successfully indexed 1 dependencies with 1247 chunks"
}
```

#### `list_indexed_dependencies`

List all dependencies that have been indexed in the knowledge base.

**Returns:**
```json
{
  "success": true,
  "dependencies": [
    {
      "name": "woocommerce",
      "version": "8.5.0",
      "type": "wordpress_plugin",
      "workspaces": ["my-store", "test-site"],
      "chunks_count": 1247,
      "indexed_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_dependencies": 1
}
```

### Status Tools

#### `get_indexing_status`

Get statistics about the index, including vector DB, graph DB, and cache metrics.

**Returns:**
```json
{
  "success": true,
  "code_db": {
    "total_chunks": 2450,
    "vectors_count": 2450,
    "status": "green"
  },
  "knowledge_db": {
    "total_chunks": 1247,
    "indexed_dependencies": ["woocommerce"]
  },
  "graph_db": {
    "enabled": true,
    "total_nodes": 2230,
    "total_relationships": 4407,
    "node_types": {
      "Function": 1459,
      "ExternalFunction": 771
    }
  },
  "index": {
    "indexed_files": 150,
    "total_chunks": 2450
  },
  "cache": {
    "enabled": true,
    "cached_embeddings": 2450,
    "total_size_mb": 18.5
  }
}
```

#### `clear_index`

Clear the entire index (useful for fresh start).

#### `get_watcher_status`

Get status of the real-time file watcher.

**Returns:**
```json
{
  "success": true,
  "enabled": true,
  "running": true,
  "watch_path": "/workspace",
  "debounce_seconds": 2.0
}
```

#### `health_check`

Check health status of all components (Ollama, Qdrant, Neo4j).

## Supported Languages

| Language   | Extensions                      | Support Level |
|------------|---------------------------------|---------------|
| Python     | `.py`, `.pyw`                   | Full          |
| TypeScript | `.ts`, `.tsx`                   | Full          |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs`   | Full          |
| PHP        | `.php`, `.phtml`                | Full          |
| Go         | `.go`                           | Full          |
| Rust       | `.rs`                           | Full          |
| Java       | `.java`                         | Full          |
| C++        | `.cpp`, `.cc`, `.hpp`, `.hh`    | Full          |
| C          | `.c`, `.h`                      | Full          |
| C#         | `.cs`                           | Full          |

## Configuration

### Environment Variables

| Variable                   | Default                          | Description                           |
|----------------------------|----------------------------------|---------------------------------------|
| `CODEBASE_PATH`            | `./sample_codebase`              | Path to codebase to index             |
| `OLLAMA_HOST`              | `http://host.docker.internal:11434` | Ollama API endpoint                |
| `EMBEDDING_MODEL`          | `embeddinggemma:latest`          | Ollama embedding model to use         |
| `QDRANT_HOST`              | `qdrant`                         | Qdrant server hostname                |
| `QDRANT_PORT`              | `6333`                           | Qdrant server port                    |
| `ENABLE_GRAPH_DB`          | `false`                          | Enable Neo4j graph database           |
| `NEO4J_URI`                | `bolt://neo4j:7687`              | Neo4j connection URI                  |
| `NEO4J_USER`               | `neo4j`                          | Neo4j username                        |
| `NEO4J_PASSWORD`           | `password`                       | Neo4j password                        |
| `INDEX_PATH`               | `/index`                         | Path for index metadata               |
| `CACHE_PATH`               | `/cache`                         | Path for embedding cache              |
| `WORKSPACE_PATH`           | `/workspace`                     | Path to mounted codebase              |
| `MAX_CHUNK_SIZE`           | `2048`                           | Maximum chunk size in characters      |
| `BATCH_SIZE`               | `32`                             | Embedding batch size                  |
| `MAX_CONCURRENT_EMBEDDINGS`| `4`                              | Concurrent embedding requests         |
| `ENABLE_FILE_WATCHER`      | `true`                           | Enable real-time file watching        |
| `WATCHER_DEBOUNCE_SECONDS` | `2.0`                            | Delay before processing file changes  |
| `LOG_LEVEL`                | `INFO`                           | Logging level                         |

### Recommended Embedding Models

- `embeddinggemma:latest` (recommended - best quality)
- `nomic-embed-text` (good balance of speed and quality)
- `mxbai-embed-large` (higher accuracy, slower)
- `all-minilm` (fastest, lower accuracy)

## Performance

### Indexing Performance

- **Medium codebase** (5K-50K files): 2-10 minutes initial indexing
- **Incremental updates**: 10-60 seconds for typical changes
- **Cache hit rate**: 80-95% on subsequent runs
- **Embedding generation**: ~100-500 chunks/minute (depends on Ollama performance)

### Search Performance

- **Latency**: Sub-second semantic search
- **Throughput**: 10-50 queries/second
- **Accuracy**: 30% better than fixed-size chunking (from research)

## Troubleshooting

### "Ollama health check failed"

1. Make sure Ollama is running: `ollama serve`
2. Pull the embedding model: `ollama pull embeddinggemma:latest`
3. Check Docker can access host: Test with `curl http://host.docker.internal:11434`

### "Qdrant connection failed"

1. Check Qdrant container is running: `docker-compose ps`
2. Check Qdrant logs: `docker-compose logs qdrant`
3. Restart services: `docker-compose restart`

### "Graph database not enabled"

1. Set `ENABLE_GRAPH_DB=true` in your `.env` file or `.mcp.json`
2. Ensure Neo4j environment variables are configured: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
3. Check Neo4j container is running: `docker-compose ps`
4. Check Neo4j logs: `docker-compose logs neo4j`
5. Test Neo4j connection: `docker exec codebase-neo4j cypher-shell -u neo4j -p codebase123 "RETURN 1"`

### "No supported files found"

1. Check `CODEBASE_PATH` is correct in `.env`
2. Verify files have supported extensions
3. Check `.gitignore` isn't excluding too much

### Slow indexing

1. Reduce `BATCH_SIZE` if running low on RAM
2. Increase `MAX_CONCURRENT_EMBEDDINGS` if you have spare CPU
3. Use `incremental=true` for re-indexing

## Development

### Running Locally (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export QDRANT_HOST=localhost
export OLLAMA_HOST=http://localhost:11434
export INDEX_PATH=./index
export CACHE_PATH=./cache
export WORKSPACE_PATH=/path/to/your/codebase

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run server
python -m src.server
```

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Quality

```bash
# Format code
black src/

# Lint code
ruff src/
```

## Architecture Details

### AST-Aware Chunking

The system uses tree-sitter to parse code into Abstract Syntax Trees (ASTs), then extracts semantic chunks that respect:
- Function boundaries
- Class definitions
- Method boundaries
- Interface/trait definitions

This achieves **30% better accuracy** than fixed-size chunking according to research (arXiv:2506.15655).

### Incremental Indexing

Uses Merkle tree-based change detection:
1. Compute Blake3 hash of each file
2. Compare with previous state
3. Only re-index changed files
4. Update vector database incrementally

Typical cache hit rates: **80-95%**

### Content-Addressable Storage

Embeddings are cached using content hashing:
```
cache_key = blake3(model_name + file_content)
```

This enables:
- Team sharing of cached embeddings
- Fast re-indexing after git operations
- Deterministic caching across machines

## Roadmap

- [x] Real-time file system watcher for instant updates
- [x] Multi-repo search with shared backend
- [x] Job-based background indexing with progress tracking
- [x] On-demand container spawning for flexible repository indexing
- [x] **Neo4j integration for relationship tracking** - Track function calls, imports, inheritance, with external dependency placeholders
- [x] **Dependency knowledge base** - Index WordPress plugins, Composer packages, npm modules
- [ ] Reranking with cross-encoders for improved accuracy
- [ ] Fine-tuned embeddings for domain-specific code
- [ ] HTTP transport for remote MCP servers
- [ ] Web UI for search and visualization
- [ ] Graph-based code navigation UI (Neo4j Browser or custom visualization)

## Research & References

Based on cutting-edge research in semantic code search:

- **cAST** (arXiv:2506.15655): AST-aware chunking methodology
- **CodeRAG** (arXiv:2504.10046): Graph-augmented retrieval
- **Model Context Protocol**: Anthropic's standard for AI tool integration
- **Qdrant**: High-performance vector database
- **tree-sitter**: Incremental parsing library

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues, questions, or feature requests, please open a GitHub issue.
