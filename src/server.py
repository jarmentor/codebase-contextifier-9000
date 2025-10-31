"""FastMCP server for semantic code search with AST-aware chunking."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .indexer.ast_chunker import ASTChunker
from .indexer.dependency_detector import DependencyDetector
from .indexer.embeddings import OllamaEmbeddings
from .indexer.file_watcher import CodebaseWatcher
from .indexer.knowledge_indexer import KnowledgeIndexer
from .indexer.merkle_tree import IncrementalIndexingSession, MerkleTreeIndexer
from .tools.index_tool import IndexingTool
from .tools.search_tool import SearchTool
from .tools.symbol_tool import SymbolTool
from .vector_db.qdrant_client import CodeVectorDB
from .vector_db.knowledge_db import KnowledgeVectorDB

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE", "/tmp/mcp-server.log")

# Create formatters and handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler (for Docker logs)
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_handler.setFormatter(formatter)

# File handler (for detailed logs)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(log_level)
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("codebase-contextifier-9000")

# Global components (initialized on startup)
vector_db: Optional[CodeVectorDB] = None
knowledge_db: Optional[KnowledgeVectorDB] = None
embeddings: Optional[OllamaEmbeddings] = None
merkle_indexer: Optional[MerkleTreeIndexer] = None
chunker: Optional[ASTChunker] = None
index_tool: Optional[IndexingTool] = None
search_tool: Optional[SearchTool] = None
symbol_tool: Optional[SymbolTool] = None
knowledge_indexer: Optional[KnowledgeIndexer] = None
file_watcher: Optional[CodebaseWatcher] = None
watcher_task: Optional[asyncio.Task] = None


def get_env_config():
    """Get configuration from environment variables."""
    return {
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        "index_path": Path(os.getenv("INDEX_PATH", "/index")),
        "cache_path": Path(os.getenv("CACHE_PATH", "/cache")),
        "workspace_path": os.getenv("WORKSPACE_PATH", "/workspace"),
        "max_chunk_size": int(os.getenv("MAX_CHUNK_SIZE", "2048")),
        "batch_size": int(os.getenv("BATCH_SIZE", "32")),
        "max_concurrent": int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "4")),
        "enable_watcher": os.getenv("ENABLE_FILE_WATCHER", "true").lower() == "true",
        "watcher_debounce": float(os.getenv("WATCHER_DEBOUNCE_SECONDS", "2.0")),
    }


async def handle_file_changes(modified_files: set, deleted_files: set) -> None:
    """Handle file changes detected by the watcher.

    Args:
        modified_files: Set of modified/created file paths
        deleted_files: Set of deleted file paths
    """
    global vector_db, embeddings, merkle_indexer, chunker

    if not all([vector_db, embeddings, merkle_indexer, chunker]):
        logger.warning("Components not initialized, skipping file change handling")
        return

    try:
        logger.info(
            f"File watcher detected changes: {len(modified_files)} modified, "
            f"{len(deleted_files)} deleted"
        )

        with IncrementalIndexingSession(merkle_indexer) as session:
            # Handle deleted files
            for file_path in deleted_files:
                chunk_ids = session.remove_file(file_path)
                if chunk_ids:
                    vector_db.delete_by_file_path(file_path)
                    logger.info(f"Removed chunks for deleted file: {file_path}")

            # Handle modified files
            total_chunks = 0
            for file_path in modified_files:
                try:
                    # Skip if file no longer exists
                    if not Path(file_path).exists():
                        continue

                    # Chunk the file
                    chunks = chunker.chunk_file(file_path)

                    if not chunks:
                        continue

                    # Generate embeddings
                    chunk_contents = [chunk.content for chunk in chunks]
                    chunk_embeddings = await embeddings.generate_embeddings_batched(
                        chunk_contents
                    )

                    # Prepare chunk metadata
                    chunk_dicts = [
                        {
                            "id": chunk.id,
                            "file_path": chunk.file_path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "language": chunk.language,
                            "chunk_type": chunk.chunk_type,
                            "content": chunk.content,
                            "context": chunk.context,
                            "repo_path": chunk.repo_path,
                        }
                        for chunk in chunks
                    ]

                    # Upsert to vector DB
                    vector_db.upsert_chunks(chunk_dicts, chunk_embeddings)

                    # Update merkle tree
                    chunk_ids = [chunk.id for chunk in chunks]
                    session.update_file(file_path, chunk_ids)

                    total_chunks += len(chunks)
                    logger.info(f"Re-indexed {file_path}: {len(chunks)} chunks")

                except Exception as e:
                    logger.error(f"Error re-indexing {file_path}: {e}")

            logger.info(
                f"File watcher update complete: {total_chunks} chunks from "
                f"{len(modified_files)} files"
            )

    except Exception as e:
        logger.error(f"Error handling file changes: {e}")


async def initialize_components():
    """Initialize all components on startup."""
    global vector_db, knowledge_db, embeddings, merkle_indexer, chunker
    global index_tool, search_tool, symbol_tool, knowledge_indexer, file_watcher, watcher_task

    config = get_env_config()
    logger.info("Initializing Codebase Contextifier 9000...")

    try:
        # Initialize vector databases
        logger.info(f"Connecting to Qdrant at {config['qdrant_host']}:{config['qdrant_port']}")
        vector_db = CodeVectorDB(
            host=config["qdrant_host"],
            port=config["qdrant_port"],
        )
        knowledge_db = KnowledgeVectorDB(
            host=config["qdrant_host"],
            port=config["qdrant_port"],
        )

        # Initialize embeddings
        logger.info(f"Connecting to Ollama at {config['ollama_host']}")
        embeddings = OllamaEmbeddings(
            host=config["ollama_host"],
            model=config["embedding_model"],
            cache_dir=config["cache_path"],
            batch_size=config["batch_size"],
            max_concurrent=config["max_concurrent"],
        )

        # Health check for Ollama
        if not await embeddings.health_check():
            logger.warning(
                f"Ollama health check failed. Make sure Ollama is running and "
                f"'{config['embedding_model']}' model is available."
            )

        # Initialize Merkle tree indexer
        logger.info(f"Initializing index at {config['index_path']}")
        merkle_indexer = MerkleTreeIndexer(config["index_path"])

        # Initialize AST chunker
        logger.info("Initializing AST chunker")
        chunker = ASTChunker(max_chunk_size=config["max_chunk_size"])

        # Initialize tools
        index_tool = IndexingTool(vector_db, embeddings, merkle_indexer, chunker)
        search_tool = SearchTool(vector_db, embeddings)
        symbol_tool = SymbolTool(chunker)
        knowledge_indexer = KnowledgeIndexer(knowledge_db, embeddings, chunker)

        # Initialize file watcher if enabled
        if config["enable_watcher"]:
            workspace_path = Path(config["workspace_path"])
            if workspace_path.exists():
                logger.info(
                    f"Initializing file watcher for {workspace_path} "
                    f"(debounce: {config['watcher_debounce']}s)"
                )
                file_watcher = CodebaseWatcher(
                    watch_path=str(workspace_path),
                    on_change_callback=handle_file_changes,
                    debounce_seconds=config["watcher_debounce"],
                    recursive=True,
                )
                logger.info("File watcher initialized successfully")
            else:
                logger.warning(
                    f"Workspace path does not exist: {workspace_path}. "
                    "File watcher will not be started."
                )
        else:
            logger.info("File watcher disabled (set ENABLE_FILE_WATCHER=true to enable)")

        logger.info("All components initialized successfully!")

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise


def start_file_watcher_sync():
    """Start the file watcher synchronously (observer thread)."""
    global file_watcher

    if file_watcher is not None:
        logger.info("Starting file watcher observer thread...")
        file_watcher.start()
        logger.info("File watcher observer is now running and monitoring for changes")
    else:
        logger.debug("File watcher not initialized, skipping startup")


async def start_file_watcher_async_task():
    """Start the async debounce processor for the file watcher."""
    global file_watcher, watcher_task

    if file_watcher is not None and file_watcher.is_running():
        logger.info("Starting file watcher debounce processor...")
        watcher_task = asyncio.create_task(file_watcher.start_debounce_processor())
        logger.info("File watcher debounce processor running")
    else:
        logger.debug("File watcher observer not running, skipping debounce processor")


def stop_file_watcher_sync():
    """Stop the file watcher synchronously."""
    global file_watcher, watcher_task

    if file_watcher is not None:
        logger.info("Stopping file watcher observer...")
        file_watcher.stop()
        logger.info("File watcher observer stopped")

        # Cancel the async task if it exists
        if watcher_task is not None and not watcher_task.done():
            logger.info("Cancelling file watcher debounce processor...")
            watcher_task.cancel()
    else:
        logger.debug("File watcher not running, skipping shutdown")




@mcp.tool()
async def search_code(
    query: str,
    limit: int = 10,
    repo_name: Optional[str] = None,
    language: Optional[str] = None,
    file_path_filter: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> dict:
    """Search for code using natural language queries with semantic understanding.

    Args:
        query: Natural language search query (e.g., "authentication logic", "error handling")
        limit: Maximum number of results to return (default: 10)
        repo_name: Filter by repository name (searches all repos if not specified)
        language: Filter by programming language (e.g., "python", "typescript", "php")
        file_path_filter: Filter by file path pattern (e.g., "src/components")
        chunk_type: Filter by chunk type (e.g., "function", "class", "method")

    Returns:
        Dictionary with search results including code snippets, file paths, and relevance scores
    """
    if not search_tool:
        return {"success": False, "error": "Server not initialized"}

    return await search_tool.search_code(query, limit, repo_name, language, file_path_filter, chunk_type)


@mcp.tool()
def get_symbols(
    file_path: str,
    symbol_type: Optional[str] = None,
) -> dict:
    """Extract symbols (functions, classes, methods) from a source file using AST parsing.

    Args:
        file_path: Path to the source file
        symbol_type: Filter by symbol type (e.g., "function", "class", "method")

    Returns:
        Dictionary with extracted symbols including names, types, and line numbers
    """
    if not symbol_tool:
        return {"success": False, "error": "Server not initialized"}

    return symbol_tool.get_symbols(file_path, symbol_type)


@mcp.tool()
def get_indexing_status() -> dict:
    """Get statistics about the current index state.

    Returns:
        Dictionary with index statistics including number of indexed files and chunks
    """
    if not search_tool or not merkle_indexer or not embeddings or not knowledge_db:
        return {"success": False, "error": "Server not initialized"}

    try:
        vector_stats = search_tool.get_stats()
        knowledge_stats = knowledge_db.get_stats()
        index_stats = merkle_indexer.get_stats()
        cache_stats = embeddings.get_cache_stats()

        return {
            "success": True,
            "code_db": vector_stats,
            "knowledge_db": knowledge_stats,
            "index": index_stats,
            "cache": cache_stats,
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def clear_index() -> dict:
    """Clear the entire index (vector database and metadata).

    Returns:
        Dictionary indicating success or failure
    """
    if not vector_db or not merkle_indexer:
        return {"success": False, "error": "Server not initialized"}

    try:
        vector_db.clear_collection()
        merkle_indexer.clear()

        return {
            "success": True,
            "message": "Index cleared successfully",
        }

    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_watcher_status() -> dict:
    """Get status of the file watcher.

    Returns:
        Dictionary with watcher status
    """
    if file_watcher is None:
        config = get_env_config()
        return {
            "success": True,
            "enabled": False,
            "running": False,
            "watcher_enabled_in_config": config["enable_watcher"],
            "workspace_path": config["workspace_path"],
            "message": "File watcher not initialized (check ENABLE_FILE_WATCHER and workspace path)",
        }

    is_running = file_watcher.is_running()
    has_task = watcher_task is not None and not watcher_task.done()

    status = {
        "success": True,
        "enabled": True,
        "running": is_running,
        "task_active": has_task,
        "watch_path": str(file_watcher.watch_path),
        "debounce_seconds": file_watcher.debounce_seconds,
        "recursive": file_watcher.recursive,
    }

    # Add pending changes info if available
    if hasattr(file_watcher, "event_handler"):
        status["pending_changes"] = {
            "modified_files": len(file_watcher.event_handler.modified_files),
            "deleted_files": len(file_watcher.event_handler.deleted_files),
            "time_since_last_change": (
                file_watcher.event_handler.time_since_last_change()
                if file_watcher.event_handler.last_change_time > 0
                else None
            ),
        }

    return status


@mcp.tool()
def health_check() -> dict:
    """Check health status of all components.

    Returns:
        Dictionary with health status of each component
    """
    health = {
        "server": True,
        "vector_db": False,
        "embeddings": False,
    }

    try:
        if vector_db:
            health["vector_db"] = vector_db.health_check()

        # Note: embeddings.health_check() is async, so we skip it here
        # It's checked during initialization

        return {
            "success": True,
            "components": health,
        }

    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return {"success": False, "error": str(e)}




@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Get the status and progress of an indexing job.

    Args:
        job_id: Job identifier returned from start_indexing_job

    Returns:
        Dictionary with job status and progress information
    """
    if not index_tool:
        return {"success": False, "error": "Server not initialized"}

    job = index_tool.job_manager.get_job(job_id)
    if not job:
        return {"success": False, "error": f"Job {job_id} not found"}

    return {
        "success": True,
        **index_tool.job_manager.get_status_dict(job),
    }


@mcp.tool()
def list_indexing_jobs() -> dict:
    """List all indexing jobs (past and present).

    Returns:
        Dictionary with list of all jobs and their statuses
    """
    if not index_tool:
        return {"success": False, "error": "Server not initialized"}

    jobs = index_tool.job_manager.list_jobs()
    job_list = [index_tool.job_manager.get_status_dict(job) for job in jobs]

    return {
        "success": True,
        "total_jobs": len(job_list),
        "jobs": job_list,
    }


@mcp.tool()
async def cancel_indexing_job(job_id: str) -> dict:
    """Cancel a running indexing job.

    Args:
        job_id: Job identifier to cancel

    Returns:
        Dictionary indicating success or failure
    """
    if not index_tool:
        return {"success": False, "error": "Server not initialized"}

    cancelled = await index_tool.job_manager.cancel_job(job_id)

    if cancelled:
        return {
            "success": True,
            "message": f"Job {job_id} cancelled successfully",
        }
    else:
        return {
            "success": False,
            "error": f"Job {job_id} not found or already completed",
        }


@mcp.tool()
async def index_repository(
    host_path: str,
    repo_name: Optional[str] = None,
    incremental: bool = True,
    exclude_patterns: Optional[str] = None,
) -> dict:
    """Index a repository from any directory on the host machine.

    The MCP server spawns a lightweight indexer container that mounts the specified
    directory and indexes it to Qdrant. This allows you to index any repo on your
    system without manually mounting it.

    By default, respects .gitignore files. You can specify additional patterns to exclude.

    Args:
        host_path: Absolute path on host machine to repository (e.g., "/Users/you/projects/my-app")
        repo_name: Unique identifier for this repository (defaults to directory name)
        incremental: Use incremental indexing (only re-index changed files, default: true)
        exclude_patterns: Comma-separated glob patterns to exclude (e.g., "wp-content/plugins/*,node_modules/*,vendor/*")

    Returns:
        Dictionary with job information including job_id for tracking progress

    Example:
        # Index a WordPress site, excluding plugins and uploads
        result = await index_repository(
            host_path="/Users/you/projects/my-site",
            repo_name="my-site",
            exclude_patterns="wp-content/plugins/*,wp-content/uploads/*,wp-includes/*"
        )
        job_id = result["job_id"]

        # Check progress
        status = await get_job_status(job_id)
    """
    if not index_tool:
        return {"success": False, "error": "Server not initialized"}

    return await index_tool.start_container_indexing(
        host_path=host_path,
        repo_name=repo_name,
        incremental=incremental,
        exclude_patterns=exclude_patterns,
    )


@mcp.tool()
def detect_dependencies(workspace_path: Optional[str] = None) -> dict:
    """Detect available dependencies in workspace (WordPress plugins, Composer, npm).

    Args:
        workspace_path: Optional workspace path (uses mounted workspace by default)

    Returns:
        Condensed summary with dependency names and versions
    """
    config = get_env_config()

    # Always use mounted workspace path (host paths don't exist inside container)
    workspace_path = config["workspace_path"]

    try:
        detector = DependencyDetector(workspace_path)
        dependencies = detector.detect_all()

        # Create condensed summaries (name + version, include slug for WP plugins)
        condensed = {}
        for dep_type, deps in dependencies.items():
            condensed_list = []
            for dep in deps:
                item = {
                    "name": dep.get("name", dep.get("slug", "unknown")),
                    "version": dep.get("version", "unknown"),
                    "type": dep.get("type", dep_type),
                }
                # For WordPress plugins/themes, include slug (what you use to index)
                if "slug" in dep and dep_type in ["wordpress_plugins", "wordpress_themes"]:
                    item["slug"] = dep["slug"]
                condensed_list.append(item)
            condensed[dep_type] = condensed_list

        return {
            "success": True,
            "workspace": workspace_path,
            **condensed,
            "summary": {
                "wordpress_plugins": len(condensed.get("wordpress_plugins", [])),
                "wordpress_themes": len(condensed.get("wordpress_themes", [])),
                "composer_packages": len(condensed.get("composer_packages", [])),
                "npm_packages": len(condensed.get("npm_packages", [])),
                "total": sum(len(deps) for deps in condensed.values()),
            },
        }

    except Exception as e:
        logger.error(f"Error detecting dependencies: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
def list_indexed_dependencies() -> dict:
    """List dependencies that have already been indexed in the knowledge base.

    Returns:
        List of indexed dependencies with chunk counts
    """
    if not knowledge_db:
        return {"success": False, "error": "Server not initialized"}

    try:
        dependencies = knowledge_db.list_dependencies()
        return {
            "success": True,
            "dependencies": dependencies,
            "total_dependencies": len(dependencies),
        }

    except Exception as e:
        logger.error(f"Error listing indexed dependencies: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def index_dependencies(
    dependency_names: list[str],
    workspace_id: str,
    workspace_path: Optional[str] = None,
) -> dict:
    """Index specific dependencies into knowledge base.

    If dependency version already exists, links to workspace instead of re-indexing.

    Args:
        dependency_names: List of dependency names/slugs to index
        workspace_id: Unique workspace identifier
        workspace_path: Optional workspace path (uses mounted workspace by default)

    Returns:
        Result with indexing details
    """
    if not knowledge_indexer:
        return {"success": False, "error": "Server not initialized"}

    config = get_env_config()

    # Always use mounted workspace path (host paths don't exist inside container)
    workspace_path = config["workspace_path"]

    try:
        result = await knowledge_indexer.index_multiple(
            dependency_names, workspace_path, workspace_id
        )
        return result

    except Exception as e:
        logger.error(f"Error indexing dependencies: {e}")
        return {"success": False, "error": str(e)}


def run_watcher_debounce_in_thread():
    """Run the file watcher's async debounce processor in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if file_watcher is not None and file_watcher.is_running():
            logger.info("Starting file watcher debounce processor in background thread...")
            loop.run_until_complete(file_watcher.start_debounce_processor())
    except Exception as e:
        logger.error(f"File watcher debounce processor error: {e}")
    finally:
        loop.close()


if __name__ == "__main__":
    import threading
    import atexit

    logger.info("Starting Codebase Contextifier 9000 MCP Server...")

    # Initialize components (runs in temporary event loop)
    asyncio.run(initialize_components())

    # Start file watcher observer (runs in watchdog thread)
    start_file_watcher_sync()

    # Start debounce processor in separate thread
    if file_watcher is not None:
        watcher_thread = threading.Thread(
            target=run_watcher_debounce_in_thread,
            daemon=True,
            name="FileWatcherDebounce"
        )
        watcher_thread.start()
        logger.info("File watcher fully initialized and running")

    # Register cleanup handler
    atexit.register(stop_file_watcher_sync)

    logger.info("Server ready!")

    # Run the MCP server (blocks until shutdown)
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
