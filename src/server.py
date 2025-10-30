"""FastMCP server for semantic code search with AST-aware chunking."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .indexer.ast_chunker import ASTChunker
from .indexer.embeddings import OllamaEmbeddings
from .indexer.file_watcher import CodebaseWatcher
from .indexer.merkle_tree import IncrementalIndexingSession, MerkleTreeIndexer
from .tools.index_tool import IndexingTool
from .tools.search_tool import SearchTool
from .tools.symbol_tool import SymbolTool
from .vector_db.qdrant_client import CodeVectorDB

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("codebase-contextifier-9000")

# Global components (initialized on startup)
vector_db: Optional[CodeVectorDB] = None
embeddings: Optional[OllamaEmbeddings] = None
merkle_indexer: Optional[MerkleTreeIndexer] = None
chunker: Optional[ASTChunker] = None
index_tool: Optional[IndexingTool] = None
search_tool: Optional[SearchTool] = None
symbol_tool: Optional[SymbolTool] = None
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
    global vector_db, embeddings, merkle_indexer, chunker
    global index_tool, search_tool, symbol_tool, file_watcher, watcher_task

    config = get_env_config()
    logger.info("Initializing Codebase Contextifier 9000...")

    try:
        # Initialize vector database
        logger.info(f"Connecting to Qdrant at {config['qdrant_host']}:{config['qdrant_port']}")
        vector_db = CodeVectorDB(
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
                logger.info("File watcher initialized (will start after server ready)")
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


@mcp.tool
async def index_repository(
    repo_path: str = "/workspace",
    exclude_patterns: Optional[list[str]] = None,
    incremental: bool = True,
) -> dict:
    """Index a code repository with AST-aware chunking and embeddings.

    Args:
        repo_path: Path to the repository to index (default: /workspace)
        exclude_patterns: List of glob patterns to exclude (e.g., ["*.test.js", "node_modules/*"])
        incremental: Use incremental indexing (only re-index changed files)

    Returns:
        Dictionary with indexing results including number of files and chunks indexed
    """
    if not index_tool:
        return {"success": False, "error": "Server not initialized"}

    return await index_tool.index_repository(repo_path, exclude_patterns, incremental)


@mcp.tool
async def search_code(
    query: str,
    limit: int = 10,
    language: Optional[str] = None,
    file_path_filter: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> dict:
    """Search for code using natural language queries with semantic understanding.

    Args:
        query: Natural language search query (e.g., "authentication logic", "error handling")
        limit: Maximum number of results to return (default: 10)
        language: Filter by programming language (e.g., "python", "typescript", "php")
        file_path_filter: Filter by file path pattern (e.g., "src/components")
        chunk_type: Filter by chunk type (e.g., "function", "class", "method")

    Returns:
        Dictionary with search results including code snippets, file paths, and relevance scores
    """
    if not search_tool:
        return {"success": False, "error": "Server not initialized"}

    return await search_tool.search_code(query, limit, language, file_path_filter, chunk_type)


@mcp.tool
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


@mcp.tool
def get_indexing_status() -> dict:
    """Get statistics about the current index state.

    Returns:
        Dictionary with index statistics including number of indexed files and chunks
    """
    if not search_tool or not merkle_indexer or not embeddings:
        return {"success": False, "error": "Server not initialized"}

    try:
        vector_stats = search_tool.get_stats()
        index_stats = merkle_indexer.get_stats()
        cache_stats = embeddings.get_cache_stats()

        return {
            "success": True,
            "vector_db": vector_stats,
            "index": index_stats,
            "cache": cache_stats,
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {"success": False, "error": str(e)}


@mcp.tool
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


@mcp.tool
def get_watcher_status() -> dict:
    """Get status of the file watcher.

    Returns:
        Dictionary with watcher status
    """
    if file_watcher is None:
        return {
            "success": True,
            "enabled": False,
            "running": False,
            "message": "File watcher not initialized (set ENABLE_FILE_WATCHER=true)",
        }

    return {
        "success": True,
        "enabled": True,
        "running": file_watcher.is_running(),
        "watch_path": str(file_watcher.watch_path),
        "debounce_seconds": file_watcher.debounce_seconds,
    }


@mcp.tool
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


# Server lifecycle
@mcp.on_startup
async def startup():
    """Run initialization on server startup."""
    global watcher_task, file_watcher

    await initialize_components()

    # Start file watcher if initialized
    if file_watcher is not None:
        logger.info("Starting file watcher...")
        watcher_task = asyncio.create_task(file_watcher.run_async())
        logger.info("File watcher running in background")


if __name__ == "__main__":
    # Run the server
    logger.info("Starting Codebase Contextifier 9000 MCP Server...")
    mcp.run()
