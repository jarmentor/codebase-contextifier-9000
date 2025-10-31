"""MCP tool for indexing code repositories."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from ..indexer.ast_chunker import ASTChunker
from ..indexer.embeddings import OllamaEmbeddings
from ..indexer.job_manager import JobManager
from ..indexer.merkle_tree import IncrementalIndexingSession, MerkleTreeIndexer
from ..vector_db.qdrant_client import CodeVectorDB

logger = logging.getLogger(__name__)


class IndexingTool:
    """Tool for indexing code repositories with AST-aware chunking."""

    def __init__(
        self,
        vector_db: CodeVectorDB,
        embeddings: OllamaEmbeddings,
        merkle_indexer: MerkleTreeIndexer,
        chunker: ASTChunker,
        job_manager: Optional[JobManager] = None,
    ):
        """Initialize indexing tool.

        Args:
            vector_db: Vector database client
            embeddings: Embeddings generator
            merkle_indexer: Merkle tree indexer for incremental updates
            chunker: AST chunker
            job_manager: Optional job manager for background indexing
        """
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.merkle_indexer = merkle_indexer
        self.chunker = chunker
        self.job_manager = job_manager or JobManager()

    async def index_repository(
        self,
        repo_path: str,
        repo_name: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> dict:
        """Index a code repository.

        Args:
            repo_path: Path to the repository to index
            repo_name: Unique identifier for this repository (defaults to basename of repo_path)
            exclude_patterns: List of glob patterns to exclude
            incremental: Whether to use incremental indexing (only re-index changed files)

        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Starting repository indexing: {repo_path}")
        repo_path_obj = Path(repo_path)

        # Auto-generate repo_name from path if not provided
        if repo_name is None:
            repo_name = repo_path_obj.name
            logger.info(f"Auto-generated repo_name: {repo_name}")
        else:
            logger.info(f"Using provided repo_name: {repo_name}")

        if not repo_path_obj.exists():
            return {"success": False, "error": f"Repository path does not exist: {repo_path}"}

        try:
            # Get all supported files
            all_files = []
            for ext in self.chunker.registry.get_supported_extensions():
                all_files.extend(repo_path_obj.rglob(f"*{ext}"))

            # Filter files
            all_file_paths = [str(f) for f in all_files if f.is_file()]

            if not all_file_paths:
                return {
                    "success": False,
                    "error": "No supported files found in repository",
                }

            logger.info(f"Found {len(all_file_paths)} supported files")

            # Incremental indexing session
            files_to_index = all_file_paths
            files_to_remove = []

            # Create repo-specific merkle indexer
            repo_index_path = self.merkle_indexer.index_path / repo_name
            repo_merkle_indexer = MerkleTreeIndexer(repo_index_path)

            if incremental:
                with IncrementalIndexingSession(repo_merkle_indexer) as session:
                    files_to_index, files_to_remove = session.plan_incremental_update(
                        all_file_paths
                    )

                    # Remove deleted files from vector DB
                    for file_path in files_to_remove:
                        chunk_ids = session.remove_file(file_path)
                        if chunk_ids:
                            self.vector_db.delete_by_file_path(file_path)

                    # Process changed/new files
                    total_chunks = 0
                    failed_files = []

                    logger.info(f"Processing {len(files_to_index)} files for indexing...")

                    for idx, file_path in enumerate(files_to_index, 1):
                        try:
                            logger.info(f"[{idx}/{len(files_to_index)}] Processing {file_path}")

                            # Chunk the file
                            chunks = self.chunker.chunk_file(
                                file_path, repo_root=str(repo_path_obj)
                            )

                            if not chunks:
                                logger.debug(f"No chunks extracted from {file_path}")
                                continue

                            logger.debug(f"Extracted {len(chunks)} chunks from {file_path}")

                            # Generate embeddings
                            chunk_contents = [chunk.content for chunk in chunks]
                            embeddings = await self.embeddings.generate_embeddings_batched(
                                chunk_contents
                            )

                            logger.debug(f"Generated embeddings for {len(embeddings)} chunks")

                            # Prepare chunk metadata
                            chunk_dicts = [
                                {
                                    "id": chunk.id,
                                    "repo_name": repo_name,
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
                            self.vector_db.upsert_chunks(chunk_dicts, embeddings)

                            # Update merkle tree
                            chunk_ids = [chunk.id for chunk in chunks]
                            session.update_file(file_path, chunk_ids)

                            total_chunks += len(chunks)
                            logger.info(f"Successfully indexed {file_path}: {len(chunks)} chunks")

                        except Exception as e:
                            logger.error(f"Error indexing {file_path}: {e}", exc_info=True)
                            failed_files.append(file_path)

                    cache_hit_rate = session.get_cache_hit_rate()

                    logger.info(
                        f"Indexing complete! Total files: {len(all_file_paths)}, "
                        f"Indexed: {len(files_to_index)}, Removed: {len(files_to_remove)}, "
                        f"Cached: {len(all_file_paths) - len(files_to_index)}, "
                        f"Total chunks: {total_chunks}, Failed: {len(failed_files)}"
                    )

                    return {
                        "success": True,
                        "total_files": len(all_file_paths),
                        "indexed_files": len(files_to_index),
                        "removed_files": len(files_to_remove),
                        "cached_files": len(all_file_paths) - len(files_to_index),
                        "total_chunks": total_chunks,
                        "cache_hit_rate": f"{cache_hit_rate:.2f}%",
                        "failed_files": failed_files,
                    }

            else:
                # Full indexing (no incremental)
                total_chunks = 0
                failed_files = []

                logger.info(f"Processing {len(all_file_paths)} files for full indexing...")

                for idx, file_path in enumerate(all_file_paths, 1):
                    try:
                        logger.info(f"[{idx}/{len(all_file_paths)}] Processing {file_path}")

                        chunks = self.chunker.chunk_file(file_path, repo_root=str(repo_path_obj))

                        if not chunks:
                            logger.debug(f"No chunks extracted from {file_path}")
                            continue

                        logger.debug(f"Extracted {len(chunks)} chunks from {file_path}")

                        chunk_contents = [chunk.content for chunk in chunks]
                        embeddings = await self.embeddings.generate_embeddings_batched(
                            chunk_contents
                        )

                        logger.debug(f"Generated embeddings for {len(embeddings)} chunks")

                        chunk_dicts = [
                            {
                                "id": chunk.id,
                                "repo_name": repo_name,
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

                        self.vector_db.upsert_chunks(chunk_dicts, embeddings)
                        total_chunks += len(chunks)
                        logger.info(f"Successfully indexed {file_path}: {len(chunks)} chunks")

                    except Exception as e:
                        logger.error(f"Error indexing {file_path}: {e}", exc_info=True)
                        failed_files.append(file_path)

                logger.info(
                    f"Full indexing complete! Total files: {len(all_file_paths)}, "
                    f"Total chunks: {total_chunks}, Failed: {len(failed_files)}"
                )

                return {
                    "success": True,
                    "total_files": len(all_file_paths),
                    "indexed_files": len(all_file_paths),
                    "total_chunks": total_chunks,
                    "failed_files": failed_files,
                }

        except Exception as e:
            logger.error(f"Error during repository indexing: {e}")
            return {"success": False, "error": str(e)}

    async def index_repository_with_progress(
        self,
        job_id: str,
        repo_path: str,
        repo_name: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> dict:
        """Index repository with progress tracking for background jobs.

        Args:
            job_id: Job identifier for progress tracking
            repo_path: Path to the repository to index
            repo_name: Unique identifier for this repository
            exclude_patterns: List of glob patterns to exclude
            incremental: Whether to use incremental indexing

        Returns:
            Dictionary with indexing results
        """
        try:
            await self.job_manager.mark_started(job_id)

            logger.info(f"Starting repository indexing: {repo_path}")
            repo_path_obj = Path(repo_path)

            # Auto-generate repo_name from path if not provided
            if repo_name is None:
                repo_name = repo_path_obj.name
                logger.info(f"Auto-generated repo_name: {repo_name}")
            else:
                logger.info(f"Using provided repo_name: {repo_name}")

            if not repo_path_obj.exists():
                error_msg = f"Repository path does not exist: {repo_path}"
                await self.job_manager.mark_failed(job_id, error_msg)
                return {"success": False, "error": error_msg}

            # Get all supported files
            all_files = []
            for ext in self.chunker.registry.get_supported_extensions():
                all_files.extend(repo_path_obj.rglob(f"*{ext}"))

            # Filter files
            all_file_paths = [str(f) for f in all_files if f.is_file()]

            if not all_file_paths:
                error_msg = "No supported files found in repository"
                await self.job_manager.mark_failed(job_id, error_msg)
                return {"success": False, "error": error_msg}

            logger.info(f"Found {len(all_file_paths)} supported files")
            await self.job_manager.update_progress(
                job_id, total_files=len(all_file_paths)
            )

            # Create repo-specific merkle indexer
            repo_index_path = self.merkle_indexer.index_path / repo_name
            repo_merkle_indexer = MerkleTreeIndexer(repo_index_path)

            files_to_index = all_file_paths
            files_to_remove = []

            if incremental:
                with IncrementalIndexingSession(repo_merkle_indexer) as session:
                    files_to_index, files_to_remove = session.plan_incremental_update(
                        all_file_paths
                    )

                    # Remove deleted files from vector DB
                    for file_path in files_to_remove:
                        chunk_ids = session.remove_file(file_path)
                        if chunk_ids:
                            self.vector_db.delete_by_file_path(file_path)

                    # Process changed/new files
                    total_chunks = 0
                    failed_files = []

                    logger.info(f"Processing {len(files_to_index)} files for indexing...")

                    for idx, file_path in enumerate(files_to_index, 1):
                        # Update progress
                        await self.job_manager.update_progress(
                            job_id,
                            current_file=idx,
                            current_file_path=file_path,
                            chunks_indexed=total_chunks,
                            failed_files=failed_files,
                        )

                        try:
                            logger.info(f"[{idx}/{len(files_to_index)}] Processing {file_path}")

                            # Chunk the file
                            chunks = self.chunker.chunk_file(
                                file_path, repo_root=str(repo_path_obj)
                            )

                            if not chunks:
                                logger.debug(f"No chunks extracted from {file_path}")
                                continue

                            logger.debug(f"Extracted {len(chunks)} chunks from {file_path}")

                            # Generate embeddings
                            chunk_contents = [chunk.content for chunk in chunks]
                            embeddings = await self.embeddings.generate_embeddings_batched(
                                chunk_contents
                            )

                            logger.debug(f"Generated embeddings for {len(embeddings)} chunks")

                            # Prepare chunk metadata
                            chunk_dicts = [
                                {
                                    "id": chunk.id,
                                    "repo_name": repo_name,
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
                            self.vector_db.upsert_chunks(chunk_dicts, embeddings)

                            # Update merkle tree
                            chunk_ids = [chunk.id for chunk in chunks]
                            session.update_file(file_path, chunk_ids)

                            total_chunks += len(chunks)
                            logger.info(f"Successfully indexed {file_path}: {len(chunks)} chunks")

                        except Exception as e:
                            logger.error(f"Error indexing {file_path}: {e}", exc_info=True)
                            failed_files.append(file_path)

                    cache_hit_rate = session.get_cache_hit_rate()

                    # Final progress update
                    await self.job_manager.update_progress(
                        job_id,
                        current_file=len(files_to_index),
                        chunks_indexed=total_chunks,
                        failed_files=failed_files,
                        cache_hit_rate=cache_hit_rate,
                    )

                    logger.info(
                        f"Indexing complete! Total files: {len(all_file_paths)}, "
                        f"Indexed: {len(files_to_index)}, Removed: {len(files_to_remove)}, "
                        f"Cached: {len(all_file_paths) - len(files_to_index)}, "
                        f"Total chunks: {total_chunks}, Failed: {len(failed_files)}"
                    )

                    result = {
                        "success": True,
                        "total_files": len(all_file_paths),
                        "indexed_files": len(files_to_index),
                        "removed_files": len(files_to_remove),
                        "cached_files": len(all_file_paths) - len(files_to_index),
                        "total_chunks": total_chunks,
                        "cache_hit_rate": f"{cache_hit_rate:.2f}%",
                        "failed_files": failed_files,
                    }

                    await self.job_manager.mark_completed(job_id)
                    return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during repository indexing: {e}", exc_info=True)
            await self.job_manager.mark_failed(job_id, error_msg)
            return {"success": False, "error": error_msg}

    async def start_background_indexing(
        self,
        repo_path: str,
        repo_name: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> dict:
        """Start indexing in the background.

        Args:
            repo_path: Path to the repository to index
            repo_name: Unique identifier for this repository
            exclude_patterns: List of glob patterns to exclude
            incremental: Whether to use incremental indexing

        Returns:
            Dictionary with job information
        """
        # Auto-generate repo_name if not provided
        if repo_name is None:
            repo_name = Path(repo_path).name

        # Create job
        job = self.job_manager.create_job(repo_name, repo_path)

        # Start indexing task
        task = asyncio.create_task(
            self.index_repository_with_progress(
                job.job_id, repo_path, repo_name, exclude_patterns, incremental
            )
        )
        job.task = task

        logger.info(f"Started background indexing job {job.job_id} for {repo_name}")

        return {
            "success": True,
            "job_id": job.job_id,
            "repo_name": repo_name,
            "status": "queued",
            "message": f"Background indexing started for '{repo_name}'",
        }

    async def start_container_indexing(
        self,
        host_path: str,
        repo_name: Optional[str] = None,
        incremental: bool = True,
        exclude_patterns: Optional[str] = None,
    ) -> dict:
        """Start indexing using a lightweight Docker container.

        This spawns a dedicated indexer container that mounts the specified
        host directory and indexes it to Qdrant.

        Args:
            host_path: Absolute path on host machine to repository
            repo_name: Unique identifier for this repository (auto-generated from path)
            incremental: Whether to use incremental indexing
            exclude_patterns: Comma-separated glob patterns to exclude

        Returns:
            Dictionary with job information
        """
        # Auto-generate repo_name if not provided
        if not repo_name:
            repo_name = Path(host_path).name

        # Create job
        job = self.job_manager.create_job(repo_name, host_path)

        # Get configuration from environment
        import os
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        enable_graph_db = os.getenv("ENABLE_GRAPH_DB", "true").lower() == "true"
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://codebase-neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "codebase123")

        # Spawn indexer container
        success = await self.job_manager.spawn_indexer_container(
            job_id=job.job_id,
            host_path=host_path,
            repo_name=repo_name,
            ollama_host=ollama_host,
            embedding_model=embedding_model,
            incremental=incremental,
            exclude_patterns=exclude_patterns,
            enable_graph_db=enable_graph_db,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )

        if success:
            logger.info(f"Started container indexing job {job.job_id} for {repo_name}")
            return {
                "success": True,
                "job_id": job.job_id,
                "repo_name": repo_name,
                "host_path": host_path,
                "status": "queued",
                "message": f"Container indexing started for '{repo_name}'",
            }
        else:
            return {
                "success": False,
                "error": "Failed to spawn indexer container",
            }
