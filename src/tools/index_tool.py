"""MCP tool for indexing code repositories."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..indexer.ast_chunker import ASTChunker
from ..indexer.embeddings import OllamaEmbeddings
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
    ):
        """Initialize indexing tool.

        Args:
            vector_db: Vector database client
            embeddings: Embeddings generator
            merkle_indexer: Merkle tree indexer for incremental updates
            chunker: AST chunker
        """
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.merkle_indexer = merkle_indexer
        self.chunker = chunker

    async def index_repository(
        self,
        repo_path: str,
        exclude_patterns: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> dict:
        """Index a code repository.

        Args:
            repo_path: Path to the repository to index
            exclude_patterns: List of glob patterns to exclude
            incremental: Whether to use incremental indexing (only re-index changed files)

        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Starting repository indexing: {repo_path}")
        repo_path_obj = Path(repo_path)

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

            if incremental:
                with IncrementalIndexingSession(self.merkle_indexer) as session:
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

                    for file_path in tqdm(files_to_index, desc="Indexing files"):
                        try:
                            # Chunk the file
                            chunks = self.chunker.chunk_file(
                                file_path, repo_root=str(repo_path_obj)
                            )

                            if not chunks:
                                continue

                            # Generate embeddings
                            chunk_contents = [chunk.content for chunk in chunks]
                            embeddings = await self.embeddings.generate_embeddings_batched(
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
                            self.vector_db.upsert_chunks(chunk_dicts, embeddings)

                            # Update merkle tree
                            chunk_ids = [chunk.id for chunk in chunks]
                            session.update_file(file_path, chunk_ids)

                            total_chunks += len(chunks)

                        except Exception as e:
                            logger.error(f"Error indexing {file_path}: {e}")
                            failed_files.append(file_path)

                    cache_hit_rate = session.get_cache_hit_rate()

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

                for file_path in tqdm(all_file_paths, desc="Indexing files"):
                    try:
                        chunks = self.chunker.chunk_file(file_path, repo_root=str(repo_path_obj))

                        if not chunks:
                            continue

                        chunk_contents = [chunk.content for chunk in chunks]
                        embeddings = await self.embeddings.generate_embeddings_batched(
                            chunk_contents
                        )

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

                        self.vector_db.upsert_chunks(chunk_dicts, embeddings)
                        total_chunks += len(chunks)

                    except Exception as e:
                        logger.error(f"Error indexing {file_path}: {e}")
                        failed_files.append(file_path)

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
