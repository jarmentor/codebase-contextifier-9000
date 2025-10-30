#!/usr/bin/env python3
"""Standalone indexer script - indexes a repository and exits."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main indexer function."""
    try:
        # Import here to avoid issues if running from different context
        from src.indexer.ast_chunker import ASTChunker
        from src.indexer.embeddings import OllamaEmbeddings
        from src.indexer.merkle_tree import IncrementalIndexingSession, MerkleTreeIndexer
        from src.vector_db.qdrant_client import CodeVectorDB

        # Get configuration from environment
        workspace_path = os.getenv("WORKSPACE_PATH", "/workspace")
        repo_name = os.getenv("REPO_NAME")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        index_path = Path(os.getenv("INDEX_PATH", "/index"))
        cache_path = Path(os.getenv("CACHE_PATH", "/cache"))
        max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "2048"))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        max_concurrent = int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "4"))
        incremental = os.getenv("INCREMENTAL", "true").lower() == "true"

        # Auto-generate repo_name if not provided
        if not repo_name:
            repo_name = Path(workspace_path).name
            logger.info(f"Auto-generated repo_name: {repo_name}")

        logger.info(f"Starting indexer for repository: {repo_name}")
        logger.info(f"Workspace path: {workspace_path}")
        logger.info(f"Qdrant: {qdrant_host}:{qdrant_port}")
        logger.info(f"Ollama: {ollama_host}")
        logger.info(f"Incremental: {incremental}")

        # Initialize components
        logger.info("Initializing components...")

        vector_db = CodeVectorDB(host=qdrant_host, port=qdrant_port)

        embeddings = OllamaEmbeddings(
            host=ollama_host,
            model=embedding_model,
            cache_dir=cache_path,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
        )

        # Health check Ollama
        if not await embeddings.health_check():
            logger.error("Ollama health check failed!")
            sys.exit(1)

        merkle_indexer = MerkleTreeIndexer(index_path)
        chunker = ASTChunker(max_chunk_size=max_chunk_size)

        logger.info("Components initialized successfully")

        # Start indexing
        repo_path_obj = Path(workspace_path)

        if not repo_path_obj.exists():
            logger.error(f"Repository path does not exist: {workspace_path}")
            sys.exit(1)

        # Get all supported files
        logger.info("Scanning for files...")
        all_files = []
        for ext in chunker.registry.get_supported_extensions():
            all_files.extend(repo_path_obj.rglob(f"*{ext}"))

        all_file_paths = [str(f) for f in all_files if f.is_file()]

        if not all_file_paths:
            logger.error("No supported files found in repository")
            sys.exit(1)

        logger.info(f"Found {len(all_file_paths)} supported files")

        # Parse and apply gitignore patterns manually (gitignore_parser library is unreliable)
        gitignore_path = repo_path_obj / ".gitignore"
        gitignore_patterns = []

        if gitignore_path.exists():
            logger.info(f"Loading gitignore from: {gitignore_path}")
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        gitignore_patterns.append(line)

            logger.info(f"Loaded {len(gitignore_patterns)} gitignore patterns")

            # Filter files using our own matcher
            filtered_paths = []
            gitignore_sample = []

            for file_path in all_file_paths:
                rel_path_str = str(Path(file_path).relative_to(repo_path_obj))
                excluded = False

                for pattern in gitignore_patterns:
                    # Skip negation patterns for now (would need more complex logic)
                    if pattern.startswith('!'):
                        continue

                    # Handle different pattern types
                    if pattern.endswith('/*'):
                        # Directory contents: wp-content/plugins/*
                        dir_prefix = pattern[:-2]
                        if rel_path_str.startswith(dir_prefix + '/'):
                            excluded = True
                            break
                    elif pattern.endswith('/'):
                        # Directory itself: vendor/
                        dir_prefix = pattern[:-1]
                        if rel_path_str.startswith(dir_prefix + '/') or rel_path_str == dir_prefix:
                            excluded = True
                            break
                    elif '/' in pattern:
                        # Path-specific pattern
                        if rel_path_str.startswith(pattern + '/') or rel_path_str == pattern:
                            excluded = True
                            break
                    else:
                        # Bare name matches anywhere: vendor matches vendor/ and any/path/vendor/
                        parts = rel_path_str.split('/')
                        if pattern in parts or rel_path_str.startswith(pattern + '/'):
                            excluded = True
                            break

                if excluded:
                    if len(gitignore_sample) < 10:
                        gitignore_sample.append(rel_path_str)
                else:
                    filtered_paths.append(file_path)

            ignored_count = len(all_file_paths) - len(filtered_paths)
            logger.info(f"Filtered {ignored_count} gitignored files out of {len(all_file_paths)}")

            if gitignore_sample:
                logger.info(f"Sample gitignored files:")
                for sample in gitignore_sample[:5]:
                    logger.info(f"  {sample}")

            all_file_paths = filtered_paths

            if not all_file_paths:
                logger.error("No files remaining after gitignore filter")
                sys.exit(1)
        else:
            logger.info("No .gitignore found, indexing all files")

        # Apply additional exclude patterns if provided
        exclude_patterns_str = os.getenv("EXCLUDE_PATTERNS", "")
        # Always add vendor and node_modules as fallback if gitignore didn't catch them
        fallback_patterns = ["vendor/**", "node_modules/**"]

        if exclude_patterns_str:
            exclude_patterns = [p.strip() for p in exclude_patterns_str.split(",") if p.strip()]
        else:
            exclude_patterns = []

        # Merge with fallback patterns (dedupe)
        all_patterns = list(set(exclude_patterns + fallback_patterns))

        if all_patterns:
            logger.info(f"Applying exclude patterns: {all_patterns}")

            filtered_paths = []
            excluded_sample = []
            for file_path in all_file_paths:
                # Make path relative to workspace for pattern matching
                rel_path = Path(file_path).relative_to(repo_path_obj)
                rel_path_str = str(rel_path)
                excluded = False

                for pattern in all_patterns:
                    # Handle different pattern types
                    if pattern.endswith('/**'):
                        # Directory recursive: vendor/** matches vendor/anything/deep
                        dir_prefix = pattern[:-3]  # Remove /**
                        if rel_path_str.startswith(dir_prefix + '/') or rel_path_str == dir_prefix:
                            excluded = True
                            if len(excluded_sample) < 5:
                                excluded_sample.append(f"{rel_path_str} (matched {pattern})")
                            break
                    elif pattern.endswith('/*'):
                        # Directory single level: vendor/* matches vendor/file but not vendor/sub/file
                        dir_prefix = pattern[:-2]
                        # Check if path starts with dir/ and has no more slashes
                        if rel_path_str.startswith(dir_prefix + '/'):
                            remainder = rel_path_str[len(dir_prefix)+1:]
                            if '/' not in remainder:
                                excluded = True
                                if len(excluded_sample) < 5:
                                    excluded_sample.append(f"{rel_path_str} (matched {pattern})")
                                break
                    elif '*' in pattern:
                        # Wildcard pattern: use Path.match()
                        if rel_path.match(pattern):
                            excluded = True
                            if len(excluded_sample) < 5:
                                excluded_sample.append(f"{rel_path_str} (matched {pattern})")
                            break
                    else:
                        # Exact match or directory name
                        if rel_path_str == pattern or rel_path_str.startswith(pattern + '/'):
                            excluded = True
                            if len(excluded_sample) < 5:
                                excluded_sample.append(f"{rel_path_str} (matched {pattern})")
                            break

                if not excluded:
                    filtered_paths.append(file_path)

            excluded_count = len(all_file_paths) - len(filtered_paths)
            logger.info(f"Filtered {excluded_count} files matching exclude patterns")

            # Log sample exclusions
            if excluded_sample:
                logger.info(f"Sample exclusions:")
                for sample in excluded_sample:
                    logger.info(f"  {sample}")

            # Log what we kept
            logger.info(f"Remaining: {len(filtered_paths)} files to index")

            all_file_paths = filtered_paths

            if not all_file_paths:
                logger.error("No files remaining after exclude patterns filter")
                sys.exit(1)

        # Create repo-specific merkle indexer
        repo_index_path = merkle_indexer.index_path / repo_name
        repo_merkle_indexer = MerkleTreeIndexer(repo_index_path)

        files_to_index = all_file_paths
        files_to_remove = []

        if incremental:
            logger.info("Using incremental indexing mode")
            with IncrementalIndexingSession(repo_merkle_indexer) as session:
                files_to_index, files_to_remove = session.plan_incremental_update(
                    all_file_paths
                )

                logger.info(
                    f"Incremental plan: {len(files_to_index)} to index, "
                    f"{len(files_to_remove)} to remove, "
                    f"{len(all_file_paths) - len(files_to_index)} cached"
                )

                # Remove deleted files from vector DB
                for file_path in files_to_remove:
                    chunk_ids = session.remove_file(file_path)
                    if chunk_ids:
                        vector_db.delete_by_file_path(file_path)
                        logger.info(f"Removed chunks for deleted file: {file_path}")

                # Process changed/new files
                total_chunks = 0
                failed_files = []

                logger.info(f"Processing {len(files_to_index)} files...")

                for idx, file_path in enumerate(files_to_index, 1):
                    try:
                        logger.info(f"[{idx}/{len(files_to_index)}] {file_path}")

                        # Chunk the file
                        chunks = chunker.chunk_file(file_path, repo_root=str(repo_path_obj))

                        if not chunks:
                            logger.debug(f"No chunks extracted from {file_path}")
                            continue

                        logger.debug(f"Extracted {len(chunks)} chunks")

                        # Generate embeddings
                        chunk_contents = [chunk.content for chunk in chunks]
                        chunk_embeddings = await embeddings.generate_embeddings_batched(
                            chunk_contents
                        )

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
                        vector_db.upsert_chunks(chunk_dicts, chunk_embeddings)

                        # Update merkle tree
                        chunk_ids = [chunk.id for chunk in chunks]
                        session.update_file(file_path, chunk_ids)

                        total_chunks += len(chunks)
                        logger.info(f"✓ Indexed {len(chunks)} chunks from {file_path}")

                    except Exception as e:
                        logger.error(f"✗ Error indexing {file_path}: {e}", exc_info=True)
                        failed_files.append(file_path)

                cache_hit_rate = session.get_cache_hit_rate()

                logger.info("=" * 80)
                logger.info("Indexing Complete!")
                logger.info(f"Repository: {repo_name}")
                logger.info(f"Total files: {len(all_file_paths)}")
                logger.info(f"Indexed: {len(files_to_index)}")
                logger.info(f"Removed: {len(files_to_remove)}")
                logger.info(f"Cached: {len(all_file_paths) - len(files_to_index)}")
                logger.info(f"Total chunks: {total_chunks}")
                logger.info(f"Failed files: {len(failed_files)}")
                logger.info(f"Cache hit rate: {cache_hit_rate:.2f}%")
                logger.info("=" * 80)

                if failed_files:
                    logger.warning(f"Failed to index {len(failed_files)} files:")
                    for failed_file in failed_files:
                        logger.warning(f"  - {failed_file}")

                # Exit with success
                sys.exit(0)

        else:
            # Full indexing mode
            logger.info("Using full indexing mode")
            total_chunks = 0
            failed_files = []

            for idx, file_path in enumerate(all_file_paths, 1):
                try:
                    logger.info(f"[{idx}/{len(all_file_paths)}] {file_path}")

                    chunks = chunker.chunk_file(file_path, repo_root=str(repo_path_obj))

                    if not chunks:
                        continue

                    chunk_contents = [chunk.content for chunk in chunks]
                    chunk_embeddings = await embeddings.generate_embeddings_batched(
                        chunk_contents
                    )

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

                    vector_db.upsert_chunks(chunk_dicts, chunk_embeddings)
                    total_chunks += len(chunks)
                    logger.info(f"✓ Indexed {len(chunks)} chunks")

                except Exception as e:
                    logger.error(f"✗ Error indexing {file_path}: {e}")
                    failed_files.append(file_path)

            logger.info("=" * 80)
            logger.info("Indexing Complete!")
            logger.info(f"Repository: {repo_name}")
            logger.info(f"Total files: {len(all_file_paths)}")
            logger.info(f"Total chunks: {total_chunks}")
            logger.info(f"Failed files: {len(failed_files)}")
            logger.info("=" * 80)

            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error during indexing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
