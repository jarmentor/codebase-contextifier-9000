"""Knowledge base indexer for dependencies (plugins, packages)."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ast_chunker import ASTChunker
from .dependency_detector import DependencyDetector
from .embeddings import OllamaEmbeddings
from ..vector_db.knowledge_db import KnowledgeVectorDB

logger = logging.getLogger(__name__)


class KnowledgeIndexer:
    """Index dependencies into knowledge base using AST chunker."""

    def __init__(
        self,
        knowledge_db: KnowledgeVectorDB,
        embeddings: OllamaEmbeddings,
        ast_chunker: ASTChunker,
    ):
        """Initialize knowledge indexer.

        Args:
            knowledge_db: Vector DB for knowledge base
            embeddings: Embedding generator
            ast_chunker: AST chunker (reused from code indexing!)
        """
        self.knowledge_db = knowledge_db
        self.embeddings = embeddings
        self.ast_chunker = ast_chunker

    async def index_dependency(
        self,
        dependency: Dict[str, Any],
        workspace_path: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a single dependency into the knowledge base.

        If the dependency version is already indexed, just links the workspace.
        This enables shared dependencies across multiple projects.

        Args:
            dependency: Dependency metadata from DependencyDetector
            workspace_path: Workspace path for context
            workspace_id: Unique workspace identifier (defaults to workspace_path)

        Returns:
            Result dict with chunks_indexed, cache_hits, etc.
        """
        dep_name = dependency.get("name", "unknown")
        dep_type = dependency.get("type", "unknown")
        dep_version = dependency.get("version", "unknown")
        dep_path = Path(dependency.get("path", ""))

        if workspace_id is None:
            workspace_id = workspace_path

        # Check if this exact version is already indexed
        if self.knowledge_db.dependency_exists(dep_name, dep_version):
            logger.info(
                f"Dependency {dep_name}:{dep_version} already indexed, linking workspace '{workspace_id}'"
            )

            chunks_updated = self.knowledge_db.add_workspace_to_dependency(
                dep_name, dep_version, workspace_id
            )

            return {
                "success": True,
                "dependency": dep_name,
                "version": dep_version,
                "type": dep_type,
                "chunks_indexed": 0,
                "chunks_linked": chunks_updated,
                "already_existed": True,
                "workspace": workspace_id,
                "message": f"Linked existing dependency to workspace '{workspace_id}'",
            }

        logger.info(f"Indexing dependency: {dep_name} v{dep_version} ({dep_type})")

        if not dep_path.exists():
            return {
                "success": False,
                "error": f"Dependency path not found: {dep_path}",
            }

        # Use AST chunker to extract code chunks
        all_chunks = []
        file_count = 0

        # Find all supported code files in the dependency
        for file_path in dep_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip minified files (no semantic value for search)
            if ".min." in file_path.name or file_path.name.endswith(".min"):
                logger.debug(f"Skipping minified file: {file_path}")
                continue

            # Check if file is supported by AST chunker
            if not self.ast_chunker.registry.is_supported_file(str(file_path)):
                continue

            # Skip test files, build artifacts, vendor directories, etc.
            path_parts = file_path.parts
            skip_patterns = [
                "test", "tests", "spec", "__tests__",  # Tests
                "dist", "build", "bundle", "public",   # Build output
                "node_modules", "vendor", "bower_components",  # Dependencies
                ".git", ".svn", ".hg",  # VCS
                "min", "minified",  # Minified dirs
                "assets/dist", "assets/build",  # Common WordPress patterns
            ]
            if any(skip in path_parts for skip in skip_patterns):
                continue

            # Skip very large files (likely minified or generated)
            if file_path.stat().st_size > 500_000:  # 500KB limit
                logger.debug(f"Skipping large file ({file_path.stat().st_size} bytes): {file_path}")
                continue

            try:
                # Chunk the file using AST
                chunks = self.ast_chunker.chunk_file(str(file_path))

                if chunks:
                    file_count += 1
                    all_chunks.extend(chunks)
                    logger.debug(
                        f"Chunked {file_path.name}: {len(chunks)} chunks, "
                        f"{file_path.stat().st_size} bytes"
                    )

            except Exception as e:
                logger.debug(f"Skipping file {file_path}: {e}")

        if not all_chunks:
            logger.warning(f"No chunks extracted from {dep_name}")
            return {
                "success": True,
                "dependency": dep_name,
                "chunks_indexed": 0,
                "message": "No indexable code found",
            }

        logger.info(f"Extracted {len(all_chunks)} chunks from {file_count} files in {dep_name}")

        # Enrich chunks with dependency metadata
        enriched_chunks = []
        for chunk in all_chunks:
            chunk_dict = {
                # Core AST metadata
                "id": chunk.id,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "context": chunk.context,
                "repo_path": workspace_path,
                # Knowledge-specific metadata
                "knowledge_type": "local_dependency",
                "dependency_type": dep_type,
                "dependency_name": dep_name,
                "dependency_version": dep_version,
                "dependency_path": str(dep_path),
                "framework": self._detect_framework(dep_type, dep_name),
                "workspaces": [workspace_id],
            }
            enriched_chunks.append(chunk_dict)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(enriched_chunks)} chunks from {dep_name}")
        chunk_contents = [chunk["content"] for chunk in enriched_chunks]
        embeddings_list = await self.embeddings.generate_embeddings_batched(chunk_contents)

        # Filter out any chunks where embedding generation failed (None values)
        valid_chunks = []
        valid_embeddings = []
        failed_count = 0
        failed_details = []

        for idx, (chunk, embedding) in enumerate(zip(enriched_chunks, embeddings_list)):
            if embedding is not None:
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
            else:
                failed_count += 1
                # Log details about failed chunks for debugging
                failed_info = {
                    "file": chunk["file_path"],
                    "chunk_type": chunk["chunk_type"],
                    "lines": f"{chunk['start_line']}-{chunk['end_line']}",
                    "content_length": len(chunk["content"]),
                }
                failed_details.append(failed_info)
                logger.warning(
                    f"Failed embedding for chunk {idx}: {failed_info['file']}:{failed_info['lines']} "
                    f"({failed_info['chunk_type']}, {failed_info['content_length']} chars)"
                )

        if failed_count > 0:
            logger.warning(
                f"Skipping {failed_count} chunks due to embedding failures for {dep_name}"
            )
            # Log summary of failed files
            failed_files = list(set([f["file"] for f in failed_details]))
            if failed_files:
                logger.warning(f"Files with failed chunks: {failed_files[:5]}")  # Show first 5

        # Upsert valid chunks to knowledge base
        if valid_chunks:
            self.knowledge_db.upsert_chunks(valid_chunks, valid_embeddings)
            logger.info(
                f"Successfully indexed {dep_name}: {len(valid_chunks)} chunks "
                f"({failed_count} failed)"
            )
        else:
            logger.error(f"No chunks could be indexed for {dep_name} - all embeddings failed")

        return {
            "success": True,
            "dependency": dep_name,
            "version": dep_version,
            "type": dep_type,
            "chunks_indexed": len(valid_chunks),
            "chunks_failed": failed_count,
            "files_processed": file_count,
            "already_existed": False,
            "workspace": workspace_id,
        }

    async def index_multiple(
        self,
        dependency_names: List[str],
        workspace_path: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index multiple dependencies by name.

        Args:
            dependency_names: List of dependency names/slugs to index
            workspace_path: Workspace path
            workspace_id: Unique workspace identifier (defaults to workspace_path)

        Returns:
            Result dict with success status and details for each dependency
        """
        detector = DependencyDetector(workspace_path)
        results = []
        total_chunks = 0
        total_linked = 0
        total_new = 0

        for dep_name in dependency_names:
            # Find the dependency
            dependency = detector.get_dependency_by_name(dep_name)

            if not dependency:
                logger.warning(f"Dependency not found: {dep_name}")
                results.append({
                    "dependency": dep_name,
                    "success": False,
                    "error": "Dependency not found in workspace",
                })
                continue

            # Index it (or link if exists)
            result = await self.index_dependency(dependency, workspace_path, workspace_id)
            results.append(result)

            if result.get("success"):
                total_chunks += result.get("chunks_indexed", 0)
                if result.get("already_existed"):
                    total_linked += 1
                else:
                    total_new += 1

        return {
            "success": True,
            "dependencies_processed": len([r for r in results if r.get("success")]),
            "new_dependencies_indexed": total_new,
            "existing_dependencies_linked": total_linked,
            "total_chunks_indexed": total_chunks,
            "details": results,
        }

    def _detect_framework(self, dep_type: str, dep_name: str) -> str:
        """Detect framework from dependency type/name.

        Args:
            dep_type: Dependency type
            dep_name: Dependency name

        Returns:
            Framework name or empty string
        """
        if dep_type in ["wordpress_plugin", "wordpress_theme"]:
            return "wordpress"

        # Detect from Composer package names
        if "symfony" in dep_name.lower():
            return "symfony"
        elif "laravel" in dep_name.lower():
            return "laravel"
        elif "drupal" in dep_name.lower():
            return "drupal"

        return ""
