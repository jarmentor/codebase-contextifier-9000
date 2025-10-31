"""Qdrant vector database client for dependency/knowledge embeddings."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)


def hex_to_uuid(hex_str: str) -> str:
    """Convert a hexadecimal string to a UUID format.

    Args:
        hex_str: Hexadecimal string (16 or 32 characters)

    Returns:
        UUID string
    """
    # Pad to 32 characters if needed
    hex_str = hex_str.ljust(32, '0')
    # Insert hyphens to make it a valid UUID format
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


class KnowledgeVectorDB:
    """Wrapper for Qdrant vector database operations for dependency/knowledge base.

    This collection stores embeddings for:
    - Local dependencies (WordPress plugins, Composer packages, npm packages)
    - External documentation (optional)

    Separate from code_embeddings collection to enable targeted searches.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "knowledge_base",
        vector_size: int = 768,  # Default for nomic-embed-text
    ):
        """Initialize Qdrant client for knowledge base.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_size: Dimension of embedding vectors
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating knowledge collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Knowledge collection {self.collection_name} created successfully")
            else:
                logger.info(f"Knowledge collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring knowledge collection: {e}")
            raise

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Insert or update knowledge chunks with their embeddings.

        Args:
            chunks: List of chunk metadata dictionaries containing:
                Core fields (from AST chunker):
                - id: unique identifier (Blake3 hash)
                - file_path: path to source file
                - start_line: starting line number
                - end_line: ending line number
                - language: programming language
                - chunk_type: function, class, method, etc.
                - content: actual code content
                - context: surrounding context (class name, etc.)

                Knowledge-specific fields:
                - knowledge_type: "local_dependency" or "external_docs"
                - dependency_type: "wordpress_plugin", "composer_package", "npm_package", etc.
                - dependency_name: name of the dependency (e.g., "woocommerce")
                - dependency_version: version string
                - dependency_path: local path to dependency
                - framework: "wordpress", "laravel", "symfony", etc. (optional)

            embeddings: List of embedding vectors corresponding to chunks

        Returns:
            Number of points upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Convert Blake3 hash to UUID format
            point_id = hex_to_uuid(chunk["id"])

            # Build payload with all metadata
            payload = {
                # Core chunk metadata (from AST)
                "chunk_id": chunk["id"],
                "file_path": chunk.get("file_path", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "language": chunk.get("language", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "content": chunk.get("content", ""),
                "context": chunk.get("context", ""),
                "repo_path": chunk.get("repo_path", ""),

                # Knowledge-specific metadata
                "knowledge_type": chunk.get("knowledge_type", "local_dependency"),
                "dependency_type": chunk.get("dependency_type", ""),
                "dependency_name": chunk.get("dependency_name", ""),
                "dependency_version": chunk.get("dependency_version", ""),
                "dependency_path": chunk.get("dependency_path", ""),
                "framework": chunk.get("framework", ""),

                # Optional fields
                "source_url": chunk.get("source_url", ""),
                "category": chunk.get("category", ""),
                "tags": chunk.get("tags", []),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert to Qdrant in batches to avoid timeouts
        batch_size = 500  # Upsert 500 points at a time
        total_upserted = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            total_upserted += len(batch)
            if len(points) > batch_size:
                logger.info(
                    f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} "
                    f"({total_upserted}/{len(points)} chunks)"
                )

        logger.info(f"Upserted {total_upserted} knowledge chunks to Qdrant")
        return total_upserted

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        dependency_name: Optional[str] = None,
        dependency_type: Optional[str] = None,
        framework: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar knowledge chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            dependency_name: Filter by dependency name (e.g., "woocommerce")
            dependency_type: Filter by dependency type (e.g., "wordpress_plugin")
            framework: Filter by framework (e.g., "wordpress")
            language: Filter by programming language

        Returns:
            List of matching chunks with metadata and scores
        """
        # Build filter conditions
        must_conditions = []

        if dependency_name:
            must_conditions.append(
                models.FieldCondition(
                    key="dependency_name",
                    match=models.MatchValue(value=dependency_name),
                )
            )

        if dependency_type:
            must_conditions.append(
                models.FieldCondition(
                    key="dependency_type",
                    match=models.MatchValue(value=dependency_type),
                )
            )

        if framework:
            must_conditions.append(
                models.FieldCondition(
                    key="framework",
                    match=models.MatchValue(value=framework),
                )
            )

        if language:
            must_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        # Build search filter
        search_filter = None
        if must_conditions:
            search_filter = models.Filter(must=must_conditions)

        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=search_filter,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.score,
                "chunk_id": result.payload.get("chunk_id"),
                "file_path": result.payload.get("file_path"),
                "start_line": result.payload.get("start_line"),
                "end_line": result.payload.get("end_line"),
                "language": result.payload.get("language"),
                "chunk_type": result.payload.get("chunk_type"),
                "content": result.payload.get("content"),
                "context": result.payload.get("context"),
                "dependency_name": result.payload.get("dependency_name"),
                "dependency_type": result.payload.get("dependency_type"),
                "dependency_version": result.payload.get("dependency_version"),
                "framework": result.payload.get("framework"),
            })

        return formatted_results

    def delete_by_dependency(self, dependency_name: str) -> int:
        """Delete all chunks for a specific dependency.

        Args:
            dependency_name: Name of the dependency to delete

        Returns:
            Number of chunks deleted
        """
        # Delete points matching the dependency name
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="dependency_name",
                            match=models.MatchValue(value=dependency_name),
                        )
                    ]
                )
            ),
        )

        logger.info(f"Deleted chunks for dependency: {dependency_name}")
        return 0  # Qdrant doesn't return count for delete operations

    def dependency_exists(self, dependency_name: str, version: str) -> bool:
        """Check if a specific dependency version is already indexed.

        Args:
            dependency_name: Name of the dependency
            version: Version string

        Returns:
            True if this exact version is indexed
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="dependency_name",
                            match=models.MatchValue(value=dependency_name),
                        ),
                        models.FieldCondition(
                            key="dependency_version",
                            match=models.MatchValue(value=version),
                        ),
                    ]
                ),
                with_payload=False,
                with_vectors=False,
            )

            return len(results[0]) > 0

        except Exception as e:
            logger.error(f"Error checking dependency existence: {e}")
            return False

    def add_workspace_to_dependency(
        self, dependency_name: str, version: str, workspace: str
    ) -> int:
        """Link a workspace to an existing dependency.

        Updates all chunks of the dependency to add workspace to their list.

        Args:
            dependency_name: Name of the dependency
            version: Version string
            workspace: Workspace identifier to add

        Returns:
            Number of chunks updated
        """
        try:
            # Get all points for this dependency
            offset = None
            updated_count = 0

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="dependency_name",
                                match=models.MatchValue(value=dependency_name),
                            ),
                            models.FieldCondition(
                                key="dependency_version",
                                match=models.MatchValue(value=version),
                            ),
                        ]
                    ),
                    with_payload=True,
                    with_vectors=False,
                )

                if not results:
                    break

                # Update each point to add workspace
                for point in results:
                    workspaces = point.payload.get("workspaces", [])
                    if workspace not in workspaces:
                        workspaces.append(workspace)

                        self.client.set_payload(
                            collection_name=self.collection_name,
                            payload={"workspaces": workspaces},
                            points=[point.id],
                        )
                        updated_count += 1

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(
                f"Added workspace '{workspace}' to {updated_count} chunks of {dependency_name}:{version}"
            )
            return updated_count

        except Exception as e:
            logger.error(f"Error adding workspace to dependency: {e}")
            return 0

    def list_dependencies(self) -> List[Dict[str, Any]]:
        """List all indexed dependencies.

        Returns:
            List of dependencies with names, types, chunk counts, and workspaces
        """
        offset = None
        dependencies = {}
        batch_size = 100

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            for point in results:
                dep_name = point.payload.get("dependency_name", "unknown")
                dep_type = point.payload.get("dependency_type", "unknown")
                dep_version = point.payload.get("dependency_version", "unknown")
                framework = point.payload.get("framework", "")
                workspaces = point.payload.get("workspaces", [])

                key = f"{dep_name}:{dep_version}"
                if key not in dependencies:
                    dependencies[key] = {
                        "name": dep_name,
                        "type": dep_type,
                        "version": dep_version,
                        "framework": framework,
                        "chunk_count": 0,
                        "workspaces": set(),
                    }

                dependencies[key]["chunk_count"] += 1
                dependencies[key]["workspaces"].update(workspaces)

            if next_offset is None:
                break
            offset = next_offset

        # Convert workspace sets to lists
        result = []
        for dep in dependencies.values():
            dep["workspaces"] = list(dep["workspaces"])
            result.append(dep)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_chunks": collection_info.points_count,
                "vector_size": self.vector_size,
            }
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "vector_size": self.vector_size,
                "error": str(e),
            }

    def clear_collection(self) -> None:
        """Delete and recreate the knowledge collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted knowledge collection: {self.collection_name}")
        except Exception:
            logger.warning(f"Knowledge collection {self.collection_name} didn't exist")

        self._ensure_collection()

    def health_check(self) -> bool:
        """Check if Qdrant is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Knowledge DB health check failed: {e}")
            return False
