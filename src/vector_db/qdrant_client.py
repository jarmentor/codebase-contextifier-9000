"""Qdrant vector database client wrapper for code embeddings."""

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


class CodeVectorDB:
    """Wrapper for Qdrant vector database operations."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "code_embeddings",
        vector_size: int = 768,  # Default for nomic-embed-text
    ):
        """Initialize Qdrant client.

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
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Insert or update code chunks with their embeddings.

        Args:
            chunks: List of chunk metadata dictionaries containing:
                - id: unique identifier (Blake3 hash)
                - file_path: path to source file
                - start_line: starting line number
                - end_line: ending line number
                - language: programming language
                - chunk_type: function, class, method, etc.
                - content: actual code content
                - context: surrounding context (class name, etc.)
            embeddings: List of embedding vectors corresponding to chunks

        Returns:
            Number of points upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Convert hex ID to UUID format for Qdrant
            chunk_id = chunk["id"]
            uuid_id = hex_to_uuid(chunk_id)

            point = PointStruct(
                id=uuid_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,  # Store original hex ID in payload for reference
                    "repo_name": chunk.get("repo_name", ""),
                    "file_path": chunk.get("file_path", ""),
                    "start_line": chunk.get("start_line", 0),
                    "end_line": chunk.get("end_line", 0),
                    "language": chunk.get("language", "unknown"),
                    "chunk_type": chunk.get("chunk_type", "unknown"),
                    "content": chunk.get("content", ""),
                    "context": chunk.get("context", ""),
                    "repo_path": chunk.get("repo_path", ""),
                },
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(f"Upserted {len(points)} chunks to Qdrant")
            return len(points)
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        repo_name_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        file_path_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar code chunks.

        Args:
            query_vector: Embedding vector of the search query
            limit: Maximum number of results to return
            repo_name_filter: Filter by repository name
            language_filter: Filter by programming language
            file_path_filter: Filter by file path (supports wildcards)
            chunk_type_filter: Filter by chunk type (function, class, etc.)

        Returns:
            List of matching chunks with metadata and scores
        """
        # Build filter conditions
        must_conditions = []

        if repo_name_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="repo_name",
                    match=models.MatchValue(value=repo_name_filter),
                )
            )

        if language_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language_filter),
                )
            )

        if file_path_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchText(text=file_path_filter),
                )
            )

        if chunk_type_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type_filter),
                )
            )

        query_filter = None
        if must_conditions:
            query_filter = models.Filter(must=must_conditions)

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "score": result.score,
                        "file_path": result.payload.get("file_path", ""),
                        "start_line": result.payload.get("start_line", 0),
                        "end_line": result.payload.get("end_line", 0),
                        "language": result.payload.get("language", "unknown"),
                        "chunk_type": result.payload.get("chunk_type", "unknown"),
                        "content": result.payload.get("content", ""),
                        "context": result.payload.get("context", ""),
                    }
                )

            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "total_points": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise

    def delete_by_file_path(self, file_path: str) -> int:
        """Delete all chunks associated with a file path.

        Args:
            file_path: Path to the file whose chunks should be deleted

        Returns:
            Number of points deleted
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=file_path),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted chunks for file: {file_path}")
            return 1  # Qdrant doesn't return count, so we return success indicator
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise

    def clear_collection(self) -> None:
        """Delete all points from the collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()  # Recreate empty collection
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
