"""MCP tool for semantic code search."""

import logging
from typing import List, Optional

from ..indexer.embeddings import OllamaEmbeddings
from ..vector_db.qdrant_client import CodeVectorDB

logger = logging.getLogger(__name__)


class SearchTool:
    """Tool for semantic code search."""

    def __init__(self, vector_db: CodeVectorDB, embeddings: OllamaEmbeddings):
        """Initialize search tool.

        Args:
            vector_db: Vector database client
            embeddings: Embeddings generator
        """
        self.vector_db = vector_db
        self.embeddings = embeddings

    async def search_code(
        self,
        query: str,
        limit: int = 10,
        repo_name: Optional[str] = None,
        language: Optional[str] = None,
        file_path_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> dict:
        """Search for code using natural language queries.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return (default: 10)
            repo_name: Filter by repository name (searches all repos if not specified)
            language: Filter by programming language (e.g., 'python', 'typescript')
            file_path_filter: Filter by file path pattern (e.g., 'src/components')
            chunk_type: Filter by chunk type (e.g., 'function', 'class')

        Returns:
            Dictionary with search results
        """
        try:
            logger.info(f"Searching for: {query}" + (f" in repo: {repo_name}" if repo_name else " (all repos)"))

            # Generate embedding for the query
            query_embeddings = await self.embeddings.generate_embeddings([query])
            query_vector = query_embeddings[0]

            # Search vector database
            results = self.vector_db.search(
                query_vector=query_vector,
                limit=limit,
                repo_name_filter=repo_name,
                language_filter=language,
                file_path_filter=file_path_filter,
                chunk_type_filter=chunk_type,
            )

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    {
                        "rank": i,
                        "score": round(result["score"], 4),
                        "repo": result.get("repo_name", "unknown"),
                        "file": result["file_path"],
                        "lines": f"{result['start_line']}-{result['end_line']}",
                        "language": result["language"],
                        "type": result["chunk_type"],
                        "context": result["context"] or "N/A",
                        "code": result["content"],
                    }
                )

            return {
                "success": True,
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "filters": {
                    "repo_name": repo_name,
                    "language": language,
                    "file_path": file_path_filter,
                    "chunk_type": chunk_type,
                },
            }

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> dict:
        """Get statistics about the indexed codebase.

        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.vector_db.get_collection_stats()

            return {
                "success": True,
                "total_chunks": stats["total_points"],
                "vectors_count": stats["vectors_count"],
                "indexed_vectors": stats["indexed_vectors_count"],
                "status": stats["status"],
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"success": False, "error": str(e)}
