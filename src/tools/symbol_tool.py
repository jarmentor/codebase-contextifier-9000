"""MCP tool for extracting symbols from code files."""

import logging
from typing import List, Optional

from ..indexer.ast_chunker import ASTChunker

logger = logging.getLogger(__name__)


class SymbolTool:
    """Tool for AST-based symbol extraction."""

    def __init__(self, chunker: ASTChunker):
        """Initialize symbol tool.

        Args:
            chunker: AST chunker for parsing code
        """
        self.chunker = chunker

    def get_symbols(
        self,
        file_path: str,
        symbol_type: Optional[str] = None,
    ) -> dict:
        """Extract symbols (functions, classes, methods) from a file.

        Args:
            file_path: Path to the file
            symbol_type: Filter by symbol type (e.g., 'function', 'class')

        Returns:
            Dictionary with extracted symbols
        """
        try:
            logger.info(f"Extracting symbols from: {file_path}")

            # Chunk the file to get all symbols
            chunks = self.chunker.chunk_file(file_path)

            if not chunks:
                return {
                    "success": False,
                    "error": "Could not parse file or no symbols found",
                }

            # Filter by symbol type if specified
            if symbol_type:
                # Use substring matching to handle different language-specific names
                # e.g., "function" matches "function", "function_declaration", "function_definition"
                chunks = [c for c in chunks if symbol_type.lower() in c.chunk_type.lower()]

            # Format symbols
            symbols = []
            for chunk in chunks:
                symbols.append(
                    {
                        "name": chunk.name or "anonymous",
                        "type": chunk.chunk_type,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "context": chunk.context or "N/A",
                        "language": chunk.language,
                    }
                )

            return {
                "success": True,
                "file_path": file_path,
                "total_symbols": len(symbols),
                "symbols": symbols,
                "filter": symbol_type,
            }

        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return {"success": False, "error": str(e)}

    def list_symbol_types(self, file_path: str) -> dict:
        """List all symbol types found in a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with symbol type counts
        """
        try:
            chunks = self.chunker.chunk_file(file_path)

            if not chunks:
                return {
                    "success": False,
                    "error": "Could not parse file",
                }

            # Count symbol types
            type_counts = {}
            for chunk in chunks:
                symbol_type = chunk.chunk_type
                type_counts[symbol_type] = type_counts.get(symbol_type, 0) + 1

            return {
                "success": True,
                "file_path": file_path,
                "symbol_types": type_counts,
            }

        except Exception as e:
            logger.error(f"Error listing symbol types: {e}")
            return {"success": False, "error": str(e)}
