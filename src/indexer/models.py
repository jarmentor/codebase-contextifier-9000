"""Data models for code analysis."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CodeChunk:
    """Represents a semantic chunk of code."""

    id: str  # Blake3 hash for content-addressable storage
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    language: str
    chunk_type: str  # function, class, method, etc.
    name: Optional[str]  # identifier name if available
    content: str
    context: str  # surrounding context (e.g., class name for methods)
    repo_path: str  # relative path from repo root


@dataclass
class CodeRelationship:
    """Represents a relationship between code entities."""

    source_id: str  # ID of source entity
    source_name: str  # Name of source entity
    source_type: str  # Type of source (function, class, method, file)
    source_file: str  # File path of source
    target_id: Optional[str]  # ID of target entity (None for external/unresolved)
    target_name: str  # Name of target entity
    target_type: str  # Type of target (function, class, module, etc.)
    relationship_type: str  # CALLS, IMPORTS, EXTENDS, IMPLEMENTS, USES, etc.
    line_number: int  # Line where relationship occurs
    metadata: Dict[str, Any]  # Additional context (e.g., import_alias, is_static)
