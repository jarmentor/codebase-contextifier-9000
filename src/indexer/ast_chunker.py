"""AST-aware code chunking using tree-sitter for semantic boundaries."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import blake3
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_php as tsphp
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp
import tree_sitter_c as tsc
import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

from .grammars import LanguageConfig, get_language_registry

logger = logging.getLogger(__name__)


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


class ASTChunker:
    """Chunk code using AST-aware parsing with tree-sitter."""

    # Language module mapping
    LANGUAGE_MODULES = {
        "python": tspython,
        "javascript": tsjavascript,
        "typescript": tstypescript,
        "php": tsphp,
        "go": tsgo,
        "rust": tsrust,
        "java": tsjava,
        "cpp": tscpp,
        "c": tsc,
        "c_sharp": tscsharp,
    }

    # Modules that use non-standard language function names
    LANGUAGE_FUNCTION_OVERRIDES = {
        "typescript": "language_typescript",
        "php": "language_php",
    }

    def __init__(self, max_chunk_size: int = 2048):
        """Initialize AST chunker.

        Args:
            max_chunk_size: Maximum size for a single chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.registry = get_language_registry()
        self.parsers: Dict[str, Parser] = {}
        self.languages: Dict[str, Language] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """Initialize tree-sitter languages."""
        for lang_name in self.registry.get_supported_languages():
            lang_config = self.registry.get_language_config(lang_name)
            if not lang_config:
                continue

            ts_lang_name = lang_config.tree_sitter_language

            try:
                # Get the language module
                module = self.LANGUAGE_MODULES.get(ts_lang_name)
                if not module:
                    logger.warning(f"No module found for language: {ts_lang_name}")
                    continue

                # Get the language function (either standard or override)
                lang_func_name = self.LANGUAGE_FUNCTION_OVERRIDES.get(ts_lang_name, "language")
                lang_func = getattr(module, lang_func_name, None)

                if not lang_func:
                    logger.warning(f"Module {ts_lang_name} has no function '{lang_func_name}'")
                    continue

                # Create language and parser
                language = Language(lang_func())
                self.languages[lang_name] = language

                parser = Parser()
                parser.language = language
                self.parsers[lang_name] = parser

                logger.debug(f"Initialized parser for {lang_name}")

            except Exception as e:
                logger.error(f"Error initializing language {lang_name}: {e}")

    def _generate_chunk_id(self, content: str, file_path: str, start_line: int) -> str:
        """Generate unique ID for a chunk using Blake3 hash.

        Args:
            content: Chunk content
            file_path: File path
            start_line: Starting line number

        Returns:
            Hexadecimal hash string
        """
        # Include file path and line number for uniqueness
        hash_input = f"{file_path}:{start_line}:{content}"
        return blake3.blake3(hash_input.encode()).hexdigest()[:16]

    def _extract_node_name(self, node: Any, lang_config: LanguageConfig) -> Optional[str]:
        """Extract the name/identifier from an AST node.

        Args:
            node: Tree-sitter node
            lang_config: Language configuration

        Returns:
            Name string or None
        """
        name_field = lang_config.get_name_field(node.type)
        if not name_field:
            return None

        # Handle nested field paths like "declarator.function_declarator.declarator"
        if "." in name_field:
            current = node
            for field in name_field.split("."):
                child = current.child_by_field_name(field)
                if not child:
                    return None
                current = child
            return current.text.decode("utf-8") if current else None
        else:
            name_node = node.child_by_field_name(name_field)
            return name_node.text.decode("utf-8") if name_node else None

    def _find_parent_context(self, node: Any, lang_config: LanguageConfig) -> str:
        """Find parent context (e.g., class name for a method).

        Args:
            node: Tree-sitter node
            lang_config: Language configuration

        Returns:
            Context string
        """
        contexts = []
        current = node.parent

        while current:
            # Check if parent is a class/interface/namespace
            if lang_config.is_chunkable_node(current.type):
                parent_name = self._extract_node_name(current, lang_config)
                if parent_name:
                    contexts.append(f"{current.type}:{parent_name}")

            current = current.parent

        return " > ".join(reversed(contexts)) if contexts else ""

    def _traverse_ast(
        self,
        node: Any,
        source_code: bytes,
        file_path: str,
        lang_config: LanguageConfig,
        repo_path: str,
    ) -> List[CodeChunk]:
        """Recursively traverse AST and extract semantic chunks.

        Args:
            node: Tree-sitter node
            source_code: Original source code bytes
            file_path: Path to the file
            lang_config: Language configuration
            repo_path: Relative path from repo root

        Returns:
            List of code chunks
        """
        chunks = []

        # Check if this node should be extracted as a chunk
        if lang_config.is_chunkable_node(node.type):
            content = source_code[node.start_byte : node.end_byte].decode("utf-8")

            # Skip if too large (would need further splitting)
            if len(content) <= self.max_chunk_size:
                name = self._extract_node_name(node, lang_config)
                context = self._find_parent_context(node, lang_config)

                chunk = CodeChunk(
                    id=self._generate_chunk_id(content, file_path, node.start_point[0] + 1),
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,  # Tree-sitter uses 0-based indexing
                    end_line=node.end_point[0] + 1,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    language=lang_config.name,
                    chunk_type=node.type,
                    name=name,
                    content=content,
                    context=context,
                    repo_path=repo_path,
                )
                chunks.append(chunk)

                logger.debug(
                    f"Extracted {node.type} '{name}' from {file_path}:{chunk.start_line}"
                )

                # Don't traverse children of extracted chunks to avoid duplication
                return chunks

        # Recursively traverse children
        for child in node.children:
            chunks.extend(
                self._traverse_ast(child, source_code, file_path, lang_config, repo_path)
            )

        return chunks

    def chunk_file(self, file_path: str, repo_root: Optional[str] = None) -> List[CodeChunk]:
        """Parse a file and extract semantic chunks.

        Args:
            file_path: Path to the file to chunk
            repo_root: Root directory of the repository (for relative paths)

        Returns:
            List of code chunks extracted from the file
        """
        # Detect language
        language = self.registry.detect_language(file_path)
        if not language:
            logger.warning(f"Unsupported file type: {file_path}")
            return []

        if language not in self.parsers:
            logger.warning(f"Parser not available for {language}")
            return []

        # Get language config
        lang_config = self.registry.get_language_config(language)
        if not lang_config:
            return []

        # Read file
        try:
            with open(file_path, "rb") as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []

        # Parse with tree-sitter
        try:
            parser = self.parsers[language]
            tree = parser.parse(source_code)
            root_node = tree.root_node

            # Check for parse errors
            if root_node.has_error:
                logger.warning(f"Parse errors in {file_path}")

            # Calculate relative path
            if repo_root:
                try:
                    repo_path = str(Path(file_path).relative_to(repo_root))
                except ValueError:
                    repo_path = file_path
            else:
                repo_path = file_path

            # Extract chunks
            chunks = self._traverse_ast(root_node, source_code, file_path, lang_config, repo_path)

            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def chunk_directory(
        self,
        directory: str,
        exclude_patterns: Optional[List[str]] = None,
        follow_gitignore: bool = True,
    ) -> List[CodeChunk]:
        """Recursively chunk all supported files in a directory.

        Args:
            directory: Directory to scan
            exclude_patterns: List of patterns to exclude (e.g., "node_modules", "*.test.js")
            follow_gitignore: Whether to respect .gitignore files

        Returns:
            List of all code chunks from the directory
        """
        from gitignore_parser import parse_gitignore

        all_chunks = []
        dir_path = Path(directory)

        # Load gitignore if present
        gitignore_matcher = None
        if follow_gitignore:
            gitignore_path = dir_path / ".gitignore"
            if gitignore_path.exists():
                try:
                    gitignore_matcher = parse_gitignore(gitignore_path)
                    logger.info(f"Loaded .gitignore from {gitignore_path}")
                except Exception as e:
                    logger.warning(f"Error parsing .gitignore: {e}")

        # Default exclude patterns
        default_excludes = {
            "node_modules",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "venv",
            ".venv",
            "dist",
            "build",
            ".next",
            ".nuxt",
            "vendor",
        }

        # Recursively find all supported files
        for file_path in dir_path.rglob("*"):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if not supported
            if not self.registry.is_supported_file(str(file_path)):
                continue

            # Skip if in default excludes
            if any(excluded in file_path.parts for excluded in default_excludes):
                continue

            # Skip if matches gitignore
            if gitignore_matcher and gitignore_matcher(str(file_path)):
                continue

            # Skip if matches exclude patterns
            if exclude_patterns:
                if any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue

            # Chunk the file
            try:
                chunks = self.chunk_file(str(file_path), repo_root=str(dir_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking {file_path}: {e}")

        logger.info(f"Extracted {len(all_chunks)} total chunks from {directory}")
        return all_chunks
