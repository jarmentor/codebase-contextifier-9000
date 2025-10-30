"""Language grammar configuration and detection for tree-sitter."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LanguageConfig:
    """Configuration for a programming language."""

    def __init__(
        self,
        name: str,
        extensions: List[str],
        tree_sitter_language: str,
        chunk_types: Dict[str, Dict],
        comment_patterns: List[str],
    ):
        """Initialize language configuration.

        Args:
            name: Language name (python, javascript, etc.)
            extensions: List of file extensions
            tree_sitter_language: Tree-sitter language identifier
            chunk_types: Dictionary mapping AST node types to their configuration
            comment_patterns: List of comment syntax patterns
        """
        self.name = name
        self.extensions = extensions
        self.tree_sitter_language = tree_sitter_language
        self.chunk_types = chunk_types
        self.comment_patterns = comment_patterns

    def is_chunkable_node(self, node_type: str) -> bool:
        """Check if a node type should be extracted as a chunk.

        Args:
            node_type: AST node type

        Returns:
            True if this node type should be chunked
        """
        return node_type in self.chunk_types

    def get_name_field(self, node_type: str) -> Optional[str]:
        """Get the field name that contains the identifier for this node type.

        Args:
            node_type: AST node type

        Returns:
            Field name or None if no specific field
        """
        if node_type in self.chunk_types:
            return self.chunk_types[node_type].get("name_field")
        return None


class LanguageRegistry:
    """Registry of language configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize language registry.

        Args:
            config_path: Path to languages.json config file
        """
        if config_path is None:
            # Default to config/languages.json relative to project root
            config_path = Path(__file__).parent.parent.parent / "config" / "languages.json"

        self.config_path = config_path
        self.languages: Dict[str, LanguageConfig] = {}
        self.extension_map: Dict[str, str] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load language configurations from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            for lang_name, lang_config in config_data.items():
                language = LanguageConfig(
                    name=lang_name,
                    extensions=lang_config["extensions"],
                    tree_sitter_language=lang_config["tree_sitter_language"],
                    chunk_types=lang_config["chunk_types"],
                    comment_patterns=lang_config["comment_patterns"],
                )
                self.languages[lang_name] = language

                # Build extension to language mapping
                for ext in language.extensions:
                    self.extension_map[ext] = lang_name

            logger.info(f"Loaded {len(self.languages)} language configurations")

        except Exception as e:
            logger.error(f"Error loading language config from {self.config_path}: {e}")
            raise

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None if not recognized
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension in self.extension_map:
            return self.extension_map[extension]

        # Handle special cases like .tsx (TypeScript React)
        if extension == ".tsx" or extension == ".jsx":
            return "typescript" if extension == ".tsx" else "javascript"

        logger.debug(f"Unknown file extension: {extension}")
        return None

    def get_language_config(self, language: str) -> Optional[LanguageConfig]:
        """Get configuration for a specific language.

        Args:
            language: Language name

        Returns:
            Language configuration or None if not found
        """
        return self.languages.get(language)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language names.

        Returns:
            List of language names
        """
        return list(self.languages.keys())

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions.

        Returns:
            List of file extensions
        """
        return list(self.extension_map.keys())

    def is_supported_file(self, file_path: str) -> bool:
        """Check if a file is supported for parsing.

        Args:
            file_path: Path to the file

        Returns:
            True if file is supported
        """
        return self.detect_language(file_path) is not None


# Global registry instance
_registry: Optional[LanguageRegistry] = None


def get_language_registry(config_path: Optional[Path] = None) -> LanguageRegistry:
    """Get the global language registry instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Language registry instance
    """
    global _registry
    if _registry is None:
        _registry = LanguageRegistry(config_path)
    return _registry
