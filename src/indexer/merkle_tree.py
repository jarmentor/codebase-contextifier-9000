"""Merkle tree-based incremental indexing for efficient change detection."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import blake3

logger = logging.getLogger(__name__)


@dataclass
class FileRecord:
    """Record of a file's indexing state."""

    path: str
    content_hash: str
    last_indexed: float  # timestamp
    chunk_count: int
    chunk_ids: List[str]


class MerkleTreeIndexer:
    """Merkle tree-based incremental indexing with content-addressable storage."""

    def __init__(self, index_path: Path):
        """Initialize Merkle tree indexer.

        Args:
            index_path: Directory to store index metadata
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.state_file = self.index_path / "index_state.json"
        self.file_records: Dict[str, FileRecord] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load index state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.file_records = {
                        path: FileRecord(**record) for path, record in data.items()
                    }
                logger.info(f"Loaded index state with {len(self.file_records)} files")
            except Exception as e:
                logger.error(f"Error loading index state: {e}")
                self.file_records = {}
        else:
            logger.info("No existing index state found, starting fresh")

    def _save_state(self) -> None:
        """Save index state to disk."""
        try:
            data = {path: asdict(record) for path, record in self.file_records.items()}
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved index state with {len(self.file_records)} files")
        except Exception as e:
            logger.error(f"Error saving index state: {e}")

    def compute_file_hash(self, file_path: str) -> str:
        """Compute Blake3 hash of a file's contents.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal hash string
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return blake3.blake3(content).hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def has_file_changed(self, file_path: str) -> bool:
        """Check if a file has changed since last indexing.

        Args:
            file_path: Path to the file

        Returns:
            True if file is new or has changed, False if unchanged
        """
        # New file
        if file_path not in self.file_records:
            return True

        # Check if file still exists
        if not Path(file_path).exists():
            return True

        # Compare content hash
        current_hash = self.compute_file_hash(file_path)
        stored_hash = self.file_records[file_path].content_hash

        return current_hash != stored_hash

    def get_changed_files(self, file_paths: List[str]) -> List[str]:
        """Get list of files that have changed or are new.

        Args:
            file_paths: List of file paths to check

        Returns:
            List of file paths that need re-indexing
        """
        changed = []
        for file_path in file_paths:
            if self.has_file_changed(file_path):
                changed.append(file_path)

        logger.info(
            f"Found {len(changed)} changed files out of {len(file_paths)} total "
            f"({len(file_paths) - len(changed)} cached)"
        )
        return changed

    def get_removed_files(self, current_file_paths: Set[str]) -> List[str]:
        """Get list of files that have been removed.

        Args:
            current_file_paths: Set of currently existing file paths

        Returns:
            List of file paths that no longer exist
        """
        removed = []
        for file_path in self.file_records.keys():
            if file_path not in current_file_paths:
                removed.append(file_path)

        if removed:
            logger.info(f"Found {len(removed)} removed files")

        return removed

    def update_file_record(
        self,
        file_path: str,
        chunk_ids: List[str],
        timestamp: Optional[float] = None,
    ) -> None:
        """Update record for an indexed file.

        Args:
            file_path: Path to the file
            chunk_ids: List of chunk IDs generated from this file
            timestamp: Indexing timestamp (defaults to current time)
        """
        import time

        if timestamp is None:
            timestamp = time.time()

        content_hash = self.compute_file_hash(file_path)

        record = FileRecord(
            path=file_path,
            content_hash=content_hash,
            last_indexed=timestamp,
            chunk_count=len(chunk_ids),
            chunk_ids=chunk_ids,
        )

        self.file_records[file_path] = record
        logger.debug(f"Updated record for {file_path} with {len(chunk_ids)} chunks")

    def remove_file_record(self, file_path: str) -> Optional[List[str]]:
        """Remove record for a deleted file.

        Args:
            file_path: Path to the file

        Returns:
            List of chunk IDs that should be deleted, or None if file not found
        """
        if file_path in self.file_records:
            record = self.file_records.pop(file_path)
            logger.debug(f"Removed record for {file_path}")
            return record.chunk_ids
        return None

    def get_file_chunk_ids(self, file_path: str) -> Optional[List[str]]:
        """Get chunk IDs for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of chunk IDs or None if file not indexed
        """
        if file_path in self.file_records:
            return self.file_records[file_path].chunk_ids
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        total_chunks = sum(record.chunk_count for record in self.file_records.values())

        return {
            "indexed_files": len(self.file_records),
            "total_chunks": total_chunks,
            "index_size_bytes": self.state_file.stat().st_size if self.state_file.exists() else 0,
        }

    def commit(self) -> None:
        """Commit current state to disk."""
        self._save_state()

    def clear(self) -> None:
        """Clear all index state."""
        self.file_records = {}
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("Cleared index state")


class IncrementalIndexingSession:
    """Context manager for incremental indexing operations."""

    def __init__(self, merkle_indexer: MerkleTreeIndexer):
        """Initialize indexing session.

        Args:
            merkle_indexer: Merkle tree indexer instance
        """
        self.indexer = merkle_indexer
        self.files_to_index: List[str] = []
        self.files_to_remove: List[str] = []

    def __enter__(self):
        """Enter indexing session."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit indexing session and commit changes."""
        if exc_type is None:
            # Success - commit changes
            self.indexer.commit()
            logger.info("Committed index changes")
        else:
            logger.error(f"Indexing session failed: {exc_val}")
        return False

    def plan_incremental_update(
        self, current_files: List[str]
    ) -> tuple[List[str], List[str]]:
        """Plan which files need to be updated.

        Args:
            current_files: List of all current file paths

        Returns:
            Tuple of (files_to_index, files_to_remove)
        """
        current_file_set = set(current_files)

        # Find changed/new files
        self.files_to_index = self.indexer.get_changed_files(current_files)

        # Find removed files
        self.files_to_remove = self.indexer.get_removed_files(current_file_set)

        logger.info(
            f"Incremental update plan: {len(self.files_to_index)} to index, "
            f"{len(self.files_to_remove)} to remove, "
            f"{len(current_files) - len(self.files_to_index)} cached"
        )

        return self.files_to_index, self.files_to_remove

    def update_file(self, file_path: str, chunk_ids: List[str]) -> None:
        """Record that a file has been indexed.

        Args:
            file_path: Path to the indexed file
            chunk_ids: List of chunk IDs generated
        """
        self.indexer.update_file_record(file_path, chunk_ids)

    def remove_file(self, file_path: str) -> Optional[List[str]]:
        """Record that a file has been removed.

        Args:
            file_path: Path to the removed file

        Returns:
            List of chunk IDs to delete
        """
        return self.indexer.remove_file_record(file_path)

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for this session.

        Returns:
            Cache hit rate as a percentage (0-100)
        """
        total = len(self.files_to_index) + (
            len(self.indexer.file_records) - len(self.files_to_remove)
        )
        if total == 0:
            return 0.0

        cached = total - len(self.files_to_index)
        return (cached / total) * 100
