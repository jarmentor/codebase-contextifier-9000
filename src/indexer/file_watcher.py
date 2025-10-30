"""Real-time file system watcher for automatic incremental re-indexing."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .grammars import get_language_registry

logger = logging.getLogger(__name__)


class CodeFileEventHandler(FileSystemEventHandler):
    """Event handler for code file changes."""

    def __init__(
        self,
        on_change_callback: Callable[[Set[str], Set[str]], None],
        debounce_seconds: float = 2.0,
    ):
        """Initialize event handler.

        Args:
            on_change_callback: Async callback function that receives (modified_files, deleted_files)
            debounce_seconds: Wait time before processing changes (to batch rapid changes)
        """
        super().__init__()
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds
        self.registry = get_language_registry()

        # Track pending changes
        self.modified_files: Set[str] = set()
        self.deleted_files: Set[str] = set()
        self.last_change_time = 0.0
        self.processing = False

        # Default excludes
        self.exclude_dirs = {
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
            ".idea",
            ".vscode",
        }

    def _should_process_file(self, file_path: str) -> bool:
        """Check if a file should be processed.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be indexed
        """
        path = Path(file_path)

        # Skip if in excluded directory
        if any(excluded in path.parts for excluded in self.exclude_dirs):
            return False

        # Skip hidden files
        if any(part.startswith(".") for part in path.parts):
            return False

        # Check if supported file type
        return self.registry.is_supported_file(file_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        if self._should_process_file(event.src_path):
            logger.debug(f"File modified: {event.src_path}")
            self.modified_files.add(event.src_path)
            self.last_change_time = time.time()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        if self._should_process_file(event.src_path):
            logger.debug(f"File created: {event.src_path}")
            self.modified_files.add(event.src_path)
            self.last_change_time = time.time()

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        if self._should_process_file(event.src_path):
            logger.debug(f"File deleted: {event.src_path}")
            # Remove from modified if it was pending
            self.modified_files.discard(event.src_path)
            self.deleted_files.add(event.src_path)
            self.last_change_time = time.time()

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events.

        Args:
            event: File system event
        """
        if event.is_directory:
            return

        # Treat as delete + create
        if hasattr(event, "src_path") and self._should_process_file(event.src_path):
            logger.debug(f"File moved from: {event.src_path}")
            self.deleted_files.add(event.src_path)
            self.last_change_time = time.time()

        if hasattr(event, "dest_path") and self._should_process_file(event.dest_path):
            logger.debug(f"File moved to: {event.dest_path}")
            self.modified_files.add(event.dest_path)
            self.last_change_time = time.time()

    def get_pending_changes(self) -> tuple[Set[str], Set[str]]:
        """Get pending changes and clear buffers.

        Returns:
            Tuple of (modified_files, deleted_files)
        """
        modified = self.modified_files.copy()
        deleted = self.deleted_files.copy()

        self.modified_files.clear()
        self.deleted_files.clear()

        return modified, deleted

    def has_pending_changes(self) -> bool:
        """Check if there are pending changes.

        Returns:
            True if there are pending changes
        """
        return bool(self.modified_files or self.deleted_files)

    def time_since_last_change(self) -> float:
        """Get time elapsed since last change.

        Returns:
            Seconds since last change
        """
        return time.time() - self.last_change_time


class CodebaseWatcher:
    """File system watcher for automatic incremental re-indexing."""

    def __init__(
        self,
        watch_path: str,
        on_change_callback: Callable[[Set[str], Set[str]], None],
        debounce_seconds: float = 2.0,
        recursive: bool = True,
    ):
        """Initialize codebase watcher.

        Args:
            watch_path: Path to directory to watch
            on_change_callback: Async callback for file changes
            debounce_seconds: Debounce time for batching changes
            recursive: Whether to watch subdirectories recursively
        """
        self.watch_path = Path(watch_path)
        self.on_change_callback = on_change_callback
        self.debounce_seconds = debounce_seconds
        self.recursive = recursive

        self.event_handler = CodeFileEventHandler(on_change_callback, debounce_seconds)
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler, str(self.watch_path), recursive=self.recursive
        )

        self._running = False
        self._debounce_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized file watcher for: {self.watch_path}")

    def start(self) -> None:
        """Start watching for file changes."""
        if not self._running:
            self.observer.start()
            self._running = True
            logger.info(f"Started watching: {self.watch_path}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self._running:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self._running = False
            logger.info("Stopped file watcher")

    def is_running(self) -> bool:
        """Check if watcher is running.

        Returns:
            True if watcher is active
        """
        return self._running

    async def start_debounce_processor(self) -> None:
        """Start background task to process debounced changes."""
        logger.info("Started debounce processor")

        while self._running:
            try:
                # Check if there are pending changes and enough time has passed
                if self.event_handler.has_pending_changes():
                    time_since_change = self.event_handler.time_since_last_change()

                    if time_since_change >= self.debounce_seconds:
                        # Process the changes
                        modified, deleted = self.event_handler.get_pending_changes()

                        if modified or deleted:
                            logger.info(
                                f"Processing changes: {len(modified)} modified, {len(deleted)} deleted"
                            )

                            try:
                                # Call the async callback
                                await self.on_change_callback(modified, deleted)
                                logger.info("Successfully processed file changes")
                            except Exception as e:
                                logger.error(f"Error processing file changes: {e}")

                # Check every 0.5 seconds
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                logger.info("Debounce processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in debounce processor: {e}")
                await asyncio.sleep(1.0)

    async def run_async(self) -> None:
        """Run the watcher with async support."""
        self.start()

        # Start debounce processor
        self._debounce_task = asyncio.create_task(self.start_debounce_processor())

        try:
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1.0)
        finally:
            if self._debounce_task:
                self._debounce_task.cancel()
                try:
                    await self._debounce_task
                except asyncio.CancelledError:
                    pass

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
