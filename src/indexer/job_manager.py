"""Background job management for indexing tasks."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for an indexing job."""

    current_file: int = 0
    total_files: int = 0
    current_file_path: str = ""
    chunks_indexed: int = 0
    failed_files: List[str] = field(default_factory=list)
    cache_hit_rate: float = 0.0


@dataclass
class IndexingJob:
    """Represents an indexing job."""

    job_id: str
    repo_name: str
    repo_path: str
    status: JobStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: JobProgress = field(default_factory=JobProgress)
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    container_id: Optional[str] = None  # Docker container ID if using container-based indexing


class JobManager:
    """Manages background indexing jobs."""

    def __init__(self):
        """Initialize job manager."""
        self.jobs: Dict[str, IndexingJob] = {}
        self._lock = asyncio.Lock()
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
            self.docker_client = None

    def create_job(self, repo_name: str, repo_path: str) -> IndexingJob:
        """Create a new indexing job.

        Args:
            repo_name: Name of the repository
            repo_path: Path to the repository

        Returns:
            Created job
        """
        job_id = str(uuid.uuid4())[:8]
        job = IndexingJob(
            job_id=job_id,
            repo_name=repo_name,
            repo_path=repo_path,
            status=JobStatus.QUEUED,
            created_at=time.time(),
        )
        self.jobs[job_id] = job
        logger.info(f"Created indexing job {job_id} for repo '{repo_name}'")
        return job

    def get_job(self, job_id: str) -> Optional[IndexingJob]:
        """Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job if found, None otherwise
        """
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[IndexingJob]:
        """List all jobs.

        Returns:
            List of all jobs
        """
        return list(self.jobs.values())

    async def update_progress(
        self,
        job_id: str,
        current_file: Optional[int] = None,
        total_files: Optional[int] = None,
        current_file_path: Optional[str] = None,
        chunks_indexed: Optional[int] = None,
        failed_files: Optional[List[str]] = None,
        cache_hit_rate: Optional[float] = None,
    ) -> None:
        """Update job progress.

        Args:
            job_id: Job identifier
            current_file: Current file number
            total_files: Total number of files
            current_file_path: Path of current file being processed
            chunks_indexed: Total chunks indexed so far
            failed_files: List of failed file paths
            cache_hit_rate: Cache hit rate percentage
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return

            if current_file is not None:
                job.progress.current_file = current_file
            if total_files is not None:
                job.progress.total_files = total_files
            if current_file_path is not None:
                job.progress.current_file_path = current_file_path
            if chunks_indexed is not None:
                job.progress.chunks_indexed = chunks_indexed
            if failed_files is not None:
                job.progress.failed_files = failed_files
            if cache_hit_rate is not None:
                job.progress.cache_hit_rate = cache_hit_rate

    async def mark_started(self, job_id: str) -> None:
        """Mark job as started.

        Args:
            job_id: Job identifier
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                logger.info(f"Job {job_id} started")

    async def mark_completed(self, job_id: str) -> None:
        """Mark job as completed.

        Args:
            job_id: Job identifier
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                logger.info(f"Job {job_id} completed")

    async def mark_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed.

        Args:
            job_id: Job identifier
            error: Error message
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = error
                logger.error(f"Job {job_id} failed: {error}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled, False if not found or already done
        """
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False

            if job.status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
                return False

            # Cancel the asyncio task if it exists
            if job.task and not job.task.done():
                job.task.cancel()

            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            logger.info(f"Job {job_id} cancelled")
            return True

    async def spawn_indexer_container(
        self,
        job_id: str,
        host_path: str,
        repo_name: str,
        qdrant_host: str = "codebase-qdrant",
        qdrant_port: int = 6333,
        ollama_host: str = "http://host.docker.internal:11434",
        embedding_model: str = "nomic-embed-text",
        incremental: bool = True,
        exclude_patterns: Optional[str] = None,
    ) -> bool:
        """Spawn a Docker container to index a repository.

        Args:
            job_id: Job identifier
            host_path: Absolute path on host machine to repository
            repo_name: Repository name
            qdrant_host: Qdrant hostname
            qdrant_port: Qdrant port
            ollama_host: Ollama API URL
            embedding_model: Embedding model name
            incremental: Use incremental indexing
            exclude_patterns: Comma-separated glob patterns to exclude

        Returns:
            True if container spawned successfully, False otherwise
        """
        if not self.docker_client:
            logger.error("Docker client not available")
            await self.mark_failed(job_id, "Docker client not available")
            return False

        try:
            # Container configuration
            image = "codebase-contextifier-9000-indexer"
            container_name = f"indexer-{job_id}"

            environment = {
                "WORKSPACE_PATH": "/workspace",
                "REPO_NAME": repo_name,
                "QDRANT_HOST": qdrant_host,
                "QDRANT_PORT": str(qdrant_port),
                "OLLAMA_HOST": ollama_host,
                "EMBEDDING_MODEL": embedding_model,
                "INDEX_PATH": "/index",
                "CACHE_PATH": "/cache",
                "LOG_LEVEL": "INFO",
                "INCREMENTAL": "true" if incremental else "false",
            }

            # Add exclude patterns if provided
            if exclude_patterns:
                environment["EXCLUDE_PATTERNS"] = exclude_patterns

            volumes = {
                host_path: {"bind": "/workspace", "mode": "ro"},
                "codebase-contextifier-9000_index_data": {"bind": "/index", "mode": "rw"},
                "codebase-contextifier-9000_cache_data": {"bind": "/cache", "mode": "rw"},
            }

            # Spawn container
            logger.info(f"Spawning indexer container for job {job_id}")
            logger.info(f"Host path: {host_path}")
            logger.info(f"Repo name: {repo_name}")

            container = self.docker_client.containers.run(
                image=image,
                name=container_name,
                environment=environment,
                volumes=volumes,
                network="codebase-contextifier-9000_default",
                extra_hosts={"host.docker.internal": "host-gateway"},
                detach=True,
                remove=True,  # Auto-remove when done
            )

            # Update job with container ID
            async with self._lock:
                job = self.jobs.get(job_id)
                if job:
                    job.container_id = container.id
                    logger.info(f"Container {container.id[:12]} spawned for job {job_id}")

            # Start monitoring task
            asyncio.create_task(self._monitor_container(job_id, container.id))

            await self.mark_started(job_id)
            return True

        except DockerException as e:
            error_msg = f"Failed to spawn container: {e}"
            logger.error(error_msg)
            await self.mark_failed(job_id, error_msg)
            return False

    async def _monitor_container(self, job_id: str, container_id: str) -> None:
        """Monitor a container and update job status.

        Args:
            job_id: Job identifier
            container_id: Docker container ID
        """
        try:
            container = self.docker_client.containers.get(container_id)

            # Wait for container to finish
            while True:
                await asyncio.sleep(2)  # Poll every 2 seconds

                try:
                    container.reload()
                    status = container.status

                    if status == "exited":
                        # Container finished
                        exit_code = container.attrs["State"]["ExitCode"]

                        if exit_code == 0:
                            logger.info(f"Container {container_id[:12]} completed successfully")
                            # TODO: Parse logs for final stats
                            await self.mark_completed(job_id)
                        else:
                            error_msg = f"Container exited with code {exit_code}"
                            logger.error(f"Container {container_id[:12]}: {error_msg}")
                            await self.mark_failed(job_id, error_msg)
                        break

                    elif status in ["dead", "removing"]:
                        error_msg = f"Container entered unexpected status: {status}"
                        logger.error(error_msg)
                        await self.mark_failed(job_id, error_msg)
                        break

                except docker.errors.NotFound:
                    # Container was removed (auto-remove on exit)
                    logger.info(f"Container {container_id[:12]} removed")
                    # Assume success if it was removed
                    await self.mark_completed(job_id)
                    break

        except Exception as e:
            error_msg = f"Error monitoring container: {e}"
            logger.error(error_msg, exc_info=True)
            await self.mark_failed(job_id, error_msg)

    def get_status_dict(self, job: IndexingJob) -> dict:
        """Convert job to status dictionary.

        Args:
            job: Job to convert

        Returns:
            Dictionary representation
        """
        progress_pct = 0.0
        if job.progress.total_files > 0:
            progress_pct = (job.progress.current_file / job.progress.total_files) * 100

        result = {
            "job_id": job.job_id,
            "repo_name": job.repo_name,
            "repo_path": job.repo_path,
            "status": job.status.value,
            "created_at": job.created_at,
            "progress": {
                "current_file": job.progress.current_file,
                "total_files": job.progress.total_files,
                "progress_pct": round(progress_pct, 2),
                "current_file_path": job.progress.current_file_path,
                "chunks_indexed": job.progress.chunks_indexed,
                "failed_files_count": len(job.progress.failed_files),
                "cache_hit_rate": f"{job.progress.cache_hit_rate:.2f}%",
            },
        }

        if job.started_at:
            result["started_at"] = job.started_at
            if job.status == JobStatus.RUNNING:
                result["elapsed_seconds"] = round(time.time() - job.started_at, 2)

        if job.completed_at:
            result["completed_at"] = job.completed_at
            result["total_seconds"] = round(job.completed_at - job.started_at, 2)

        if job.error:
            result["error"] = job.error

        return result
