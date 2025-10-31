"""Embedding generation using Ollama for local LLM inference."""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import blake3
import httpx

logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """Generate embeddings using Ollama's local embedding models."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        max_concurrent: int = 4,
        max_tokens: int = 2048,
    ):
        """Initialize Ollama embeddings client.

        Args:
            host: Ollama API host URL
            model: Name of the embedding model to use
            cache_dir: Directory for caching embeddings (None to disable)
            batch_size: Number of texts to process in parallel
            max_concurrent: Maximum concurrent requests to Ollama
            max_tokens: Maximum token length for model (default: 2048 for nomic-embed-text)
        """
        self.host = host.rstrip("/")
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop_id: Optional[int] = None
        self._semaphore = None

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at: {self.cache_dir}")

        logger.info(f"Initialized Ollama embeddings with model: {model} (max_tokens: {max_tokens})")

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create an httpx client for the current event loop.

        Returns:
            httpx.AsyncClient instance for current event loop
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)

            # Create new client if we don't have one or if we're in a different event loop
            if self._client is None or self._client_loop_id != loop_id:
                # Clean up old client if it exists
                if self._client is not None:
                    try:
                        # Don't await since it might be from a different loop
                        pass
                    except Exception:
                        pass

                self._client = httpx.AsyncClient(timeout=60.0)
                self._client_loop_id = loop_id
                logger.debug(f"Created new httpx client for event loop {loop_id}")

            # Recreate semaphore if needed
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self.max_concurrent)

            return self._client
        except RuntimeError:
            # No running event loop, create a basic client
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=60.0)
            return self._client

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text.

        Uses Blake3 hash of (text + model + version) for content-addressable storage.

        Args:
            text: Text to generate cache key for

        Returns:
            Hexadecimal hash string
        """
        # Include model name to invalidate cache if model changes
        cache_input = f"{self.model}:{text}"
        return blake3.blake3(cache_input.encode()).hexdigest()

    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Retrieve cached embedding if available.

        Args:
            cache_key: Cache key for the embedding

        Returns:
            Cached embedding vector or None if not found
        """
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return data["embedding"]
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                return None

        return None

    def _save_cached_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Save embedding to cache.

        Args:
            cache_key: Cache key for the embedding
            embedding: Embedding vector to cache
        """
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"embedding": embedding}, f)
            logger.debug(f"Cached embedding for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within model's token limit.

        Uses a conservative estimate of 3 characters per token.
        Leaves 20% buffer to account for tokenization overhead and safety margin.

        Args:
            text: Text to truncate

        Returns:
            Truncated text if needed, original text otherwise
        """
        # Very conservative estimate: 3 chars per token, with 20% safety buffer
        # For 2048 token limit: 2048 * 3 * 0.8 = ~4915 chars max
        max_chars = int(self.max_tokens * 3 * 0.8)

        if len(text) > max_chars:
            truncated = text[:max_chars]
            logger.warning(
                f"Truncated text from {len(text)} to {len(truncated)} chars "
                f"to fit {self.max_tokens} token limit (preview: {text[:100]}...)"
            )
            return truncated

        return text

    async def _generate_embedding_single(
        self, text: str, max_retries: int = 3
    ) -> List[float]:
        """Generate embedding for a single text using Ollama API.

        Args:
            text: Text to generate embedding for
            max_retries: Maximum number of retry attempts for transient errors

        Returns:
            Embedding vector

        Raises:
            httpx.HTTPError: If API request fails after all retries
        """
        # Truncate text to fit model's context window
        original_len = len(text)
        text = self._truncate_text(text)

        client = self._get_client()
        semaphore = self._semaphore or asyncio.Semaphore(self.max_concurrent)

        async with semaphore:
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.host}/api/embeddings",
                        json={"model": self.model, "prompt": text},
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["embedding"]
                except httpx.HTTPStatusError as e:
                    last_error = e
                    # Retry on 500 errors (server overload) with exponential backoff
                    if e.response.status_code >= 500 and attempt < max_retries - 1:
                        wait_time = 2**attempt  # 1s, 2s, 4s
                        logger.warning(
                            f"Ollama 500 error (attempt {attempt + 1}/{max_retries}), "
                            f"text_len={len(text)} (original={original_len}), retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    # Log final failure with text details
                    logger.error(
                        f"Ollama API error {e.response.status_code}: text_len={len(text)} "
                        f"(original={original_len}), preview: {text[:200]}..."
                    )
                    # Don't retry on 4xx errors (client errors)
                    raise
                except httpx.HTTPError as e:
                    logger.error(f"Ollama API error: {e}")
                    raise
                except KeyError as e:
                    logger.error(f"Unexpected API response format: {e}")
                    raise

            # If we exhausted all retries
            logger.error(
                f"Failed after {max_retries} attempts, text_len={len(text)} "
                f"(original={original_len})"
            )
            raise last_error

    async def generate_embeddings(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to generate embeddings for
            use_cache: Whether to use cached embeddings

        Returns:
            List of embedding vectors corresponding to input texts
        """
        embeddings = []
        texts_to_generate = []
        cache_keys = []

        # Check cache first
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_keys.append(cache_key)

            if use_cache and self.cache_dir:
                cached = self._get_cached_embedding(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    continue

            # Need to generate this embedding
            texts_to_generate.append(text)
            embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if texts_to_generate:
            logger.info(
                f"Generating {len(texts_to_generate)} embeddings "
                f"({len(texts) - len(texts_to_generate)} cached)"
            )

            tasks = [self._generate_embedding_single(text) for text in texts_to_generate]
            # Use return_exceptions=True to allow partial success
            generated = await asyncio.gather(*tasks, return_exceptions=True)

            # Fill in generated embeddings and cache them
            # Skip any that failed (will be exceptions)
            gen_idx = 0
            failed_count = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    result = generated[gen_idx]
                    # Check if this embedding generation failed
                    if isinstance(result, Exception):
                        failed_count += 1
                        logger.warning(
                            f"Failed to generate embedding for chunk {i}: {result}"
                        )
                        # Leave as None - will be filtered out later
                        embeddings[i] = None
                    else:
                        embeddings[i] = result
                        if use_cache and self.cache_dir:
                            self._save_cached_embedding(cache_keys[i], result)
                    gen_idx += 1

            if failed_count > 0:
                logger.warning(
                    f"{failed_count}/{len(texts_to_generate)} embeddings failed, "
                    f"continuing with successful ones"
                )

        return embeddings

    async def generate_embeddings_batched(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Generate embeddings in batches for better performance.

        Args:
            texts: List of texts to generate embeddings for
            use_cache: Whether to use cached embeddings

        Returns:
            List of embedding vectors corresponding to input texts
        """
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(f"Processing batch {i // self.batch_size + 1}")
            batch_embeddings = await self.generate_embeddings(batch, use_cache=use_cache)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def health_check(self) -> bool:
        """Check if Ollama is healthy and the model is available.

        Returns:
            True if healthy, False otherwise
        """
        try:
            client = self._get_client()

            # Check if Ollama is running
            response = await client.get(f"{self.host}/api/tags")
            response.raise_for_status()

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            # Check for exact match or match with :latest suffix
            model_found = self.model in model_names or f"{self.model}:latest" in model_names

            if not model_found:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {model_names}"
                )
                logger.info(f"Run: ollama pull {self.model}")
                return False

            logger.info(f"Ollama health check passed. Model '{self.model}' is available.")
            return True

        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "cached_embeddings": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing httpx client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
