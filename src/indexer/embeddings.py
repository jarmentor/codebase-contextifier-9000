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
    ):
        """Initialize Ollama embeddings client.

        Args:
            host: Ollama API host URL
            model: Name of the embedding model to use
            cache_dir: Directory for caching embeddings (None to disable)
            batch_size: Number of texts to process in parallel
            max_concurrent: Maximum concurrent requests to Ollama
        """
        self.host = host.rstrip("/")
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.client = httpx.AsyncClient(timeout=60.0)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at: {self.cache_dir}")

        logger.info(f"Initialized Ollama embeddings with model: {model}")

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

    async def _generate_embedding_single(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama API.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector

        Raises:
            httpx.HTTPError: If API request fails
        """
        async with self._semaphore:
            try:
                response = await self.client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
            except httpx.HTTPError as e:
                logger.error(f"Ollama API error: {e}")
                raise
            except KeyError as e:
                logger.error(f"Unexpected API response format: {e}")
                raise

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
            generated = await asyncio.gather(*tasks)

            # Fill in generated embeddings and cache them
            gen_idx = 0
            for i, emb in enumerate(embeddings):
                if emb is None:
                    embeddings[i] = generated[gen_idx]
                    if use_cache and self.cache_dir:
                        self._save_cached_embedding(cache_keys[i], generated[gen_idx])
                    gen_idx += 1

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
            # Check if Ollama is running
            response = await self.client.get(f"{self.host}/api/tags")
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
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
