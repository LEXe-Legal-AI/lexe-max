"""
LEXE Knowledge Base - OpenRouter Embedding Client

Embedder per articoli normativi via OpenRouter API.
Usa text-embedding-3-small (1536 dims) - benchmark winner.

IMPORTANTE: Se OpenRouter non disponibile, la pipeline deve fermarsi.
NO FALLBACK a modelli locali.
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Constants
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/text-embedding-3-small"
DEFAULT_DIMS = 1536
MAX_BATCH_SIZE = 100  # OpenRouter supports large batches
MAX_RETRIES = 3


class OpenRouterUnavailableError(Exception):
    """Raised when OpenRouter API is not available and there's no fallback."""

    pass


class OpenRouterRateLimitError(Exception):
    """Raised when hitting rate limits."""

    pass


@dataclass
class EmbeddingResult:
    """Risultato embedding singolo."""

    text_hash: str
    embedding: list[float]
    dims: int
    model: str
    latency_ms: float


@dataclass
class BatchEmbeddingResult:
    """Risultato batch embedding."""

    results: list[EmbeddingResult]
    total_tokens: int
    total_latency_ms: float
    failed_indices: list[int] = field(default_factory=list)


class OpenRouterEmbedder:
    """
    Client per generazione embedding via OpenRouter.

    Usa text-embedding-3-small come unico modello (benchmark winner).
    NO FALLBACK - se OpenRouter non disponibile, raise error.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Inizializza embedder.

        Args:
            api_key: OpenRouter API key (default from env OPENROUTER_API_KEY)
            model: Modello embedding (default text-embedding-3-small)
            timeout: Timeout per request
            max_retries: Numero max retry

        Raises:
            ValueError: Se API key non fornita e non in environment
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

        # Cache embeddings per evitare ricalcolo
        self._cache: dict[str, list[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Ottieni o crea client HTTP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Chiudi client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Calcola hash SHA256 del testo per cache/dedup."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed_single(
        self,
        text: str,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """
        Genera embedding per singolo testo.

        Args:
            text: Testo da embeddare
            use_cache: Usa cache in-memory

        Returns:
            EmbeddingResult con embedding

        Raises:
            OpenRouterUnavailableError: Se API non disponibile
        """
        text_hash = self.compute_text_hash(text)

        # Check cache
        if use_cache and text_hash in self._cache:
            self._cache_hits += 1
            return EmbeddingResult(
                text_hash=text_hash,
                embedding=self._cache[text_hash],
                dims=len(self._cache[text_hash]),
                model=self.model,
                latency_ms=0.0,
            )

        self._cache_misses += 1
        start = time.time()
        client = await self._get_client()

        payload = {
            "model": self.model,
            "input": text,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lexe.pro",
            "X-Title": "LEXE Knowledge Base",
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{OPENROUTER_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 429:
                    # Rate limit - wait and retry
                    wait_time = 2**attempt
                    logger.warning(
                        "Rate limited, waiting",
                        wait_seconds=wait_time,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()
                embedding = data["data"][0]["embedding"]
                latency_ms = (time.time() - start) * 1000

                # Cache result
                if use_cache:
                    self._cache[text_hash] = embedding

                logger.debug(
                    "Embedding generated",
                    model=self.model,
                    dims=len(embedding),
                    latency_ms=round(latency_ms, 1),
                )

                return EmbeddingResult(
                    text_hash=text_hash,
                    embedding=embedding,
                    dims=len(embedding),
                    model=self.model,
                    latency_ms=latency_ms,
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    "Embedding request failed",
                    model=self.model,
                    attempt=attempt + 1,
                    status=e.response.status_code,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

            except httpx.ConnectError as e:
                last_error = e
                logger.error(
                    "Cannot connect to OpenRouter",
                    error=str(e),
                )
                raise OpenRouterUnavailableError(f"Cannot connect to OpenRouter API: {e}") from e

            except Exception as e:
                last_error = e
                logger.error(
                    "Embedding error",
                    model=self.model,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise OpenRouterUnavailableError(
            f"Failed to generate embedding after {self.max_retries} attempts: {last_error}"
        )

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = MAX_BATCH_SIZE,
        use_cache: bool = True,
    ) -> BatchEmbeddingResult:
        """
        Genera embedding per batch di testi.

        Args:
            texts: Lista testi
            batch_size: Dimensione batch (max 100)
            use_cache: Usa cache in-memory

        Returns:
            BatchEmbeddingResult con tutti gli embedding

        Raises:
            OpenRouterUnavailableError: Se API non disponibile
        """
        start = time.time()
        all_results: list[EmbeddingResult] = []
        failed_indices: list[int] = []
        total_tokens = 0

        # Pre-compute hashes and check cache
        text_hashes = [self.compute_text_hash(t) for t in texts]
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, (text, text_hash) in enumerate(zip(texts, text_hashes, strict=False)):
            if use_cache and text_hash in self._cache:
                self._cache_hits += 1
                all_results.append(
                    EmbeddingResult(
                        text_hash=text_hash,
                        embedding=self._cache[text_hash],
                        dims=len(self._cache[text_hash]),
                        model=self.model,
                        latency_ms=0.0,
                    )
                )
            else:
                self._cache_misses += 1
                uncached_indices.append(i)
                uncached_texts.append(text)
                all_results.append(None)  # Placeholder

        if not uncached_texts:
            # All cached!
            return BatchEmbeddingResult(
                results=all_results,
                total_tokens=0,
                total_latency_ms=0.0,
            )

        # Process uncached texts in batches
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lexe.pro",
            "X-Title": "LEXE Knowledge Base",
        }

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_original_indices = uncached_indices[batch_start:batch_end]

            payload = {
                "model": self.model,
                "input": batch_texts,
            }

            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        f"{OPENROUTER_BASE_URL}/embeddings",
                        headers=headers,
                        json=payload,
                    )

                    if response.status_code == 429:
                        wait_time = 2**attempt
                        logger.warning(
                            "Rate limited on batch, waiting",
                            wait_seconds=wait_time,
                            batch_start=batch_start,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    # Sort by index and extract embeddings
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])

                    for j, item in enumerate(sorted_data):
                        embedding = item["embedding"]
                        original_idx = batch_original_indices[j]
                        text_hash = text_hashes[original_idx]

                        # Cache
                        if use_cache:
                            self._cache[text_hash] = embedding

                        all_results[original_idx] = EmbeddingResult(
                            text_hash=text_hash,
                            embedding=embedding,
                            dims=len(embedding),
                            model=self.model,
                            latency_ms=0.0,  # Batch latency computed at end
                        )

                    # Track tokens if available
                    if "usage" in data:
                        total_tokens += data["usage"].get("total_tokens", 0)

                    break  # Success, exit retry loop

                except httpx.ConnectError as e:
                    raise OpenRouterUnavailableError(
                        f"Cannot connect to OpenRouter API: {e}"
                    ) from e

                except Exception as e:
                    logger.warning(
                        "Batch embedding failed",
                        batch_start=batch_start,
                        batch_size=len(batch_texts),
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        # Mark all in this batch as failed
                        for idx in batch_original_indices:
                            failed_indices.append(idx)
                        logger.error(
                            "Batch failed after retries",
                            batch_start=batch_start,
                            error=str(e),
                        )

        total_latency_ms = (time.time() - start) * 1000

        logger.info(
            "Batch embedding completed",
            model=self.model,
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            processed=len(uncached_texts) - len(failed_indices),
            failed=len(failed_indices),
            latency_ms=round(total_latency_ms, 1),
        )

        return BatchEmbeddingResult(
            results=[r for r in all_results if r is not None],
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            failed_indices=failed_indices,
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Ritorna statistiche cache."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cached_embeddings": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Svuota cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


async def create_embedder_from_env() -> OpenRouterEmbedder:
    """
    Crea embedder da environment variables.

    Raises:
        ValueError: Se OPENROUTER_API_KEY non configurato
    """
    return OpenRouterEmbedder()


async def validate_openrouter_connection() -> bool:
    """
    Valida connessione OpenRouter con test embedding.

    Returns:
        True se connessione OK
    """
    try:
        embedder = await create_embedder_from_env()
        result = await embedder.embed_single("Test connessione LEXE KB")
        await embedder.close()

        if result.dims != DEFAULT_DIMS:
            logger.warning(
                "Unexpected embedding dims",
                expected=DEFAULT_DIMS,
                actual=result.dims,
            )
            return False

        logger.info(
            "OpenRouter connection validated",
            model=DEFAULT_MODEL,
            dims=result.dims,
            latency_ms=round(result.latency_ms, 1),
        )
        return True

    except Exception as e:
        logger.error(
            "OpenRouter connection validation failed",
            error=str(e),
        )
        return False
