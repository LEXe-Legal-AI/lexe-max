"""
LEXE Knowledge Base - Multi-Embedding Generator

Generazione multi-embedding per benchmark A/B.
Supporta 4 modelli e 3 canali per massima.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from ..config import EMBEDDING_DIMS, EmbeddingChannel, EmbeddingModel

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingRequest:
    """Request per embedding singolo."""

    text: str
    model: EmbeddingModel
    channel: EmbeddingChannel
    massima_id: str | None = None


@dataclass
class EmbeddingResult:
    """Risultato embedding."""

    embedding: list[float]
    model: EmbeddingModel
    channel: EmbeddingChannel
    dims: int
    massima_id: str | None = None
    latency_ms: float = 0.0


class EmbeddingClient:
    """
    Client per generazione embedding via LiteLLM.

    Supporta batching e retry.
    """

    # Mapping modello -> nome LiteLLM
    MODEL_NAMES = {
        EmbeddingModel.QWEN3: "qwen3-embedding",
        EmbeddingModel.E5_LARGE: "multilingual-e5-large",
        EmbeddingModel.BGE_M3: "bge-m3",
        EmbeddingModel.LEGAL_BERT_IT: "legal-bert-it",
    }

    def __init__(
        self,
        litellm_url: str = "http://localhost:4000/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Inizializza client.

        Args:
            litellm_url: URL base LiteLLM
            timeout: Timeout per request
            max_retries: Numero max retry
        """
        self.litellm_url = litellm_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Ottieni o crea client HTTP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Chiudi client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed_single(
        self,
        text: str,
        model: EmbeddingModel,
    ) -> list[float]:
        """
        Genera embedding singolo.

        Args:
            text: Testo da embeddare
            model: Modello da usare

        Returns:
            Vettore embedding
        """
        import time

        client = await self._get_client()
        model_name = self.MODEL_NAMES.get(model, str(model.value))

        payload = {
            "model": model_name,
            "input": text,
        }

        start = time.time()

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.litellm_url}/embeddings",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                embedding = data["data"][0]["embedding"]

                latency = (time.time() - start) * 1000
                logger.debug(
                    "Embedding generated",
                    model=model.value,
                    dims=len(embedding),
                    latency_ms=round(latency, 1),
                )

                return embedding

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Embedding request failed",
                    model=model.value,
                    attempt=attempt + 1,
                    status=e.response.status_code,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

            except Exception as e:
                logger.error(
                    "Embedding error",
                    model=model.value,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError(f"Failed to generate embedding after {self.max_retries} attempts")

    async def embed_batch(
        self,
        texts: list[str],
        model: EmbeddingModel,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Genera embedding per batch di testi.

        Args:
            texts: Lista testi
            model: Modello
            batch_size: Dimensione batch

        Returns:
            Lista vettori embedding
        """
        import time

        client = await self._get_client()
        model_name = self.MODEL_NAMES.get(model, str(model.value))

        all_embeddings = []
        start = time.time()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            payload = {
                "model": model_name,
                "input": batch,
            }

            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        f"{self.litellm_url}/embeddings",
                        json=payload,
                    )
                    response.raise_for_status()

                    data = response.json()
                    # Ordina per index
                    embeddings = sorted(data["data"], key=lambda x: x["index"])
                    batch_embeddings = [e["embedding"] for e in embeddings]

                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    logger.warning(
                        "Batch embedding failed",
                        batch_start=i,
                        batch_size=len(batch),
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        raise

        latency = (time.time() - start) * 1000
        logger.info(
            "Batch embedding completed",
            model=model.value,
            total=len(texts),
            latency_ms=round(latency, 1),
        )

        return all_embeddings


class MultiEmbedder:
    """
    Generatore multi-embedding per massime.

    Genera embedding per:
    - 4 modelli (Qwen3, E5, BGE, LEGAL-BERT)
    - 3 canali (testo, tema, contesto)
    """

    def __init__(
        self,
        client: EmbeddingClient,
        models: list[EmbeddingModel] | None = None,
        channels: list[EmbeddingChannel] | None = None,
    ):
        """
        Inizializza multi-embedder.

        Args:
            client: Client embedding
            models: Modelli da usare (default tutti)
            channels: Canali da generare (default tutti)
        """
        self.client = client
        self.models = models or list(EmbeddingModel)
        self.channels = channels or list(EmbeddingChannel)

    def _get_channel_text(
        self,
        massima: dict[str, Any],
        channel: EmbeddingChannel,
    ) -> str:
        """
        Ottieni testo per canale specifico.

        Args:
            massima: Dict con campi massima
            channel: Canale richiesto

        Returns:
            Testo da embeddare
        """
        if channel == EmbeddingChannel.TESTO:
            return massima.get("testo", "")

        elif channel == EmbeddingChannel.TEMA:
            # Usa tema se disponibile, altrimenti estrai
            tema = massima.get("tema")
            if tema:
                return tema

            # Fallback: estrai da testo
            from .massima_extractor import extract_tema

            return extract_tema(massima.get("testo", ""))

        elif channel == EmbeddingChannel.CONTESTO:
            # Combina sezione + tipo + materia
            parts = []
            if massima.get("section_context"):
                parts.append(massima["section_context"])
            if massima.get("tipo"):
                parts.append(f"Materia: {massima['tipo']}")
            if massima.get("materia"):
                parts.append(massima["materia"])
            return " | ".join(parts) if parts else massima.get("testo", "")[:200]

        return massima.get("testo", "")

    async def embed_massima(
        self,
        massima: dict[str, Any],
    ) -> list[EmbeddingResult]:
        """
        Genera tutti gli embedding per una massima.

        Args:
            massima: Dict con campi massima (testo, tema, section_context, etc.)

        Returns:
            Lista EmbeddingResult per ogni (model, channel)
        """
        results = []
        massima_id = str(massima.get("id", ""))

        for model in self.models:
            for channel in self.channels:
                text = self._get_channel_text(massima, channel)

                if not text:
                    logger.warning(
                        "Empty text for embedding",
                        massima_id=massima_id,
                        model=model.value,
                        channel=channel.value,
                    )
                    continue

                try:
                    import time

                    start = time.time()
                    embedding = await self.client.embed_single(text, model)
                    latency = (time.time() - start) * 1000

                    result = EmbeddingResult(
                        embedding=embedding,
                        model=model,
                        channel=channel,
                        dims=len(embedding),
                        massima_id=massima_id,
                        latency_ms=latency,
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(
                        "Failed to embed massima",
                        massima_id=massima_id,
                        model=model.value,
                        channel=channel.value,
                        error=str(e),
                    )

        return results

    async def embed_batch_single_model(
        self,
        massime: list[dict[str, Any]],
        model: EmbeddingModel,
        channel: EmbeddingChannel,
        batch_size: int = 32,
    ) -> list[EmbeddingResult]:
        """
        Embed batch di massime con singolo modello/canale.

        Piu' efficiente per benchmark A/B.

        Args:
            massime: Lista massime
            model: Modello
            channel: Canale
            batch_size: Dimensione batch

        Returns:
            Lista risultati
        """
        texts = [self._get_channel_text(m, channel) for m in massime]
        ids = [str(m.get("id", "")) for m in massime]

        embeddings = await self.client.embed_batch(texts, model, batch_size)

        results = []
        for _i, (emb, massima_id) in enumerate(zip(embeddings, ids, strict=False)):
            result = EmbeddingResult(
                embedding=emb,
                model=model,
                channel=channel,
                dims=len(emb),
                massima_id=massima_id,
            )
            results.append(result)

        return results


async def validate_embedding_dims(
    model: EmbeddingModel,
    client: EmbeddingClient,
) -> bool:
    """
    Valida che modello ritorni dimensioni attese.

    Args:
        model: Modello da validare
        client: Client embedding

    Returns:
        True se dimensioni corrette
    """
    expected_dims = EMBEDDING_DIMS.get(model)
    if not expected_dims:
        logger.warning("Unknown expected dims for model", model=model.value)
        return True

    test_text = "Test di validazione dimensioni embedding."

    try:
        embedding = await client.embed_single(test_text, model)
        actual_dims = len(embedding)

        if actual_dims != expected_dims:
            logger.error(
                "Embedding dims mismatch",
                model=model.value,
                expected=expected_dims,
                actual=actual_dims,
            )
            return False

        logger.info(
            "Embedding dims validated",
            model=model.value,
            dims=actual_dims,
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to validate embedding dims",
            model=model.value,
            error=str(e),
        )
        return False


async def create_embedding_client_from_env() -> EmbeddingClient:
    """Crea client da environment variables."""
    from ..config import KBSettings

    settings = KBSettings()
    return EmbeddingClient(
        litellm_url=settings.kb_litellm_url,
    )
