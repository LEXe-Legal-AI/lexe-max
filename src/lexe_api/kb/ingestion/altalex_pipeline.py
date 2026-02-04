"""
LEXE Knowledge Base - Altalex Normativa Ingestion Pipeline

Pipeline completa per ingestionare codici e leggi italiane da PDF Altalex.
Architettura a stadi con backpressure control.

Pipeline:
    PDF (marker-pdf) → JSON → Chunk (marker_chunker) → Embed (OpenRouter) → Store (PostgreSQL)

IMPORTANTE: Se OpenRouter non disponibile, la pipeline si FERMA. NO fallback locale.

Usage:
    # Singolo file
    python -m lexe_api.kb.ingestion.altalex_pipeline single path/to/file.json CC

    # Batch directory
    python -m lexe_api.kb.ingestion.altalex_pipeline batch path/to/json/dir --commit

    # Validate only (no DB)
    python -m lexe_api.kb.ingestion.altalex_pipeline validate path/to/file.json CC
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

from .altalex_store import AltalexStore, StoreResult
from .marker_chunker import ExtractedArticle, MarkerChunker
from .openrouter_embedder import (
    BatchEmbeddingResult,
    OpenRouterEmbedder,
    OpenRouterUnavailableError,
)

logger = structlog.get_logger(__name__)


class PipelineStage(str, Enum):
    """Stadi della pipeline."""
    INIT = "init"
    CHUNK = "chunk"
    VALIDATE = "validate"
    EMBED = "embed"
    STORE = "store"
    COMPLETE = "complete"
    FAILED = "failed"


class ArticleStatus(str, Enum):
    """Stati articolo nella pipeline."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    EMBEDDED = "embedded"
    STORED = "stored"
    QUARANTINE = "quarantine"


@dataclass
class PipelineConfig:
    """Configurazione pipeline."""
    # Embedding
    embed_batch_size: int = 50  # Batch size per OpenRouter
    embed_channel: str = "testo"  # Channel embedding

    # Database
    db_batch_size: int = 100  # Batch size per INSERT
    commit: bool = False  # Commit transazioni

    # Validation
    min_testo_length: int = 10  # Minimo caratteri testo
    max_testo_length: int = 50000  # Massimo caratteri testo

    # Quarantine
    quarantine_on_error: bool = True  # Salva articoli invalidi per review

    # Checkpoint
    checkpoint_dir: str = "./checkpoints"
    enable_checkpoint: bool = True


@dataclass
class ArticleRecord:
    """Record articolo per pipeline."""
    article: ExtractedArticle
    status: ArticleStatus = ArticleStatus.PENDING
    embedding: list[float] | None = None
    embedding_hash: str | None = None
    db_id: UUID | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineStats:
    """Statistiche pipeline."""
    total_articles: int = 0
    valid_articles: int = 0
    invalid_articles: int = 0
    embedded_articles: int = 0
    stored_articles: int = 0
    quarantine_articles: int = 0

    # Timing
    chunk_time_ms: float = 0.0
    validate_time_ms: float = 0.0
    embed_time_ms: float = 0.0
    store_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Embedding
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_articles": self.total_articles,
            "valid_articles": self.valid_articles,
            "invalid_articles": self.invalid_articles,
            "embedded_articles": self.embedded_articles,
            "stored_articles": self.stored_articles,
            "quarantine_articles": self.quarantine_articles,
            "chunk_time_ms": round(self.chunk_time_ms, 1),
            "validate_time_ms": round(self.validate_time_ms, 1),
            "embed_time_ms": round(self.embed_time_ms, 1),
            "store_time_ms": round(self.store_time_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_tokens": self.total_tokens,
        }


@dataclass
class PipelineResult:
    """Risultato pipeline."""
    source_file: str
    codice: str
    stage: PipelineStage
    stats: PipelineStats
    records: list[ArticleRecord]
    document_id: UUID | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "codice": self.codice,
            "stage": self.stage.value,
            "stats": self.stats.to_dict(),
            "document_id": str(self.document_id) if self.document_id else None,
            "error_message": self.error_message,
            "articles_summary": {
                "valid": self.stats.valid_articles,
                "invalid": self.stats.invalid_articles,
                "embedded": self.stats.embedded_articles,
                "stored": self.stats.stored_articles,
            }
        }


class AltalexPipeline:
    """
    Pipeline di ingestion per normativa Altalex.

    Stadi:
    1. Chunk: marker_chunker.py (JSON → Articles)
    2. Validate: Validazione articoli
    3. Embed: OpenRouter embedding (NO FALLBACK)
    4. Store: PostgreSQL UPSERT
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedder: OpenRouterEmbedder | None = None,
        db_pool: Any | None = None,
        store: AltalexStore | None = None,
    ):
        """
        Inizializza pipeline.

        Args:
            config: Configurazione pipeline
            embedder: Embedder OpenRouter (creato se None)
            db_pool: Pool connessioni DB (opzionale, per creare store)
            store: AltalexStore preconfigurato (priorità su db_pool)
        """
        self.config = config or PipelineConfig()
        self._embedder = embedder
        self._db_pool = db_pool
        self._chunker = MarkerChunker()

        # Store: usa quello passato, o crea da db_pool
        if store is not None:
            self._store = store
        elif db_pool is not None:
            self._store = AltalexStore(db_pool)
        else:
            self._store = None

    async def _get_embedder(self) -> OpenRouterEmbedder:
        """Ottieni o crea embedder."""
        if self._embedder is None:
            self._embedder = OpenRouterEmbedder()
        return self._embedder

    async def close(self) -> None:
        """Chiudi risorse."""
        if self._embedder:
            await self._embedder.close()

    def _validate_article(self, article: ExtractedArticle) -> tuple[bool, list[str]]:
        """
        Valida singolo articolo.

        Returns:
            (is_valid, errors)
        """
        errors = []

        # 1. Numero articolo presente
        if not article.articolo_num:
            errors.append("articolo_num vuoto")

        # 2. Numero normalizzato valido
        if article.articolo_num_norm is None or article.articolo_num_norm < 1:
            errors.append(f"articolo_num_norm invalido: {article.articolo_num_norm}")

        # 3. Testo presente e lunghezza ragionevole
        testo = article.testo or ""
        if len(testo) < self.config.min_testo_length:
            errors.append(f"testo troppo corto: {len(testo)} chars")
        if len(testo) > self.config.max_testo_length:
            errors.append(f"testo troppo lungo: {len(testo)} chars")

        # 4. Content hash presente
        if not article.content_hash:
            errors.append("content_hash mancante")

        return len(errors) == 0, errors

    async def _chunk_stage(
        self,
        json_path: Path,
        codice: str,
    ) -> list[ArticleRecord]:
        """
        Stage 1: Chunking JSON → Articles.

        Args:
            json_path: Path al file JSON marker
            codice: Codice documento (CC, CP, GDPR, etc.)

        Returns:
            Lista ArticleRecord
        """
        logger.info("Starting chunk stage", path=str(json_path), codice=codice)

        articles = self._chunker.process_file(json_path, codice)

        records = []
        for article in articles:
            record = ArticleRecord(article=article)
            # Copia warnings dal chunker
            if article.warnings:
                record.warnings.extend(article.warnings)
            records.append(record)

        logger.info(
            "Chunk stage complete",
            total=len(records),
            with_warnings=sum(1 for r in records if r.warnings),
        )

        return records

    async def _validate_stage(
        self,
        records: list[ArticleRecord],
    ) -> list[ArticleRecord]:
        """
        Stage 2: Validazione articoli.

        Args:
            records: Lista ArticleRecord da validare

        Returns:
            Lista aggiornata con status
        """
        logger.info("Starting validate stage", total=len(records))

        valid_count = 0
        invalid_count = 0

        for record in records:
            is_valid, errors = self._validate_article(record.article)

            if is_valid:
                record.status = ArticleStatus.VALID
                valid_count += 1
            else:
                record.status = ArticleStatus.INVALID
                record.errors.extend(errors)
                invalid_count += 1

        logger.info(
            "Validate stage complete",
            valid=valid_count,
            invalid=invalid_count,
        )

        return records

    async def _embed_stage(
        self,
        records: list[ArticleRecord],
    ) -> tuple[list[ArticleRecord], BatchEmbeddingResult | None]:
        """
        Stage 3: Generazione embedding via OpenRouter.

        IMPORTANTE: Se OpenRouter non disponibile, raise error.
        NO FALLBACK a modelli locali.

        Args:
            records: Lista ArticleRecord validati

        Returns:
            (records aggiornati, batch_result)

        Raises:
            OpenRouterUnavailableError: Se API non disponibile
        """
        # Filtra solo articoli validi
        valid_records = [r for r in records if r.status == ArticleStatus.VALID]

        if not valid_records:
            logger.warning("No valid articles to embed")
            return records, None

        logger.info("Starting embed stage", total=len(valid_records))

        embedder = await self._get_embedder()

        # Prepara testi per embedding (usa testo_context se disponibile)
        texts = []
        for record in valid_records:
            text = record.article.testo_context or record.article.testo or ""
            # Truncate se troppo lungo (OpenAI ha limite ~8k tokens)
            if len(text) > 8000:
                text = text[:8000]
            texts.append(text)

        # Genera embeddings in batch
        batch_result = await embedder.embed_batch(
            texts,
            batch_size=self.config.embed_batch_size,
        )

        # Aggiorna records con embeddings
        embedded_count = 0
        for i, record in enumerate(valid_records):
            if i < len(batch_result.results):
                result = batch_result.results[i]
                if result:
                    record.embedding = result.embedding
                    record.embedding_hash = result.text_hash
                    record.status = ArticleStatus.EMBEDDED
                    embedded_count += 1
            else:
                record.errors.append("Embedding generation failed")
                record.status = ArticleStatus.QUARANTINE

        logger.info(
            "Embed stage complete",
            embedded=embedded_count,
            failed=len(valid_records) - embedded_count,
        )

        return records, batch_result

    async def _store_stage(
        self,
        records: list[ArticleRecord],
        codice: str,
        source_file: str,
        include_valid: bool = False,
    ) -> tuple[list[ArticleRecord], UUID | None]:
        """
        Stage 4: Store in PostgreSQL con UPSERT.

        Args:
            records: Lista ArticleRecord con embeddings
            codice: Codice documento
            source_file: Path file sorgente
            include_valid: Include VALID articles (without embeddings)

        Returns:
            (records aggiornati, document_id)
        """
        if self._store is None:
            logger.warning("No store configured, skipping store stage")
            return records, None

        # Get records to store - EMBEDDED, or VALID if include_valid=True
        statuses_to_store = {ArticleStatus.EMBEDDED}
        if include_valid:
            statuses_to_store.add(ArticleStatus.VALID)

        records_to_store = [r for r in records if r.status in statuses_to_store]

        if not records_to_store:
            logger.warning("No articles to store")
            return records, None

        logger.info("Starting store stage", total=len(records_to_store))

        # Prepare articles with embeddings for batch store
        articles_with_embeddings = [
            (record.article, record.embedding)
            for record in records_to_store
        ]

        # Batch UPSERT
        result = await self._store.store_batch(
            articles=articles_with_embeddings,
            codice=codice,
            embedding_model="openai/text-embedding-3-small",
            source_file=source_file,
            batch_size=self.config.db_batch_size,
        )

        # Update records with stored IDs
        stored_map = {sa.articolo: sa for sa in result.articles}
        for record in records_to_store:
            stored = stored_map.get(record.article.articolo_num)
            if stored:
                record.db_id = stored.id
                record.status = ArticleStatus.STORED

        logger.info(
            "Store stage complete",
            inserted=result.inserted,
            updated=result.updated,
            failed=result.failed,
        )

        # Return first stored ID as document reference
        document_id = result.articles[0].id if result.articles else None
        return records, document_id

    async def _quarantine_stage(
        self,
        records: list[ArticleRecord],
        codice: str,
        source_file: str,
    ) -> list[ArticleRecord]:
        """
        Salva articoli invalidi/falliti in quarantine per review manuale.

        Args:
            records: Lista ArticleRecord
            codice: Codice documento
            source_file: Path file sorgente

        Returns:
            Records aggiornati
        """
        quarantine_records = [
            r for r in records
            if r.status in (ArticleStatus.INVALID, ArticleStatus.QUARANTINE)
        ]

        if not quarantine_records:
            return records

        logger.info(
            "Quarantine articles",
            count=len(quarantine_records),
            codice=codice,
        )

        for record in quarantine_records:
            record.status = ArticleStatus.QUARANTINE

            # Log to database if store available
            if self._store is not None:
                await self._store.log_ingestion(
                    source_file=source_file,
                    codice=codice,
                    articolo=record.article.articolo_num,
                    stage="validate",
                    status="quarantine",
                    message="; ".join(record.errors) if record.errors else None,
                    details={
                        "warnings": record.warnings,
                        "testo_length": len(record.article.testo or ""),
                        "rubrica": record.article.rubrica,
                    },
                )

            logger.debug(
                "Quarantine article",
                articolo=record.article.articolo_num,
                errors=record.errors,
                warnings=record.warnings,
            )

        return records

    async def run(
        self,
        json_path: str | Path,
        codice: str,
        skip_embed: bool = False,
        skip_store: bool = False,
    ) -> PipelineResult:
        """
        Esegui pipeline completa.

        Args:
            json_path: Path al file JSON marker
            codice: Codice documento (CC, CP, GDPR, etc.)
            skip_embed: Skip fase embedding
            skip_store: Skip fase storage

        Returns:
            PipelineResult

        Raises:
            OpenRouterUnavailableError: Se OpenRouter non disponibile
            FileNotFoundError: Se file JSON non esiste
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        start_time = time.time()
        stats = PipelineStats()
        current_stage = PipelineStage.INIT
        records: list[ArticleRecord] = []
        document_id: UUID | None = None
        error_message: str | None = None

        try:
            # ============================================================
            # STAGE 1: CHUNKING
            # ============================================================
            current_stage = PipelineStage.CHUNK
            chunk_start = time.time()

            records = await self._chunk_stage(json_path, codice)
            stats.total_articles = len(records)
            stats.chunk_time_ms = (time.time() - chunk_start) * 1000

            # ============================================================
            # STAGE 2: VALIDATION
            # ============================================================
            current_stage = PipelineStage.VALIDATE
            validate_start = time.time()

            records = await self._validate_stage(records)
            stats.valid_articles = sum(1 for r in records if r.status == ArticleStatus.VALID)
            stats.invalid_articles = sum(1 for r in records if r.status == ArticleStatus.INVALID)
            stats.validate_time_ms = (time.time() - validate_start) * 1000

            # ============================================================
            # STAGE 3: EMBEDDING
            # ============================================================
            if not skip_embed:
                current_stage = PipelineStage.EMBED
                embed_start = time.time()

                records, batch_result = await self._embed_stage(records)

                if batch_result:
                    stats.total_tokens = batch_result.total_tokens

                # Get cache stats
                embedder = await self._get_embedder()
                cache_stats = embedder.get_cache_stats()
                stats.cache_hits = cache_stats["hits"]
                stats.cache_misses = cache_stats["misses"]

                stats.embedded_articles = sum(
                    1 for r in records if r.status == ArticleStatus.EMBEDDED
                )
                stats.embed_time_ms = (time.time() - embed_start) * 1000

            # ============================================================
            # STAGE 4: STORAGE
            # ============================================================
            if not skip_store and self._store is not None:
                current_stage = PipelineStage.STORE
                store_start = time.time()

                records, document_id = await self._store_stage(
                    records, codice, str(json_path),
                    include_valid=skip_embed,  # Store VALID articles if embed was skipped
                )
                stats.stored_articles = sum(
                    1 for r in records if r.status == ArticleStatus.STORED
                )
                stats.store_time_ms = (time.time() - store_start) * 1000

            # ============================================================
            # QUARANTINE
            # ============================================================
            if self.config.quarantine_on_error:
                records = await self._quarantine_stage(records, codice, str(json_path))
                stats.quarantine_articles = sum(
                    1 for r in records if r.status == ArticleStatus.QUARANTINE
                )

            current_stage = PipelineStage.COMPLETE

        except OpenRouterUnavailableError as e:
            current_stage = PipelineStage.FAILED
            error_message = f"OpenRouter unavailable: {e}"
            logger.error(
                "Pipeline failed - OpenRouter unavailable",
                error=str(e),
                stage=current_stage.value,
            )
            raise  # Re-raise - NO FALLBACK

        except Exception as e:
            current_stage = PipelineStage.FAILED
            error_message = str(e)
            logger.error(
                "Pipeline failed",
                error=str(e),
                stage=current_stage.value,
            )

        stats.total_time_ms = (time.time() - start_time) * 1000

        result = PipelineResult(
            source_file=str(json_path),
            codice=codice,
            stage=current_stage,
            stats=stats,
            records=records,
            document_id=document_id,
            error_message=error_message,
        )

        logger.info(
            "Pipeline complete",
            codice=codice,
            stage=current_stage.value,
            stats=stats.to_dict(),
        )

        return result


async def create_pipeline_with_db(
    config: PipelineConfig | None = None,
) -> AltalexPipeline:
    """
    Create pipeline con connessione DB dalla Knowledge Base.

    Args:
        config: Configurazione pipeline

    Returns:
        AltalexPipeline con store configurato

    Raises:
        Exception: Se connessione DB fallisce
    """
    from ...database import get_kb_pool

    pool = await get_kb_pool()
    store = AltalexStore(pool)

    return AltalexPipeline(
        config=config,
        store=store,
        db_pool=pool,
    )


async def run_single(
    json_path: str,
    codice: str,
    skip_embed: bool = False,
    skip_store: bool = True,
) -> PipelineResult:
    """
    Convenience function per processare singolo file.

    Args:
        json_path: Path al file JSON marker
        codice: Codice documento
        skip_embed: Skip embedding
        skip_store: Skip storage (default True per test)

    Returns:
        PipelineResult
    """
    pipeline = AltalexPipeline()
    try:
        return await pipeline.run(
            json_path=json_path,
            codice=codice,
            skip_embed=skip_embed,
            skip_store=skip_store,
        )
    finally:
        await pipeline.close()


async def validate_only(
    json_path: str,
    codice: str,
) -> PipelineResult:
    """
    Solo validazione, no embedding o storage.

    Args:
        json_path: Path al file JSON marker
        codice: Codice documento

    Returns:
        PipelineResult con solo chunk + validate
    """
    return await run_single(
        json_path=json_path,
        codice=codice,
        skip_embed=True,
        skip_store=True,
    )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Altalex Normativa Ingestion Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Single file
    single_parser = subparsers.add_parser("single", help="Process single JSON file")
    single_parser.add_argument("json_path", help="Path to marker JSON file")
    single_parser.add_argument("codice", help="Document code (CC, CP, GDPR, etc.)")
    single_parser.add_argument(
        "--skip-embed", action="store_true", help="Skip embedding stage"
    )
    single_parser.add_argument(
        "--skip-store", action="store_true", help="Skip storage stage"
    )

    # Validate only
    validate_parser = subparsers.add_parser("validate", help="Validate only (no embed/store)")
    validate_parser.add_argument("json_path", help="Path to marker JSON file")
    validate_parser.add_argument("codice", help="Document code")

    # Batch
    batch_parser = subparsers.add_parser("batch", help="Process directory of JSON files")
    batch_parser.add_argument("json_dir", help="Directory with marker JSON files")
    batch_parser.add_argument(
        "--commit", action="store_true", help="Commit to database"
    )

    args = parser.parse_args()

    if args.command == "single":
        result = asyncio.run(run_single(
            json_path=args.json_path,
            codice=args.codice,
            skip_embed=args.skip_embed,
            skip_store=args.skip_store,
        ))
        print(json.dumps(result.to_dict(), indent=2))

    elif args.command == "validate":
        result = asyncio.run(validate_only(
            json_path=args.json_path,
            codice=args.codice,
        ))
        print(json.dumps(result.to_dict(), indent=2))

    elif args.command == "batch":
        print("Batch processing not yet implemented")
        # TODO: Implement batch processing


if __name__ == "__main__":
    main()
