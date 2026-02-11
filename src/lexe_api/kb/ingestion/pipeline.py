"""
LEXE Knowledge Base - Ingestion Pipeline Orchestrator

Pipeline completa: PDF -> OCR -> Parse -> Extract -> Dedup -> Embed -> Store.
Supporta job queue idempotente con retry.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog

from ..config import EmbeddingChannel, KBConfig
from .citation_parser import extract_all_citations
from .deduplicator import Deduplicator
from .embedder import EmbeddingClient, MultiEmbedder
from .extractor import ExtractionResult, extract_pdf_with_quality
from .massima_extractor import (
    ExtractedMassima,
    extract_massime_from_elements,
    extract_tema,
)
from .parser import ParsedDocument, parse_document_structure

logger = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    """Stati possibili per job ingestion."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    PARSING = "parsing"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class IngestionJob:
    """Job di ingestion per singolo documento."""

    id: UUID
    source_path: str
    anno: int
    volume: int
    tipo: str  # 'civile' | 'penale'

    status: JobStatus = JobStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3

    # Progress
    current_step: str = ""
    progress_pct: float = 0.0

    # Results
    document_id: UUID | None = None
    massime_count: int = 0
    embeddings_count: int = 0
    duplicates_found: int = 0

    # Metrics
    ocr_quality_score: float | None = None
    extraction_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Errors
    error_message: str | None = None
    error_step: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializza job."""
        return {
            "id": str(self.id),
            "source_path": self.source_path,
            "anno": self.anno,
            "volume": self.volume,
            "tipo": self.tipo,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "current_step": self.current_step,
            "progress_pct": self.progress_pct,
            "document_id": str(self.document_id) if self.document_id else None,
            "massime_count": self.massime_count,
            "embeddings_count": self.embeddings_count,
            "duplicates_found": self.duplicates_found,
            "ocr_quality_score": self.ocr_quality_score,
            "extraction_time_ms": self.extraction_time_ms,
            "total_time_ms": self.total_time_ms,
            "error_message": self.error_message,
            "error_step": self.error_step,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class PipelineResult:
    """Risultato pipeline per documento."""

    job: IngestionJob
    document_id: UUID | None
    extraction: ExtractionResult | None
    parsed: ParsedDocument | None
    massime: list[ExtractedMassima]
    citations: list[dict]
    embeddings_generated: int
    duplicates: list[dict]


def compute_source_hash(path: str) -> str:
    """Calcola hash del file sorgente per idempotenza."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class IngestionPipeline:
    """
    Pipeline di ingestion completa.

    Fasi:
    1. Extract: OCR con unstructured
    2. Parse: Struttura gerarchica
    3. Process: Estrai massime e citazioni
    4. Dedup: Hash + similarity
    5. Embed: Multi-model multi-channel
    6. Store: Salva in DB
    """

    def __init__(
        self,
        config: KBConfig,
        embedding_client: EmbeddingClient | None = None,
        deduplicator: Deduplicator | None = None,
    ):
        """
        Inizializza pipeline.

        Args:
            config: Configurazione KB
            embedding_client: Client per embedding (opzionale)
            deduplicator: Deduplicatore (opzionale, creato se None)
        """
        self.config = config
        self.embedding_client = embedding_client
        self.deduplicator = deduplicator or Deduplicator(
            near_threshold=config.settings.kb_near_dedup_threshold,
        )

        # Multi-embedder (creato on-demand)
        self._multi_embedder: MultiEmbedder | None = None

    async def _get_multi_embedder(self) -> MultiEmbedder:
        """Ottieni o crea multi-embedder."""
        if self._multi_embedder is None:
            if self.embedding_client is None:
                from .embedder import create_embedding_client_from_env

                self.embedding_client = await create_embedding_client_from_env()

            self._multi_embedder = MultiEmbedder(
                client=self.embedding_client,
                models=[self.config.settings.kb_default_model],  # Start con default
                channels=[EmbeddingChannel.TESTO, EmbeddingChannel.TEMA],
            )

        return self._multi_embedder

    async def run(
        self,
        job: IngestionJob,
        skip_embedding: bool = False,
        store_callback: Any | None = None,
    ) -> PipelineResult:
        """
        Esegui pipeline completa per job.

        Args:
            job: Job da processare
            skip_embedding: Skip fase embedding (per test)
            store_callback: Callback per storage (async function)

        Returns:
            PipelineResult con tutti i dati
        """
        start_time = time.time()
        job.started_at = datetime.utcnow()
        job.status = JobStatus.EXTRACTING

        extraction = None
        parsed = None
        massime: list[ExtractedMassima] = []
        all_citations: list[dict] = []
        duplicates: list[dict] = []
        embeddings_generated = 0

        try:
            # ============================================================
            # FASE 1: ESTRAZIONE OCR
            # ============================================================
            job.current_step = "Estrazione PDF con OCR"
            job.progress_pct = 10

            logger.info(
                "Starting extraction",
                job_id=str(job.id),
                path=job.source_path,
            )

            extraction = await extract_pdf_with_quality(
                path=job.source_path,
                strategy=self.config.settings.kb_ocr_strategy,
                languages=self.config.settings.kb_ocr_languages,
            )

            job.extraction_time_ms = extraction.extraction_time_ms
            job.ocr_quality_score = extraction.metrics.quality_score

            # Check qualita' OCR
            if not extraction.metrics.is_acceptable:
                logger.warning(
                    "Low OCR quality",
                    job_id=str(job.id),
                    score=extraction.metrics.quality_score,
                    valid_chars=extraction.metrics.valid_chars_ratio,
                )

            # ============================================================
            # FASE 2: PARSING STRUTTURA
            # ============================================================
            job.status = JobStatus.PARSING
            job.current_step = "Parsing struttura gerarchica"
            job.progress_pct = 30

            parsed = parse_document_structure(extraction.elements)

            logger.info(
                "Structure parsed",
                job_id=str(job.id),
                sections=len(parsed.all_sections),
                orphans=len(parsed.orphan_elements),
            )

            # ============================================================
            # FASE 3: ESTRAZIONE MASSIME
            # ============================================================
            job.status = JobStatus.PROCESSING
            job.current_step = "Estrazione massime"
            job.progress_pct = 50

            # Estrai massime da ogni sezione
            for section in parsed.all_sections:
                section_massime = extract_massime_from_elements(
                    elements=section.elements,
                    section=section,
                    context_window=1,
                )
                massime.extend(section_massime)

            # Estrai anche da orphan elements
            orphan_massime = extract_massime_from_elements(
                elements=parsed.orphan_elements,
                section=None,
                context_window=1,
            )
            massime.extend(orphan_massime)

            logger.info(
                "Massime extracted",
                job_id=str(job.id),
                total=len(massime),
                with_citation=sum(1 for m in massime if m.citation_complete),
            )

            # ============================================================
            # FASE 4: ESTRAZIONE CITAZIONI
            # ============================================================
            job.current_step = "Estrazione citazioni"
            job.progress_pct = 60

            for massima in massime:
                citations = extract_all_citations(massima.testo)
                for c in citations:
                    all_citations.append(
                        {
                            "massima_hash": massima.content_hash,
                            "citation": c.to_dict(),
                        }
                    )

            logger.info(
                "Citations extracted",
                job_id=str(job.id),
                total=len(all_citations),
            )

            # ============================================================
            # FASE 5: DEDUPLICAZIONE
            # ============================================================
            job.current_step = "Deduplicazione"
            job.progress_pct = 70

            items_to_dedup = [(m.content_hash, m.testo_normalizzato) for m in massime]
            dedup_result = self.deduplicator.deduplicate_batch(items_to_dedup)

            job.duplicates_found = dedup_result.exact_duplicates + dedup_result.near_duplicates
            duplicates = [
                {
                    "original_hash": m.original_hash,
                    "duplicate_hash": m.duplicate_hash,
                    "type": m.match_type,
                    "similarity": m.similarity,
                }
                for m in dedup_result.matches
            ]

            # Filtra massime uniche
            duplicate_hashes = {m.duplicate_hash for m in dedup_result.matches}
            unique_massime = [m for m in massime if m.content_hash not in duplicate_hashes]

            logger.info(
                "Deduplication done",
                job_id=str(job.id),
                total=len(massime),
                unique=len(unique_massime),
                duplicates=job.duplicates_found,
            )

            # ============================================================
            # FASE 6: EMBEDDING
            # ============================================================
            if not skip_embedding and unique_massime:
                job.status = JobStatus.EMBEDDING
                job.current_step = "Generazione embedding"
                job.progress_pct = 80

                multi_embedder = await self._get_multi_embedder()

                for massima in unique_massime:
                    massima_dict = {
                        "id": massima.content_hash,
                        "testo": massima.testo,
                        "tema": extract_tema(massima.testo),
                        "section_context": massima.section_context,
                        "tipo": job.tipo,
                    }

                    try:
                        results = await multi_embedder.embed_massima(massima_dict)
                        embeddings_generated += len(results)
                    except Exception as e:
                        logger.error(
                            "Embedding failed for massima",
                            massima_hash=massima.content_hash[:16],
                            error=str(e),
                        )

                logger.info(
                    "Embeddings generated",
                    job_id=str(job.id),
                    total=embeddings_generated,
                )

            # ============================================================
            # FASE 7: STORAGE
            # ============================================================
            if store_callback:
                job.status = JobStatus.STORING
                job.current_step = "Salvataggio database"
                job.progress_pct = 90

                document_id = await store_callback(
                    job=job,
                    extraction=extraction,
                    parsed=parsed,
                    massime=unique_massime,
                    citations=all_citations,
                )
                job.document_id = document_id

            # ============================================================
            # COMPLETATO
            # ============================================================
            job.status = JobStatus.COMPLETED
            job.current_step = "Completato"
            job.progress_pct = 100
            job.massime_count = len(unique_massime)
            job.embeddings_count = embeddings_generated
            job.completed_at = datetime.utcnow()
            job.total_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Pipeline completed",
                job_id=str(job.id),
                massime=job.massime_count,
                embeddings=job.embeddings_count,
                duplicates=job.duplicates_found,
                total_time_ms=round(job.total_time_ms, 1),
            )

            return PipelineResult(
                job=job,
                document_id=job.document_id,
                extraction=extraction,
                parsed=parsed,
                massime=unique_massime,
                citations=all_citations,
                embeddings_generated=embeddings_generated,
                duplicates=duplicates,
            )

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.error_step = job.current_step
            job.total_time_ms = (time.time() - start_time) * 1000

            logger.error(
                "Pipeline failed",
                job_id=str(job.id),
                step=job.current_step,
                error=str(e),
            )

            # Retry logic
            if job.retry_count < job.max_retries:
                job.status = JobStatus.RETRYING
                job.retry_count += 1
                logger.info(
                    "Scheduling retry",
                    job_id=str(job.id),
                    retry_count=job.retry_count,
                )

            return PipelineResult(
                job=job,
                document_id=None,
                extraction=extraction,
                parsed=parsed,
                massime=massime,
                citations=all_citations,
                embeddings_generated=embeddings_generated,
                duplicates=duplicates,
            )


def create_job(
    source_path: str,
    anno: int,
    volume: int,
    tipo: str,
    max_retries: int = 3,
) -> IngestionJob:
    """
    Crea nuovo job di ingestion.

    Args:
        source_path: Path al file PDF
        anno: Anno massimario
        volume: Volume
        tipo: "civile" o "penale"
        max_retries: Max retry

    Returns:
        IngestionJob configurato
    """
    return IngestionJob(
        id=uuid4(),
        source_path=source_path,
        anno=anno,
        volume=volume,
        tipo=tipo,
        max_retries=max_retries,
    )


async def run_single_document(
    path: str,
    anno: int,
    volume: int,
    tipo: str,
    config: KBConfig | None = None,
    skip_embedding: bool = False,
) -> PipelineResult:
    """
    Convenience function per processare singolo documento.

    Args:
        path: Path PDF
        anno: Anno
        volume: Volume
        tipo: civile/penale
        config: Config (default from env)
        skip_embedding: Skip embedding

    Returns:
        PipelineResult
    """
    if config is None:
        config = KBConfig.from_env()

    job = create_job(
        source_path=path,
        anno=anno,
        volume=volume,
        tipo=tipo,
    )

    pipeline = IngestionPipeline(config=config)
    return await pipeline.run(job, skip_embedding=skip_embedding)
