"""
LEXE Knowledge Base - Pydantic Models
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .config import EmbeddingChannel, EmbeddingModel, SystemConfig

# ============================================================
# Document Models
# ============================================================


class DocumentBase(BaseModel):
    """Base model per documenti PDF."""

    source_path: str
    anno: int
    volume: int
    tipo: str  # 'civile' | 'penale'
    titolo: str | None = None
    pagine: int | None = None


class DocumentCreate(DocumentBase):
    """Model per creazione documento."""

    source_hash: str


class OCRQualityMetrics(BaseModel):
    """Metriche qualita' OCR."""

    valid_chars_ratio: float = Field(ge=0, le=1)
    italian_tokens_ratio: float = Field(ge=0, le=1)
    citation_regex_success: float = Field(ge=0, le=1)

    @property
    def quality_score(self) -> float:
        """Score complessivo OCR."""
        return (
            self.valid_chars_ratio * 0.4
            + self.italian_tokens_ratio * 0.4
            + self.citation_regex_success * 0.2
        )


class Document(DocumentBase):
    """Model documento completo."""

    id: UUID
    source_hash: str
    ocr_quality_score: float | None = None
    ocr_valid_chars_ratio: float | None = None
    ocr_italian_tokens_ratio: float | None = None
    ocr_citation_regex_success: float | None = None
    processed_at: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Section Models
# ============================================================


class SectionBase(BaseModel):
    """Base model per sezioni gerarchiche."""

    level: int  # 1=parte, 2=capitolo, 3=sezione, 4=sottosezione
    titolo: str
    pagina_inizio: int | None = None
    pagina_fine: int | None = None
    section_path: str  # "PARTE I > Cap. 1 > Sez. 2"


class SectionCreate(SectionBase):
    """Model per creazione sezione."""

    document_id: UUID
    parent_id: UUID | None = None


class Section(SectionBase):
    """Model sezione completo."""

    id: UUID
    document_id: UUID
    parent_id: UUID | None = None
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Massima Models
# ============================================================


class CitationNormalized(BaseModel):
    """Citazione normalizzata."""

    sezione: str | None = None  # "Sez. U", "Sez. 1"
    numero: str | None = None  # "12345"
    anno: int | None = None
    data_decisione: date | None = None
    rv: str | None = None  # "Rv. 123456-01"
    relatore: str | None = None

    @property
    def is_complete(self) -> bool:
        """Citazione completa se ha almeno sezione, numero e anno."""
        return all([self.sezione, self.numero, self.anno])


class MassimaBase(BaseModel):
    """Base model per massime atomiche."""

    testo: str  # Chunk A: massima pulita
    testo_con_contesto: str | None = None  # Chunk B: con blocchi attigui
    citation: CitationNormalized = Field(default_factory=CitationNormalized)
    pagina_inizio: int | None = None
    pagina_fine: int | None = None
    tipo: str | None = None  # civile/penale
    materia: str | None = None
    keywords: list[str] = Field(default_factory=list)


class MassimaCreate(MassimaBase):
    """Model per creazione massima."""

    document_id: UUID
    section_id: UUID | None = None
    testo_normalizzato: str
    content_hash: str


class Massima(MassimaBase):
    """Model massima completo."""

    id: UUID
    document_id: UUID
    section_id: UUID | None = None
    testo_normalizzato: str
    content_hash: str
    importance_score: float = 0.5
    confirmation_count: int = 1
    citation_extracted: bool = False
    citation_complete: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MassimaWithEvidence(MassimaBase):
    """Massima con chunk A (retrieval) e B (generation)."""

    chunk_a: str  # Testo pulito
    chunk_b: str  # Con contesto OCR


# ============================================================
# Embedding Models
# ============================================================


class EmbeddingCreate(BaseModel):
    """Model per creazione embedding."""

    massima_id: UUID
    model: EmbeddingModel
    channel: EmbeddingChannel
    embedding: list[float]
    dims: int


class Embedding(EmbeddingCreate):
    """Model embedding completo."""

    id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Citation Models
# ============================================================


class CitationExtracted(BaseModel):
    """Citazione/norma estratta da massima."""

    tipo: str  # 'pronuncia', 'norma', 'regolamento_ue', 'direttiva_ue'
    raw_text: str
    # Pronuncia
    sezione: str | None = None
    numero: str | None = None
    anno: int | None = None
    data_decisione: date | None = None
    rv: str | None = None
    # Norma
    articolo: str | None = None
    comma: str | None = None
    codice: str | None = None  # "c.c.", "c.p.c."
    # EU
    regolamento: str | None = None
    direttiva: str | None = None
    celex: str | None = None


class CitationCreate(CitationExtracted):
    """Model per creazione citation."""

    massima_id: UUID


class Citation(CitationExtracted):
    """Model citation completo."""

    id: UUID
    massima_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# Search Models
# ============================================================


class SearchFilters(BaseModel):
    """Filtri per ricerca."""

    anno_min: int | None = None
    anno_max: int | None = None
    tipo: str | None = None  # civile/penale
    sezione: str | None = None
    materia: str | None = None
    codice: str | None = None  # Per citazioni norme


class SearchRequest(BaseModel):
    """Request per ricerca KB."""

    query: str
    system: SystemConfig = SystemConfig.S2_HYBRID
    model: EmbeddingModel = EmbeddingModel.QWEN3
    channel: EmbeddingChannel = EmbeddingChannel.TESTO
    limit: int = Field(default=20, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    use_graph_expansion: bool = False
    use_rerank: bool = False
    temporal_boost: bool = False


class SearchResult(BaseModel):
    """Risultato singolo ricerca."""

    massima: Massima
    score: float
    rrf_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    graph_expanded: bool = False


class SearchResponse(BaseModel):
    """Response ricerca KB."""

    query: str
    system: SystemConfig
    results: list[SearchResult]
    total: int
    filters_applied: SearchFilters
    latency_ms: float


# ============================================================
# Ingestion Models
# ============================================================


class IngestionJobStatus(BaseModel):
    """Stato job ingestion."""

    id: UUID
    document_id: UUID | None = None
    status: str  # pending, processing, completed, failed, retrying
    retry_count: int = 0
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime


class IngestionRequest(BaseModel):
    """Request per ingestion documento."""

    source_path: str
    anno: int
    volume: int
    tipo: str  # civile/penale


class IngestionResponse(BaseModel):
    """Response ingestion."""

    job_id: UUID
    status: str
    message: str


# ============================================================
# Benchmark Models
# ============================================================


class BenchmarkMetrics(BaseModel):
    """Metriche benchmark."""

    # Retrieval
    recall_at_20: float
    mrr_at_10: float
    precision_at_5: float
    # Answer quality
    groundedness: float
    citation_completeness: float
    contradiction_rate: float
    # System
    latency_p95_ms: float
    cost_per_query: float | None = None
    # OCR
    ocr_valid_chars_ratio: float | None = None
    ocr_citation_regex_success: float | None = None
    citation_pinpoint_accuracy: float | None = None

    @property
    def final_score(self) -> float:
        """Calcola score finale pesato."""
        retrieval = (self.recall_at_20 * 16 + self.mrr_at_10 * 16 + self.precision_at_5 * 8) / 40
        answer = (
            self.groundedness * 16
            + self.citation_completeness * 16
            + (1 - self.contradiction_rate) * 8
        ) / 40
        # Normalizza latency (assume 1000ms max)
        system = 1 - min(self.latency_p95_ms / 1000, 1)
        # OCR (se disponibile)
        ocr = 1.0
        if self.ocr_valid_chars_ratio is not None:
            ocr = (self.ocr_valid_chars_ratio * 2 + (self.ocr_citation_regex_success or 0) * 1) / 3
        # Citation pinpoint
        citation = self.citation_pinpoint_accuracy or 1.0

        return (
            retrieval * 0.40 + answer * 0.40 + system * 0.10 + ocr * 0.05 + citation * 0.05
        ) * 100


class BenchmarkRun(BaseModel):
    """Run benchmark completo."""

    id: UUID
    config_name: str
    embedding_model: EmbeddingModel
    system_config: SystemConfig
    metrics: BenchmarkMetrics
    query_count: int
    created_at: datetime

    class Config:
        from_attributes = True
