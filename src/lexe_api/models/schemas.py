"""LEXe API Schemas.

Pydantic models for request/response validation.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class ActType(str, Enum):
    """Italian act types."""

    LEGGE = "legge"
    DECRETO_LEGGE = "decreto legge"
    DECRETO_LEGISLATIVO = "decreto legislativo"
    DPR = "d.p.r."
    REGIO_DECRETO = "regio decreto"
    CODICE_CIVILE = "codice civile"
    CODICE_PENALE = "codice penale"
    CODICE_PROCEDURA_CIVILE = "codice procedura civile"
    CODICE_PROCEDURA_PENALE = "codice procedura penale"
    COSTITUZIONE = "costituzione"


class EuActType(str, Enum):
    """European act types."""

    REGOLAMENTO = "regolamento"
    DIRETTIVA = "direttiva"
    DECISIONE = "decisione"
    TRATTATO = "trattato"
    RACCOMANDAZIONE = "raccomandazione"


class DocumentVersion(str, Enum):
    """Document version types."""

    VIGENTE = "vigente"
    ORIGINALE = "originale"


class ToolState(str, Enum):
    """Tool health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# Normattiva Models
# =============================================================================


class NormattivaRequest(BaseModel):
    """Request to search Normattiva."""

    act_type: str = Field(..., description="Tipo atto: legge, decreto legislativo, etc.")
    date: str | None = Field(None, description="Data atto YYYY-MM-DD o YYYY")
    act_number: str | None = Field(None, description="Numero atto")
    article: str | None = Field(None, description="Numero articolo")
    version: DocumentVersion = Field(
        default=DocumentVersion.VIGENTE,
        description="Versione: vigente o originale",
    )


class VigenzaResponse(BaseModel):
    """Vigenza (validity) check response."""

    urn: str = Field(..., description="URN Normattiva")
    is_vigente: bool = Field(..., description="Articolo ancora vigente")
    abrogato_da: str | None = Field(None, description="URN che lo abroga")
    modificato_da: list[str] = Field(default_factory=list, description="URN modifiche")
    data_verifica: datetime = Field(..., description="Data verifica")


class NormattivaResponse(BaseModel):
    """Response from Normattiva search."""

    success: bool = Field(default=True)
    urn: str = Field(..., description="URN Normattiva")
    codice_redazionale: str | None = Field(None, description="Codice redazionale")
    data_gu: str | None = Field(None, description="Data Gazzetta Ufficiale")

    # Content
    title: str | None = Field(None, description="Titolo atto")
    text: str | None = Field(None, description="Testo articolo/atto")

    # Vigenza
    vigente: bool = Field(default=True, description="Articolo vigente")
    note_modifiche: str | None = Field(None, description="Note su modifiche")

    # Metadata
    act_type: str | None = None
    act_number: str | None = None
    act_date: date | None = None
    article: str | None = None
    version: str = "vigente"

    # Source info
    source: str = "normattiva"
    cached: bool = Field(default=False, description="Risultato da cache")
    scraped_at: datetime | None = None


# =============================================================================
# EUR-Lex Models
# =============================================================================


class EurLexRequest(BaseModel):
    """Request to search EUR-Lex."""

    act_type: EuActType = Field(..., description="Tipo atto EU")
    year: int = Field(..., ge=1950, le=2100, description="Anno pubblicazione")
    number: int = Field(..., ge=1, description="Numero atto")
    article: str | None = Field(None, description="Articolo specifico")
    language: str = Field(default="ita", description="Lingua: ita, eng, fra, deu, spa")


class EurLexResponse(BaseModel):
    """Response from EUR-Lex search."""

    success: bool = Field(default=True)
    celex: str = Field(..., description="CELEX ID")
    eli: str | None = Field(None, description="ELI identifier")

    # Content
    title: str | None = Field(None, description="Titolo atto")
    text: str | None = Field(None, description="Testo articolo/atto")
    preamble: str | None = Field(None, description="Preambolo")

    # Metadata
    act_type: str | None = None
    year: int | None = None
    number: int | None = None
    article: str | None = None
    publication_date: date | None = None
    entry_into_force: date | None = None

    # Status
    in_force: bool = Field(default=True, description="In vigore")

    # Source info
    source: str = "eurlex"
    language: str = "ita"
    cached: bool = False
    scraped_at: datetime | None = None


# =============================================================================
# InfoLex (Brocardi) Models
# =============================================================================


class InfoLexRequest(BaseModel):
    """Request to search InfoLex/Brocardi."""

    act_type: str = Field(..., description="Tipo atto: codice civile, codice penale, etc.")
    article: str = Field(..., description="Numero articolo")
    include_massime: bool = Field(default=True, description="Includere massime")
    include_relazioni: bool = Field(default=False, description="Includere relazioni")
    include_footnotes: bool = Field(default=False, description="Includere note")


class MassimaResponse(BaseModel):
    """Case law summary (massima)."""

    id: UUID | None = None
    autorita: str = Field(..., description="Autorità: Cass. civ., Corte Cost., etc.")
    sezione: str | None = Field(None, description="Sezione")
    numero: str | None = Field(None, description="Numero sentenza")
    data: date | None = Field(None, description="Data sentenza")
    testo: str = Field(..., description="Testo massima")
    principio: str | None = Field(None, description="Principio di diritto")
    keywords: list[str] = Field(default_factory=list)
    materia: str | None = None


class InfoLexResponse(BaseModel):
    """Response from InfoLex/Brocardi search."""

    success: bool = Field(default=True)

    # Article info
    act_type: str | None = None
    article: str | None = None
    article_title: str | None = None
    article_text: str | None = None

    # Brocardi content
    massime: list[MassimaResponse] = Field(default_factory=list)
    relazioni: list[dict[str, Any]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    spiegazione: str | None = Field(None, description="Spiegazione Brocardi")

    # Source
    brocardi_url: str | None = None
    source: str = "brocardi"
    cached: bool = False
    scraped_at: datetime | None = None


# =============================================================================
# Generic Document Models
# =============================================================================


class DocumentResponse(BaseModel):
    """Generic document response."""

    id: UUID | None = None
    source: str
    urn: str | None = None
    act_type: str | None = None
    act_number: str | None = None
    article: str | None = None
    title: str | None = None
    content: str | None = None
    is_vigente: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    scraped_at: datetime | None = None


# =============================================================================
# Health Models
# =============================================================================


class HealthStatus(BaseModel):
    """Service health status."""

    status: str = "healthy"
    version: str
    database: bool = True
    cache: bool = True
    tools: dict[str, bool] = Field(default_factory=dict)


class ToolHealthResponse(BaseModel):
    """Individual tool health status."""

    tool_name: str
    state: ToolState = ToolState.HEALTHY
    circuit_state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error_message: str | None = None
    fallback_available: bool = True


# =============================================================================
# KB Massime Search Models
# =============================================================================


class KBSearchRequest(BaseModel):
    """Request to search KB massime."""

    query: str = Field(..., min_length=3, description="Search query in natural language")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum relevance score threshold"
    )
    filters: dict[str, Any] | None = Field(
        None, description="Optional filters (materia, sezione, anno, etc.)"
    )


class KBMassimaResult(BaseModel):
    """Single KB search result."""

    massima_id: UUID
    testo: str = Field(..., description="Testo della massima")
    sezione: str | None = Field(None, description="Sezione (e.g., Sez. III)")
    numero: str | None = Field(None, description="Numero sentenza")
    anno: int | None = Field(None, description="Anno sentenza")
    rv: str | None = Field(None, description="RV (numero identificativo)")
    materia: str | None = Field(None, description="Materia/categoria")

    # Search scores
    score: float = Field(..., description="Combined relevance score")
    dense_score: float | None = Field(None, description="Dense (embedding) score")
    sparse_score: float | None = Field(None, description="Sparse (BM25) score")
    rank: int = Field(..., description="Result rank (1-based)")


class KBSearchResponse(BaseModel):
    """Response from KB massime search."""

    success: bool = Field(default=True)
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Number of results returned")
    results: list[KBMassimaResult] = Field(default_factory=list)

    # Search metadata
    search_mode: str = Field(default="hybrid", description="Search mode used")
    source: str = "kb_massime"


# =============================================================================
# Error Models
# =============================================================================


class ErrorResponse(BaseModel):
    """API error response."""

    success: bool = False
    error: str
    error_type: str | None = None
    details: dict[str, Any] | None = None
