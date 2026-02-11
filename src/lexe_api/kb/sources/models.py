# lexe_api/kb/sources/models.py
"""
Pydantic models for Legal Source Adapters.

These models define the CROSS-VALIDATION CONTRACT:
Every adapter (Normattiva, StudioCataldi, Brocardi, etc.) MUST produce
these standardized outputs for consistent ingestion and comparison.
"""

import hashlib
import re
import unicodedata
from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class TrustLevel(str, Enum):
    """Source trust level for cross-validation hierarchy."""

    CANONICAL = "canonical"  # Normattiva, Gazzetta - fonte di verità
    EDITORIAL = "editorial"  # Brocardi - arricchimento strutturato
    MIRROR = "mirror"  # StudioCataldi - mirror locale


class DiffType(str, Enum):
    """Types of differences found during cross-validation."""

    EXACT = "exact"  # Hash identici
    FORMATTING = "formatting"  # Solo formattazione (spazi, newline)
    MINOR = "minor"  # Punteggiatura, typo minori
    SUBSTANTIVE = "substantive"  # Differenze di contenuto
    VERSION = "version"  # Versione diversa (multivigenza)


class ValidationAction(str, Enum):
    """Actions after cross-validation."""

    USE_CANONICAL = "use_canonical"
    FLAG_FOR_REVIEW = "flag_for_review"


# ============================================================
# NORMALIZATION HELPERS
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalizza testo per confronto hash.

    - Rimuove spazi multipli e newline extra
    - Normalizza punteggiatura
    - Rimuove accenti (per uniformità)
    - Lowercase
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove accents for comparison
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Normalize punctuation
    text = re.sub(r"[''`]", "'", text)
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"–—", "-", text)

    return text


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ============================================================
# ARTICLE EXTRACT (Normativa)
# ============================================================


class ArticleExtract(BaseModel):
    """
    Contract che ogni adapter deve produrre per articoli di codice.

    Permette cross-validation tra fonti diverse (Normattiva vs StudioCataldi)
    confrontando content_hash.
    """

    # ═══ IDENTIFICAZIONE ═══
    codice: str = Field(..., description="CC, CP, CPC, CPP, COST, CDS, etc.")
    articolo: str = Field(..., description="2043, 575, 360-bis")
    comma: str | None = Field(None, description="1, 2, bis, ter")

    # URN:NIR (standard italiano per identificativi normativi)
    urn_nir: str | None = Field(None, description="urn:nir:stato:legge:1942-03-16;262:art2043")

    # ═══ CONTENUTO ═══
    rubrica: str | None = Field(None, description="Risarcimento per fatto illecito")
    testo: str = Field(..., description="Testo articolo completo")
    testo_normalizzato: str | None = Field(None, description="Per hash/search")

    # ═══ GERARCHIA ═══
    libro: str | None = None
    titolo: str | None = None
    capo: str | None = None
    sezione: str | None = None

    # ═══ VIGENZA (Multivigenza) ═══
    data_vigenza_da: date | None = None
    data_vigenza_a: date | None = Field(None, description="None = vigente")
    nota_modifica: str | None = Field(None, description="Modificato da L. 123/2020")

    # ═══ SOURCE TRACKING ═══
    source: Literal["normattiva", "gazzetta", "studiocataldi", "brocardi"]
    source_url: str | None = None
    source_file: str | None = Field(None, description="Per mirror locali")
    retrieved_at: datetime

    # ═══ CITAZIONI (estratte dopo) ═══
    citations_raw: list[str] | None = Field(None, description="['art. 2044 c.c.', 'L. 123/2020']")

    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA256 del testo normalizzato per cross-validation."""
        return compute_content_hash(self.testo)

    @computed_field
    @property
    def is_vigente(self) -> bool:
        """True se articolo attualmente vigente."""
        return self.data_vigenza_a is None

    def model_post_init(self, __context) -> None:
        """Compute normalized text if not provided."""
        if self.testo_normalizzato is None:
            object.__setattr__(self, "testo_normalizzato", normalize_text(self.testo))


# ============================================================
# VALIDATION RESULT (Cross-Check)
# ============================================================


class ValidationResult(BaseModel):
    """
    Risultato cross-validation tra due fonti.

    Livello 1: Hash comparison (deterministico, gratis)
    Livello 2: Semantic diff con LLM (solo se hash diversi)
    """

    # Fonti confrontate
    source_a: str = Field(..., description="Fonte canonica (normattiva)")
    source_b: str = Field(..., description="Fonte mirror (studiocataldi)")

    # Articolo identificativo
    codice: str
    articolo: str
    comma: str | None = None

    # Risultato
    hash_match: bool = Field(..., description="True se hash identici")
    diff_type: DiffType | None = Field(None, description="Tipo di differenza se hash diversi")
    diff_summary: str | None = Field(None, description="Breve descrizione diff")

    # LLM analysis (solo se hash_match=False)
    llm_analyzed: bool = False
    llm_model: str | None = None
    llm_confidence: float | None = Field(None, ge=0.0, le=1.0)

    # Action
    action: ValidationAction = ValidationAction.USE_CANONICAL

    @computed_field
    @property
    def needs_review(self) -> bool:
        """True se richiede review manuale."""
        return (
            self.diff_type == DiffType.SUBSTANTIVE
            or self.action == ValidationAction.FLAG_FOR_REVIEW
        )


# ============================================================
# BROCARDI EXTRACT
# ============================================================


class BrocardiExtract(BaseModel):
    """
    Contract per brocardi latini.

    Fonte principale: brocardi.it/brocardi
    """

    # ═══ CONTENUTO ═══
    latino: str = Field(..., description="Ad impossibilia nemo tenetur")
    italiano: str | None = Field(None, description="Traduzione italiana")
    significato: str | None = Field(None, description="Spiegazione operativa")
    esempi_uso: list[str] | None = Field(None, description="Esempi pratici")

    # ═══ CLASSIFICAZIONE ═══
    tags: list[str] | None = Field(
        None, description="['contratti', 'obbligazioni', 'impossibilità']"
    )
    categoria: Literal["principio", "massima", "locuzione", "adagio", "brocardo"] | None = None
    area_diritto: (
        Literal["civile", "penale", "processuale", "amministrativo", "generale"] | None
    ) = None

    # ═══ SOURCE ═══
    source: str
    source_url: str | None = None
    attribution: str | None = Field(None, description="Autore/fonte storica")
    retrieved_at: datetime

    @computed_field
    @property
    def content_hash(self) -> str:
        """Hash per deduplicazione."""
        text = f"{self.latino}|{self.italiano or ''}"
        return compute_content_hash(text)


# ============================================================
# DIZIONARIO EXTRACT
# ============================================================


class DizionarioExtract(BaseModel):
    """
    Contract per voci dizionario giuridico.

    Fonte principale: brocardi.it/dizionario
    """

    # ═══ CONTENUTO ═══
    voce: str = Field(..., description="Capacità giuridica")
    voce_normalizzata: str | None = Field(None, description="Per search/dedup")
    definizione: str = Field(..., description="Definizione completa")
    definizione_breve: str | None = Field(None, description="One-liner")

    # ═══ COLLEGAMENTI ═══
    sinonimi: list[str] | None = Field(None, description="['capacità di diritto']")
    contrari: list[str] | None = None
    vedi_anche: list[str] | None = Field(
        None, description="['Capacità di agire', 'Persona giuridica']"
    )

    # ═══ CLASSIFICAZIONE ═══
    area: Literal["civile", "penale", "processuale", "amministrativo", "comunitario"] | None = None
    sotto_area: str | None = Field(None, description="obbligazioni, famiglia, successioni")
    tags: list[str] | None = None
    livello: Literal["base", "intermedio", "avanzato", "tecnico"] = "base"

    # ═══ SOURCE ═══
    source: str
    source_url: str | None = None
    retrieved_at: datetime

    def model_post_init(self, __context) -> None:
        """Compute normalized voce if not provided."""
        if self.voce_normalizzata is None:
            object.__setattr__(self, "voce_normalizzata", normalize_text(self.voce))

    @computed_field
    @property
    def content_hash(self) -> str:
        """Hash per deduplicazione."""
        return compute_content_hash(self.voce + "|" + self.definizione)


# ============================================================
# MIRROR SOURCE (Per tracking cross-validation)
# ============================================================


class MirrorSource(BaseModel):
    """
    Tracciamento di una fonte mirror per un articolo.

    Usato per registrare quale fonte ha fornito quale versione
    e se ha passato la cross-validation.
    """

    source: Literal["studiocataldi", "brocardi", "altalex"]
    source_url: str | None = None
    source_file: str | None = None
    retrieved_at: datetime

    text_hash: str
    match_status: Literal["exact", "format_diff", "content_diff", "pending"]
    diff_summary: str | None = None


# ============================================================
# LEGAL NUMBER (Per Number-Anchored Graph)
# ============================================================


class LegalNumberType(str, Enum):
    """Tipi di numeri legali per il grafo."""

    ARTICLE = "article"  # art. 2043 c.c.
    LAW = "law"  # L. 241/1990
    DECREE = "decree"  # D.Lgs. 165/2001, D.P.R. 380/2001
    SENTENCE = "sentence"  # Cass. 12345/2020
    REGULATION = "regulation"  # Reg. UE 679/2016


class LegalNumberExtract(BaseModel):
    """
    Numero legale estratto da un testo.

    Rappresenta un ANCHOR POINT deterministico per il grafo.
    """

    # ═══ RAW ═══
    raw_text: str = Field(..., description="art. 2043 c.c.")

    # ═══ PARSED ═══
    number_type: LegalNumberType
    codice: str | None = Field(None, description="CC, CP, L, DLGS, CASS")
    numero: str | None = Field(None, description="2043, 241, 12345")
    anno: int | None = None
    comma: str | None = None
    articolo: str | None = None  # Per leggi: articolo citato

    # ═══ CANONICAL ═══
    canonical_id: str = Field(..., description="CC:2043, L:241:1990, CASS:12345:2020")

    # ═══ CONTEXT ═══
    context_span: str | None = Field(None, description="Frase che contiene la citazione")
    position_start: int | None = None
    position_end: int | None = None

    @computed_field
    @property
    def is_code_article(self) -> bool:
        """True se è un articolo di codice (CC, CP, CPC, etc.)."""
        return self.number_type == LegalNumberType.ARTICLE

    @computed_field
    @property
    def is_external_norm(self) -> bool:
        """True se è una norma esterna (legge, decreto, etc.)."""
        return self.number_type in (LegalNumberType.LAW, LegalNumberType.DECREE)
