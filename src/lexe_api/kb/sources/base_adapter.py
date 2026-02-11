# lexe_api/kb/sources/base_adapter.py
"""
Base Legal Source Adapter Interface.

Abstract interface che ogni adapter (Normattiva, StudioCataldi, Brocardi, etc.)
DEVE implementare per garantire interoperabilità e cross-validation.

Trust Hierarchy:
- CANONICAL (Normattiva, Gazzetta): Fonte di verità, usata per verificare altri
- EDITORIAL (Brocardi): Arricchimento strutturato, buona qualità
- MIRROR (StudioCataldi): Mirror locale, richiede SEMPRE cross-check
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

import structlog

from lexe_api.kb.sources.models import (
    ArticleExtract,
    BrocardiExtract,
    DizionarioExtract,
    TrustLevel,
    ValidationResult,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# ============================================================
# EXCEPTIONS
# ============================================================


class AdapterError(Exception):
    """Base exception for adapter errors."""

    def __init__(self, source: str, message: str):
        self.source = source
        self.message = message
        super().__init__(f"[{source}] {message}")


class FetchError(AdapterError):
    """Error fetching content from source."""

    pass


class ParseError(AdapterError):
    """Error parsing content from source."""

    pass


class RateLimitError(AdapterError):
    """Rate limit exceeded on source."""

    def __init__(self, source: str, retry_after: int | None = None):
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(source, msg)


class ArticleNotFoundError(AdapterError):
    """Article not found in source."""

    def __init__(self, source: str, codice: str, articolo: str):
        self.codice = codice
        self.articolo = articolo
        super().__init__(source, f"Article {codice}:{articolo} not found")


# ============================================================
# CODICE MAPPING (Standard codes)
# ============================================================

CODICE_FULL_NAMES = {
    "CC": "Codice Civile",
    "CP": "Codice Penale",
    "CPC": "Codice di Procedura Civile",
    "CPP": "Codice di Procedura Penale",
    "COST": "Costituzione",
    "CDS": "Codice della Strada",
    "CDPR": "Codice della Privacy",
    "CCONS": "Codice del Consumo",
    "CAPPALTI": "Codice degli Appalti",
    "CAMB": "Codice dell'Ambiente",
    "CNAV": "Codice della Navigazione",
    "CNAUT": "Codice della Nautica",
    "TUEL": "Testo Unico Enti Locali",
    "CAD": "Codice Amministrazione Digitale",
    "TUB": "Testo Unico Bancario",
    "TUF": "Testo Unico Finanza",
}

# URN:NIR base references (data promulgazione)
URN_NIR_BASES = {
    "CC": "urn:nir:stato:regio.decreto:1942-03-16;262",
    "CP": "urn:nir:stato:regio.decreto:1930-10-19;1398",
    "CPC": "urn:nir:stato:regio.decreto:1940-10-28;1443",
    "CPP": "urn:nir:stato:decreto.presidente.repubblica:1988-09-22;447",
    "COST": "urn:nir:stato:costituzione:1947-12-27",
    "CDS": "urn:nir:stato:decreto.legislativo:1992-04-30;285",
}


# ============================================================
# PROGRESS CALLBACK TYPE
# ============================================================

ProgressCallback = Callable[[int, int, str | None], None]
"""
Callback per progress updates durante fetch batch.

Args:
    current: Numero corrente di items processati
    total: Numero totale di items
    message: Messaggio opzionale (es. "Processing art. 2043...")
"""


# ============================================================
# BASE ADAPTER INTERFACE
# ============================================================


class BaseLegalSourceAdapter(ABC):
    """
    Interface astratta che ogni adapter deve implementare.

    Ogni adapter rappresenta UNA fonte di dati legali e deve:
    1. Dichiarare il proprio trust_level (canonical/editorial/mirror)
    2. Implementare fetch per articoli singoli e batch
    3. Produrre output conforme al contract (ArticleExtract, etc.)
    """

    # =========================================================================
    # PROPERTIES (da implementare)
    # =========================================================================

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Nome identificativo della fonte.

        Examples: 'normattiva', 'gazzetta', 'studiocataldi', 'brocardi'
        """
        pass

    @property
    @abstractmethod
    def trust_level(self) -> TrustLevel:
        """
        Livello di attendibilità della fonte.

        - CANONICAL: Normattiva, Gazzetta (fonte ufficiale)
        - EDITORIAL: Brocardi (buona struttura, aggiornato)
        - MIRROR: StudioCataldi (mirror locale, richiede verifica)
        """
        pass

    @property
    def base_url(self) -> str | None:
        """Base URL della fonte (se online)."""
        return None

    @property
    def requires_throttling(self) -> bool:
        """True se la fonte richiede throttling (rate limit)."""
        return True

    @property
    def requests_per_second(self) -> float:
        """Max requests per secondo (default: 1.0 per rispettare ToS)."""
        return 1.0

    # =========================================================================
    # ABSTRACT METHODS - Articoli
    # =========================================================================

    @abstractmethod
    async def fetch_article(
        self,
        codice: str,
        articolo: str,
        comma: str | None = None,
    ) -> ArticleExtract | None:
        """
        Fetch singolo articolo dalla fonte.

        Args:
            codice: Codice (CC, CP, CPC, etc.)
            articolo: Numero articolo (2043, 575, 360-bis)
            comma: Comma specifico (opzionale)

        Returns:
            ArticleExtract o None se non trovato

        Raises:
            FetchError: Se errore di rete/parsing
            RateLimitError: Se rate limit superato
        """
        pass

    @abstractmethod
    async def fetch_codice(
        self,
        codice: str,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ArticleExtract]:
        """
        Fetch tutti gli articoli di un codice.

        Args:
            codice: Codice da fetchare (CC, CP, etc.)
            progress_callback: Callback per progress updates

        Returns:
            Lista di ArticleExtract per ogni articolo

        Raises:
            FetchError: Se errore critico
        """
        pass

    @abstractmethod
    async def list_codici(self) -> list[str]:
        """
        Lista codici disponibili da questa fonte.

        Returns:
            Lista di codici (es. ['CC', 'CP', 'CPC', ...])
        """
        pass

    # =========================================================================
    # ABSTRACT METHODS - Streaming (opzionale, override per fonti grandi)
    # =========================================================================

    async def stream_codice(
        self,
        codice: str,
        progress_callback: ProgressCallback | None = None,
    ) -> AsyncIterator[ArticleExtract]:
        """
        Stream articoli di un codice (per fonti grandi).

        Default: chiama fetch_codice e yielda.
        Override per streaming vero con fonti paginate.
        """
        articles = await self.fetch_codice(codice, progress_callback)
        for article in articles:
            yield article

    # =========================================================================
    # OPTIONAL METHODS - Brocardi / Dizionario
    # =========================================================================

    async def fetch_brocardo(self, latino: str) -> BrocardiExtract | None:
        """
        Fetch singolo brocardo latino.

        Default: NotImplemented (solo Brocardi adapter lo implementa)
        """
        return None

    async def fetch_all_brocardi(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> list[BrocardiExtract]:
        """
        Fetch tutti i brocardi disponibili.

        Default: lista vuota (solo Brocardi adapter lo implementa)
        """
        return []

    async def fetch_voce_dizionario(self, voce: str) -> DizionarioExtract | None:
        """
        Fetch singola voce dizionario.

        Default: NotImplemented
        """
        return None

    async def fetch_all_dizionario(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> list[DizionarioExtract]:
        """
        Fetch tutte le voci dizionario.

        Default: lista vuota
        """
        return []

    # =========================================================================
    # HELPER METHODS (condivisi tra adapters)
    # =========================================================================

    def build_urn_nir(
        self,
        codice: str,
        articolo: str,
        comma: str | None = None,
    ) -> str | None:
        """
        Genera URN:NIR per un articolo.

        Format: {base}:art{num}[~com{comma}]
        Example: urn:nir:stato:regio.decreto:1942-03-16;262:art2043

        Returns:
            URN:NIR string o None se codice non supportato
        """
        base = URN_NIR_BASES.get(codice.upper())
        if not base:
            return None

        # Normalize articolo (rimuovi "bis", "ter" dal numero, aggiungi come suffisso)
        art_num = articolo.lower().replace("-", "")

        urn = f"{base}:art{art_num}"

        if comma:
            urn += f"~com{comma}"

        return urn

    def get_codice_full_name(self, codice: str) -> str:
        """Ritorna nome completo del codice."""
        return CODICE_FULL_NAMES.get(codice.upper(), codice)

    def log_fetch(self, codice: str, articolo: str, success: bool) -> None:
        """Log fetch operation."""
        if success:
            logger.debug(
                "Article fetched",
                source=self.source_name,
                codice=codice,
                articolo=articolo,
            )
        else:
            logger.warning(
                "Article fetch failed",
                source=self.source_name,
                codice=codice,
                articolo=articolo,
            )

    # =========================================================================
    # CONTEXT MANAGER (per cleanup risorse)
    # =========================================================================

    async def __aenter__(self) -> "BaseLegalSourceAdapter":
        """Setup resources (es. HTTP client)."""
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup resources."""


# ============================================================
# CROSS-VALIDATION HELPER
# ============================================================


class CrossValidator:
    """
    Helper per cross-validation tra fonti.

    Livello 1: Hash comparison (gratis, deterministico)
    Livello 2: Semantic diff con LLM (solo se hash diversi)
    """

    def __init__(
        self,
        canonical_adapter: BaseLegalSourceAdapter,
        mirror_adapter: BaseLegalSourceAdapter,
    ):
        if canonical_adapter.trust_level != TrustLevel.CANONICAL:
            logger.warning(
                "Canonical adapter should have CANONICAL trust level",
                source=canonical_adapter.source_name,
                trust_level=canonical_adapter.trust_level,
            )

        self.canonical = canonical_adapter
        self.mirror = mirror_adapter

    async def validate_article(
        self,
        codice: str,
        articolo: str,
        comma: str | None = None,
    ) -> ValidationResult:
        """
        Valida articolo confrontando fonte canonica e mirror.

        1. Fetch da entrambe le fonti
        2. Confronta hash
        3. Se diversi → semantic diff (TIER 2 LLM)
        """
        from lexe_api.kb.sources.models import DiffType, ValidationAction

        # Fetch da entrambe le fonti
        canonical_article = await self.canonical.fetch_article(codice, articolo, comma)
        mirror_article = await self.mirror.fetch_article(codice, articolo, comma)

        # Base result
        result = ValidationResult(
            source_a=self.canonical.source_name,
            source_b=self.mirror.source_name,
            codice=codice,
            articolo=articolo,
            comma=comma,
            hash_match=False,
            action=ValidationAction.USE_CANONICAL,
        )

        # Handle missing articles
        if canonical_article is None:
            result.diff_type = DiffType.SUBSTANTIVE
            result.diff_summary = "Article not found in canonical source"
            result.action = ValidationAction.FLAG_FOR_REVIEW
            return result

        if mirror_article is None:
            result.diff_type = DiffType.SUBSTANTIVE
            result.diff_summary = "Article not found in mirror source"
            # Still use canonical, but note the discrepancy
            return result

        # Compare hashes
        if canonical_article.content_hash == mirror_article.content_hash:
            result.hash_match = True
            result.diff_type = DiffType.EXACT
            logger.debug(
                "Hash match",
                codice=codice,
                articolo=articolo,
                hash=canonical_article.content_hash[:16],
            )
            return result

        # Hash mismatch → need semantic diff
        result.hash_match = False
        result.diff_summary = (
            f"Hash mismatch: canonical={canonical_article.content_hash[:16]}... "
            f"mirror={mirror_article.content_hash[:16]}..."
        )

        # TODO: Call TIER 2 LLM for semantic diff
        # For now, assume formatting difference
        result.diff_type = DiffType.FORMATTING
        result.llm_analyzed = False

        return result

    async def validate_codice(
        self,
        codice: str,
        sample_rate: float = 0.05,
        risk_articles: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ValidationResult]:
        """
        Valida un intero codice con sampling.

        Args:
            codice: Codice da validare
            sample_rate: Percentuale random sampling (default 5%)
            risk_articles: Articoli ad alto rischio da validare sempre
            progress_callback: Callback per progress

        Returns:
            Lista di ValidationResult per articoli validati
        """
        import random

        # Fetch lista articoli dal mirror (più veloce)
        mirror_articles = await self.mirror.fetch_codice(codice, progress_callback)

        # Build sample set
        sample_indices = set()

        # Random sampling
        n_random = int(len(mirror_articles) * sample_rate)
        sample_indices.update(random.sample(range(len(mirror_articles)), n_random))

        # Risk-based sampling
        if risk_articles:
            for i, art in enumerate(mirror_articles):
                if art.articolo in risk_articles:
                    sample_indices.add(i)

        # Validate sample
        results = []
        for i in sorted(sample_indices):
            art = mirror_articles[i]
            result = await self.validate_article(codice, art.articolo, art.comma)
            results.append(result)

            if progress_callback:
                progress_callback(
                    len(results),
                    len(sample_indices),
                    f"Validated {art.articolo}",
                )

        return results
