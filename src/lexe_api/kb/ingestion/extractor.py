"""
LEXE Knowledge Base - PDF Extractor

Estrazione PDF con unstructured (hi_res OCR) e metriche qualita'.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Stopwords italiane per calcolo metriche OCR
ITALIAN_STOPWORDS = {
    "di",
    "a",
    "da",
    "in",
    "con",
    "su",
    "per",
    "tra",
    "fra",
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",
    "e",
    "che",
    "non",
    "è",
    "sono",
    "essere",
    "del",
    "della",
    "dei",
    "delle",
    "al",
    "alla",
    "ai",
    "alle",
    "dal",
    "dalla",
    "dai",
    "dalle",
    "nel",
    "nella",
    "nei",
    "nelle",
    "sul",
    "sulla",
    "sui",
    "sulle",
    "questo",
    "questa",
    "questi",
    "queste",
    "quello",
    "quella",
    "quelli",
    "quelle",
    "come",
    "quando",
    "dove",
    "perché",
    "se",
    "anche",
    "più",
    "ma",
    "però",
    "quindi",
    "così",
    "sia",
    "o",
    "ed",
    "cui",
    "quale",
    "quali",
}

# Pattern citazione Cassazione
CITATION_PATTERN = re.compile(
    r"Sez\.?\s*[UuLl0-9\-]+",
    re.IGNORECASE,
)


@dataclass
class OCRQualityMetrics:
    """Metriche qualita' OCR."""

    valid_chars_ratio: float = 0.0
    italian_tokens_ratio: float = 0.0
    citation_regex_success: float = 0.0
    total_chars: int = 0
    total_words: int = 0
    citations_found: int = 0
    citations_expected: int = 0

    @property
    def quality_score(self) -> float:
        """Score qualita' complessivo (0-1)."""
        return (
            self.valid_chars_ratio * 0.4
            + self.italian_tokens_ratio * 0.4
            + self.citation_regex_success * 0.2
        )

    @property
    def is_acceptable(self) -> bool:
        """Qualita' accettabile per processing."""
        return self.quality_score >= 0.6


@dataclass
class ExtractedElement:
    """Elemento estratto da PDF."""

    text: str
    category: str  # "Title", "NarrativeText", "ListItem", "Table", etc.
    page_number: int | None = None
    coordinates: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Risultato estrazione PDF."""

    elements: list[ExtractedElement]
    metrics: OCRQualityMetrics
    source_path: str
    page_count: int = 0
    extraction_time_ms: float = 0.0


def calculate_ocr_metrics(text: str) -> OCRQualityMetrics:
    """
    Calcola metriche qualita' OCR dal testo estratto.

    Args:
        text: Testo completo estratto

    Returns:
        Metriche OCR
    """
    if not text:
        return OCRQualityMetrics()

    # 1. Valid chars ratio
    valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?()-\"'")
    valid_chars_ratio = valid_chars / len(text) if text else 0

    # 2. Italian tokens ratio
    words = re.findall(r"\b[a-zA-ZàèéìòùÀÈÉÌÒÙ]+\b", text.lower())
    if words:
        # Token italiano se e' stopword o ha lunghezza > 2
        italian_words = sum(1 for w in words if w in ITALIAN_STOPWORDS or len(w) > 2)
        italian_ratio = italian_words / len(words)
    else:
        italian_ratio = 0

    # 3. Citation regex success
    citations_found = len(CITATION_PATTERN.findall(text))
    # Stima citazioni attese dal conteggio "Sez" (puo' essere in altri contesti)
    citations_expected = text.lower().count("sez")
    if citations_expected > 0:
        citation_success = min(citations_found / citations_expected, 1.0)
    else:
        citation_success = 1.0 if citations_found == 0 else 0.5

    return OCRQualityMetrics(
        valid_chars_ratio=valid_chars_ratio,
        italian_tokens_ratio=italian_ratio,
        citation_regex_success=citation_success,
        total_chars=len(text),
        total_words=len(words),
        citations_found=citations_found,
        citations_expected=citations_expected,
    )


async def extract_pdf_with_quality(
    path: str | Path,
    strategy: str = "hi_res",
    languages: list[str] | None = None,
) -> ExtractionResult:
    """
    Estrai PDF con unstructured e calcola metriche qualita' OCR.

    Args:
        path: Path al file PDF
        strategy: Strategia unstructured ("hi_res" per OCR, "fast" per testo nativo)
        languages: Lingue per OCR (default ["ita"])

    Returns:
        ExtractionResult con elementi e metriche
    """
    import time

    from unstructured.partition.pdf import partition_pdf

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if languages is None:
        languages = ["ita"]

    start_time = time.time()

    logger.info(
        "Starting PDF extraction",
        path=str(path),
        strategy=strategy,
        languages=languages,
    )

    try:
        # Estrai con unstructured
        elements = partition_pdf(
            filename=str(path),
            strategy=strategy,
            languages=languages,
            infer_table_structure=True,
            include_page_breaks=True,
            extract_images_in_pdf=False,  # Non estraiamo immagini
        )

        # Converti in nostri ExtractedElement
        extracted: list[ExtractedElement] = []
        max_page = 0

        for elem in elements:
            page_num = getattr(elem.metadata, "page_number", None)
            if page_num and page_num > max_page:
                max_page = page_num

            extracted.append(
                ExtractedElement(
                    text=str(elem),
                    category=elem.category,
                    page_number=page_num,
                    coordinates=getattr(elem.metadata, "coordinates", None),
                    metadata={
                        "element_id": getattr(elem, "element_id", None),
                        "parent_id": getattr(elem.metadata, "parent_id", None),
                    },
                )
            )

        # Calcola metriche OCR sul testo completo
        all_text = " ".join(e.text for e in extracted if e.text)
        metrics = calculate_ocr_metrics(all_text)

        extraction_time = (time.time() - start_time) * 1000

        logger.info(
            "PDF extraction completed",
            path=str(path),
            elements_count=len(extracted),
            pages=max_page,
            ocr_quality_score=round(metrics.quality_score, 3),
            extraction_time_ms=round(extraction_time, 1),
        )

        return ExtractionResult(
            elements=extracted,
            metrics=metrics,
            source_path=str(path),
            page_count=max_page,
            extraction_time_ms=extraction_time,
        )

    except Exception as e:
        logger.error(
            "PDF extraction failed",
            path=str(path),
            error=str(e),
        )
        raise


def extract_pdf_sync(
    path: str | Path,
    strategy: str = "hi_res",
    languages: list[str] | None = None,
) -> ExtractionResult:
    """
    Versione sincrona di extract_pdf_with_quality.

    Per uso in contesti non-async (es. Temporal activities).
    """
    import asyncio

    return asyncio.get_event_loop().run_until_complete(
        extract_pdf_with_quality(path, strategy, languages)
    )


def group_elements_by_page(
    elements: list[ExtractedElement],
) -> dict[int, list[ExtractedElement]]:
    """
    Raggruppa elementi per pagina.
    """
    by_page: dict[int, list[ExtractedElement]] = {}
    for elem in elements:
        page = elem.page_number or 0
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(elem)
    return by_page


def filter_elements_by_category(
    elements: list[ExtractedElement],
    categories: list[str],
) -> list[ExtractedElement]:
    """
    Filtra elementi per categoria.

    Categories comuni: "Title", "NarrativeText", "ListItem", "Table"
    """
    return [e for e in elements if e.category in categories]
