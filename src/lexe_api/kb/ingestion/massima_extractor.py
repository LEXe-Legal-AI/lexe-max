"""
LEXE Knowledge Base - Massima Extractor

Estrazione massime atomiche con evidenza (Chunk A + B).
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import structlog

from .cleaner import clean_legal_text, compute_content_hash, normalize_for_hash
from .extractor import ExtractedElement
from .parser import SectionNode, get_section_context

logger = structlog.get_logger(__name__)


# ============================================================
# Pattern Citazione Cassazione
# ============================================================

# Pattern principale per citazione Cassazione
# Esempi:
# - "Sez. U, Sentenza n. 12345 del 01/02/2020, Rv. 123456-01 - Relatore: ROSSI"
# - "Sez. 1, n. 9876/2019"
# - "Sezioni Unite, 15 marzo 2021, n. 7890"
CITATION_PATTERN = re.compile(
    r"""
    (?:Sez\.?|Sezione|Sezioni)\s*
    (U(?:nite)?|L(?:av)?|[0-9]+(?:\s*-\s*[0-9]+)?)\s*
    (?:,\s*)?
    (?:(?:Sentenza|Ordinanza|Decreto)\s+)?
    n\.?\s*
    ([0-9]+)
    (?:\s*/\s*|\s+del\s+)
    (?:(\d{1,2})[/.-](\d{1,2})[/.-])?
    (\d{4})
    (?:,?\s*Rv\.?\s*([0-9]+(?:-[0-9]+)?))?
    (?:[,\s]*[-–—]\s*Relatore:?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Pattern semplificato per citazione breve
CITATION_SIMPLE_PATTERN = re.compile(
    r"(?:Sez\.?|Sezione)\s*(U|L|[0-9]+)[,\s]+n\.?\s*([0-9]+)[/\s]+(\d{4})",
    re.IGNORECASE,
)

# Pattern per RV isolato
RV_PATTERN = re.compile(
    r"Rv\.?\s*([0-9]{5,6}(?:-[0-9]{2})?)",
    re.IGNORECASE,
)

# Pattern per data decisione
DATE_PATTERN = re.compile(
    r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})"
)


@dataclass
class ExtractedCitation:
    """Citazione estratta da massima."""

    sezione: str | None = None  # "U", "1", "L", etc.
    numero: str | None = None   # Numero sentenza
    anno: int | None = None
    data_decisione: date | None = None
    rv: str | None = None
    relatore: str | None = None
    raw_text: str = ""

    @property
    def is_complete(self) -> bool:
        """Citazione ha almeno sezione, numero e anno."""
        return all([self.sezione, self.numero, self.anno])

    @property
    def normalized_sezione(self) -> str | None:
        """Sezione normalizzata (U, L, 1, 2, etc.)."""
        if not self.sezione:
            return None
        s = self.sezione.upper().strip()
        if s.startswith("U"):
            return "U"
        if s.startswith("L"):
            return "L"
        # Rimuovi spazi e trattini
        return re.sub(r"[\s-]+", "", s)

    def to_dict(self) -> dict:
        """Converti a dizionario per serializzazione."""
        return {
            "sezione": self.normalized_sezione,
            "numero": self.numero,
            "anno": self.anno,
            "data_decisione": self.data_decisione.isoformat() if self.data_decisione else None,
            "rv": self.rv,
            "relatore": self.relatore,
        }


@dataclass
class ExtractedMassima:
    """Massima estratta con evidenza."""

    # Chunk A: Massima pulita (per retrieval)
    testo: str
    testo_normalizzato: str
    content_hash: str

    # Chunk B: Con contesto OCR (per generation)
    testo_con_contesto: str

    # Citazione
    citation: ExtractedCitation

    # Contesto editoriale
    section_context: str
    section_path: str | None = None

    # Metadati
    page_start: int | None = None
    page_end: int | None = None
    element_index: int = 0

    # Quality flags
    citation_complete: bool = False
    text_quality_score: float = 0.0


def normalize_sezione(sezione: str) -> str:
    """Normalizza codice sezione."""
    s = sezione.upper().strip()
    # Sezioni Unite
    if s in ("U", "UNITE", "UNITA"):
        return "U"
    # Sezione Lavoro
    if s in ("L", "LAV", "LAVORO"):
        return "L"
    # Numeriche
    return re.sub(r"[\s-]+", "", s)


def extract_citation(text: str) -> ExtractedCitation:
    """
    Estrai citazione da testo massima.

    Args:
        text: Testo contenente citazione

    Returns:
        ExtractedCitation con campi popolati
    """
    citation = ExtractedCitation()

    # Prova pattern completo
    match = CITATION_PATTERN.search(text)
    if match:
        citation.sezione = match.group(1)
        citation.numero = match.group(2)

        # Data (se presente nel pattern)
        day = match.group(3)
        month = match.group(4)
        year = match.group(5)

        if day and month and year:
            try:
                citation.data_decisione = date(
                    int(year), int(month), int(day)
                )
            except ValueError:
                pass
        citation.anno = int(year) if year else None

        citation.rv = match.group(6)
        citation.relatore = match.group(7)
        citation.raw_text = match.group(0)

        return citation

    # Prova pattern semplice
    match = CITATION_SIMPLE_PATTERN.search(text)
    if match:
        citation.sezione = match.group(1)
        citation.numero = match.group(2)
        citation.anno = int(match.group(3))
        citation.raw_text = match.group(0)

    # Cerca RV isolato se non trovato
    if not citation.rv:
        rv_match = RV_PATTERN.search(text)
        if rv_match:
            citation.rv = rv_match.group(1)

    # Cerca data isolata se non trovata
    if not citation.data_decisione:
        date_match = DATE_PATTERN.search(text)
        if date_match:
            try:
                citation.data_decisione = date(
                    int(date_match.group(3)),
                    int(date_match.group(2)),
                    int(date_match.group(1)),
                )
            except ValueError:
                pass

    return citation


def is_massima_text(element: ExtractedElement) -> bool:
    """
    Verifica se elemento e' probabilmente una massima.

    Criteri:
    - Categoria NarrativeText o UncategorizedText
    - Lunghezza > 100 caratteri
    - Contiene pattern citazione O inizia con pattern tipico
    """
    if not element.text:
        return False

    text = element.text.strip()

    # Troppo corto
    if len(text) < 100:
        return False

    # Categorie che non sono massime
    if element.category in ("Title", "PageBreak", "Header", "Footer"):
        return False

    # Deve contenere citazione O iniziare con pattern tipico
    has_citation = bool(CITATION_PATTERN.search(text) or CITATION_SIMPLE_PATTERN.search(text))

    # Pattern tipici inizio massima
    typical_starts = (
        "In tema di",
        "In materia di",
        "La sentenza",
        "L'ordinanza",
        "Il principio",
        "Ai fini",
        "Nel caso",
        "Qualora",
        "Ove",
        "Allorche'",
        "Allorché",
    )
    starts_typical = any(text.lower().startswith(s.lower()) for s in typical_starts)

    return has_citation or starts_typical


def extract_massime_from_elements(
    elements: list[ExtractedElement],
    section: SectionNode | None = None,
    context_window: int = 1,
) -> list[ExtractedMassima]:
    """
    Estrai massime da lista elementi con evidenza.

    Args:
        elements: Lista elementi (da sezione o documento)
        section: Sezione di appartenenza (per contesto)
        context_window: Numero elementi adiacenti per Chunk B

    Returns:
        Lista massime estratte
    """
    massime: list[ExtractedMassima] = []

    for i, element in enumerate(elements):
        if not is_massima_text(element):
            continue

        # Chunk A: Testo pulito
        testo = clean_legal_text(element.text)
        testo_normalizzato = normalize_for_hash(testo)
        content_hash = compute_content_hash(testo)

        # Chunk B: Con contesto OCR (elementi adiacenti)
        context_parts = []

        # Elementi precedenti
        for j in range(max(0, i - context_window), i):
            if elements[j].text:
                context_parts.append(elements[j].text.strip())

        # Elemento corrente
        context_parts.append(element.text.strip())

        # Elementi successivi
        for j in range(i + 1, min(len(elements), i + 1 + context_window)):
            if elements[j].text:
                context_parts.append(elements[j].text.strip())

        testo_con_contesto = "\n\n".join(context_parts)

        # Estrai citazione
        citation = extract_citation(element.text)

        # Contesto editoriale
        if section:
            section_context = get_section_context(section)
            section_path = section.section_path
        else:
            section_context = ""
            section_path = None

        # Calcola quality score basato su caratteri validi
        valid_chars = sum(
            1 for c in testo
            if c.isalnum() or c.isspace() or c in '.,;:!?()-"\''
        )
        text_quality = valid_chars / len(testo) if testo else 0

        massima = ExtractedMassima(
            testo=testo,
            testo_normalizzato=testo_normalizzato,
            content_hash=content_hash,
            testo_con_contesto=testo_con_contesto,
            citation=citation,
            section_context=section_context,
            section_path=section_path,
            page_start=element.page_number,
            page_end=element.page_number,
            element_index=i,
            citation_complete=citation.is_complete,
            text_quality_score=text_quality,
        )

        massime.append(massima)

    logger.info(
        "Massime extracted",
        total=len(massime),
        with_complete_citation=sum(1 for m in massime if m.citation_complete),
        section=section.section_path if section else "orphan",
    )

    return massime


def extract_tema(testo: str) -> str:
    """
    Estrai tema/titolo breve da massima.

    Cerca pattern "In tema di X" o usa prima frase.

    Args:
        testo: Testo massima

    Returns:
        Tema estratto (max 200 chars)
    """
    if not testo:
        return ""

    # Pattern "In tema di..."
    match = re.search(r"[Ii]n tema di\s+([^,.]+)", testo)
    if match:
        tema = match.group(1).strip()
        return f"In tema di {tema}"[:200]

    # Pattern "In materia di..."
    match = re.search(r"[Ii]n materia di\s+([^,.]+)", testo)
    if match:
        tema = match.group(1).strip()
        return f"In materia di {tema}"[:200]

    # Fallback: prima frase
    sentences = re.split(r"[.!?]", testo)
    if sentences:
        first = sentences[0].strip()
        return first[:200]

    return testo[:200]


def calculate_massima_importance(
    massima: ExtractedMassima,
    citation_count: int = 0,
    is_sezioni_unite: bool = False,
) -> float:
    """
    Calcola score di importanza per massima.

    Fattori:
    - Sezioni Unite: boost importante
    - Citazione completa: indica qualita' OCR
    - Numero citazioni (se noto)
    - Recency

    Returns:
        Score 0-1
    """
    score = 0.5  # Base

    # Sezioni Unite = +0.2
    if is_sezioni_unite or (
        massima.citation.normalized_sezione and
        massima.citation.normalized_sezione == "U"
    ):
        score += 0.2

    # Citazione completa = +0.1
    if massima.citation_complete:
        score += 0.1

    # Citation count (normalizzato)
    if citation_count > 0:
        score += min(citation_count / 10, 0.1)

    # Recency boost (ultimi 5 anni)
    if massima.citation.anno:
        current_year = date.today().year
        years_old = current_year - massima.citation.anno
        if years_old <= 5:
            score += 0.1 * (1 - years_old / 5)

    return min(score, 1.0)
