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


# ============================================================
# Citation-Anchored Extraction (per Civile)
# ============================================================

# Pattern citazione compositi per anchoring
CITATION_ANCHOR_PATTERNS = [
    # Rv. con numero (priorità alta)
    re.compile(r"Rv\.?\s*\d{5,6}(?:-\d{2})?", re.IGNORECASE),
    # Sez. con numero sentenza
    re.compile(
        r"Sez\.?\s*(?:U(?:n)?|L(?:av)?|[IVX0-9]+)[,\s]+(?:(?:Sent|Ord)\.?\s+)?n\.?\s*\d+[/\s]+\d{4}",
        re.IGNORECASE,
    ),
    # Cass. civ/pen con numero
    re.compile(
        r"Cass\.?\s*(?:civ|pen)?\.?\s*(?:,?\s*(?:Sez\.?\s*)?(?:U|L|[IVX0-9]+)[,\s]+)?n\.?\s*\d+[/\s]+\d{4}",
        re.IGNORECASE,
    ),
    # Sentenza/Ordinanza n. XXXX/YYYY
    re.compile(
        r"(?:sent|ord|sentenza|ordinanza)\.?\s*n\.?\s*\d+[/\s]+\d{4}",
        re.IGNORECASE,
    ),
]


@dataclass
class CitationAnchor:
    """Ancora di citazione trovata nel testo."""
    pattern_name: str
    match_text: str
    start_pos: int
    end_pos: int
    line_number: int


def find_citation_anchors(text: str) -> list[CitationAnchor]:
    """
    Trova tutte le citazioni nel testo con posizioni.

    Returns:
        Lista di CitationAnchor ordinate per posizione.
    """
    anchors: list[CitationAnchor] = []
    seen_positions: set[tuple[int, int]] = set()

    pattern_names = ["rv", "sez", "cass", "sent"]

    for pattern, name in zip(CITATION_ANCHOR_PATTERNS, pattern_names):
        for match in pattern.finditer(text):
            pos = (match.start(), match.end())
            if pos not in seen_positions:
                seen_positions.add(pos)

                # Calcola line number
                line_num = text[:match.start()].count("\n") + 1

                anchors.append(
                    CitationAnchor(
                        pattern_name=name,
                        match_text=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        line_number=line_num,
                    )
                )

    # Ordina per posizione
    anchors.sort(key=lambda a: a.start_pos)
    return anchors


def split_into_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split testo in frasi con posizioni start/end.

    Returns:
        Lista di (sentence_text, start_pos, end_pos)
    """
    # Abbreviazioni comuni che non terminano frase
    ABBREVIATIONS = {
        "sez", "cass", "sent", "ord", "rv", "art", "artt",
        "pag", "pagg", "cfr", "nn", "vol", "ss", "segg",
        "cit", "op", "loc", "es", "ecc", "dott", "prof",
        "avv", "ing", "sig", "sigg", "s.p.a", "s.r.l",
    }

    sentences: list[tuple[str, int, int]] = []
    last_end = 0

    # Pattern semplice: punto seguito da spazio e maiuscola, o fine testo
    sentence_end = re.compile(r'\.(?:\s+[A-Z]|\s*$)')

    for match in sentence_end.finditer(text):
        # Controlla se il punto è dopo un'abbreviazione
        start = match.start()
        # Trova la parola prima del punto
        word_before = ""
        i = start - 1
        while i >= 0 and (text[i].isalpha() or text[i] == '.'):
            word_before = text[i] + word_before
            i -= 1

        word_before_clean = word_before.lower().rstrip('.')

        # Se è un'abbreviazione, salta
        if word_before_clean in ABBREVIATIONS:
            continue

        end_pos = match.start() + 1  # Include il punto
        sentence = text[last_end:end_pos].strip()
        if sentence:
            sentences.append((sentence, last_end, end_pos))
        last_end = end_pos

    # Ultima frase senza punto finale
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append((remaining, last_end, len(text)))

    return sentences


def extract_window_around_citation(
    text: str,
    anchor: CitationAnchor,
    sentences: list[tuple[str, int, int]],
    before: int = 2,
    after: int = 1,
) -> tuple[str, int, int]:
    """
    Estrai finestra di testo attorno a una citazione.

    Args:
        text: Testo completo
        anchor: Citazione trovata
        sentences: Lista frasi pre-calcolate
        before: Numero frasi prima della citazione
        after: Numero frasi dopo la citazione

    Returns:
        (window_text, start_pos, end_pos)
    """
    # Trova la frase che contiene la citazione
    citation_sentence_idx = -1
    for i, (sent, start, end) in enumerate(sentences):
        if start <= anchor.start_pos < end:
            citation_sentence_idx = i
            break

    if citation_sentence_idx == -1:
        # Fallback: ritorna contesto basato su caratteri
        start = max(0, anchor.start_pos - 500)
        end = min(len(text), anchor.end_pos + 200)
        return text[start:end].strip(), start, end

    # Calcola range frasi
    start_idx = max(0, citation_sentence_idx - before)
    end_idx = min(len(sentences), citation_sentence_idx + after + 1)

    # Estrai window
    window_start = sentences[start_idx][1]
    window_end = sentences[end_idx - 1][2]

    window_text = text[window_start:window_end].strip()

    return window_text, window_start, window_end


def should_split_block(
    anchors: list[CitationAnchor],
    text_length: int,
    min_distance: int = 300,
) -> bool:
    """
    Determina se un blocco con multiple citazioni va splittato.

    Split se:
    - Più di 1 citazione
    - Citazioni distanti almeno min_distance caratteri
    - Testo lungo (>800 chars per citazione)
    """
    if len(anchors) <= 1:
        return False

    # Check distanza tra citazioni
    for i in range(1, len(anchors)):
        dist = anchors[i].start_pos - anchors[i - 1].end_pos
        if dist >= min_distance:
            return True

    # Check lunghezza media per citazione
    avg_len = text_length / len(anchors)
    if avg_len > 800:
        return True

    return False


def extract_massime_citation_anchored(
    text: str,
    page_number: int | None = None,
    section_context: str = "",
    section_path: str | None = None,
    window_before: int = 2,
    window_after: int = 1,
    split_on_multiple: bool = True,
    min_distance_split: int = 300,
    min_length: int = 120,
    max_length: int = 3000,
) -> list[ExtractedMassima]:
    """
    Estrai massime ancorandosi ai pattern di citazione.

    Questa modalità è ottimale per documenti Civile dove le massime
    sono integrate nel testo narrativo con citazioni sparse.

    Args:
        text: Testo raw da una pagina o blocco
        page_number: Numero pagina (se disponibile)
        section_context: Contesto editoriale
        section_path: Path sezione
        window_before: Frasi da includere prima della citazione
        window_after: Frasi da includere dopo la citazione
        split_on_multiple: Se True, split blocchi con multiple citazioni
        min_distance_split: Distanza minima tra citazioni per split
        min_length: Lunghezza minima massima estratta
        max_length: Lunghezza massima (tronca se supera)

    Returns:
        Lista ExtractedMassima
    """
    massime: list[ExtractedMassima] = []

    # Trova tutte le citazioni
    anchors = find_citation_anchors(text)

    if not anchors:
        logger.debug("citation_anchored: no anchors found", text_len=len(text))
        return []

    # Pre-calcola frasi
    sentences = split_into_sentences(text)

    # Decide se splittare
    if split_on_multiple and should_split_block(anchors, len(text), min_distance_split):
        # Estrai una massima per ogni citazione
        for i, anchor in enumerate(anchors):
            window_text, start_pos, end_pos = extract_window_around_citation(
                text, anchor, sentences, window_before, window_after
            )

            # Verifica lunghezza
            if len(window_text) < min_length:
                continue
            if len(window_text) > max_length:
                window_text = window_text[:max_length] + "..."

            # Crea massima
            massima = _create_massima_from_window(
                window_text=window_text,
                anchor=anchor,
                page_number=page_number,
                section_context=section_context,
                section_path=section_path,
                element_index=i,
            )
            massime.append(massima)
    else:
        # Usa prima citazione come ancora, estrai tutto il blocco
        anchor = anchors[0]
        window_text = text.strip()

        if len(window_text) < min_length:
            return []
        if len(window_text) > max_length:
            window_text = window_text[:max_length] + "..."

        massima = _create_massima_from_window(
            window_text=window_text,
            anchor=anchor,
            page_number=page_number,
            section_context=section_context,
            section_path=section_path,
            element_index=0,
        )
        massime.append(massima)

    logger.debug(
        "citation_anchored extracted",
        n_anchors=len(anchors),
        n_massime=len(massime),
        split=split_on_multiple and len(anchors) > 1,
    )

    return massime


def _create_massima_from_window(
    window_text: str,
    anchor: CitationAnchor,
    page_number: int | None,
    section_context: str,
    section_path: str | None,
    element_index: int,
) -> ExtractedMassima:
    """Helper: crea ExtractedMassima da window estratto."""
    # Pulisci e normalizza
    testo = clean_legal_text(window_text)
    testo_normalizzato = normalize_for_hash(testo)
    content_hash = compute_content_hash(testo)

    # Estrai citazione dal window (più preciso)
    citation = extract_citation(window_text)

    # Quality score
    valid_chars = sum(
        1 for c in testo
        if c.isalnum() or c.isspace() or c in '.,;:!?()-"\''
    )
    text_quality = valid_chars / len(testo) if testo else 0

    return ExtractedMassima(
        testo=testo,
        testo_normalizzato=testo_normalizzato,
        content_hash=content_hash,
        testo_con_contesto=window_text,  # Raw con contesto
        citation=citation,
        section_context=section_context,
        section_path=section_path,
        page_start=page_number,
        page_end=page_number,
        element_index=element_index,
        citation_complete=citation.is_complete,
        text_quality_score=text_quality,
    )


def extract_massime_from_pdf_text(
    pages: list[tuple[int, str]],
    extraction_mode: str = "standard",
    toc_skip_pages: int = 0,
    section_context: str = "",
    section_path: str | None = None,
    gate_config: dict | None = None,
) -> list[ExtractedMassima]:
    """
    Estrai massime da testo PDF (multi-pagina).

    Supporta diverse modalità di estrazione:
    - standard: element-based (legacy)
    - citation_anchored: ancora su pattern citazione (raccomandato per Civile)

    Args:
        pages: Lista di (page_num, page_text)
        extraction_mode: "standard" o "citation_anchored"
        toc_skip_pages: Salta prime N pagine (TOC)
        section_context: Contesto editoriale
        section_path: Path sezione
        gate_config: Configurazione gate (min_length, max_length, etc.)

    Returns:
        Lista ExtractedMassima
    """
    gate = gate_config or {}
    min_length = gate.get("min_length", 150)
    max_length = gate.get("max_length", 3000)
    window_before = gate.get("citation_window_before", 2)
    window_after = gate.get("citation_window_after", 1)
    split_on_multiple = gate.get("split_on_multiple_citations", True)

    massime: list[ExtractedMassima] = []

    for page_num, page_text in pages:
        # Skip TOC pages
        if page_num <= toc_skip_pages:
            continue

        # Skip pagine quasi vuote
        if len(page_text.strip()) < 100:
            continue

        if extraction_mode == "citation_anchored":
            page_massime = extract_massime_citation_anchored(
                text=page_text,
                page_number=page_num,
                section_context=section_context,
                section_path=section_path,
                window_before=window_before,
                window_after=window_after,
                split_on_multiple=split_on_multiple,
                min_length=min_length,
                max_length=max_length,
            )
            massime.extend(page_massime)

        # Per "standard" mode, usa extract_massime_from_elements (richiede elementi)

    logger.info(
        "PDF text extraction complete",
        mode=extraction_mode,
        pages_processed=len(pages) - toc_skip_pages,
        massime_extracted=len(massime),
    )

    return massime
