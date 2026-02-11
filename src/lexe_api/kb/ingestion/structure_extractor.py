# lexe_api/kb/ingestion/structure_extractor.py
"""
Structure Extractor - Regex + Heuristics for Legal Text.

Estrae struttura gerarchica da testi legali puliti:
- Libri, Titoli, Capi, Sezioni, Articoli
- Numeri articolo, commi, rubrica
- Identificazione tipo documento (codice, legge, TU)

Costo: $0 (puro Python, nessun LLM)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# ENUMS & TYPES
# ============================================================


class DocumentType(str, Enum):
    """Tipo di documento normativo."""

    CODICE = "codice"  # Codice Civile, Penale, etc.
    LEGGE = "legge"  # Legge ordinaria
    DECRETO_LEGISLATIVO = "dlgs"  # D.Lgs.
    DECRETO_LEGGE = "dl"  # D.L.
    DPR = "dpr"  # D.P.R.
    COSTITUZIONE = "costituzione"
    TESTO_UNICO = "tu"  # Testo Unico
    REGOLAMENTO = "regolamento"
    UNKNOWN = "unknown"


class HierarchyLevel(int, Enum):
    """Livelli gerarchia normativa."""

    LIBRO = 1
    PARTE = 2
    TITOLO = 3
    CAPO = 4
    SEZIONE = 5
    PARAGRAFO = 6
    ARTICOLO = 7


class Position(NamedTuple):
    """Posizione nel testo."""

    start: int
    end: int


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class HierarchyNode:
    """Nodo della gerarchia normativa."""

    level: HierarchyLevel
    number: str  # "I", "1", "2-bis"
    title: str | None  # Rubrica
    position: Position | None = None
    children: list["HierarchyNode"] = field(default_factory=list)

    @property
    def full_path(self) -> str:
        """Path completo nella gerarchia."""
        level_names = {
            HierarchyLevel.LIBRO: "Libro",
            HierarchyLevel.PARTE: "Parte",
            HierarchyLevel.TITOLO: "Titolo",
            HierarchyLevel.CAPO: "Capo",
            HierarchyLevel.SEZIONE: "Sezione",
            HierarchyLevel.PARAGRAFO: "§",
            HierarchyLevel.ARTICOLO: "Art.",
        }
        name = level_names.get(self.level, str(self.level))
        return f"{name} {self.number}"


@dataclass
class ArticleStructure:
    """Struttura estratta di un articolo."""

    # Identificazione
    articolo: str  # "2043", "360-bis"
    rubrica: str | None  # "Risarcimento per fatto illecito"

    # Contenuto
    testo: str  # Testo completo articolo
    commi: list[str]  # Lista commi separati

    # Posizione nella gerarchia
    libro: str | None = None
    parte: str | None = None
    titolo: str | None = None
    capo: str | None = None
    sezione: str | None = None

    # Posizione nel file
    position: Position | None = None


@dataclass
class DocumentStructure:
    """Struttura completa di un documento normativo."""

    doc_type: DocumentType
    codice: str | None  # "CC", "CP", etc.
    title: str | None

    # Gerarchia
    hierarchy: list[HierarchyNode]
    articles: list[ArticleStructure]

    # Stats
    article_count: int = 0


# ============================================================
# REGEX PATTERNS
# ============================================================


class LegalPatterns:
    """Collezione pattern regex per testi legali italiani."""

    # Numeri romani
    ROMAN_NUMERAL = r"(?:[IVXLCDM]+)"

    # Numeri articolo (include bis, ter, etc.)
    ARTICLE_SUFFIX = r"(?:-?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|quaterdecies|quindecies|sexdecies))?"

    # Pattern articolo base
    ARTICLE_NUMBER = rf"(\d+){ARTICLE_SUFFIX}"

    # Intestazione articolo
    ARTICLE_HEADER = re.compile(
        rf"^\s*(?:Art(?:icolo)?\.?)\s*({ARTICLE_NUMBER})"
        rf"(?:\s*[\.\-–—]\s*(.+?))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Intestazione articolo alternativa (numero da solo)
    ARTICLE_HEADER_ALT = re.compile(rf"^\s*({ARTICLE_NUMBER})\.\s*(.+?)\s*$", re.MULTILINE)

    # Comma numerato
    COMMA_PATTERN = re.compile(
        r"^\s*(\d+)[\.\)]\s*(.+?)(?=^\s*\d+[\.\)]|\Z)", re.MULTILINE | re.DOTALL
    )

    # Lettera in comma
    LETTERA_PATTERN = re.compile(
        r"^\s*([a-z])\)\s*(.+?)(?=^\s*[a-z]\)|\Z)", re.MULTILINE | re.DOTALL
    )

    # Gerarchia: Libro
    LIBRO_PATTERN = re.compile(
        rf"^\s*LIBRO\s+({ROMAN_NUMERAL}|\d+)"
        rf"(?:\s*[\.\-–—]\s*(.+?))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Gerarchia: Parte
    PARTE_PATTERN = re.compile(
        rf"^\s*PARTE\s+({ROMAN_NUMERAL}|\d+)"
        rf"(?:\s*[\.\-–—]\s*(.+?))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Gerarchia: Titolo
    TITOLO_PATTERN = re.compile(
        rf"^\s*TITOLO\s+({ROMAN_NUMERAL}|\d+)"
        rf"(?:\s*[\.\-–—]\s*(.+?))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Gerarchia: Capo
    CAPO_PATTERN = re.compile(
        rf"^\s*CAPO\s+({ROMAN_NUMERAL}|\d+)"
        rf"(?:\s*[\.\-–—]\s*(.+?))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Gerarchia: Sezione
    SEZIONE_PATTERN = re.compile(
        rf"^\s*SEZIONE\s+({ROMAN_NUMERAL}|\d+)"
        rf"(?:\s*[\.\-–—]\s*(.+?))?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Tipo documento
    DOC_TYPE_PATTERNS = {
        DocumentType.CODICE: re.compile(
            r"codice\s+(?:civile|penale|della?\s+(?:strada|navigazione|privacy|consumo))",
            re.IGNORECASE,
        ),
        DocumentType.COSTITUZIONE: re.compile(
            r"costituzione\s+(?:della\s+)?(?:repubblica\s+)?italiana", re.IGNORECASE
        ),
        DocumentType.DECRETO_LEGISLATIVO: re.compile(
            r"d(?:ecreto)?\.?\s*l(?:eg)?(?:islativo)?\.?\s*\d+", re.IGNORECASE
        ),
        DocumentType.LEGGE: re.compile(r"legge\s+\d+", re.IGNORECASE),
        DocumentType.DPR: re.compile(r"d\.?p\.?r\.?\s*\d+", re.IGNORECASE),
        DocumentType.TESTO_UNICO: re.compile(r"testo\s+unico", re.IGNORECASE),
    }

    # Codice abbreviato
    CODICE_PATTERNS = {
        "CC": re.compile(r"\b(?:c(?:od)?\.?\s*c(?:iv)?\.?|codice\s+civile)\b", re.I),
        "CP": re.compile(r"\b(?:c(?:od)?\.?\s*p(?:en)?\.?|codice\s+penale)\b", re.I),
        "CPC": re.compile(
            r"\b(?:c\.?\s*p\.?\s*c\.?|codice\s+(?:di\s+)?procedura\s+civile)\b", re.I
        ),
        "CPP": re.compile(
            r"\b(?:c\.?\s*p\.?\s*p\.?|codice\s+(?:di\s+)?procedura\s+penale)\b", re.I
        ),
        "COST": re.compile(r"\b(?:cost\.?|costituzione)\b", re.I),
        "CDS": re.compile(r"\b(?:c\.?\s*d\.?\s*s\.?|codice\s+della\s+strada)\b", re.I),
    }


# ============================================================
# STRUCTURE EXTRACTOR
# ============================================================


class StructureExtractor:
    """
    Estrae struttura da testi legali usando regex + heuristics.

    Pipeline:
    1. Identifica tipo documento
    2. Trova nodi gerarchia (Libro, Titolo, Capo, etc.)
    3. Estrae articoli con rubrica e testo
    4. Segmenta commi all'interno degli articoli
    """

    def __init__(self):
        self.patterns = LegalPatterns()

    def extract(self, text: str, filename: str | None = None) -> DocumentStructure:
        """
        Estrae struttura completa da testo.

        Args:
            text: Testo pulito da analizzare
            filename: Nome file (per inferire codice)

        Returns:
            DocumentStructure con gerarchia e articoli
        """
        # Detect document type
        doc_type = self._detect_doc_type(text)

        # Detect codice from filename or content
        codice = self._detect_codice(text, filename)

        # Extract title
        title = self._extract_title(text)

        # Extract hierarchy nodes
        hierarchy = self._extract_hierarchy(text)

        # Extract articles
        articles = self._extract_articles(text, hierarchy)

        return DocumentStructure(
            doc_type=doc_type,
            codice=codice,
            title=title,
            hierarchy=hierarchy,
            articles=articles,
            article_count=len(articles),
        )

    def extract_single_article(self, text: str) -> ArticleStructure | None:
        """
        Estrae struttura da testo di singolo articolo.

        Usa quando sai già che il testo è un solo articolo.
        """
        # Try to find article header
        match = self.patterns.ARTICLE_HEADER.search(text)

        if match:
            articolo = match.group(1)
            rubrica = match.group(2)
            # Text after header
            article_text = text[match.end() :].strip()
        else:
            # Try alternative pattern
            match_alt = self.patterns.ARTICLE_HEADER_ALT.search(text)
            if match_alt:
                articolo = match_alt.group(1)
                rubrica = match_alt.group(2)
                article_text = text[match_alt.end() :].strip()
            else:
                # Can't parse, return whole text
                return None

        # Extract commi
        commi = self._extract_commi(article_text)

        return ArticleStructure(
            articolo=articolo,
            rubrica=rubrica,
            testo=article_text,
            commi=commi,
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _detect_doc_type(self, text: str) -> DocumentType:
        """Rileva tipo documento dal testo."""
        # Check first 500 chars for type indicators
        header = text[:500].lower()

        for doc_type, pattern in self.patterns.DOC_TYPE_PATTERNS.items():
            if pattern.search(header):
                return doc_type

        return DocumentType.UNKNOWN

    def _detect_codice(self, text: str, filename: str | None) -> str | None:
        """Rileva codice abbreviato."""
        # First try filename
        if filename:
            filename_lower = filename.lower()
            if "codice-civile" in filename_lower or "codicecivile" in filename_lower:
                return "CC"
            if "codice-penale" in filename_lower or "codicepenale" in filename_lower:
                return "CP"
            if "costituzione" in filename_lower:
                return "COST"
            if "procedura-civile" in filename_lower:
                return "CPC"
            if "procedura-penale" in filename_lower:
                return "CPP"
            if "strada" in filename_lower:
                return "CDS"

        # Then try content
        header = text[:1000]
        for codice, pattern in self.patterns.CODICE_PATTERNS.items():
            if pattern.search(header):
                return codice

        return None

    def _extract_title(self, text: str) -> str | None:
        """Estrae titolo documento."""
        # First non-empty line is usually title
        lines = text.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and not line.startswith("Art"):
                return line

        return None

    def _extract_hierarchy(self, text: str) -> list[HierarchyNode]:
        """Estrae nodi gerarchia."""
        nodes = []

        # Find all hierarchy markers
        hierarchy_patterns = [
            (HierarchyLevel.LIBRO, self.patterns.LIBRO_PATTERN),
            (HierarchyLevel.PARTE, self.patterns.PARTE_PATTERN),
            (HierarchyLevel.TITOLO, self.patterns.TITOLO_PATTERN),
            (HierarchyLevel.CAPO, self.patterns.CAPO_PATTERN),
            (HierarchyLevel.SEZIONE, self.patterns.SEZIONE_PATTERN),
        ]

        for level, pattern in hierarchy_patterns:
            for match in pattern.finditer(text):
                number = match.group(1)
                title = match.group(2) if match.lastindex >= 2 else None

                nodes.append(
                    HierarchyNode(
                        level=level,
                        number=number,
                        title=title.strip() if title else None,
                        position=Position(match.start(), match.end()),
                    )
                )

        # Sort by position
        nodes.sort(key=lambda n: n.position.start if n.position else 0)

        return nodes

    def _extract_articles(
        self,
        text: str,
        hierarchy: list[HierarchyNode],
    ) -> list[ArticleStructure]:
        """Estrae tutti gli articoli dal testo."""
        articles = []

        # Find all article headers
        matches = list(self.patterns.ARTICLE_HEADER.finditer(text))

        if not matches:
            # Try alternative pattern
            matches = list(self.patterns.ARTICLE_HEADER_ALT.finditer(text))

        for i, match in enumerate(matches):
            articolo = match.group(1)
            rubrica = match.group(2) if match.lastindex >= 2 else None

            # Text until next article or end
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[start:end].strip()

            # Find hierarchy context
            libro, parte, titolo, capo, sezione = self._find_hierarchy_context(
                match.start(), hierarchy
            )

            # Extract commi
            commi = self._extract_commi(article_text)

            articles.append(
                ArticleStructure(
                    articolo=articolo,
                    rubrica=rubrica.strip() if rubrica else None,
                    testo=article_text,
                    commi=commi,
                    libro=libro,
                    parte=parte,
                    titolo=titolo,
                    capo=capo,
                    sezione=sezione,
                    position=Position(match.start(), end),
                )
            )

        return articles

    def _find_hierarchy_context(
        self,
        position: int,
        hierarchy: list[HierarchyNode],
    ) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        """Trova contesto gerarchia per una posizione."""
        libro = parte = titolo = capo = sezione = None

        # Find most recent node of each level before position
        for node in hierarchy:
            if node.position and node.position.start < position:
                if node.level == HierarchyLevel.LIBRO:
                    libro = node.full_path
                elif node.level == HierarchyLevel.PARTE:
                    parte = node.full_path
                elif node.level == HierarchyLevel.TITOLO:
                    titolo = node.full_path
                elif node.level == HierarchyLevel.CAPO:
                    capo = node.full_path
                elif node.level == HierarchyLevel.SEZIONE:
                    sezione = node.full_path

        return libro, parte, titolo, capo, sezione

    def _extract_commi(self, article_text: str) -> list[str]:
        """Estrae commi da testo articolo."""
        commi = []

        # Try numbered comma pattern
        matches = list(self.patterns.COMMA_PATTERN.finditer(article_text))

        if matches:
            for match in matches:
                comma_num = match.group(1)
                comma_text = match.group(2).strip()
                commi.append(f"{comma_num}. {comma_text}")
        else:
            # No numbered commi, treat as single comma
            if article_text.strip():
                commi.append(article_text.strip())

        return commi


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def extract_structure(text: str, filename: str | None = None) -> DocumentStructure:
    """
    Convenience function per estrarre struttura.

    Args:
        text: Testo pulito
        filename: Nome file (opzionale)

    Returns:
        DocumentStructure
    """
    extractor = StructureExtractor()
    return extractor.extract(text, filename)


def extract_article(text: str) -> ArticleStructure | None:
    """
    Convenience function per estrarre singolo articolo.

    Args:
        text: Testo singolo articolo

    Returns:
        ArticleStructure o None
    """
    extractor = StructureExtractor()
    return extractor.extract_single_article(text)
