# lexe_api/kb/ingestion/legal_numbers_extractor.py
"""
Legal Numbers Extractor - ANCHOR POINTS per il Knowledge Graph.

Estrae riferimenti numerici legali dal testo:
- Articoli di codice: "art. 2043 c.c.", "art. 575 c.p."
- Leggi: "L. 241/1990", "Legge 7 agosto 1990, n. 241"
- Decreti: "D.Lgs. 165/2001", "D.P.R. 380/2001"
- Sentenze: "Cass. 12345/2020", "Cass. Sez. Un. 8770/2020"

Questi numeri sono ANCHOR POINTS deterministici per il grafo!

Costo: $0 (puro regex, nessun LLM)
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import structlog

from lexe_api.kb.ingestion.urn_generator import CanonicalIdGenerator

logger = structlog.get_logger(__name__)


# ============================================================
# ENUMS & TYPES
# ============================================================


class LegalNumberType(str, Enum):
    """Tipi di numeri legali."""

    ARTICLE = "article"  # art. 2043 c.c.
    LAW = "law"  # L. 241/1990
    LEGISLATIVE_DECREE = "dlgs"  # D.Lgs. 165/2001
    LAW_DECREE = "dl"  # D.L. 18/2020
    DPR = "dpr"  # D.P.R. 380/2001
    SENTENCE = "sentence"  # Cass. 12345/2020
    EU_REGULATION = "reg_eu"  # Reg. UE 679/2016
    EU_DIRECTIVE = "dir_eu"  # Dir. 2016/680/UE
    CONSTITUTIONAL = "cost"  # art. 3 Cost.


class Position(NamedTuple):
    """Posizione nel testo."""

    start: int
    end: int


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class LegalNumber:
    """
    Numero legale estratto - ANCHOR POINT per il grafo.

    Ogni LegalNumber diventa un nodo nel Number-Anchored Knowledge Graph.
    """

    # Raw extraction
    raw_text: str  # "art. 2043 c.c."
    position: Position  # Posizione nel testo

    # Parsed components
    number_type: LegalNumberType
    codice: str | None = None  # CC, CP, L, DLGS, CASS
    numero: str | None = None  # 2043, 241, 12345
    anno: int | None = None  # 1990, 2020
    articolo: str | None = None  # Per leggi: articolo citato
    comma: str | None = None
    sezione: str | None = None  # Per sentenze: Sez. Un., Sez. 1

    # Generated IDs
    canonical_id: str = ""  # CC:2043, L:241:1990
    urn_nir: str | None = None  # urn:nir:stato:...

    # Context
    context_span: str | None = None  # Frase contenente la citazione

    @property
    def is_code_article(self) -> bool:
        """True se è un articolo di codice."""
        return self.number_type == LegalNumberType.ARTICLE

    @property
    def is_law(self) -> bool:
        """True se è una legge o decreto."""
        return self.number_type in (
            LegalNumberType.LAW,
            LegalNumberType.LEGISLATIVE_DECREE,
            LegalNumberType.LAW_DECREE,
            LegalNumberType.DPR,
        )

    @property
    def is_sentence(self) -> bool:
        """True se è una sentenza."""
        return self.number_type == LegalNumberType.SENTENCE

    @property
    def is_eu_norm(self) -> bool:
        """True se è norma UE."""
        return self.number_type in (
            LegalNumberType.EU_REGULATION,
            LegalNumberType.EU_DIRECTIVE,
        )


@dataclass
class ExtractionResult:
    """Risultato estrazione numeri legali."""

    numbers: list[LegalNumber] = field(default_factory=list)
    text_length: int = 0

    @property
    def count(self) -> int:
        return len(self.numbers)

    @property
    def unique_canonical_ids(self) -> set[str]:
        return {n.canonical_id for n in self.numbers if n.canonical_id}

    def by_type(self, number_type: LegalNumberType) -> list[LegalNumber]:
        return [n for n in self.numbers if n.number_type == number_type]


# ============================================================
# REGEX PATTERNS
# ============================================================


class LegalPatterns:
    """Collezione pattern regex per numeri legali italiani."""

    # Suffissi articolo (bis, ter, etc.)
    SUFFIX = r"(?:[\-\s]?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?"

    # ═══ ARTICOLI DI CODICE ═══

    # Pattern: "art. 2043 c.c." / "artt. 1 e 2 c.c." / "articolo 575 c.p."
    ARTICLE_CODICE = re.compile(
        rf"\b(?:art(?:t)?(?:icol[oi])?\.?)\s*"  # art./artt./articolo/articoli
        rf"(\d+{SUFFIX})"  # numero articolo
        rf"(?:\s*(?:e|,)\s*(\d+{SUFFIX}))?"  # secondo articolo opzionale
        rf"\s+"  # spazio
        rf"(c\.?\s*c\.?|c\.?\s*p\.?|c\.?\s*p\.?\s*c\.?|c\.?\s*p\.?\s*p\.?|cost\.?|c\.?\s*d\.?\s*s\.?|"
        rf"cod(?:ice)?\s+civ(?:ile)?|cod(?:ice)?\s+pen(?:ale)?|"
        rf"cod(?:ice)?\s+(?:di\s+)?proc(?:edura)?\s+civ(?:ile)?|"
        rf"cod(?:ice)?\s+(?:di\s+)?proc(?:edura)?\s+pen(?:ale)?|"
        rf"costituzione)",
        re.IGNORECASE,
    )

    # Pattern per "dell'art. 2043" (senza codice esplicito, per contesto)
    ARTICLE_DELL = re.compile(
        rf"\b(?:dell[''']?\s*)?(?:art(?:icol[oi])?\.?)\s*"
        rf"(\d+{SUFFIX})"
        rf"(?:\s*,?\s*comm[ao]\.?\s*(\d+))?",
        re.IGNORECASE,
    )

    # ═══ LEGGI E DECRETI ═══

    # Pattern: "L. 241/1990" / "Legge 7 agosto 1990, n. 241"
    LEGGE = re.compile(
        r"\b(?:L(?:egge)?\.?)\s*"
        r"(?:(\d{1,2})\s+(\w+)\s+)??"  # data opzionale
        r"(\d{4})?"  # anno se prima di /
        r"[,\s]*n\.?\s*(\d+)"  # numero
        r"(?:\s*/\s*(\d{4}))?",  # /anno
        re.IGNORECASE,
    )

    # Pattern: "D.Lgs. 165/2001" / "decreto legislativo 30 marzo 2001, n. 165"
    DLGS = re.compile(
        r"\b(?:D\.?\s*Lgs\.?|decreto\s+legislativo)\s*"
        r"(?:(\d{1,2})\s+(\w+)\s+)??"  # data opzionale
        r"(\d{4})?"  # anno se prima di /
        r"[,\s]*n\.?\s*(\d+)"  # numero
        r"(?:\s*/\s*(\d{4}))?",  # /anno
        re.IGNORECASE,
    )

    # Pattern: "D.L. 18/2020" / "decreto legge"
    DL = re.compile(
        r"\b(?:D\.?\s*L\.?|decreto[\-\s]+legge)\s*"
        r"(?:(\d{1,2})\s+(\w+)\s+)??"
        r"(\d{4})?"
        r"[,\s]*n\.?\s*(\d+)"
        r"(?:\s*/\s*(\d{4}))?",
        re.IGNORECASE,
    )

    # Pattern: "D.P.R. 380/2001"
    DPR = re.compile(
        r"\b(?:D\.?\s*P\.?\s*R\.?|decreto\s+del\s+presidente\s+della\s+repubblica)\s*"
        r"(?:(\d{1,2})\s+(\w+)\s+)??"
        r"(\d{4})?"
        r"[,\s]*n\.?\s*(\d+)"
        r"(?:\s*/\s*(\d{4}))?",
        re.IGNORECASE,
    )

    # ═══ SENTENZE CASSAZIONE ═══

    # Pattern: "Cass. 12345/2020" / "Cass. Sez. Un. n. 8770/2020"
    CASSAZIONE = re.compile(
        r"\b(?:Cass(?:azione)?\.?)\s*"
        r"(?:(Sez\.?\s*(?:Un\.?|I{1,3}|IV|V|VI(?:\-\d)?|\d))\s*)??"  # sezione
        r"(?:,?\s*(?:n\.?|sent\.?)\s*)?"
        r"(\d+)"  # numero
        r"(?:\s*/\s*(\d{4}))?",  # /anno
        re.IGNORECASE,
    )

    # Pattern Rv: "Rv. 639966-01"
    RV = re.compile(r"\bRv\.?\s*(\d+)(?:[\-/](\d+))?", re.IGNORECASE)

    # ═══ NORME UE ═══

    # Pattern: "Reg. UE 679/2016" / "Regolamento (UE) 2016/679"
    REG_UE = re.compile(
        r"\b(?:Reg(?:olamento)?\.?)\s*"
        r"(?:\(?\s*(?:UE|CE)\s*\)?)\s*"
        r"(?:(\d{4})\s*/\s*)??"  # anno/ (formato nuovo)
        r"(\d+)"  # numero
        r"(?:\s*/\s*(\d{4}))?",  # /anno (formato vecchio)
        re.IGNORECASE,
    )

    # Pattern: "Dir. 2016/680/UE" / "Direttiva (UE) 2016/680"
    DIR_UE = re.compile(
        r"\b(?:Dir(?:ettiva)?\.?)\s*"
        r"(?:\(?\s*(?:UE|CE)\s*\)?)\s*"
        r"(?:(\d{4})\s*/\s*)??"
        r"(\d+)"
        r"(?:\s*/\s*(\d{4}))?"
        r"(?:\s*/\s*(?:UE|CE))?",
        re.IGNORECASE,
    )

    # ═══ COSTITUZIONE ═══

    COSTITUZIONE = re.compile(
        rf"\b(?:art(?:icol[oi])?\.?)\s*"
        rf"(\d+{SUFFIX})"
        rf"\s+(?:Cost\.?|Costituzione)",
        re.IGNORECASE,
    )


# ============================================================
# CODE MAPPING
# ============================================================

CODICE_MAPPING = {
    # Codice Civile
    "c.c.": "CC",
    "cc": "CC",
    "c. c.": "CC",
    "codice civile": "CC",
    "cod. civ.": "CC",
    "cod civ": "CC",
    # Codice Penale
    "c.p.": "CP",
    "cp": "CP",
    "c. p.": "CP",
    "codice penale": "CP",
    "cod. pen.": "CP",
    "cod pen": "CP",
    # Codice Procedura Civile
    "c.p.c.": "CPC",
    "cpc": "CPC",
    "c. p. c.": "CPC",
    "codice procedura civile": "CPC",
    "codice di procedura civile": "CPC",
    # Codice Procedura Penale
    "c.p.p.": "CPP",
    "cpp": "CPP",
    "c. p. p.": "CPP",
    "codice procedura penale": "CPP",
    "codice di procedura penale": "CPP",
    # Costituzione
    "cost.": "COST",
    "cost": "COST",
    "costituzione": "COST",
    # Codice della Strada
    "c.d.s.": "CDS",
    "cds": "CDS",
    "codice della strada": "CDS",
}


# ============================================================
# LEGAL NUMBERS EXTRACTOR
# ============================================================


class LegalNumbersExtractor:
    """
    Estrae numeri legali dal testo.

    Questi numeri diventano ANCHOR POINTS nel Number-Anchored Graph,
    permettendo collegamenti deterministici tra documenti.
    """

    def __init__(self):
        self.patterns = LegalPatterns()
        self.canonical_gen = CanonicalIdGenerator()

    def extract(self, text: str, include_context: bool = True) -> ExtractionResult:
        """
        Estrae tutti i numeri legali dal testo.

        Args:
            text: Testo da analizzare
            include_context: Se True, include la frase di contesto

        Returns:
            ExtractionResult con tutti i numeri trovati
        """
        numbers = []

        # Extract each type
        numbers.extend(self._extract_articles(text, include_context))
        numbers.extend(self._extract_laws(text, include_context))
        numbers.extend(self._extract_sentences(text, include_context))
        numbers.extend(self._extract_eu_norms(text, include_context))
        numbers.extend(self._extract_constitution(text, include_context))

        # Sort by position
        numbers.sort(key=lambda n: n.position.start)

        # Remove duplicates (same canonical_id at overlapping positions)
        numbers = self._deduplicate(numbers)

        return ExtractionResult(
            numbers=numbers,
            text_length=len(text),
        )

    def extract_canonical_ids(self, text: str) -> set[str]:
        """
        Estrae solo i canonical IDs (per lookup veloce).

        Args:
            text: Testo da analizzare

        Returns:
            Set di canonical IDs
        """
        result = self.extract(text, include_context=False)
        return result.unique_canonical_ids

    def iter_numbers(self, text: str) -> Iterator[LegalNumber]:
        """
        Iterator sui numeri legali (memory-efficient).
        """
        result = self.extract(text, include_context=False)
        yield from result.numbers

    # =========================================================================
    # EXTRACTION METHODS
    # =========================================================================

    def _extract_articles(
        self,
        text: str,
        include_context: bool,
    ) -> list[LegalNumber]:
        """Estrae articoli di codice."""
        numbers = []

        for match in self.patterns.ARTICLE_CODICE.finditer(text):
            articolo1 = match.group(1)
            articolo2 = match.group(2)  # Secondo articolo opzionale
            codice_raw = match.group(3).lower().replace(" ", "")

            codice = self._normalize_codice(codice_raw)
            if not codice:
                continue

            # First article
            canonical_id = self.canonical_gen.generate_for_article(codice, articolo1)
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.ARTICLE,
                    codice=codice,
                    numero=articolo1,
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

            # Second article if present
            if articolo2:
                canonical_id2 = self.canonical_gen.generate_for_article(codice, articolo2)
                numbers.append(
                    LegalNumber(
                        raw_text=match.group(0),
                        position=Position(match.start(), match.end()),
                        number_type=LegalNumberType.ARTICLE,
                        codice=codice,
                        numero=articolo2,
                        canonical_id=canonical_id2,
                    )
                )

        return numbers

    def _extract_laws(
        self,
        text: str,
        include_context: bool,
    ) -> list[LegalNumber]:
        """Estrae leggi e decreti."""
        numbers = []

        # Leggi
        for match in self.patterns.LEGGE.finditer(text):
            numero = match.group(4)
            anno = match.group(5) or match.group(3)

            if not anno:
                continue

            canonical_id = self.canonical_gen.generate_for_law("L", numero, int(anno))
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.LAW,
                    codice="L",
                    numero=numero,
                    anno=int(anno),
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        # D.Lgs.
        for match in self.patterns.DLGS.finditer(text):
            numero = match.group(4)
            anno = match.group(5) or match.group(3)

            if not anno:
                continue

            canonical_id = self.canonical_gen.generate_for_law("DLGS", numero, int(anno))
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.LEGISLATIVE_DECREE,
                    codice="DLGS",
                    numero=numero,
                    anno=int(anno),
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        # D.L.
        for match in self.patterns.DL.finditer(text):
            numero = match.group(4)
            anno = match.group(5) or match.group(3)

            if not anno:
                continue

            canonical_id = self.canonical_gen.generate_for_law("DL", numero, int(anno))
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.LAW_DECREE,
                    codice="DL",
                    numero=numero,
                    anno=int(anno),
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        # D.P.R.
        for match in self.patterns.DPR.finditer(text):
            numero = match.group(4)
            anno = match.group(5) or match.group(3)

            if not anno:
                continue

            canonical_id = self.canonical_gen.generate_for_law("DPR", numero, int(anno))
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.DPR,
                    codice="DPR",
                    numero=numero,
                    anno=int(anno),
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        return numbers

    def _extract_sentences(
        self,
        text: str,
        include_context: bool,
    ) -> list[LegalNumber]:
        """Estrae sentenze Cassazione."""
        numbers = []

        for match in self.patterns.CASSAZIONE.finditer(text):
            sezione = match.group(1)
            numero = match.group(2)
            anno = match.group(3)

            if not anno:
                # Try to infer anno from context
                continue

            canonical_id = self.canonical_gen.generate_for_sentence(
                "CASS", numero, int(anno), sezione
            )
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.SENTENCE,
                    codice="CASS",
                    numero=numero,
                    anno=int(anno),
                    sezione=sezione,
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        return numbers

    def _extract_eu_norms(
        self,
        text: str,
        include_context: bool,
    ) -> list[LegalNumber]:
        """Estrae norme UE."""
        numbers = []

        # Regolamenti
        for match in self.patterns.REG_UE.finditer(text):
            anno1 = match.group(1)
            numero = match.group(2)
            anno2 = match.group(3)

            anno = anno1 or anno2
            if not anno:
                continue

            canonical_id = f"REG_UE:{numero}:{anno}"
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.EU_REGULATION,
                    codice="REG_UE",
                    numero=numero,
                    anno=int(anno),
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        # Direttive
        for match in self.patterns.DIR_UE.finditer(text):
            anno1 = match.group(1)
            numero = match.group(2)
            anno2 = match.group(3)

            anno = anno1 or anno2
            if not anno:
                continue

            canonical_id = f"DIR_UE:{numero}:{anno}"
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.EU_DIRECTIVE,
                    codice="DIR_UE",
                    numero=numero,
                    anno=int(anno),
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        return numbers

    def _extract_constitution(
        self,
        text: str,
        include_context: bool,
    ) -> list[LegalNumber]:
        """Estrae articoli Costituzione."""
        numbers = []

        for match in self.patterns.COSTITUZIONE.finditer(text):
            articolo = match.group(1)

            canonical_id = self.canonical_gen.generate_for_article("COST", articolo)
            numbers.append(
                LegalNumber(
                    raw_text=match.group(0),
                    position=Position(match.start(), match.end()),
                    number_type=LegalNumberType.CONSTITUTIONAL,
                    codice="COST",
                    numero=articolo,
                    canonical_id=canonical_id,
                    context_span=self._get_context(text, match.start(), match.end())
                    if include_context
                    else None,
                )
            )

        return numbers

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _normalize_codice(self, codice_raw: str) -> str | None:
        """Normalizza codice raw a formato standard."""
        codice_clean = codice_raw.lower().replace(".", "").replace(" ", "")

        # Direct mapping
        for pattern, code in CODICE_MAPPING.items():
            pattern_clean = pattern.lower().replace(".", "").replace(" ", "")
            if codice_clean == pattern_clean:
                return code

        # Fuzzy matching
        if "civil" in codice_clean:
            return "CC"
        if "penal" in codice_clean and "proc" not in codice_clean:
            return "CP"
        if "proc" in codice_clean and "civ" in codice_clean:
            return "CPC"
        if "proc" in codice_clean and "pen" in codice_clean:
            return "CPP"
        if "cost" in codice_clean:
            return "COST"
        if "strad" in codice_clean:
            return "CDS"

        return None

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 100,
    ) -> str:
        """Estrae contesto intorno alla citazione."""
        # Expand to sentence boundaries
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)

        # Find sentence start
        for i in range(start, ctx_start, -1):
            if text[i] in ".!?\n":
                ctx_start = i + 1
                break

        # Find sentence end
        for i in range(end, ctx_end):
            if text[i] in ".!?\n":
                ctx_end = i + 1
                break

        return text[ctx_start:ctx_end].strip()

    def _deduplicate(self, numbers: list[LegalNumber]) -> list[LegalNumber]:
        """Rimuove duplicati (stesso canonical_id a posizioni sovrapposte)."""
        if not numbers:
            return numbers

        result = []
        seen_positions = set()

        for num in numbers:
            key = (num.canonical_id, num.position.start)
            if key not in seen_positions:
                result.append(num)
                seen_positions.add(key)

        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_extractor = LegalNumbersExtractor()


def extract_legal_numbers(text: str) -> ExtractionResult:
    """Estrae numeri legali dal testo."""
    return _extractor.extract(text)


def extract_canonical_ids(text: str) -> set[str]:
    """Estrae solo i canonical IDs."""
    return _extractor.extract_canonical_ids(text)


def iter_legal_numbers(text: str) -> Iterator[LegalNumber]:
    """Iterator sui numeri legali."""
    return _extractor.iter_numbers(text)
