# lexe_api/kb/ingestion/urn_generator.py
"""
URN:NIR Generator - Italian Legal Document Identifiers.

Genera identificativi conformi allo standard URN:NIR (Norme In Rete).
Standard ufficiale italiano per identificare documenti normativi.

Reference:
- https://www.normattiva.it/uri-res/N2Ls
- Circolare AIPA n. 35/2001

Format:
    urn:nir:{autorità}:{tipo}:{data};{numero}[:art{articolo}][~{componente}]

Examples:
    urn:nir:stato:legge:1990-08-07;241                    # Legge 241/1990
    urn:nir:stato:legge:1990-08-07;241:art1               # Art. 1 L. 241/1990
    urn:nir:stato:regio.decreto:1942-03-16;262:art2043    # Art. 2043 c.c.
    urn:nir:stato:costituzione:1947-12-27                  # Costituzione
    urn:nir:stato:decreto.legislativo:2001-03-30;165       # D.Lgs. 165/2001

Costo: $0 (puro Python, lookup table + pattern)
"""

import re
from dataclasses import dataclass
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# URN:NIR KNOWLEDGE BASE
# ============================================================


class CodeReference(NamedTuple):
    """Riferimento a un codice italiano."""

    short_code: str  # CC, CP, CPC, etc.
    full_name: str  # Codice Civile
    authority: str  # stato
    act_type: str  # regio.decreto, legge, etc.
    promulgation_date: str  # YYYY-MM-DD
    act_number: str  # numero atto
    urn_base: str  # URN completo senza articolo


# Mapping codici italiani -> URN:NIR base
CODICI_ITALIANI = {
    "CC": CodeReference(
        short_code="CC",
        full_name="Codice Civile",
        authority="stato",
        act_type="regio.decreto",
        promulgation_date="1942-03-16",
        act_number="262",
        urn_base="urn:nir:stato:regio.decreto:1942-03-16;262",
    ),
    "CP": CodeReference(
        short_code="CP",
        full_name="Codice Penale",
        authority="stato",
        act_type="regio.decreto",
        promulgation_date="1930-10-19",
        act_number="1398",
        urn_base="urn:nir:stato:regio.decreto:1930-10-19;1398",
    ),
    "CPC": CodeReference(
        short_code="CPC",
        full_name="Codice di Procedura Civile",
        authority="stato",
        act_type="regio.decreto",
        promulgation_date="1940-10-28",
        act_number="1443",
        urn_base="urn:nir:stato:regio.decreto:1940-10-28;1443",
    ),
    "CPP": CodeReference(
        short_code="CPP",
        full_name="Codice di Procedura Penale",
        authority="stato",
        act_type="decreto.presidente.repubblica",
        promulgation_date="1988-09-22",
        act_number="447",
        urn_base="urn:nir:stato:decreto.presidente.repubblica:1988-09-22;447",
    ),
    "COST": CodeReference(
        short_code="COST",
        full_name="Costituzione della Repubblica Italiana",
        authority="stato",
        act_type="costituzione",
        promulgation_date="1947-12-27",
        act_number="",
        urn_base="urn:nir:stato:costituzione:1947-12-27",
    ),
    "CDS": CodeReference(
        short_code="CDS",
        full_name="Codice della Strada",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="1992-04-30",
        act_number="285",
        urn_base="urn:nir:stato:decreto.legislativo:1992-04-30;285",
    ),
    "CDPR": CodeReference(
        short_code="CDPR",
        full_name="Codice della Privacy",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="2003-06-30",
        act_number="196",
        urn_base="urn:nir:stato:decreto.legislativo:2003-06-30;196",
    ),
    "CCONS": CodeReference(
        short_code="CCONS",
        full_name="Codice del Consumo",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="2005-09-06",
        act_number="206",
        urn_base="urn:nir:stato:decreto.legislativo:2005-09-06;206",
    ),
    "CAPPALTI": CodeReference(
        short_code="CAPPALTI",
        full_name="Codice dei Contratti Pubblici",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="2023-03-31",
        act_number="36",
        urn_base="urn:nir:stato:decreto.legislativo:2023-03-31;36",
    ),
    "CAMB": CodeReference(
        short_code="CAMB",
        full_name="Codice dell'Ambiente",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="2006-04-03",
        act_number="152",
        urn_base="urn:nir:stato:decreto.legislativo:2006-04-03;152",
    ),
    "CNAV": CodeReference(
        short_code="CNAV",
        full_name="Codice della Navigazione",
        authority="stato",
        act_type="regio.decreto",
        promulgation_date="1942-03-30",
        act_number="327",
        urn_base="urn:nir:stato:regio.decreto:1942-03-30;327",
    ),
    "CAD": CodeReference(
        short_code="CAD",
        full_name="Codice dell'Amministrazione Digitale",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="2005-03-07",
        act_number="82",
        urn_base="urn:nir:stato:decreto.legislativo:2005-03-07;82",
    ),
    "TUB": CodeReference(
        short_code="TUB",
        full_name="Testo Unico Bancario",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="1993-09-01",
        act_number="385",
        urn_base="urn:nir:stato:decreto.legislativo:1993-09-01;385",
    ),
    "TUF": CodeReference(
        short_code="TUF",
        full_name="Testo Unico della Finanza",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="1998-02-24",
        act_number="58",
        urn_base="urn:nir:stato:decreto.legislativo:1998-02-24;58",
    ),
    "TUEL": CodeReference(
        short_code="TUEL",
        full_name="Testo Unico Enti Locali",
        authority="stato",
        act_type="decreto.legislativo",
        promulgation_date="2000-08-18",
        act_number="267",
        urn_base="urn:nir:stato:decreto.legislativo:2000-08-18;267",
    ),
}

# Mapping tipi atto -> formato URN
ACT_TYPES_URN = {
    "legge": "legge",
    "l": "legge",
    "decreto.legislativo": "decreto.legislativo",
    "dlgs": "decreto.legislativo",
    "d.lgs.": "decreto.legislativo",
    "decreto.legge": "decreto.legge",
    "dl": "decreto.legge",
    "d.l.": "decreto.legge",
    "decreto.presidente.repubblica": "decreto.presidente.repubblica",
    "dpr": "decreto.presidente.repubblica",
    "d.p.r.": "decreto.presidente.repubblica",
    "regio.decreto": "regio.decreto",
    "rd": "regio.decreto",
    "r.d.": "regio.decreto",
    "costituzione": "costituzione",
    "cost": "costituzione",
}


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class ParsedURN:
    """URN parsato in componenti."""

    authority: str  # "stato", "regione:lombardia", etc.
    act_type: str  # "legge", "decreto.legislativo", etc.
    date: str  # "1990-08-07"
    number: str  # "241"
    article: str | None = None  # "1", "2043", "360bis"
    comma: str | None = None  # "1", "2"
    component: str | None = None  # "allegato1", etc.

    @property
    def urn(self) -> str:
        """Ricostruisce URN completo."""
        base = f"urn:nir:{self.authority}:{self.act_type}:{self.date};{self.number}"

        if self.article:
            base += f":art{self.article}"

        if self.comma:
            base += f"~com{self.comma}"

        if self.component:
            base += f"~{self.component}"

        return base


# ============================================================
# URN GENERATOR
# ============================================================


class URNGenerator:
    """
    Generatore URN:NIR per documenti normativi italiani.

    Supporta:
    - Codici (CC, CP, CPC, CPP, CDS, etc.)
    - Leggi e decreti (L. 241/1990, D.Lgs. 165/2001)
    - Articoli, commi, componenti
    """

    # Pattern per normalizzare numero articolo
    ARTICLE_SUFFIX_PATTERN = re.compile(
        r"^(\d+)[\-\s]*(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)$",
        re.IGNORECASE,
    )

    def __init__(self):
        self.codici = CODICI_ITALIANI

    def generate_for_codice(
        self,
        codice: str,
        articolo: str,
        comma: str | None = None,
    ) -> str | None:
        """
        Genera URN per articolo di codice.

        Args:
            codice: Codice abbreviato (CC, CP, CPC, etc.)
            articolo: Numero articolo (2043, 360-bis)
            comma: Numero comma opzionale

        Returns:
            URN:NIR string o None se codice non supportato
        """
        codice_upper = codice.upper()

        if codice_upper not in self.codici:
            logger.warning("Unknown codice", codice=codice)
            return None

        ref = self.codici[codice_upper]
        art_normalized = self._normalize_article(articolo)

        urn = f"{ref.urn_base}:art{art_normalized}"

        if comma:
            urn += f"~com{comma}"

        return urn

    def generate_for_law(
        self,
        act_type: str,
        number: str | int,
        year: int,
        month: int | None = None,
        day: int | None = None,
        article: str | None = None,
        comma: str | None = None,
    ) -> str:
        """
        Genera URN per legge o decreto.

        Args:
            act_type: Tipo atto (legge, dlgs, dpr, dl)
            number: Numero atto
            year: Anno
            month: Mese (opzionale, default 1)
            day: Giorno (opzionale, default 1)
            article: Articolo opzionale
            comma: Comma opzionale

        Returns:
            URN:NIR string
        """
        # Normalize act type
        act_type_urn = ACT_TYPES_URN.get(act_type.lower(), act_type.lower())

        # Build date
        month = month or 1
        day = day or 1
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        urn = f"urn:nir:stato:{act_type_urn}:{date_str};{number}"

        if article:
            art_normalized = self._normalize_article(article)
            urn += f":art{art_normalized}"

        if comma:
            urn += f"~com{comma}"

        return urn

    def parse(self, urn: str) -> ParsedURN | None:
        """
        Parsa URN in componenti.

        Args:
            urn: URN:NIR string

        Returns:
            ParsedURN o None se formato non valido
        """
        # Pattern: urn:nir:{authority}:{type}:{date};{number}[:art{art}][~{component}]
        pattern = re.compile(
            r"^urn:nir:"
            r"([^:]+):"  # authority
            r"([^:]+):"  # type
            r"(\d{4}-\d{2}-\d{2});"  # date
            r"(\d+)"  # number
            r"(?::art(\w+))?"  # article (optional)
            r"(?:~com(\w+))?"  # comma (optional)
            r"(?:~(\w+))?$"  # component (optional)
        )

        match = pattern.match(urn.lower())
        if not match:
            return None

        return ParsedURN(
            authority=match.group(1),
            act_type=match.group(2),
            date=match.group(3),
            number=match.group(4),
            article=match.group(5),
            comma=match.group(6),
            component=match.group(7),
        )

    def get_normattiva_url(self, urn: str) -> str:
        """
        Genera URL Normattiva da URN.

        Args:
            urn: URN:NIR string

        Returns:
            URL Normattiva
        """
        # Normattiva usa formato: https://www.normattiva.it/uri-res/N2Ls?{urn}
        return f"https://www.normattiva.it/uri-res/N2Ls?{urn}"

    def get_codice_info(self, codice: str) -> CodeReference | None:
        """Restituisce info su un codice."""
        return self.codici.get(codice.upper())

    def list_codici(self) -> list[str]:
        """Lista tutti i codici supportati."""
        return list(self.codici.keys())

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _normalize_article(self, articolo: str) -> str:
        """
        Normalizza numero articolo per URN.

        - "2043" -> "2043"
        - "360-bis" -> "360bis"
        - "360 bis" -> "360bis"
        """
        articolo = articolo.lower().strip()

        # Remove spaces and hyphens before suffix
        match = self.ARTICLE_SUFFIX_PATTERN.match(articolo.replace("-", "").replace(" ", ""))
        if match:
            return match.group(1) + match.group(2)

        # Simple cleanup
        return articolo.replace("-", "").replace(" ", "")


# ============================================================
# CANONICAL ID GENERATOR (For Number-Anchored Graph)
# ============================================================


class CanonicalIdGenerator:
    """
    Genera ID canonici per il Number-Anchored Graph.

    Format: {CODICE}:{ARTICOLO} per codici
            {TIPO}:{NUMERO}:{ANNO} per leggi/decreti

    Examples:
        CC:2043
        CP:575
        L:241:1990
        DLGS:165:2001
    """

    def generate_for_article(self, codice: str, articolo: str) -> str:
        """
        Genera canonical ID per articolo di codice.

        Args:
            codice: CC, CP, CPC, etc.
            articolo: 2043, 575, 360-bis

        Returns:
            Canonical ID (es. "CC:2043")
        """
        articolo_clean = articolo.lower().replace("-", "").replace(" ", "")
        return f"{codice.upper()}:{articolo_clean}"

    def generate_for_law(
        self,
        act_type: str,
        number: str | int,
        year: int,
    ) -> str:
        """
        Genera canonical ID per legge/decreto.

        Args:
            act_type: L, DLGS, DL, DPR
            number: Numero atto
            year: Anno

        Returns:
            Canonical ID (es. "L:241:1990")
        """
        type_normalized = self._normalize_act_type(act_type)
        return f"{type_normalized}:{number}:{year}"

    def generate_for_sentence(
        self,
        court: str,
        number: str | int,
        year: int,
        section: str | None = None,
    ) -> str:
        """
        Genera canonical ID per sentenza.

        Args:
            court: CASS, CORTE_COST, etc.
            number: Numero sentenza
            year: Anno
            section: Sezione opzionale

        Returns:
            Canonical ID (es. "CASS:12345:2020")
        """
        court_normalized = court.upper().replace(" ", "_")
        base = f"{court_normalized}:{number}:{year}"

        if section:
            base = f"{court_normalized}:{section}:{number}:{year}"

        return base

    def parse(self, canonical_id: str) -> dict | None:
        """
        Parsa canonical ID in componenti.

        Returns:
            Dict con componenti o None se invalido
        """
        parts = canonical_id.split(":")

        if len(parts) < 2:
            return None

        {"type": parts[0]}

        # Articolo di codice: CC:2043
        if parts[0] in CODICI_ITALIANI and len(parts) == 2:
            return {
                "type": "article",
                "codice": parts[0],
                "articolo": parts[1],
            }

        # Legge/Decreto: L:241:1990
        if parts[0] in ("L", "DLGS", "DL", "DPR") and len(parts) == 3:
            return {
                "type": "law",
                "act_type": parts[0],
                "number": parts[1],
                "year": int(parts[2]),
            }

        # Sentenza: CASS:12345:2020
        if parts[0] in ("CASS", "CORTE_COST") and len(parts) >= 3:
            return {
                "type": "sentence",
                "court": parts[0],
                "number": parts[-2],
                "year": int(parts[-1]),
                "section": parts[1] if len(parts) > 3 else None,
            }

        return None

    def _normalize_act_type(self, act_type: str) -> str:
        """Normalizza tipo atto per canonical ID."""
        mapping = {
            "legge": "L",
            "l": "L",
            "decreto legislativo": "DLGS",
            "decreto.legislativo": "DLGS",
            "dlgs": "DLGS",
            "d.lgs.": "DLGS",
            "decreto legge": "DL",
            "decreto.legge": "DL",
            "dl": "DL",
            "d.l.": "DL",
            "decreto presidente repubblica": "DPR",
            "decreto.presidente.repubblica": "DPR",
            "dpr": "DPR",
            "d.p.r.": "DPR",
        }
        return mapping.get(act_type.lower(), act_type.upper())


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

_urn_generator = URNGenerator()
_canonical_generator = CanonicalIdGenerator()


def generate_urn(
    codice: str,
    articolo: str,
    comma: str | None = None,
) -> str | None:
    """Genera URN:NIR per articolo di codice."""
    return _urn_generator.generate_for_codice(codice, articolo, comma)


def generate_canonical_id(codice: str, articolo: str) -> str:
    """Genera canonical ID per articolo."""
    return _canonical_generator.generate_for_article(codice, articolo)


def parse_urn(urn: str) -> ParsedURN | None:
    """Parsa URN in componenti."""
    return _urn_generator.parse(urn)


def get_normattiva_url(urn: str) -> str:
    """Genera URL Normattiva da URN."""
    return _urn_generator.get_normattiva_url(urn)
