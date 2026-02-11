"""
LEXE Knowledge Base - Citation Parser

Estrazione citazioni, norme e riferimenti da massime.
"""

import contextlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# Tipi di Citazione
# ============================================================

CitationType = Literal[
    "pronuncia",  # Altra sentenza/ordinanza
    "norma",  # Articolo di legge/codice
    "regolamento_ue",  # Regolamento UE
    "direttiva_ue",  # Direttiva UE
    "trattato",  # Trattato internazionale
]


@dataclass
class ParsedCitation:
    """Citazione/norma parsata."""

    tipo: CitationType
    raw_text: str

    # Per pronuncia
    sezione: str | None = None
    numero: str | None = None
    anno: int | None = None
    data_decisione: date | None = None
    rv: str | None = None
    autorita: str | None = None  # "Cassazione", "Corte Cost.", etc.

    # Per norma
    articolo: str | None = None
    comma: str | None = None
    lettera: str | None = None
    codice: str | None = None  # "c.c.", "c.p.c.", "c.p.", etc.
    legge: str | None = None  # "l. 241/1990", "d.lgs. 196/2003"

    # Per EU
    regolamento: str | None = None  # "2016/679"
    direttiva: str | None = None  # "2006/123"
    celex: str | None = None  # Identificativo EUR-Lex

    # Metadati
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Converti a dizionario."""
        result = {
            "tipo": self.tipo,
            "raw_text": self.raw_text,
        }
        # Aggiungi solo campi non-None
        for field_name in [
            "sezione",
            "numero",
            "anno",
            "data_decisione",
            "rv",
            "autorita",
            "articolo",
            "comma",
            "lettera",
            "codice",
            "legge",
            "regolamento",
            "direttiva",
            "celex",
            "confidence",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, date):
                    result[field_name] = value.isoformat()
                else:
                    result[field_name] = value
        return result


# ============================================================
# Pattern per Norme Italiane
# ============================================================

# Codici italiani
CODICE_PATTERNS = {
    "c.c.": re.compile(
        r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*"
        r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
        r"(?:,?\s*(?:lett\.?|lettera)\s*([a-z]))?\s*"
        r"(?:c\.?\s*c\.?|cod\.?\s*civ\.?|codice\s+civile)",
        re.IGNORECASE,
    ),
    "c.p.c.": re.compile(
        r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*"
        r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
        r"(?:,?\s*(?:lett\.?|lettera)\s*([a-z]))?\s*"
        r"(?:c\.?\s*p\.?\s*c\.?|cod\.?\s*proc\.?\s*civ\.?|codice\s+di\s+procedura\s+civile)",
        re.IGNORECASE,
    ),
    "c.p.": re.compile(
        r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*"
        r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
        r"(?:,?\s*(?:lett\.?|lettera)\s*([a-z]))?\s*"
        r"(?:c\.?\s*p\.?(?!\s*c)|cod\.?\s*pen\.?|codice\s+penale)",
        re.IGNORECASE,
    ),
    "c.p.p.": re.compile(
        r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*"
        r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
        r"(?:,?\s*(?:lett\.?|lettera)\s*([a-z]))?\s*"
        r"(?:c\.?\s*p\.?\s*p\.?|cod\.?\s*proc\.?\s*pen\.?|codice\s+di\s+procedura\s+penale)",
        re.IGNORECASE,
    ),
    "disp. att. c.p.c.": re.compile(
        r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?)\s*"
        r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
        r"disp\.?\s*att\.?\s*c\.?\s*p\.?\s*c\.?",
        re.IGNORECASE,
    ),
    "cost.": re.compile(
        r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*"
        r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
        r"(?:Cost\.?|Costituzione)",
        re.IGNORECASE,
    ),
}

# Leggi generiche
LEGGE_PATTERN = re.compile(
    r"(?:art\.?|artt\.?)\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\s*"
    r"(?:,?\s*(?:co\.?|comma)\s*(\d+(?:\s*[-–]\s*\d+)?))?\s*"
    r"(?:,?\s*(?:lett\.?|lettera)\s*([a-z]))?\s*"
    r"(?:della?\s+|,\s*)?"
    r"((?:l\.?|legge|d\.?\s*l(?:gs)?\.?|decreto\s*legislativo|d\.?\s*p\.?\s*r\.?)\s*"
    r"(?:n\.?\s*)?(\d+)\s*/\s*(\d{4}))",
    re.IGNORECASE,
)

# Pattern per solo riferimento legge (senza articolo)
LEGGE_REF_PATTERN = re.compile(
    r"(?:l\.?|legge|d\.?\s*l(?:gs)?\.?|decreto\s*legislativo|d\.?\s*p\.?\s*r\.?)\s*"
    r"(?:n\.?\s*)?(\d+)\s*/\s*(\d{4})",
    re.IGNORECASE,
)


# ============================================================
# Pattern per Norme EU
# ============================================================

REGOLAMENTO_UE_PATTERN = re.compile(
    r"(?:Reg\.?|Regolamento)\s*"
    r"(?:\(?\s*(?:UE|CE|CEE)\s*\)?\s*)?"
    r"(?:n\.?\s*)?"
    r"(\d{4})\s*/\s*(\d+)|(\d+)\s*/\s*(\d{4})",
    re.IGNORECASE,
)

DIRETTIVA_UE_PATTERN = re.compile(
    r"(?:Dir\.?|Direttiva)\s*"
    r"(?:\(?\s*(?:UE|CE|CEE)\s*\)?\s*)?"
    r"(?:n\.?\s*)?"
    r"(\d{4})\s*/\s*(\d+)|(\d+)\s*/\s*(\d{4})",
    re.IGNORECASE,
)


# ============================================================
# Pattern per Citazioni Giurisprudenziali
# ============================================================

# Cassazione
CASS_PATTERN = re.compile(
    r"(?:Cass\.?|Cassazione)\s*"
    r"(?:civ\.?|pen\.?)?\s*"
    r"(?:,?\s*(?:Sez\.?|Sezione|Sezioni)\s*(U(?:nite)?|L(?:av)?|[0-9]+(?:\s*-\s*[0-9]+)?))?\s*"
    r"(?:,?\s*(?:Sent\.?|Sentenza|Ord\.?|Ordinanza))?\s*"
    r"(?:,?\s*n\.?\s*(\d+))?\s*"
    r"(?:[/\s]+del\s+|[/\s]+)?"
    r"(?:(\d{1,2})[/.-](\d{1,2})[/.-])?"
    r"(\d{4})?",
    re.IGNORECASE,
)

# Corte Costituzionale
CORTE_COST_PATTERN = re.compile(
    r"(?:Corte\s+Cost\.?|C\.?\s*Cost\.?|Corte\s+Costituzionale)\s*"
    r"(?:,?\s*(?:Sent\.?|Sentenza|Ord\.?|Ordinanza))?\s*"
    r"(?:,?\s*n\.?\s*(\d+))?\s*"
    r"(?:[/\s]+del\s+|[/\s]+)?"
    r"(?:(\d{1,2})[/.-](\d{1,2})[/.-])?"
    r"(\d{4})?",
    re.IGNORECASE,
)

# Consiglio di Stato
CDS_PATTERN = re.compile(
    r"(?:Cons\.?\s*(?:di\s+)?Stato|C\.?\s*d\.?\s*S\.?)\s*"
    r"(?:,?\s*Sez\.?\s*([IVX0-9]+))?\s*"
    r"(?:,?\s*(?:Sent\.?|Sentenza))?\s*"
    r"(?:,?\s*n\.?\s*(\d+))?\s*"
    r"(?:[/\s]+del\s+|[/\s]+)?"
    r"(?:(\d{1,2})[/.-](\d{1,2})[/.-])?"
    r"(\d{4})?",
    re.IGNORECASE,
)

# CGUE
CGUE_PATTERN = re.compile(
    r"(?:CGUE|Corte\s+(?:di\s+)?Giustizia\s*(?:UE|EU)?|C\.?\s*Giust\.?)\s*"
    r"(?:,?\s*causa\s*(?:C-)?(\d+/\d+))?"
    r"(?:,?\s*(?:Sent\.?|Sentenza))?\s*"
    r"(?:,?\s*del\s+)?"
    r"(?:(\d{1,2})[/.-](\d{1,2})[/.-])?"
    r"(\d{4})?",
    re.IGNORECASE,
)


def extract_codice_citations(text: str) -> list[ParsedCitation]:
    """Estrai citazioni di articoli da codici."""
    citations = []

    for codice, pattern in CODICE_PATTERNS.items():
        for match in pattern.finditer(text):
            citation = ParsedCitation(
                tipo="norma",
                raw_text=match.group(0),
                articolo=match.group(1),
                comma=match.group(2) if match.lastindex >= 2 else None,
                lettera=match.group(3) if match.lastindex >= 3 else None,
                codice=codice,
            )
            citations.append(citation)

    return citations


def extract_legge_citations(text: str) -> list[ParsedCitation]:
    """Estrai citazioni di leggi."""
    citations = []

    for match in LEGGE_PATTERN.finditer(text):
        citation = ParsedCitation(
            tipo="norma",
            raw_text=match.group(0),
            articolo=match.group(1),
            comma=match.group(2) if match.lastindex >= 2 else None,
            lettera=match.group(3) if match.lastindex >= 3 else None,
            legge=match.group(4),  # "l. 241/1990"
        )
        citations.append(citation)

    return citations


def extract_eu_citations(text: str) -> list[ParsedCitation]:
    """Estrai citazioni norme EU."""
    citations = []

    # Regolamenti
    for match in REGOLAMENTO_UE_PATTERN.finditer(text):
        # Pattern cattura in ordine diverso a seconda del formato
        if match.group(1) and match.group(2):
            reg = f"{match.group(1)}/{match.group(2)}"
        elif match.group(3) and match.group(4):
            reg = f"{match.group(3)}/{match.group(4)}"
        else:
            continue

        citation = ParsedCitation(
            tipo="regolamento_ue",
            raw_text=match.group(0),
            regolamento=reg,
        )
        citations.append(citation)

    # Direttive
    for match in DIRETTIVA_UE_PATTERN.finditer(text):
        if match.group(1) and match.group(2):
            dir_num = f"{match.group(1)}/{match.group(2)}"
        elif match.group(3) and match.group(4):
            dir_num = f"{match.group(3)}/{match.group(4)}"
        else:
            continue

        citation = ParsedCitation(
            tipo="direttiva_ue",
            raw_text=match.group(0),
            direttiva=dir_num,
        )
        citations.append(citation)

    return citations


def extract_pronuncia_citations(text: str) -> list[ParsedCitation]:
    """Estrai citazioni di pronunce giurisprudenziali."""
    citations = []

    # Cassazione
    for match in CASS_PATTERN.finditer(text):
        sezione = match.group(1)
        numero = match.group(2)
        day = match.group(3)
        month = match.group(4)
        year = match.group(5)

        data_decisione = None
        if day and month and year:
            with contextlib.suppress(ValueError):
                data_decisione = date(int(year), int(month), int(day))

        anno = int(year) if year else None

        if numero or anno:  # Almeno uno dei due
            citation = ParsedCitation(
                tipo="pronuncia",
                raw_text=match.group(0),
                autorita="Cassazione",
                sezione=sezione,
                numero=numero,
                anno=anno,
                data_decisione=data_decisione,
            )
            citations.append(citation)

    # Corte Costituzionale
    for match in CORTE_COST_PATTERN.finditer(text):
        numero = match.group(1)
        day = match.group(2)
        month = match.group(3)
        year = match.group(4)

        data_decisione = None
        if day and month and year:
            with contextlib.suppress(ValueError):
                data_decisione = date(int(year), int(month), int(day))

        if numero or year:
            citation = ParsedCitation(
                tipo="pronuncia",
                raw_text=match.group(0),
                autorita="Corte Costituzionale",
                numero=numero,
                anno=int(year) if year else None,
                data_decisione=data_decisione,
            )
            citations.append(citation)

    # Consiglio di Stato
    for match in CDS_PATTERN.finditer(text):
        sezione = match.group(1)
        numero = match.group(2)
        day = match.group(3)
        month = match.group(4)
        year = match.group(5)

        data_decisione = None
        if day and month and year:
            with contextlib.suppress(ValueError):
                data_decisione = date(int(year), int(month), int(day))

        if numero or year:
            citation = ParsedCitation(
                tipo="pronuncia",
                raw_text=match.group(0),
                autorita="Consiglio di Stato",
                sezione=sezione,
                numero=numero,
                anno=int(year) if year else None,
                data_decisione=data_decisione,
            )
            citations.append(citation)

    # CGUE
    for match in CGUE_PATTERN.finditer(text):
        causa = match.group(1)
        day = match.group(2)
        month = match.group(3)
        year = match.group(4)

        data_decisione = None
        if day and month and year:
            with contextlib.suppress(ValueError):
                data_decisione = date(int(year), int(month), int(day))

        if causa or year:
            citation = ParsedCitation(
                tipo="pronuncia",
                raw_text=match.group(0),
                autorita="CGUE",
                numero=causa,
                anno=int(year) if year else None,
                data_decisione=data_decisione,
            )
            citations.append(citation)

    return citations


def extract_all_citations(text: str) -> list[ParsedCitation]:
    """
    Estrai tutte le citazioni da un testo.

    Args:
        text: Testo massima

    Returns:
        Lista citazioni trovate
    """
    if not text:
        return []

    all_citations = []

    # Estrai per tipo
    all_citations.extend(extract_codice_citations(text))
    all_citations.extend(extract_legge_citations(text))
    all_citations.extend(extract_eu_citations(text))
    all_citations.extend(extract_pronuncia_citations(text))

    # Rimuovi duplicati (stesso raw_text)
    seen = set()
    unique_citations = []
    for c in all_citations:
        if c.raw_text not in seen:
            seen.add(c.raw_text)
            unique_citations.append(c)

    logger.debug(
        "Citations extracted",
        total=len(unique_citations),
        norme=sum(1 for c in unique_citations if c.tipo == "norma"),
        pronunce=sum(1 for c in unique_citations if c.tipo == "pronuncia"),
        eu=sum(1 for c in unique_citations if c.tipo in ("regolamento_ue", "direttiva_ue")),
    )

    return unique_citations


def get_cited_norms(citations: list[ParsedCitation]) -> list[str]:
    """
    Ottieni lista normalizzata di norme citate.

    Utile per costruire edge APPLIES nel graph.
    """
    norms = []
    for c in citations:
        if c.tipo == "norma":
            if c.codice:
                norm_id = f"art. {c.articolo} {c.codice}"
            elif c.legge:
                norm_id = f"art. {c.articolo} {c.legge}"
            else:
                continue

            if c.comma:
                norm_id += f", co. {c.comma}"
            if c.lettera:
                norm_id += f", lett. {c.lettera}"

            norms.append(norm_id.lower())

    return list(set(norms))


def get_cited_pronounce(citations: list[ParsedCitation]) -> list[str]:
    """
    Ottieni lista normalizzata di pronunce citate.

    Utile per costruire edge CITES nel graph.
    """
    pronounce = []
    for c in citations:
        if c.tipo == "pronuncia" and c.autorita and c.numero:
            p_id = f"{c.autorita}"
            if c.sezione:
                p_id += f" sez. {c.sezione}"
            p_id += f" n. {c.numero}"
            if c.anno:
                p_id += f"/{c.anno}"
            pronounce.append(p_id.lower())

    return list(set(pronounce))
