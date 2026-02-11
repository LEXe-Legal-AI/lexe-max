"""
LEXE Knowledge Base - Norm Extractor

Estrae riferimenti normativi dalle massime:
- Codici: CC, CPC, CP, CPP, COST
- Leggi: LEGGE, DLGS, DPR, DL
- Testi unici: TUB, TUF, TULPS, CAD

Pattern ottimizzati per testo_normalizzato (già pulito).
"""

import re
from dataclasses import dataclass

# Suffissi articolo
_SUFFIX = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)"
_SUFFIX_CAPTURE = rf"({_SUFFIX})"


@dataclass(frozen=True)
class NormRef:
    """Riferimento normativo estratto."""

    id: str  # canonical ID: CC:2043, CPC:360:bis, LEGGE:241:1990
    code: str  # CC, CPC, CP, CPP, COST, LEGGE, DLGS, DPR, DL
    article: str | None  # for codes: 2043, 360
    suffix: str | None  # bis, ter, quater...
    number: str | None  # for laws: 241, 50
    year: int | None  # for laws: 1990, 2016
    full_ref: str  # human readable: "art. 2043 c.c."
    context_span: str  # text context around citation


def _canon_id(
    code: str,
    article: str | None,
    suffix: str | None,
    number: str | None,
    year: int | None,
) -> str:
    """Build canonical norm ID."""
    if code in {"LEGGE", "DLGS", "DPR", "DL"}:
        return f"{code}:{number}:{year}"
    if suffix:
        return f"{code}:{article}:{suffix}"
    return f"{code}:{article}"


def _clean(s: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", s).strip()


def _get_context(text: str, start: int, end: int, window: int = 80) -> str:
    """Extract context window around match."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return _clean(text[ctx_start:ctx_end])


# Compiled patterns for efficiency
# Order matters: more specific patterns first

PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # ===== CODICI =====
    # art. 2043 c.c. / art 2043 cc / artt. 2043-2045 c.c.
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)(?:\s*[-–]\s*\d+)?\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:c\.?\s*c\.?|cod\.?\s*civ\.?|codice\s+civile)\b",
            re.IGNORECASE,
        ),
        "CC",
        "code",
    ),
    # art. 360 c.p.c. / art 360 cpc
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)(?:\s*[-–]\s*\d+)?\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:c\.?\s*p\.?\s*c\.?|cod\.?\s*proc\.?\s*civ\.?|codice\s+di\s+procedura\s+civile)\b",
            re.IGNORECASE,
        ),
        "CPC",
        "code",
    ),
    # art. 640 c.p. / art 640 cp (NOT c.p.c. or c.p.p.)
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)(?:\s*[-–]\s*\d+)?\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:c\.?\s*p\.?|cod\.?\s*pen\.?|codice\s+penale)(?![.\s]*[cp])\b",
            re.IGNORECASE,
        ),
        "CP",
        "code",
    ),
    # art. 384 c.p.p. / art 384 cpp
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)(?:\s*[-–]\s*\d+)?\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:c\.?\s*p\.?\s*p\.?|cod\.?\s*proc\.?\s*pen\.?|codice\s+di\s+procedura\s+penale)\b",
            re.IGNORECASE,
        ),
        "CPP",
        "code",
    ),
    # art. 111 Cost. / art 111 della Costituzione
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)(?:\s*[-–]\s*\d+)?\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:cost\.?|della\s+costituzione|costituzione)\b",
            re.IGNORECASE,
        ),
        "COST",
        "code",
    ),
    # ===== TESTI UNICI =====
    # art. 10 TUB / art 10 t.u.b.
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:t\.?\s*u\.?\s*b\.?|testo\s+unico\s+bancario)\b",
            re.IGNORECASE,
        ),
        "TUB",
        "code",
    ),
    # art. 21 TUF / art 21 t.u.f.
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:t\.?\s*u\.?\s*f\.?|testo\s+unico\s+finanza)\b",
            re.IGNORECASE,
        ),
        "TUF",
        "code",
    ),
    # art. 5 CAD / art 5 c.a.d.
    (
        re.compile(
            rf"\b(?:artt?\.?\s*)(\d+)\s*{_SUFFIX_CAPTURE}?\s*"
            rf"(?:c\.?\s*a\.?\s*d\.?|codice\s+amministrazione\s+digitale)\b",
            re.IGNORECASE,
        ),
        "CAD",
        "code",
    ),
    # ===== LEGGI =====
    # L. 241/1990, L. n. 241/1990, legge 241/1990, legge n. 241 del 1990
    (
        re.compile(
            r"\b(?:l\.|legge)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*)(\d{4})\b",
            re.IGNORECASE,
        ),
        "LEGGE",
        "law",
    ),
    # legge 7 agosto 1990, n. 241 (forma estesa)
    (
        re.compile(
            r"\blegge\s+\d+\s+\w+\s+(\d{4})\s*,?\s*n\.?\s*(\d+)\b",
            re.IGNORECASE,
        ),
        "LEGGE",
        "law_ext",
    ),
    # d.lgs. 50/2016, d.lgs. n. 50/2016, decreto legislativo 50/2016
    (
        re.compile(
            r"\b(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*)(\d{4})\b",
            re.IGNORECASE,
        ),
        "DLGS",
        "law",
    ),
    # d.lgs. 10 settembre 2003, n. 276 (forma estesa)
    (
        re.compile(
            r"\b(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s+\d+\s+\w+\s+(\d{4})\s*,?\s*n\.?\s*(\d+)\b",
            re.IGNORECASE,
        ),
        "DLGS",
        "law_ext",
    ),
    # d.p.r. 445/2000, d.p.r. n. 445/2000
    (
        re.compile(
            r"\b(?:d\.?\s*p\.?\s*r\.?|decreto\s+del\s+presidente\s+della\s+repubblica)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*)(\d{4})\b",
            re.IGNORECASE,
        ),
        "DPR",
        "law",
    ),
    # d.l. 18/2020, decreto legge 18/2020
    (
        re.compile(
            r"\b(?:d\.?\s*l\.?|decreto\s+legge)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*)(\d{4})\b",
            re.IGNORECASE,
        ),
        "DL",
        "law",
    ),
]


def extract_norms(testo: str) -> list[NormRef]:
    """
    Extract norm references from massima text.

    Args:
        testo: Normalized massima text (testo_normalizzato)

    Returns:
        List of unique NormRef objects
    """
    if not testo:
        return []

    out: dict[str, NormRef] = {}

    for pattern, code, ptype in PATTERNS:
        for m in pattern.finditer(testo):
            _clean(m.group(0))
            context = _get_context(testo, m.start(), m.end())

            if ptype == "code":
                # Codes: article (+ optional suffix)
                article = m.group(1)
                suffix = m.group(2).lower() if m.group(2) else None
                nid = _canon_id(code, article, suffix, None, None)

                suffix_str = f" {suffix}" if suffix else ""
                code_abbr = {
                    "CC": "c.c.",
                    "CPC": "c.p.c.",
                    "CP": "c.p.",
                    "CPP": "c.p.p.",
                    "COST": "Cost.",
                    "TUB": "TUB",
                    "TUF": "TUF",
                    "CAD": "CAD",
                }.get(code, code)
                full_ref = f"art. {article}{suffix_str} {code_abbr}"

                out[nid] = NormRef(
                    id=nid,
                    code=code,
                    article=article,
                    suffix=suffix,
                    number=None,
                    year=None,
                    full_ref=full_ref,
                    context_span=context,
                )

            elif ptype == "law":
                # Laws: number/year
                number = m.group(1)
                year = int(m.group(2))
                nid = _canon_id(code, None, None, number, year)

                code_abbr = {
                    "LEGGE": "L.",
                    "DLGS": "D.Lgs.",
                    "DPR": "D.P.R.",
                    "DL": "D.L.",
                }.get(code, code)
                full_ref = f"{code_abbr} n. {number}/{year}"

                out[nid] = NormRef(
                    id=nid,
                    code=code,
                    article=None,
                    suffix=None,
                    number=number,
                    year=year,
                    full_ref=full_ref,
                    context_span=context,
                )

            elif ptype == "law_ext":
                # Extended law format: year first, then number
                year = int(m.group(1))
                number = m.group(2)
                nid = _canon_id(code, None, None, number, year)

                code_abbr = {
                    "LEGGE": "L.",
                    "DLGS": "D.Lgs.",
                }.get(code, code)
                full_ref = f"{code_abbr} n. {number}/{year}"

                out[nid] = NormRef(
                    id=nid,
                    code=code,
                    article=None,
                    suffix=None,
                    number=number,
                    year=year,
                    full_ref=full_ref,
                    context_span=context,
                )

    return list(out.values())


def _normalize_dirty_query(query: str) -> str:
    """
    Normalize dirty query for better norm detection.

    Handles common variations:
    - "art 2043 cc" -> "art. 2043 c.c."
    - "2043cc" -> "2043 c.c."
    - "d lgs 165 2001" -> "d.lgs. 165/2001"
    - "111 cost" -> "art. 111 cost."
    """
    q = query.lower().strip()

    # Normalize code abbreviations
    replacements = [
        # Codici senza punti
        (r"\bcc\b", "c.c."),
        (r"\bcpc\b", "c.p.c."),
        (r"\bcpp\b", "c.p.p."),
        (r"\bcp\b(?!\s*[cp])", "c.p."),  # cp but not cpc/cpp
        (r"\bcost\b", "cost."),
        (r"\btub\b", "t.u.b."),
        (r"\btuf\b", "t.u.f."),
        (r"\bcad\b", "c.a.d."),
        # Leggi senza punti
        (r"\bdlgs\b", "d.lgs."),
        (r"\bdpr\b", "d.p.r."),
        (r"\bdl\b(?!\s*[gp])", "d.l."),  # dl but not dlgs/dpr
        # Spazi mancanti
        (r"(\d+)(c\.?c\.?|c\.?p\.?c\.?|c\.?p\.?|cost\.?)", r"\1 \2"),
        # Anno senza slash
        (r"(\d+)\s+(\d{4})\b", r"\1/\2"),
    ]

    for pattern, replacement in replacements:
        q = re.sub(pattern, replacement, q)

    return q


def parse_norm_query(query: str) -> dict | None:
    """
    Parse a query to detect norm reference.

    Handles both clean and dirty queries:
    - "art. 2043 c.c." (clean)
    - "art 2043 cc" (dirty)
    - "2043 codice civile" (natural)

    Returns dict with keys: code, article, suffix, number, year
    or None if no norm detected.
    """
    # Try original query first
    result = _parse_norm_query_inner(query)
    if result:
        return result

    # Try normalized version for dirty queries
    normalized = _normalize_dirty_query(query)
    if normalized != query.lower().strip():
        return _parse_norm_query_inner(normalized)

    return None


def _parse_norm_query_inner(query: str) -> dict | None:
    """Inner parsing logic."""
    q = query.lower().strip()

    # art. 2043 c.c. / 2043 cc / art 2043 codice civile
    m = re.search(
        rf"(?:art\.?\s*)?(\d+)\s*({_SUFFIX})?\s*(?:c\.?\s*c\.?|cc|codice\s*civile)\b",
        q,
    )
    if m:
        return {
            "code": "CC",
            "article": m.group(1),
            "suffix": m.group(2),
            "number": None,
            "year": None,
        }

    # art. 360 c.p.c. / 360 cpc
    m = re.search(
        rf"(?:art\.?\s*)?(\d+)\s*({_SUFFIX})?\s*(?:c\.?\s*p\.?\s*c\.?|cpc)\b",
        q,
    )
    if m:
        return {
            "code": "CPC",
            "article": m.group(1),
            "suffix": m.group(2),
            "number": None,
            "year": None,
        }

    # art. 384 c.p.p. / 384 cpp (CHECK BEFORE c.p. to avoid false positive)
    m = re.search(
        rf"(?:art\.?\s*)?(\d+)\s*({_SUFFIX})?\s*(?:c\.?\s*p\.?\s*p\.?|cpp)\b",
        q,
    )
    if m:
        return {
            "code": "CPP",
            "article": m.group(1),
            "suffix": m.group(2),
            "number": None,
            "year": None,
        }

    # art. 640 c.p. / 640 cp (checked after c.p.p.)
    m = re.search(
        rf"(?:art\.?\s*)?(\d+)\s*({_SUFFIX})?\s*(?:c\.?\s*p\.?|cp)\b",
        q,
    )
    if m:
        return {
            "code": "CP",
            "article": m.group(1),
            "suffix": m.group(2),
            "number": None,
            "year": None,
        }

    # art. 111 cost / costituzione
    m = re.search(
        rf"(?:art\.?\s*)?(\d+)\s*({_SUFFIX})?\s*(?:cost\.?|costituzione)\b",
        q,
    )
    if m:
        return {
            "code": "COST",
            "article": m.group(1),
            "suffix": m.group(2),
            "number": None,
            "year": None,
        }

    # legge 241/1990, l. 241/1990, legge 241 1990, l 241 del 1990
    m = re.search(
        r"(?:l\.?|legge)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*|\s+)(\d{4})\b",
        q,
    )
    if m:
        return {
            "code": "LEGGE",
            "article": None,
            "suffix": None,
            "number": m.group(1),
            "year": int(m.group(2)),
        }

    # d.lgs. 50/2016, dlgs 50 2016
    m = re.search(
        r"(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*|\s+)(\d{4})\b",
        q,
    )
    if m:
        return {
            "code": "DLGS",
            "article": None,
            "suffix": None,
            "number": m.group(1),
            "year": int(m.group(2)),
        }

    # d.p.r. 445/2000
    m = re.search(
        r"(?:d\.?\s*p\.?\s*r\.?)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*|\s+)(\d{4})\b",
        q,
    )
    if m:
        return {
            "code": "DPR",
            "article": None,
            "suffix": None,
            "number": m.group(1),
            "year": int(m.group(2)),
        }

    # d.l. 18/2020
    m = re.search(
        r"(?:d\.?\s*l\.?|decreto\s+legge)\s*(?:n\.?\s*)?(\d+)\s*(?:/|del\s*|\s+)(\d{4})\b",
        q,
    )
    if m:
        return {
            "code": "DL",
            "article": None,
            "suffix": None,
            "number": m.group(1),
            "year": int(m.group(2)),
        }

    return None


def norm_to_canonical_id(parsed: dict) -> str:
    """Convert parsed norm dict to canonical ID."""
    return _canon_id(
        parsed["code"],
        parsed.get("article"),
        parsed.get("suffix"),
        parsed.get("number"),
        parsed.get("year"),
    )
