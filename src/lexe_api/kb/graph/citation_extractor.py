"""
LEXE Knowledge Base - Citation Extractor v3.2.1

Two-step citation extraction for building the citation graph:
1. Extract mentions from text (pattern matching)
2. Resolve mentions to massima_id (DB lookup with fallback cascade)

Features (v3.2.1):
- RV normalization (lstrip zeros, handle suffixes like -01)
- Numero normalization (lstrip zeros)
- Semantic deduplication (same source+target+subtype = 1 edge)
- Weight calculation for pruning
- Evidence tracking for debugging
"""

import contextlib
import re
from dataclasses import dataclass, field
from uuid import UUID

import structlog

from lexe_api.kb.graph import RELATION_INDICATORS, RelationSubtype

logger = structlog.get_logger(__name__)


# ============================================================
# CITATION PATTERNS
# ============================================================

# Rv. 639966 or Rv. 639966-01 or Rv.639966
RV_PATTERN = re.compile(r"Rv\.?\s*(\d{5,7})(?:-\d+)?", re.IGNORECASE)

# Cass. Sez. Un., n. 12345/2020 or Sez. 1, n. 12345/2020
SEZ_NUM_ANNO_PATTERN = re.compile(
    r"(?:Cass\.?\s+)?Sez\.?\s*([A-Za-z0-9\-]+)\.?[,\s]+n\.?\s*(\d+)[/\s]+(\d{4})",
    re.IGNORECASE,
)

# n. 12345 del 2020 or n. 12345/2020
NUM_ANNO_PATTERN = re.compile(
    r"n\.?\s*(\d+)\s+(?:del\s+)?(\d{4})",
    re.IGNORECASE,
)

# Sentenza n. 12345 del 2020
SENTENZA_PATTERN = re.compile(
    r"(?:sent(?:enza)?|ord(?:inanza)?)\s*n\.?\s*(\d+)\s+(?:del\s+)?(\d{4})",
    re.IGNORECASE,
)

CITATION_PATTERNS = [
    (RV_PATTERN, "rv"),
    (SEZ_NUM_ANNO_PATTERN, "sez_num_anno"),
    (NUM_ANNO_PATTERN, "num_anno"),
    (SENTENZA_PATTERN, "sentenza"),
]


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class CitationMention:
    """Raw mention extracted from text."""

    # Parsed fields
    rv: str | None = None
    rv_raw: str | None = None  # Original before normalization
    sezione: str | None = None
    numero: str | None = None
    numero_raw: str | None = None  # Original before normalization
    numero_norm: str | None = None  # After lstrip("0")
    anno: int | None = None

    # Context
    raw_span: str = ""
    position: int = 0
    pattern_type: str = ""

    # Relation indicator (detected from surrounding text)
    indicator: str | None = None  # CONFIRMS, DISTINGUISHES, OVERRULES


@dataclass
class ResolvedCitation:
    """Citation resolved to target massima."""

    source_id: UUID
    target_id: UUID
    relation_type: str = "CITES"  # Always CITES for now
    relation_subtype: str | None = None  # CONFIRMS, DISTINGUISHES, OVERRULES

    # Scoring (v3.2.1)
    confidence: float = 1.0
    weight: float = 1.0

    # Evidence (v3.2.1) - for debugging
    evidence: dict = field(default_factory=dict)

    context_span: str = ""


# ============================================================
# NORMALIZATION FUNCTIONS (v3.2.1 Migliorie #3)
# ============================================================


def normalize_numero(numero: str) -> str:
    """
    Normalize numero: lstrip zeros but keep "0" if all zeros.
    "09337" -> "9337", "0000" -> "0"
    """
    if not numero:
        return numero
    stripped = numero.lstrip("0")
    return stripped if stripped else "0"


def normalize_rv(rv: str) -> str:
    """
    Normalize RV: extract core numeric, remove suffixes.
    "639966-01" -> "639966"
    "0639966" -> "639966"
    """
    if not rv:
        return rv
    # Extract just the digits
    match = re.match(r"0*(\d{5,7})", rv)
    if match:
        return match.group(1)
    return rv


def normalize_sezione(sezione: str) -> str:
    """
    Normalize sezione: standardize format.
    "Un" -> "U", "Unite" -> "U", "1" -> "1"
    """
    if not sezione:
        return sezione
    sezione = sezione.strip()
    if sezione.lower() in ("un", "unite", "u"):
        return "U"
    if sezione.lower() in ("lav", "l"):
        return "L"
    return sezione


# ============================================================
# RELATION INDICATOR DETECTION
# ============================================================


def detect_indicator(text: str, position: int, window: int = 50) -> str | None:
    """
    Detect relation indicator from text surrounding the citation.

    Args:
        text: Full text
        position: Position of citation match
        window: Characters to look before the match

    Returns:
        RelationSubtype value or None
    """
    # Get text before the citation
    start = max(0, position - window)
    before_text = text[start:position].lower()

    # Check indicators in priority order (OVERRULES > DISTINGUISHES > CONFIRMS)
    for subtype in [
        RelationSubtype.OVERRULES,
        RelationSubtype.DISTINGUISHES,
        RelationSubtype.CONFIRMS,
    ]:
        patterns = RELATION_INDICATORS.get(subtype, [])
        for pattern in patterns:
            if re.search(pattern, before_text, re.IGNORECASE):
                return subtype.value

    return None


# ============================================================
# STEP 1: EXTRACT MENTIONS
# ============================================================


def extract_mentions(testo: str) -> list[CitationMention]:
    """
    Step 1: Extract all citation mentions from text.

    Returns list of CitationMention with normalized fields.
    """
    if not testo:
        return []

    mentions = []

    for pattern, pattern_type in CITATION_PATTERNS:
        for match in pattern.finditer(testo):
            mention = _parse_match(match, pattern_type)
            if mention:
                mention.position = match.start()
                mention.raw_span = match.group(0)
                mention.pattern_type = pattern_type
                mention.indicator = detect_indicator(testo, match.start())
                mentions.append(mention)

    logger.debug(
        "Mentions extracted",
        count=len(mentions),
        by_type={
            pt: sum(1 for m in mentions if m.pattern_type == pt)
            for pt in ["rv", "sez_num_anno", "num_anno", "sentenza"]
        },
    )

    return mentions


def _parse_match(match: re.Match, pattern_type: str) -> CitationMention | None:
    """Parse regex match into CitationMention based on pattern type."""
    mention = CitationMention()

    if pattern_type == "rv":
        rv_raw = match.group(1)
        mention.rv_raw = rv_raw
        mention.rv = normalize_rv(rv_raw)

    elif pattern_type == "sez_num_anno":
        mention.sezione = normalize_sezione(match.group(1))
        numero_raw = match.group(2)
        mention.numero_raw = numero_raw
        mention.numero_norm = normalize_numero(numero_raw)
        mention.numero = mention.numero_norm  # Use normalized as primary
        with contextlib.suppress(ValueError, TypeError):
            mention.anno = int(match.group(3))

    elif pattern_type in ("num_anno", "sentenza"):
        numero_raw = match.group(1)
        mention.numero_raw = numero_raw
        mention.numero_norm = normalize_numero(numero_raw)
        mention.numero = mention.numero_norm
        with contextlib.suppress(ValueError, TypeError):
            mention.anno = int(match.group(2))

    return mention


# ============================================================
# STEP 2: RESOLVE MENTIONS
# ============================================================


async def resolve_mention(
    mention: CitationMention,
    conn,  # asyncpg.Connection
    source_massima_id: UUID | None = None,
) -> tuple[UUID | None, str]:
    """
    Step 2: Resolve mention to massima_id with cascade fallback.

    Cascade order (v3.2.1 Miglioria #4):
    1. Exact RV match
    2. RV raw match (if different from normalized)
    3. RV text fallback (search in testo)
    4. (sezione, numero_norm, anno) match
    5. (sezione, numero_raw, anno) match (if different)
    6. (numero_norm, anno) match (less precise)

    Returns:
        (target_id, resolver_used) - resolver_used for evidence tracking
    """
    # 1. Exact RV match (normalized)
    if mention.rv:
        result = await conn.fetchval(
            "SELECT id FROM kb.massime WHERE rv = $1 AND is_active = TRUE",
            mention.rv,
        )
        if result:
            # Skip self-references
            if source_massima_id and result == source_massima_id:
                return None, "self_reference"
            return result, "rv_exact"

        # 2. RV raw match (if different)
        if mention.rv_raw and mention.rv_raw != mention.rv:
            result = await conn.fetchval(
                "SELECT id FROM kb.massime WHERE rv = $1 AND is_active = TRUE",
                mention.rv_raw,
            )
            if result:
                if source_massima_id and result == source_massima_id:
                    return None, "self_reference"
                return result, "rv_raw"

        # 3. RV text fallback (search in testo_normalizzato)
        result = await conn.fetchval(
            """
            SELECT id FROM kb.massime
            WHERE testo_normalizzato ILIKE $1
            AND is_active = TRUE
            LIMIT 1
            """,
            f"%Rv. {mention.rv}%",
        )
        if result:
            if source_massima_id and result == source_massima_id:
                return None, "self_reference"
            return result, "rv_text_fallback"

    # 4. (sezione, numero_norm, anno) match
    if mention.sezione and mention.numero_norm and mention.anno:
        result = await conn.fetchval(
            """
            SELECT id FROM kb.massime
            WHERE sezione = $1 AND numero = $2 AND anno = $3 AND is_active = TRUE
            LIMIT 1
            """,
            mention.sezione,
            mention.numero_norm,
            mention.anno,
        )
        if result:
            if source_massima_id and result == source_massima_id:
                return None, "self_reference"
            return result, "sez_num_anno"

        # 5. Try with raw numero
        if mention.numero_raw and mention.numero_raw != mention.numero_norm:
            result = await conn.fetchval(
                """
                SELECT id FROM kb.massime
                WHERE sezione = $1 AND numero = $2 AND anno = $3 AND is_active = TRUE
                LIMIT 1
                """,
                mention.sezione,
                mention.numero_raw,
                mention.anno,
            )
            if result:
                if source_massima_id and result == source_massima_id:
                    return None, "self_reference"
                return result, "sez_num_anno_raw"

    # 6. (numero_norm, anno) match - less precise, higher chance of false positives
    if mention.numero_norm and mention.anno:
        result = await conn.fetchval(
            """
            SELECT id FROM kb.massime
            WHERE numero = $1 AND anno = $2 AND is_active = TRUE
            LIMIT 1
            """,
            mention.numero_norm,
            mention.anno,
        )
        if result:
            if source_massima_id and result == source_massima_id:
                return None, "self_reference"
            return result, "num_anno"

    return None, "unresolved"


# ============================================================
# WEIGHT CALCULATION (v3.2.1 Miglioria #1)
# ============================================================


def compute_weight(mention: CitationMention, resolver: str) -> float:
    """
    Compute weight for edge pruning.

    Weight >= 0.6 -> keep edge
    Weight < 0.6 -> consider discarding

    Factors:
    - Resolver type (rv_exact > rv_raw > fallback > num_anno)
    - Indicator presence (OVERRULES/DISTINGUISHES boost signal)
    """
    base = 1.0

    # Resolver penalty
    if resolver == "rv_exact":
        base = 1.0
    elif resolver == "rv_raw":
        base = 0.95
    elif resolver == "rv_text_fallback":
        base = 0.85
    elif resolver == "sez_num_anno":
        base = 0.9
    elif resolver == "sez_num_anno_raw":
        base = 0.85
    elif resolver == "num_anno":
        base = 0.7  # Less reliable

    # Indicator boost (signal importante)
    if mention.indicator in ("OVERRULES", "DISTINGUISHES"):
        base = min(base * 1.1, 1.0)

    return round(base, 2)


def build_evidence(mention: CitationMention, resolver: str) -> dict:
    """
    Build evidence dict for debugging.

    Contains:
    - pattern: what was matched (RV or sez/num/anno)
    - indicator: detected relation indicator
    - resolver: which resolution method succeeded
    """
    pattern = None
    if mention.rv:
        pattern = f"Rv.{mention.rv}"
        if mention.rv_raw and mention.rv_raw != mention.rv:
            pattern += f" (raw: {mention.rv_raw})"
    elif mention.sezione and mention.numero and mention.anno:
        pattern = f"Sez.{mention.sezione}/n.{mention.numero}/{mention.anno}"
    elif mention.numero and mention.anno:
        pattern = f"n.{mention.numero}/{mention.anno}"

    return {
        "pattern": pattern,
        "indicator": mention.indicator,
        "resolver": resolver,
        "raw_span": mention.raw_span[:100] if mention.raw_span else None,
    }


# ============================================================
# DEDUPLICATION (v3.2.1 Miglioria #2)
# ============================================================


def dedupe_mentions(
    resolved: list[tuple[UUID, CitationMention, UUID, str]],
) -> list[ResolvedCitation]:
    """
    Deduplicate mentions per massima.

    Rule: (source_id, target_id, subtype) -> 1 edge
    Keep: max(confidence), best context_span (first with indicator or shortest)

    Args:
        resolved: List of (source_id, mention, target_id, resolver) tuples

    Returns:
        Deduplicated list of ResolvedCitation
    """
    # Key: (source_id, target_id, subtype)
    seen: dict[tuple, ResolvedCitation] = {}

    for source_id, mention, target_id, resolver in resolved:
        subtype = mention.indicator
        key = (source_id, target_id, subtype)

        confidence = 1.0 if "rv" in resolver else 0.9
        weight = compute_weight(mention, resolver)
        evidence = build_evidence(mention, resolver)

        citation = ResolvedCitation(
            source_id=source_id,
            target_id=target_id,
            relation_type="CITES",
            relation_subtype=subtype,
            confidence=confidence,
            weight=weight,
            evidence=evidence,
            context_span=mention.raw_span,
        )

        if key in seen:
            # Keep if better confidence
            if (
                confidence > seen[key].confidence
                or confidence == seen[key].confidence
                and subtype
                and not seen[key].relation_subtype
            ):
                seen[key] = citation
        else:
            seen[key] = citation

    return list(seen.values())


# ============================================================
# FULL EXTRACTION PIPELINE
# ============================================================


async def extract_citations_for_massima(
    massima_id: UUID,
    testo: str,
    conn,  # asyncpg.Connection
) -> list[ResolvedCitation]:
    """
    Full pipeline: extract and resolve all citations for a massima.

    Returns deduplicated list of ResolvedCitation.
    """
    # Step 1: Extract mentions
    mentions = extract_mentions(testo)

    if not mentions:
        return []

    # Step 2: Resolve each mention
    resolved = []
    for mention in mentions:
        target_id, resolver = await resolve_mention(mention, conn, massima_id)
        if target_id:
            resolved.append((massima_id, mention, target_id, resolver))

    # Step 3: Deduplicate
    citations = dedupe_mentions(resolved)

    logger.debug(
        "Citations extracted for massima",
        massima_id=str(massima_id)[:8],
        mentions=len(mentions),
        resolved=len(resolved),
        deduplicated=len(citations),
    )

    return citations
