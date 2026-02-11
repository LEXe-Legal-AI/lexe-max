#!/usr/bin/env python
"""
Article Classification Module v3

Tre assi indipendenti per classificazione articoli:
1. ArticleIdentityClass - per ordinamento, range, dedup, coverage
2. ArticleQualityClass - per validi/deboli e decisioni enrichment
3. LifecycleStatus - stato editoriale (vigente, abrogato, storico)

Regole formali derivate da analisi empirica su:
- Codice Civile (3,208 articoli, 3 sospetti)
- Codice Crisi Impresa (415 articoli, 1 sospetto)

References:
- SCHEMA_KB_OVERVIEW.md
- ARTICLE_EXTRACTION_STRATEGIES
- KB_V3_UNIFIED_SCHEMA.md
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import hashlib


# ==============================================================================
# IDENTITY CLASSES
# ==============================================================================

class ArticleIdentityClass(Enum):
    """
    Classe identità articolo - per ordinamento, range, dedup, coverage.

    Non mescolare con qualità! L'identità riguarda COSA è l'articolo,
    non SE è estratto bene.
    """
    BASE = "base"           # Articolo numerico puro: 1, 2043, 360
    SUFFIX = "suffix"       # Articolo con suffisso latino: 2635-ter, 360-bis
    SPECIAL = "special"     # Fuori schema numerico: disp. transitorie, allegati


# ==============================================================================
# QUALITY CLASSES
# ==============================================================================

class ArticleQualityClass(Enum):
    """
    Classe qualità articolo - per validi/deboli e decisioni enrichment.

    Mappatura verso tabella standard:
    - validi = VALID_STRONG + VALID_SHORT
    - deboli = WEAK + EMPTY
    - invalidi = INVALID (contati a parte, non gonfiano totale)
    """
    VALID_STRONG = "valid_strong"   # Testo presente, struttura coerente, no warning critici
    VALID_SHORT = "valid_short"     # Testo corto ma semanticamente pieno
    WEAK = "weak"                   # Testo presente ma probabilmente incompleto
    EMPTY = "empty"                 # Vuoto, placeholder, "abrogato", "omissis"
    INVALID = "invalid"             # Estrazione rotta, duplicato, heading catturato


# ==============================================================================
# LIFECYCLE STATUS
# ==============================================================================

class LifecycleStatus(Enum):
    """
    Stato editoriale articolo - separato da qualità estrazione.

    Un articolo abrogato può essere corto o "omissis" ma NON è un errore
    di parsing. Evita falsi positivi nel quality check.
    """
    CURRENT = "current"         # Articolo vigente
    HISTORICAL = "historical"   # Versione precedente (multivigenza)
    REPEALED = "repealed"       # Articolo abrogato
    UNKNOWN = "unknown"         # Stato non verificato (default iniziale)


# ==============================================================================
# WARNING TYPES
# ==============================================================================

class WarningType(Enum):
    """Tipi di warning durante classificazione."""
    # Critical (portano a WEAK o INVALID)
    EMPTY_TEXT = "empty_text"
    TRUNCATED_TEXT = "truncated_text"
    MID_LINE_HEADING = "mid_line_heading"
    SECTION_HEADER_CAPTURED = "section_header_captured"
    DUPLICATE_CONTENT = "duplicate_content"

    # Warning (informativi, non cambiano classe)
    SHORT_TEXT = "short_text"
    MISSING_RUBRICA = "missing_rubrica"
    LOW_NORMATIVE_DENSITY = "low_normative_density"
    ABROGATO_MARKER = "abrogato_marker"
    OMISSIS_MARKER = "omissis_marker"

    # Info
    SUFFIX_ARTICLE = "suffix_article"
    SPECIAL_ARTICLE = "special_article"


# ==============================================================================
# LATIN SUFFIXES
# ==============================================================================

LATIN_SUFFIXES = [
    "bis", "ter", "quater", "quinquies", "sexies", "septies", "octies",
    "novies", "nonies", "decies", "undecies", "duodecies", "terdecies",
    "quaterdecies", "quinquiesdecies", "sexiesdecies", "septiesdecies",
    "octiesdecies"
]

SUFFIX_PATTERN = "|".join(LATIN_SUFFIXES)
SUFFIX_REGEX = re.compile(rf"^(\d+)[-\s]?({SUFFIX_PATTERN})$", re.IGNORECASE)

# Ordine suffissi per sort
SUFFIX_ORDER = {s: i for i, s in enumerate(LATIN_SUFFIXES)}


# ==============================================================================
# THRESHOLDS
# ==============================================================================

# Lunghezza testo
MIN_TEXT_LENGTH = 10                # Sotto = EMPTY
SHORT_TEXT_THRESHOLD = 50           # Sotto = potenziale VALID_SHORT
STRONG_TEXT_THRESHOLD = 150         # Sopra = sicuramente VALID_STRONG

# Pattern normativi (indicatori di testo "vero")
NORMATIVE_PATTERNS = [
    r'\bcomma\b',
    r'\bart\.\s*\d+',
    r'\bc\.c\.',
    r'\bc\.p\.',
    r'\bd\.lgs\.',
    r'\blegge\s+\d+',
    r'\bai\s+sensi\b',
    r'\bè\s+punito\b',
    r'\bsono\s+puniti\b',
    r'\bobbligo\b',
    r'\bdiritto\b',
    r'\bcontratto\b',
    r'\bresponsabil',
    r'\brisarcimento\b',
]

# Pattern di articoli vuoti/abrogati
EMPTY_PATTERNS = [
    r'^\s*\[?\s*abrogat[oa]\s*\]?\s*$',
    r'^\s*\[?\s*omissis\s*\]?\s*$',
    r'^\s*\[?\s*soppresso\s*\]?\s*$',
    r'^\s*\[?\s*riservat[oa]\s*\]?\s*$',
    r'^\s*-+\s*$',
    r'^\s*\.+\s*$',
]

# Pattern mid-line headings (trappole frequenti)
MID_LINE_HEADING_PATTERNS = [
    r'^##\s*(LIBRO|PARTE|TITOLO|CAPO|SEZIONE)\s',
    r'^(LIBRO|PARTE|TITOLO|CAPO|SEZIONE)\s+[IVXLCDM]+\s*[-–—]',
    r'^\*\*\s*(Libro|Parte|Titolo|Capo|Sezione)\s',
]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ArticleWarning:
    """Warning singolo."""
    type: WarningType
    severity: str  # 'error', 'warning', 'info'
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ClassifiedArticle:
    """Articolo con classificazione completa."""
    # Identificazione
    raw_num: str                        # "2043-bis", "disp. trans."
    base_num: Optional[int]             # 2043, None per SPECIAL
    suffix: Optional[str]               # "bis", None
    sort_key: str                       # "002043.02" (02=bis)

    # Classi
    identity_class: ArticleIdentityClass
    quality_class: ArticleQualityClass

    # Contenuto
    rubrica: Optional[str]
    testo: str
    testo_hash: str

    # Warnings
    warnings: list[ArticleWarning] = field(default_factory=list)

    # Metadata
    source_file: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.quality_class in (ArticleQualityClass.VALID_STRONG, ArticleQualityClass.VALID_SHORT)

    @property
    def is_weak(self) -> bool:
        return self.quality_class in (ArticleQualityClass.WEAK, ArticleQualityClass.EMPTY)

    @property
    def is_invalid(self) -> bool:
        return self.quality_class == ArticleQualityClass.INVALID

    def to_dict(self) -> dict:
        return {
            "raw_num": self.raw_num,
            "base_num": self.base_num,
            "suffix": self.suffix,
            "sort_key": self.sort_key,
            "identity_class": self.identity_class.value,
            "quality_class": self.quality_class.value,
            "rubrica": self.rubrica,
            "testo_preview": self.testo[:100] + "..." if len(self.testo) > 100 else self.testo,
            "testo_length": len(self.testo),
            "testo_hash": self.testo_hash,
            "warnings": [
                {"type": w.type.value, "severity": w.severity, "message": w.message}
                for w in self.warnings
            ],
            "is_valid": self.is_valid,
            "is_weak": self.is_weak,
            "is_invalid": self.is_invalid,
        }


# ==============================================================================
# PARSING FUNCTIONS
# ==============================================================================

def parse_article_number(raw_num: str) -> tuple[Optional[int], Optional[str], ArticleIdentityClass]:
    """
    Parse numero articolo in componenti.

    Returns:
        (base_num, suffix, identity_class)

    Examples:
        "2043" -> (2043, None, BASE)
        "2635-ter" -> (2635, "ter", SUFFIX)
        "360 bis" -> (360, "bis", SUFFIX)
        "disp. trans." -> (None, None, SPECIAL)
    """
    raw_num = raw_num.strip()

    # Prova pattern SUFFIX
    match = SUFFIX_REGEX.match(raw_num)
    if match:
        base = int(match.group(1))
        suffix = match.group(2).lower()
        return base, suffix, ArticleIdentityClass.SUFFIX

    # Prova pattern BASE (solo numero)
    if raw_num.isdigit():
        return int(raw_num), None, ArticleIdentityClass.BASE

    # Prova numero con separatore (es. "360-bis" non matchato sopra)
    match = re.match(r'^(\d+)[-\s]+(.+)$', raw_num)
    if match:
        base = int(match.group(1))
        rest = match.group(2).lower().strip()
        if rest in LATIN_SUFFIXES:
            return base, rest, ArticleIdentityClass.SUFFIX

    # Fallback: prova estrarre solo il numero
    match = re.match(r'^(\d+)', raw_num)
    if match:
        return int(match.group(1)), None, ArticleIdentityClass.BASE

    # SPECIAL: disposizioni transitorie, allegati, etc.
    return None, None, ArticleIdentityClass.SPECIAL


def make_sort_key(base_num: Optional[int], suffix: Optional[str]) -> str:
    """
    Genera sort key per ordinamento naturale.

    Format: NNNNNN.SS
    - NNNNNN: numero paddato a 6 cifre
    - SS: ordine suffisso (00=nessuno, 01=bis, 02=ter, ...)

    Examples:
        (1, None) -> "000001.00"
        (2043, None) -> "002043.00"
        (2043, "bis") -> "002043.01"
        (2043, "ter") -> "002043.02"
        (None, None) -> "999999.99" (SPECIAL in fondo)
    """
    if base_num is None:
        return "999999.99"

    suffix_order = SUFFIX_ORDER.get(suffix.lower(), 99) if suffix else 0
    return f"{base_num:06d}.{suffix_order:02d}"


def compute_text_hash(text: str) -> str:
    """Calcola hash del testo normalizzato per dedup."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ==============================================================================
# CLASSIFICATION FUNCTIONS
# ==============================================================================

def classify_quality(
    testo: str,
    rubrica: Optional[str] = None,
    identity_class: ArticleIdentityClass = ArticleIdentityClass.BASE,
) -> tuple[ArticleQualityClass, list[ArticleWarning]]:
    """
    Classifica qualità articolo.

    Returns:
        (quality_class, warnings)
    """
    warnings = []
    testo = testo.strip() if testo else ""
    testo_len = len(testo)

    # 1. Check EMPTY patterns
    if not testo or testo_len < MIN_TEXT_LENGTH:
        warnings.append(ArticleWarning(
            type=WarningType.EMPTY_TEXT,
            severity="error",
            message=f"Testo vuoto o troppo corto ({testo_len} chars)",
            details={"length": testo_len}
        ))
        return ArticleQualityClass.EMPTY, warnings

    # Check abrogato/omissis
    for pattern in EMPTY_PATTERNS:
        if re.match(pattern, testo, re.IGNORECASE):
            marker = "abrogato" if "abrog" in testo.lower() else "omissis/placeholder"
            warnings.append(ArticleWarning(
                type=WarningType.ABROGATO_MARKER if "abrog" in testo.lower() else WarningType.OMISSIS_MARKER,
                severity="info",
                message=f"Articolo {marker}",
                details={"text": testo[:50]}
            ))
            return ArticleQualityClass.EMPTY, warnings

    # 2. Check INVALID patterns (mid-line headings, section headers)
    for pattern in MID_LINE_HEADING_PATTERNS:
        if re.match(pattern, testo, re.IGNORECASE | re.MULTILINE):
            warnings.append(ArticleWarning(
                type=WarningType.SECTION_HEADER_CAPTURED,
                severity="error",
                message="Header di sezione catturato come articolo",
                details={"text_preview": testo[:100]}
            ))
            return ArticleQualityClass.INVALID, warnings

    # 3. Calcola normative density
    normative_matches = sum(1 for p in NORMATIVE_PATTERNS if re.search(p, testo, re.IGNORECASE))
    normative_density = normative_matches / max(1, testo_len / 100)

    # 4. Determina classe
    if testo_len >= STRONG_TEXT_THRESHOLD:
        # Testo lungo -> VALID_STRONG (a meno di warning critici)
        if normative_density < 0.1:
            warnings.append(ArticleWarning(
                type=WarningType.LOW_NORMATIVE_DENSITY,
                severity="info",
                message=f"Bassa densità normativa ({normative_density:.2f})",
                details={"density": normative_density, "matches": normative_matches}
            ))
        return ArticleQualityClass.VALID_STRONG, warnings

    elif testo_len >= SHORT_TEXT_THRESHOLD:
        # Testo medio -> dipende da struttura
        # Articoli con suffisso tendono ad essere brevi ma validi
        if identity_class == ArticleIdentityClass.SUFFIX:
            return ArticleQualityClass.VALID_SHORT, warnings

        if normative_density >= 0.3:
            return ArticleQualityClass.VALID_SHORT, warnings

        # Check se sembra completo (termina con punto, ha struttura)
        if testo.rstrip().endswith('.') and '.' in testo[:-1]:
            return ArticleQualityClass.VALID_SHORT, warnings

        warnings.append(ArticleWarning(
            type=WarningType.SHORT_TEXT,
            severity="warning",
            message=f"Testo corto ({testo_len} chars), potrebbe essere incompleto",
            details={"length": testo_len}
        ))
        return ArticleQualityClass.WEAK, warnings

    else:
        # Testo corto -> VALID_SHORT se sembra completo, altrimenti WEAK
        # Definizioni secche e rinvii sono spesso molto corti ma validi
        if testo.rstrip().endswith('.'):
            # Sembra una frase completa
            if normative_density >= 0.5 or len(testo.split()) >= 5:
                return ArticleQualityClass.VALID_SHORT, warnings

        warnings.append(ArticleWarning(
            type=WarningType.SHORT_TEXT,
            severity="warning",
            message=f"Testo molto corto ({testo_len} chars)",
            details={"length": testo_len}
        ))
        return ArticleQualityClass.WEAK, warnings


def classify_article(
    raw_num: str,
    testo: str,
    rubrica: Optional[str] = None,
    source_file: Optional[str] = None,
) -> ClassifiedArticle:
    """
    Classificazione completa di un articolo.

    Args:
        raw_num: Numero articolo come da estrazione ("2043", "2635-ter")
        testo: Testo dell'articolo
        rubrica: Rubrica/titolo opzionale
        source_file: File sorgente opzionale

    Returns:
        ClassifiedArticle con tutte le informazioni
    """
    # 1. Parse identity
    base_num, suffix, identity_class = parse_article_number(raw_num)

    # 2. Sort key
    sort_key = make_sort_key(base_num, suffix)

    # 3. Quality classification
    quality_class, warnings = classify_quality(testo, rubrica, identity_class)

    # 4. Add identity-related warnings
    if identity_class == ArticleIdentityClass.SUFFIX:
        warnings.append(ArticleWarning(
            type=WarningType.SUFFIX_ARTICLE,
            severity="info",
            message=f"Articolo con suffisso latino: {suffix}",
            details={"suffix": suffix}
        ))
    elif identity_class == ArticleIdentityClass.SPECIAL:
        warnings.append(ArticleWarning(
            type=WarningType.SPECIAL_ARTICLE,
            severity="info",
            message="Articolo fuori schema numerico standard",
            details={"raw_num": raw_num}
        ))

    # 5. Check rubrica
    if not rubrica:
        warnings.append(ArticleWarning(
            type=WarningType.MISSING_RUBRICA,
            severity="info",
            message="Rubrica mancante",
            details={}
        ))

    # 6. Text hash
    testo_hash = compute_text_hash(testo)

    return ClassifiedArticle(
        raw_num=raw_num,
        base_num=base_num,
        suffix=suffix,
        sort_key=sort_key,
        identity_class=identity_class,
        quality_class=quality_class,
        rubrica=rubrica,
        testo=testo,
        testo_hash=testo_hash,
        warnings=warnings,
        source_file=source_file,
    )


# ==============================================================================
# BATCH STATISTICS
# ==============================================================================

@dataclass
class BatchStats:
    """
    Statistiche batch per documento.

    Colonne standard:
    - documento, dal, al, totale, validi, deboli
    - invalidi, duplicati (consigliate)
    - missing_expected, extra_unexpected, coverage_pct (se expected set disponibile)
    - sospetti (count articoli con warning critici)
    """
    documento: str

    # Range (calcolato da sort_key)
    dal: str                            # Primo articolo
    al: str                             # Ultimo articolo

    # Conteggi
    totale: int = 0                     # Articoli unici estratti (inclusi SUFFIX)
    validi: int = 0                     # VALID_STRONG + VALID_SHORT
    deboli: int = 0                     # WEAK + EMPTY
    invalidi: int = 0                   # INVALID (contati a parte)
    duplicati: int = 0                  # Hash duplicati

    # Expected set (opzionale)
    expected_count: Optional[int] = None
    missing_expected: int = 0
    extra_unexpected: int = 0
    coverage_pct: Optional[float] = None

    # Sospetti
    sospetti: int = 0                   # Articoli con warning critici

    # Breakdown qualità
    valid_strong: int = 0
    valid_short: int = 0
    weak: int = 0
    empty: int = 0

    # Breakdown identità
    base_count: int = 0
    suffix_count: int = 0
    special_count: int = 0

    def to_dict(self) -> dict:
        return {
            "documento": self.documento,
            "dal": self.dal,
            "al": self.al,
            "totale": self.totale,
            "validi": self.validi,
            "deboli": self.deboli,
            "invalidi": self.invalidi,
            "duplicati": self.duplicati,
            "sospetti": self.sospetti,
            "coverage_pct": self.coverage_pct,
            "breakdown": {
                "valid_strong": self.valid_strong,
                "valid_short": self.valid_short,
                "weak": self.weak,
                "empty": self.empty,
            },
            "identity": {
                "base": self.base_count,
                "suffix": self.suffix_count,
                "special": self.special_count,
            }
        }

    def to_table_row(self) -> str:
        """Riga per tabella CSV/markdown."""
        coverage = f"{self.coverage_pct:.1f}%" if self.coverage_pct else "N/A"
        return f"{self.documento};{self.dal};{self.al};{self.totale};{self.validi};{self.deboli};{self.invalidi};{self.sospetti};{coverage}"

    @staticmethod
    def table_header() -> str:
        return "documento;dal;al;totale;validi;deboli;invalidi;sospetti;coverage"


def compute_batch_stats(
    documento: str,
    articles: list[ClassifiedArticle],
    expected_articles: Optional[set[int]] = None,
) -> BatchStats:
    """
    Calcola statistiche batch per un documento.

    Args:
        documento: Nome/codice documento
        articles: Lista di articoli classificati
        expected_articles: Set opzionale di numeri articolo attesi

    Returns:
        BatchStats completo
    """
    if not articles:
        return BatchStats(documento=documento, dal="N/A", al="N/A")

    # Sort by sort_key
    sorted_articles = sorted(articles, key=lambda a: a.sort_key)

    # Range
    dal = sorted_articles[0].raw_num
    al = sorted_articles[-1].raw_num

    # Initialize stats
    stats = BatchStats(
        documento=documento,
        dal=dal,
        al=al,
        totale=len(articles),
    )

    # Count by quality
    seen_hashes = set()
    critical_warnings = {WarningType.SECTION_HEADER_CAPTURED, WarningType.DUPLICATE_CONTENT}

    for art in articles:
        # Quality counts
        if art.quality_class == ArticleQualityClass.VALID_STRONG:
            stats.valid_strong += 1
            stats.validi += 1
        elif art.quality_class == ArticleQualityClass.VALID_SHORT:
            stats.valid_short += 1
            stats.validi += 1
        elif art.quality_class == ArticleQualityClass.WEAK:
            stats.weak += 1
            stats.deboli += 1
        elif art.quality_class == ArticleQualityClass.EMPTY:
            stats.empty += 1
            stats.deboli += 1
        elif art.quality_class == ArticleQualityClass.INVALID:
            stats.invalidi += 1

        # Identity counts
        if art.identity_class == ArticleIdentityClass.BASE:
            stats.base_count += 1
        elif art.identity_class == ArticleIdentityClass.SUFFIX:
            stats.suffix_count += 1
        elif art.identity_class == ArticleIdentityClass.SPECIAL:
            stats.special_count += 1

        # Duplicates
        if art.testo_hash in seen_hashes:
            stats.duplicati += 1
        seen_hashes.add(art.testo_hash)

        # Sospetti (warning critici)
        if any(w.type in critical_warnings for w in art.warnings):
            stats.sospetti += 1

    # Expected set comparison
    if expected_articles:
        stats.expected_count = len(expected_articles)
        extracted_base_nums = {a.base_num for a in articles if a.base_num is not None}
        stats.missing_expected = len(expected_articles - extracted_base_nums)
        stats.extra_unexpected = len(extracted_base_nums - expected_articles)
        stats.coverage_pct = (len(extracted_base_nums & expected_articles) / len(expected_articles)) * 100

    return stats


# ==============================================================================
# MAIN (test)
# ==============================================================================

if __name__ == "__main__":
    print("=== ARTICLE CLASSIFICATION v2 ===\n")

    # Test cases
    test_cases = [
        ("2043", "Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno.", "Risarcimento per fatto illecito"),
        ("2635-ter", "Le pene previste dai commi precedenti sono raddoppiate se il fatto è commesso...", None),
        ("188", "", None),  # Empty
        ("1020", "## Libro III - Della proprietà\n\n## CAPO II - DELLE SERVITU'", None),  # Header captured
        ("424", "[Abrogato]", "Articolo abrogato"),
        ("5", "La minore età cessa al compimento del diciottesimo anno.", "Maggiore età"),
    ]

    articles = []
    for raw_num, testo, rubrica in test_cases:
        art = classify_article(raw_num, testo, rubrica)
        articles.append(art)

        print(f"Art. {raw_num}:")
        print(f"  Identity: {art.identity_class.value}, Sort: {art.sort_key}")
        print(f"  Quality: {art.quality_class.value}")
        print(f"  Valid: {art.is_valid}, Weak: {art.is_weak}, Invalid: {art.is_invalid}")
        if art.warnings:
            print(f"  Warnings: {[w.type.value for w in art.warnings]}")
        print()

    # Compute batch stats
    print("=== BATCH STATS ===\n")
    stats = compute_batch_stats("TEST", articles)
    print(BatchStats.table_header())
    print(stats.to_table_row())
    print()
    print(f"Breakdown: {stats.to_dict()['breakdown']}")
    print(f"Identity: {stats.to_dict()['identity']}")
