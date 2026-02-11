#!/usr/bin/env python
"""
Article Quality Classification Module

Definisce le classi formali per la validazione degli articoli estratti.
Basato su segnali strutturali, non su header come fonte di verità.

Classi Match (pre-estrazione):
- MATCH_HEADER: Inizio reale di un articolo
- MATCH_REFERENCE: Citazione nel corpo
- MATCH_AMBIGUOUS: Non classificabile

Classi Articolo (post-estrazione):
- ARTICLE_VALID: Buono per RAG ingestion
- ARTICLE_WEAK: Presente ma semanticamente debole
- ARTICLE_INVALID: Da escludere

Warning Types:
- EMPTY_TEXT: Testo vuoto (abrogato o conversione)
- SHORT_TEXT: Testo troppo corto (< soglia)
- MODIFICATION_NOTE: Sembra nota di modifica
- OUT_OF_ORDER: Posizione anomala senza suffisso
- MISSING_IN_RANGE: Gap numerico nel range
- LOW_NORMATIVE_DENSITY: Articolo di coordinamento/rinvio
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ==============================================================================
# ENUMS
# ==============================================================================

class MatchClass(Enum):
    """Classificazione del match pre-estrazione."""
    HEADER = "HEADER"
    REFERENCE = "REFERENCE"
    AMBIGUOUS = "AMBIGUOUS"


class ArticleQuality(Enum):
    """Classificazione qualità articolo post-estrazione."""
    VALID = "VALID"       # Buono per RAG
    WEAK = "WEAK"         # Presente ma debole
    INVALID = "INVALID"   # Da escludere


class WarningType(Enum):
    """Tipi di warning per articoli."""
    EMPTY_TEXT = "empty_text"
    SHORT_TEXT = "short_text"
    MODIFICATION_NOTE = "modification_note"
    OUT_OF_ORDER = "out_of_order"
    MISSING_IN_RANGE = "missing_in_range"
    LOW_NORMATIVE_DENSITY = "low_normative_density"
    CONVERSION_ERROR = "conversion_error"
    ABROGATED = "abrogated"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ArticleWarning:
    """Warning associato a un articolo."""
    type: WarningType
    severity: str  # "info", "warning", "error"
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "severity": self.severity,
            "message": self.message,
            "details": self.details
        }


@dataclass
class QualityAssessment:
    """Risultato della valutazione qualità di un articolo."""
    quality: ArticleQuality
    warnings: list[ArticleWarning] = field(default_factory=list)
    confidence: float = 1.0
    normative_density: str = "normal"  # "high", "normal", "low"

    def to_dict(self) -> dict:
        return {
            "quality": self.quality.value,
            "warnings": [w.to_dict() for w in self.warnings],
            "confidence": self.confidence,
            "normative_density": self.normative_density
        }


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Soglie
MIN_TEXT_LENGTH = 20          # Sotto = EMPTY_TEXT warning (testo quasi vuoto)
SHORT_TEXT_THRESHOLD = 100    # Sotto = SHORT_TEXT warning informativo
WEAK_TEXT_THRESHOLD = 25      # Sotto = classificato WEAK (articoli reali brevi sono ~30+)

# Pattern per note di modifica (portano a INVALID)
MODIFICATION_PATTERNS = [
    r"^-\s*[a-z]\)\s+al\s",           # "- a) al comma..."
    r"^[a-z]\)\s+all[''']articolo",   # "a) all'articolo..."
    r"^[a-z]\)\s+al\s+comma",         # "a) al comma..."
    r"^la\s+lettera\s+[a-z]\)",       # "la lettera a)..."
    r"^il\s+comma\s+\d+\s+è",         # "il comma 1 è..."
    r"^dopo\s+il\s+comma",            # "dopo il comma..."
    r"^sono\s+aggiunti",              # "sono aggiunti..."
    r"^è\s+inserito",                 # "è inserito..."
]
MODIFICATION_RE = re.compile("|".join(MODIFICATION_PATTERNS), re.IGNORECASE)

# Pattern per articoli di coordinamento (LOW_NORMATIVE_DENSITY)
COORDINATION_PATTERNS = [
    r"sono\s+apportate\s+le\s+seguenti\s+modificazioni",
    r"è\s+modificato\s+come\s+segue",
    r"sono\s+abrogate\s+le\s+seguenti",
    r"le\s+parole\s+.*\s+sono\s+sostituite",
    r"dopo\s+le\s+parole\s+.*\s+sono\s+inserite",
]
COORDINATION_RE = re.compile("|".join(COORDINATION_PATTERNS), re.IGNORECASE)

# Pattern per articoli abrogati
ABROGATED_PATTERNS = [
    r"^\s*\(?\s*abrogato\s*\)?\s*$",
    r"^\s*\(?\s*soppresso\s*\)?\s*$",
    r"^\s*\(?\s*omissis\s*\)?\s*$",
]
ABROGATED_RE = re.compile("|".join(ABROGATED_PATTERNS), re.IGNORECASE)

# Pattern per header di sezione erroneamente estratti (INVALID)
SECTION_HEADER_PATTERNS = [
    r"^##\s*SEZIONE\s",           # "## SEZIONE I -..."
    r"^##\s*CAPO\s",              # "## CAPO I -..."
    r"^##\s*TITOLO\s",            # "## TITOLO I -..."
    r"^##\s*LIBRO\s",             # "## LIBRO I -..."
    r"^##\s*PARTE\s",             # "## PARTE I -..."
    r"^-\s*SEZIONE\s",            # "- SEZIONE I"
    r"^-\s*CAPO\s",               # "- CAPO I"
]
SECTION_HEADER_RE = re.compile("|".join(SECTION_HEADER_PATTERNS), re.IGNORECASE)


# ==============================================================================
# CLASSIFICATION FUNCTIONS
# ==============================================================================

def assess_article_quality(
    articolo_num: str,
    testo: str,
    position: int,
    max_position: int,
    article_max: int,
    has_suffix: bool = False,
) -> QualityAssessment:
    """
    Valuta la qualità di un articolo estratto.

    Args:
        articolo_num: Numero articolo (es. "123" o "123-bis")
        testo: Testo dell'articolo
        position: Posizione nel documento
        max_position: Posizione massima nel documento
        article_max: Numero articolo massimo nel documento
        has_suffix: Se l'articolo ha suffisso (bis, ter, etc.)

    Returns:
        QualityAssessment con quality, warnings e metadata
    """
    warnings = []
    quality = ArticleQuality.VALID
    normative_density = "normal"

    text = (testo or "").strip()
    text_len = len(text)

    # Parse numero base
    try:
        base_num = int(re.match(r"(\d+)", articolo_num).group(1))
    except (AttributeError, ValueError):
        base_num = 0

    # --- CHECK 1: Testo vuoto ---
    if text_len == 0:
        # Potrebbe essere abrogato o errore di conversione
        warnings.append(ArticleWarning(
            type=WarningType.EMPTY_TEXT,
            severity="warning",
            message="Testo vuoto - possibile abrogazione o errore conversione",
            details={"text_length": 0}
        ))
        quality = ArticleQuality.WEAK

    # --- CHECK 2: Articolo esplicitamente abrogato ---
    elif ABROGATED_RE.match(text):
        warnings.append(ArticleWarning(
            type=WarningType.ABROGATED,
            severity="info",
            message="Articolo abrogato",
            details={"text": text[:50]}
        ))
        quality = ArticleQuality.WEAK
        normative_density = "low"

    # --- CHECK 3: Testo molto corto ---
    elif text_len < WEAK_TEXT_THRESHOLD:
        warnings.append(ArticleWarning(
            type=WarningType.SHORT_TEXT,
            severity="warning",
            message=f"Testo molto corto ({text_len} chars)",
            details={"text_length": text_len, "threshold": WEAK_TEXT_THRESHOLD}
        ))
        quality = ArticleQuality.WEAK

    # --- CHECK 4: Nota di modifica (INVALID) ---
    elif MODIFICATION_RE.match(text):
        warnings.append(ArticleWarning(
            type=WarningType.MODIFICATION_NOTE,
            severity="error",
            message="Sembra una nota di modifica, non un articolo",
            details={"text_preview": text[:100]}
        ))
        quality = ArticleQuality.INVALID

    # --- CHECK 4b: Header di sezione erroneamente estratto (INVALID) ---
    elif SECTION_HEADER_RE.match(text):
        warnings.append(ArticleWarning(
            type=WarningType.CONVERSION_ERROR,
            severity="error",
            message="Header di sezione erroneamente estratto come articolo",
            details={"text_preview": text[:100]}
        ))
        quality = ArticleQuality.INVALID

    # --- CHECK 5: Articolo di coordinamento ---
    elif COORDINATION_RE.search(text):
        warnings.append(ArticleWarning(
            type=WarningType.LOW_NORMATIVE_DENSITY,
            severity="info",
            message="Articolo di coordinamento - bassa densità normativa",
            details={"type": "coordination"}
        ))
        # Rimane VALID ma con normative_density = low
        normative_density = "low"

    # --- CHECK 6: Out of order (solo senza suffisso) ---
    if max_position > 0 and not has_suffix:
        position_pct = position / max_position
        # Nell'ultimo 10% con numero < 50% del max
        if position_pct > 0.9 and base_num < (article_max * 0.5):
            warnings.append(ArticleWarning(
                type=WarningType.OUT_OF_ORDER,
                severity="warning",
                message=f"Posizione anomala: Art. {base_num} al {position_pct:.0%} del documento",
                details={
                    "position_pct": round(position_pct, 2),
                    "base_num": base_num,
                    "article_max": article_max
                }
            ))
            if quality == ArticleQuality.VALID:
                quality = ArticleQuality.WEAK

    # --- CHECK 7: Testo corto ma non critico (solo warning) ---
    if SHORT_TEXT_THRESHOLD > text_len >= WEAK_TEXT_THRESHOLD:
        warnings.append(ArticleWarning(
            type=WarningType.SHORT_TEXT,
            severity="info",
            message=f"Testo relativamente corto ({text_len} chars)",
            details={"text_length": text_len}
        ))

    return QualityAssessment(
        quality=quality,
        warnings=warnings,
        confidence=1.0 if not warnings else max(0.5, 1.0 - 0.1 * len(warnings)),
        normative_density=normative_density
    )


def classify_articles_batch(articles: list[dict], article_max: int = None) -> list[dict]:
    """
    Classifica un batch di articoli e aggiunge quality assessment.

    Args:
        articles: Lista di articoli con campi standard
        article_max: Numero articolo massimo (auto-detect se None)

    Returns:
        Lista di articoli con campo 'quality' aggiunto
    """
    if not articles:
        return []

    # Auto-detect max
    if article_max is None:
        article_max = max(a.get("articolo_num_norm", 0) for a in articles)

    max_position = max(a.get("position", 0) for a in articles) or 1

    results = []
    for a in articles:
        assessment = assess_article_quality(
            articolo_num=a.get("articolo_num", ""),
            testo=a.get("testo", ""),
            position=a.get("position", 0),
            max_position=max_position,
            article_max=article_max,
            has_suffix=bool(a.get("articolo_suffix"))
        )

        # Aggiungi assessment all'articolo
        a_copy = dict(a)
        a_copy["quality"] = assessment.to_dict()
        results.append(a_copy)

    return results


def generate_quality_summary(articles: list[dict]) -> dict:
    """
    Genera summary delle metriche di qualità.

    Returns:
        Dict con dal, al, totale, validi, deboli, invalidi
    """
    if not articles:
        return {
            "dal": 0,
            "al": 0,
            "totale": 0,
            "validi": 0,
            "deboli": 0,
            "invalidi": 0,
            "by_warning_type": {}
        }

    base_nums = [a.get("articolo_num_norm", 0) for a in articles]

    # Count by quality
    valid_count = 0
    weak_count = 0
    invalid_count = 0
    warning_types = {}

    for a in articles:
        q = a.get("quality", {})
        quality = q.get("quality", "VALID")

        if quality == "VALID":
            valid_count += 1
        elif quality == "WEAK":
            weak_count += 1
        elif quality == "INVALID":
            invalid_count += 1

        # Count warning types
        for w in q.get("warnings", []):
            wtype = w.get("type", "unknown")
            warning_types[wtype] = warning_types.get(wtype, 0) + 1

    return {
        "dal": min(base_nums) if base_nums else 0,
        "al": max(base_nums) if base_nums else 0,
        "totale": len(articles),
        "validi": valid_count,
        "deboli": weak_count,
        "invalidi": invalid_count,
        "by_warning_type": warning_types
    }


# ==============================================================================
# MAIN (test)
# ==============================================================================

if __name__ == "__main__":
    # Test
    test_articles = [
        {"articolo_num": "1", "articolo_num_norm": 1, "testo": "Questo è un articolo normale con testo sufficiente.", "position": 100},
        {"articolo_num": "2", "articolo_num_norm": 2, "testo": "", "position": 200},
        {"articolo_num": "3", "articolo_num_norm": 3, "testo": "(Abrogato)", "position": 300},
        {"articolo_num": "4", "articolo_num_norm": 4, "testo": "Corto", "position": 400},
        {"articolo_num": "373", "articolo_num_norm": 373, "testo": "Al decreto sono apportate le seguenti modificazioni...", "position": 500},
    ]

    classified = classify_articles_batch(test_articles)
    summary = generate_quality_summary(classified)

    print("=== QUALITY SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== ARTICLES ===")
    for a in classified:
        q = a["quality"]
        print(f"  Art. {a['articolo_num']}: {q['quality']} (density: {q['normative_density']})")
        for w in q["warnings"]:
            print(f"    - [{w['severity']}] {w['type']}: {w['message']}")
