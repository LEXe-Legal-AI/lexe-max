#!/usr/bin/env python
"""
Enrichment Policy Module

Policy per arricchimento articoli da fonti esterne:
1. Studio Cataldi (locale) - 24 codici, illimitato
2. Brocardi (web) - CC/CP/CPC/CPP/CCI/COST/GDPR, rate limited
3. Normattiva (API) - validazione vigenza

Regole:
- PDF Altalex è sempre fonte primaria (text_primary)
- Enrichment solo per WEAK/EMPTY o watchlist
- Mai sovrascrivere senza traccia (provenance)
- Similarity score per decidere quale testo usare
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import difflib


# ==============================================================================
# CONFIGURATION
# ==============================================================================

STUDIO_CATALDI_ROOT = Path("C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa")

# Mapping Studio Cataldi folder -> codice standard
STUDIO_CATALDI_MAP = {
    'testo-unico-bancario': 'TUB',
    'testo-unico-intermediazione-finanziaria': 'TUF',
    'testo-unico-imposte-sui-redditi': 'TUIR',
    'testo-unico-edilizia': 'TUE',
    'testo-unico-enti-locali': 'TUEL',
    'testo-unico-immigrazione': 'TUI',
    'testo-unico-sicurezza-sul-lavoro': 'TUSL',
    'testo-unico-spese-giustizia': 'TUSG',
    'testo-unico-iva': 'DIVA',
    'codice-degli-appalti': 'CAPP',
    'codice-dei-beni-culturali': 'CBCP',
    'codice-del-consumo': 'CCONS',
    'codice-dell-ambiente': 'CAMB',
    'codice-della-privacy': 'CPRIV',
    'codice-della-proprieta-industriale': 'CPI',
    'codicedellastrada': 'CDS',
    'codice-delle-assicurazioni-private': 'CAP',
    'codice-della-navigazione': 'CNAV',
    'statuto-dei-lavoratori': 'SL',
    'ordinamento-penitenziario': 'OP',
    'legge-fallimentare': 'LF',
    'diritto-internazionale-privato': 'LDIP',
    'legge-divorzio': 'LDIV',
    'legge-locazioni-abitative': 'LLOC',
}

# Reverse mapping
CODE_TO_FOLDER = {v: k for k, v in STUDIO_CATALDI_MAP.items()}

# Codici che necessitano Brocardi (non in Studio Cataldi)
BROCARDI_CODES = {'CC', 'CP', 'CPC', 'CPP', 'CCI', 'COST', 'GDPR'}

# Soglie
WEAK_TEXT_THRESHOLD = 300      # Sotto = trigger enrichment
SHORT_RUBRICA_THRESHOLD = 6    # Rubrica < 6 chars = trigger
SIMILARITY_CONFIRM = 0.70      # >= conferma forte
SIMILARITY_PARTIAL = 0.40      # >= conferma parziale
SIMILARITY_CONFLICT = 0.40     # < conflitto serio

# Watchlist: articoli chiave da arricchire sempre
# Formato: {codice: [articoli]}
WATCHLIST = {
    'CC': [1, 2, 1218, 1223, 1224, 1226, 1227, 2043, 2049, 2050, 2051, 2052, 2054, 2055, 2056, 2057, 2058, 2059,
           2086, 2087, 2094, 2095, 2096, 2104, 2222, 2229, 2230, 2325, 2380, 2381, 2392, 2393, 2394, 2395, 2409,
           2476, 2477, 2484, 2497, 2740, 2929, 2930, 2932],
    'CP': [40, 41, 42, 43, 56, 57, 58, 59, 61, 62, 110, 185, 316, 317, 318, 319, 323, 336, 337, 340, 368, 369,
           416, 423, 572, 575, 576, 582, 588, 589, 590, 591, 594, 595, 612, 615, 624, 628, 629, 640, 646],
    'CCI': [1, 2, 3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 40, 44, 46, 47, 48, 49, 54, 55,
            56, 57, 84, 85, 121, 125, 161, 162, 163, 164, 165, 166, 167, 168, 169, 172, 173, 174, 175, 176, 185,
            186, 189, 190, 191, 192, 193, 194, 195, 196, 197, 268, 282, 283, 284, 285, 286, 287, 288, 289, 290],
    'TUB': [1, 10, 11, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 144],
    'TUF': [1, 5, 6, 18, 19, 21, 23, 24, 25, 94, 95, 96, 97, 100, 101, 102, 103, 104, 105, 106, 180, 181, 182,
            183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195],
}


# ==============================================================================
# ENUMS
# ==============================================================================

class EnrichmentSource(Enum):
    """Fonte di arricchimento."""
    STUDIO_CATALDI = "studio_cataldi"
    BROCARDI = "brocardi"
    NORMATTIVA = "normattiva"
    NONE = "none"


class EnrichmentStatus(Enum):
    """Stato dell'arricchimento."""
    OK = "ok"
    NOT_FOUND = "not_found"
    MISMATCH = "mismatch"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    SKIPPED = "skipped"


class PreferredSource(Enum):
    """Fonte preferita per il testo finale."""
    PRIMARY = "primary"      # Usa text_primary (PDF)
    ENRICHED = "enriched"    # Usa text_enriched (web/locale)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class EnrichmentResult:
    """Risultato di un singolo arricchimento."""
    source: EnrichmentSource
    status: EnrichmentStatus
    text_enriched: Optional[str] = None
    rubrica_enriched: Optional[str] = None
    similarity_score: float = 0.0
    confidence: float = 0.0
    error_message: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "status": self.status.value,
            "text_enriched": self.text_enriched,
            "rubrica_enriched": self.rubrica_enriched,
            "similarity_score": round(self.similarity_score, 3),
            "confidence": round(self.confidence, 2),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class EnrichedArticle:
    """Articolo con dati di arricchimento."""
    # Identificazione
    article_key: str           # es. "CC:2043" o "CC:2043-bis"
    doc_id: str                # es. "CC"
    base_num: int              # es. 2043
    suffix: Optional[str]      # es. "bis" o None

    # Testi
    text_primary: str          # Da PDF
    text_enriched: Optional[str] = None
    text_preferred: Optional[str] = None
    preferred_source: PreferredSource = PreferredSource.PRIMARY

    # Rubriche
    rubrica_primary: Optional[str] = None
    rubrica_enriched: Optional[str] = None

    # Enrichment info
    enrichment: Optional[EnrichmentResult] = None
    trigger_reason: str = ""   # Perché è stato arricchito

    # Quality (from article_quality module)
    quality_class: str = "VALID"
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "article_key": self.article_key,
            "doc_id": self.doc_id,
            "base_num": self.base_num,
            "suffix": self.suffix,
            "text_primary": self.text_primary,
            "text_enriched": self.text_enriched,
            "text_preferred": self.text_preferred,
            "preferred_source": self.preferred_source.value,
            "rubrica_primary": self.rubrica_primary,
            "rubrica_enriched": self.rubrica_enriched,
            "enrichment": self.enrichment.to_dict() if self.enrichment else None,
            "trigger_reason": self.trigger_reason,
            "quality_class": self.quality_class,
            "warnings": self.warnings,
        }


# ==============================================================================
# TRIGGER LOGIC
# ==============================================================================

def should_enrich(
    doc_id: str,
    base_num: int,
    suffix: Optional[str],
    text: str,
    rubrica: Optional[str],
    quality_class: str,
) -> tuple[bool, str]:
    """
    Determina se un articolo deve essere arricchito.

    Returns:
        (should_enrich, reason)
    """
    text_len = len((text or "").strip())
    rubrica_len = len((rubrica or "").strip())

    # A. WEAK o EMPTY
    if quality_class in ("WEAK", "EMPTY"):
        return True, f"quality_{quality_class.lower()}"

    # B. Rubrica mancante o troppo corta
    if rubrica_len < SHORT_RUBRICA_THRESHOLD:
        return True, "rubrica_missing_or_short"

    # C. Testo corto con segnali di fusione
    if text_len < 1200 and text and re.search(r"Art\.\s*\d+\.", text):
        return True, "text_fused"

    # D. In watchlist
    if doc_id in WATCHLIST:
        if base_num in WATCHLIST[doc_id]:
            return True, "watchlist"

    return False, ""


def get_enrichment_source(doc_id: str) -> EnrichmentSource:
    """
    Determina quale fonte usare per l'arricchimento.
    """
    if doc_id in BROCARDI_CODES:
        return EnrichmentSource.BROCARDI

    if doc_id in CODE_TO_FOLDER:
        return EnrichmentSource.STUDIO_CATALDI

    return EnrichmentSource.NONE


# ==============================================================================
# SIMILARITY
# ==============================================================================

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calcola similarity tra due testi.

    Usa SequenceMatcher per un calcolo veloce.
    Per produzione, considera embedding cosine.
    """
    if not text1 or not text2:
        return 0.0

    # Normalizza
    t1 = re.sub(r'\s+', ' ', text1.lower().strip())
    t2 = re.sub(r'\s+', ' ', text2.lower().strip())

    return difflib.SequenceMatcher(None, t1, t2).ratio()


def decide_preferred_source(
    text_primary: str,
    text_enriched: Optional[str],
    quality_class: str,
    similarity_score: float,
) -> tuple[PreferredSource, str]:
    """
    Decide quale fonte usare per il testo finale.

    Returns:
        (preferred_source, reason)
    """
    # Se non c'è enriched, usa primary
    if not text_enriched:
        return PreferredSource.PRIMARY, "no_enriched_text"

    # Se primary è VALID, preferisci primary (enriched solo per rubrica)
    if quality_class == "VALID":
        return PreferredSource.PRIMARY, "primary_valid"

    # Se WEAK/EMPTY e similarity >= 0.40, usa enriched
    if quality_class in ("WEAK", "EMPTY"):
        if similarity_score >= SIMILARITY_PARTIAL:
            return PreferredSource.ENRICHED, f"weak_enriched_sim_{similarity_score:.2f}"
        else:
            # Conflitto: non sostituire, flag per review
            return PreferredSource.PRIMARY, f"conflict_sim_{similarity_score:.2f}"

    return PreferredSource.PRIMARY, "default"


# ==============================================================================
# ARTICLE KEY
# ==============================================================================

def make_article_key(doc_id: str, base_num: int, suffix: Optional[str] = None) -> str:
    """Genera chiave canonica articolo."""
    if suffix:
        return f"{doc_id}:{base_num}-{suffix}"
    return f"{doc_id}:{base_num}"


def parse_article_key(key: str) -> tuple[str, int, Optional[str]]:
    """Parse chiave articolo in (doc_id, base_num, suffix)."""
    match = re.match(r"([A-Z]+):(\d+)(?:-([a-z]+))?", key)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return "", 0, None


# ==============================================================================
# STATS
# ==============================================================================

@dataclass
class EnrichmentStats:
    """Statistiche di arricchimento per documento."""
    doc_id: str
    total: int = 0
    enriched_count: int = 0
    weak_fixed_count: int = 0   # WEAK/EMPTY -> preferred=enriched
    conflict_count: int = 0      # similarity < 0.40
    not_found_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "total": self.total,
            "enriched_count": self.enriched_count,
            "weak_fixed_count": self.weak_fixed_count,
            "conflict_count": self.conflict_count,
            "not_found_count": self.not_found_count,
            "error_count": self.error_count,
        }


def compute_enrichment_stats(articles: list[EnrichedArticle], doc_id: str) -> EnrichmentStats:
    """Calcola statistiche di arricchimento."""
    stats = EnrichmentStats(doc_id=doc_id, total=len(articles))

    for a in articles:
        if a.enrichment:
            if a.enrichment.status == EnrichmentStatus.OK:
                stats.enriched_count += 1
                if a.preferred_source == PreferredSource.ENRICHED:
                    stats.weak_fixed_count += 1
                if a.enrichment.similarity_score < SIMILARITY_CONFLICT:
                    stats.conflict_count += 1
            elif a.enrichment.status == EnrichmentStatus.NOT_FOUND:
                stats.not_found_count += 1
            elif a.enrichment.status == EnrichmentStatus.ERROR:
                stats.error_count += 1

    return stats


# ==============================================================================
# MAIN (test)
# ==============================================================================

if __name__ == "__main__":
    # Test
    print("=== ENRICHMENT POLICY ===")
    print(f"Studio Cataldi codes: {len(CODE_TO_FOLDER)}")
    print(f"Brocardi codes: {BROCARDI_CODES}")
    print(f"Watchlist articles: {sum(len(v) for v in WATCHLIST.values())}")

    # Test trigger
    print("\n=== TRIGGER TESTS ===")
    tests = [
        ("CC", 2043, None, "Testo lungo...", "Responsabilità civile", "VALID"),
        ("CC", 188, None, "", None, "EMPTY"),
        ("TUB", 106, None, "Testo corto", None, "WEAK"),
        ("CC", 1, None, "Lungo testo completo...", "Fonti del diritto", "VALID"),
    ]

    for doc_id, base, suffix, text, rubrica, quality in tests:
        should, reason = should_enrich(doc_id, base, suffix, text, rubrica, quality)
        source = get_enrichment_source(doc_id)
        print(f"  {doc_id}:{base} -> enrich={should} ({reason}), source={source.value}")

    # Test similarity
    print("\n=== SIMILARITY TESTS ===")
    t1 = "Il danno ingiusto cagionato ad altri obbliga chi ha commesso il fatto a risarcire il danno."
    t2 = "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno."
    sim = calculate_similarity(t1, t2)
    print(f"  Similarity: {sim:.3f}")
