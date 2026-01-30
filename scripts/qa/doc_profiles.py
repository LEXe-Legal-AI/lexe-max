#!/usr/bin/env python3
"""
Document Profiles - Configurazioni per tipo di documento.

Basato su analisi diagnostica di 8 PDF rappresentativi.
Ogni profilo definisce: skip_pages, gate, chunk_caps, patterns.
"""

from dataclasses import dataclass, field
from typing import Literal

ProfileType = Literal[
    "structured_by_title",      # Massimari civili standard
    "structured_parent_child",  # Approfondimenti tematici
    "baseline_toc_filter",      # Massimari penali con TOC pesante
    "legacy_layout",            # Documenti pre-2014
    "list_pure",                # Elenchi puri di massime brevi
    "mixed_hybrid",             # Struttura mista
]


@dataclass
class GateConfig:
    """Configurazione gate per filtrare chunks."""
    min_length: int = 150
    max_length: int = 50000
    max_citation_ratio: float = 0.03
    bad_starts: list[str] = field(default_factory=lambda: [
        ", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO", "pag.", "..."
    ])
    skip_types: list[str] = field(default_factory=lambda: [
        "Header", "Footer", "PageNumber"
    ])


@dataclass
class ChunkConfig:
    """Configurazione chunking."""
    strategy: str = "by_title"  # by_title, by_similarity, pattern_based, parent_child
    max_chars: int = 8000
    min_chars: int = 200
    overlap: int = 200
    # Per parent_child
    parent_max_chars: int = 60000
    child_max_chars: int = 3000
    child_min_chars: int = 500


@dataclass
class SkipConfig:
    """Pagine da saltare."""
    skip_start: int = 0       # Prime N pagine (copertina, front matter)
    skip_end: int = 0         # Ultime N pagine (appendici)
    skip_toc_pages: bool = True
    toc_detection_threshold: float = 0.3  # % pagina con segnali TOC


@dataclass
class DocumentProfile:
    """Profilo completo per un tipo di documento."""
    name: ProfileType
    description: str
    gate: GateConfig
    chunk: ChunkConfig
    skip: SkipConfig
    # Pattern per identificare inizio massima
    massima_anchors: list[str] = field(default_factory=list)
    # Pattern per filtrare rumore
    noise_patterns: list[str] = field(default_factory=list)
    # Metriche soglia per assegnare questo profilo
    thresholds: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# PROFILI PREDEFINITI
# ═══════════════════════════════════════════════════════════════════

PROFILES: dict[ProfileType, DocumentProfile] = {

    # ───────────────────────────────────────────────────────────────
    # STRUCTURED BY_TITLE - Massimari civili standard (2014-2024)
    # Esempi: 2014 Mass civile, 2016 Massimario Civile, 2018 Massimario Civile
    # ───────────────────────────────────────────────────────────────
    "structured_by_title": DocumentProfile(
        name="structured_by_title",
        description="Massimari civili con struttura regolare, by_title funziona bene",
        gate=GateConfig(
            min_length=150,
            max_length=30000,
            max_citation_ratio=0.03,
        ),
        chunk=ChunkConfig(
            strategy="by_title",
            max_chars=8000,
            min_chars=200,
        ),
        skip=SkipConfig(
            skip_start=5,   # Copertina, indice
            skip_end=3,     # Appendici
            skip_toc_pages=True,
        ),
        massima_anchors=[
            r"^Sez\.\s*[IVX\d]+",
            r"^SEZIONE\s+[IVX]+",
            r"^N\.\s*\d+",
            r"^Ordinanza\s+n\.",
            r"^Sentenza\s+n\.",
        ],
        noise_patterns=[
            r"^INDICE\s*(GENERALE|ANALITICO)?",
            r"^\d+\s*\.{3,}\s*\d+$",  # TOC lines: 123 ... 456
            r"^pag\.\s*\d+",
        ],
        thresholds={
            "avg_chars_per_page": (3500, 4500),
            "empty_pages_pct": (0, 5),
            "toc_signal_pages": (30, 80),
            "std_dev": (800, 1300),
        },
    ),

    # ───────────────────────────────────────────────────────────────
    # STRUCTURED PARENT_CHILD - Approfondimenti tematici
    # Esempi: Volume IV_2021 Approfondimenti, Volume III_2017 Approfondimenti
    # ───────────────────────────────────────────────────────────────
    "structured_parent_child": DocumentProfile(
        name="structured_parent_child",
        description="Approfondimenti con sezioni lunghe, richiede parent-child",
        gate=GateConfig(
            min_length=300,  # Più alto, contenuto denso
            max_length=80000,
            max_citation_ratio=0.05,
        ),
        chunk=ChunkConfig(
            strategy="parent_child",
            max_chars=60000,  # Parent grande
            min_chars=500,
            parent_max_chars=60000,
            child_max_chars=3000,
            child_min_chars=500,
        ),
        skip=SkipConfig(
            skip_start=8,   # Front matter più lungo
            skip_end=5,
            skip_toc_pages=True,
        ),
        massima_anchors=[
            r"^CAPITOLO\s+[IVX\d]+",
            r"^SEZIONE\s+[IVX]+",
            r"^PARTE\s+[IVX]+",
            r"^\d+\.\s+[A-Z]",  # 1. TITOLO
        ],
        noise_patterns=[
            r"^INDICE",
            r"^\d+\s*\.{3,}",
        ],
        thresholds={
            "avg_chars_per_page": (3500, 5500),
            "empty_pages_pct": (0, 7),
            "low_text_pct": (0, 3),
        },
    ),

    # ───────────────────────────────────────────────────────────────
    # BASELINE TOC_FILTER - Massimari penali con TOC infiltrato
    # Esempi: Volume I_2017 Massimario Penale, Rassegna Penale 2013
    # ───────────────────────────────────────────────────────────────
    "baseline_toc_filter": DocumentProfile(
        name="baseline_toc_filter",
        description="Documenti con alto rischio TOC infiltration, filtro aggressivo",
        gate=GateConfig(
            min_length=120,  # Più permissivo
            max_length=25000,
            max_citation_ratio=0.04,
            bad_starts=[", del", ", dep.", ", Rv.", "INDICE", "SOMMARIO",
                       "pag.", "...", "Indice delle sentenze", "SENTENZE CITATE"],
        ),
        chunk=ChunkConfig(
            strategy="by_title",
            max_chars=6000,  # Cap più basso per evitare monoliti
            min_chars=150,
        ),
        skip=SkipConfig(
            skip_start=10,  # Più pagine indice
            skip_end=15,    # Appendici, indici sentenze
            skip_toc_pages=True,
            toc_detection_threshold=0.25,  # Più sensibile
        ),
        massima_anchors=[
            r"^Sez\.\s*[IVX\d]+",
            r"^SEZIONE\s+[IVX]+",
            r"^N\.\s*\d+",
        ],
        noise_patterns=[
            r"^INDICE",
            r"^\d+\s*\.{3,}",
            r"^Sentenze citate",
            r"^SENTENZE\s+CITATE",
            r"^Indice\s+delle\s+sentenze",
        ],
        thresholds={
            "toc_signal_pages": (80, 200),  # Alto TOC
            "avg_chars_per_page": (2500, 4000),
        },
    ),

    # ───────────────────────────────────────────────────────────────
    # LEGACY LAYOUT - Documenti pre-2014
    # Esempi: Rassegna penale 2010, Rassegna civile 2010-2013
    # ───────────────────────────────────────────────────────────────
    "legacy_layout": DocumentProfile(
        name="legacy_layout",
        description="Documenti vecchi con layout problematico, cleaning aggressivo",
        gate=GateConfig(
            min_length=100,  # Più basso per non perdere contenuto
            max_length=20000,
            max_citation_ratio=0.05,
        ),
        chunk=ChunkConfig(
            strategy="by_title",
            max_chars=5000,
            min_chars=100,
            overlap=300,  # Più overlap per compensare split imprecisi
        ),
        skip=SkipConfig(
            skip_start=3,
            skip_end=5,
            skip_toc_pages=True,
        ),
        massima_anchors=[
            r"^Sez\.\s*[IVX\d]+",
            r"^N\.\s*\d+",
            r"^La Corte",
            r"^In tema",
        ],
        noise_patterns=[
            r"^INDICE",
            r"^\d+\s*\.{3,}",
            r"^_{3,}",  # Linee underscore
            r"^-{3,}",  # Linee dash
        ],
        thresholds={
            "low_text_pct": (3, 10),  # Più pagine povere
            "avg_chars_per_page": (2000, 3500),
        },
    ),

    # ───────────────────────────────────────────────────────────────
    # LIST PURE - Elenchi puri di massime (raro, ma possibile)
    # ───────────────────────────────────────────────────────────────
    "list_pure": DocumentProfile(
        name="list_pure",
        description="Elenco puro di massime brevi senza commentary",
        gate=GateConfig(
            min_length=80,  # Massime brevi OK
            max_length=5000,
            max_citation_ratio=0.02,
        ),
        chunk=ChunkConfig(
            strategy="pattern_based",  # Usa anchor patterns
            max_chars=3000,
            min_chars=80,
        ),
        skip=SkipConfig(
            skip_start=5,
            skip_end=3,
            skip_toc_pages=True,
        ),
        massima_anchors=[
            r"^Sez\.\s*[IVX\d]+",
            r"^N\.\s*\d+",
            r"^\d+\)\s+",  # 1) 2) etc
        ],
        noise_patterns=[
            r"^INDICE",
        ],
        thresholds={
            "avg_chars_per_page": (2000, 3500),
            "std_dev": (0, 700),  # Bassa varianza = uniforme
        },
    ),

    # ───────────────────────────────────────────────────────────────
    # MIXED HYBRID - Struttura che cambia, richiede adaptive
    # ───────────────────────────────────────────────────────────────
    "mixed_hybrid": DocumentProfile(
        name="mixed_hybrid",
        description="Documento con struttura mista, chunking adattivo",
        gate=GateConfig(
            min_length=120,
            max_length=40000,
            max_citation_ratio=0.04,
        ),
        chunk=ChunkConfig(
            strategy="by_similarity",  # Adattivo semantico
            max_chars=8000,
            min_chars=200,
        ),
        skip=SkipConfig(
            skip_start=5,
            skip_end=5,
            skip_toc_pages=True,
        ),
        massima_anchors=[
            r"^Sez\.\s*[IVX\d]+",
            r"^SEZIONE",
            r"^CAPITOLO",
            r"^N\.\s*\d+",
            r"^La Corte",
        ],
        noise_patterns=[
            r"^INDICE",
            r"^\d+\s*\.{3,}",
        ],
        thresholds={
            "std_dev": (1200, 2000),  # Alta varianza
        },
    ),
}


def get_profile(name: ProfileType) -> DocumentProfile:
    """Ritorna un profilo per nome."""
    return PROFILES[name]


def suggest_profile_from_metrics(
    avg_chars: float,
    std_dev: float,
    empty_pct: float,
    low_text_pct: float,
    toc_pages: int,
    total_pages: int,
    anno: int | None = None,
) -> tuple[ProfileType, float]:
    """
    Suggerisce il profilo migliore basato sulle metriche.

    Returns:
        Tuple di (profile_name, confidence)
    """
    scores: dict[ProfileType, float] = {name: 0.0 for name in PROFILES}

    # Legacy check
    if anno and anno <= 2013:
        scores["legacy_layout"] += 3.0

    # Low text check
    if low_text_pct > 3:
        scores["legacy_layout"] += 2.0

    # TOC heavy check
    toc_ratio = toc_pages / total_pages if total_pages > 0 else 0
    if toc_ratio > 0.2:
        scores["baseline_toc_filter"] += 3.0
    elif toc_ratio > 0.1:
        scores["baseline_toc_filter"] += 1.5

    # Avg chars check
    if avg_chars > 4500:
        scores["structured_parent_child"] += 2.0
    elif avg_chars < 3000:
        if std_dev < 700:
            scores["list_pure"] += 2.0
        else:
            scores["legacy_layout"] += 1.0
    else:
        scores["structured_by_title"] += 2.0

    # Std dev check (varianza)
    if std_dev > 1200:
        scores["mixed_hybrid"] += 2.0
        scores["structured_parent_child"] += 1.0
    elif std_dev < 700:
        scores["list_pure"] += 1.5
        scores["structured_by_title"] += 1.0

    # Empty pages
    if empty_pct > 5:
        scores["legacy_layout"] += 1.0

    # Trova migliore
    best_profile = max(scores, key=scores.get)
    best_score = scores[best_profile]

    # Confidence basata su quanto il best supera gli altri
    total_score = sum(scores.values())
    confidence = best_score / total_score if total_score > 0 else 0.5

    return best_profile, min(confidence, 0.95)


# ═══════════════════════════════════════════════════════════════════
# CONFIG PER I 8 PDF DIAGNOSTICATI
# ═══════════════════════════════════════════════════════════════════

DIAGNOSED_CONFIGS = {
    "Volume I_2017_Massimario_Penale.pdf": {
        "profile": "baseline_toc_filter",
        "skip_start": 15,
        "skip_end": 20,  # Indice sentenze citate
        "gate_min_length": 120,
        "chunk_max": 6000,
        "notes": "129 pagine TOC, rischio infiltration alto, filtro indice e appendici",
    },

    "2014 Mass civile Vol 1 pagg 408.pdf": {
        "profile": "structured_by_title",
        "skip_start": 8,
        "skip_end": 5,
        "gate_min_length": 150,
        "chunk_max": 8000,
        "notes": "Testo uniforme, struttura presente, by_title OK con skip indice",
    },

    "Volume I_2016_Massimario_Civile_1_372.pdf": {
        "profile": "structured_by_title",
        "skip_start": 10,
        "skip_end": 5,
        "gate_min_length": 150,
        "chunk_max": 8000,
        "notes": "Baseline massimario, TOC removal necessario, by_title o anchor split",
    },

    "Volume I_2018_Massimario_Civile_1_358 con Copertina.pdf": {
        "profile": "structured_by_title",
        "skip_start": 12,  # Copertina + front matter
        "skip_end": 5,
        "gate_min_length": 150,
        "chunk_max": 8000,
        "notes": "Copertina presente, skip front matter, detector titoli finti da indice",
    },

    "Rassegna Penale 2013.pdf": {
        "profile": "baseline_toc_filter",  # NOT list_pure!
        "skip_start": 5,
        "skip_end": 5,
        "gate_min_length": 100,  # Meno aggressivo
        "chunk_max": 6000,
        "notes": "Discorsivo, NON lista pura, chunking per sezione, gate meno aggressivo",
    },

    "Rassegna penale 2010.pdf": {
        "profile": "legacy_layout",
        "skip_start": 3,
        "skip_end": 5,
        "gate_min_length": 80,  # Più basso
        "chunk_max": 5000,
        "notes": "Legacy 2010, 4.9% low text, cleaning aggressivo, controlla header/footer",
    },

    "Volume IV_2021_Massimario_Civile_Approfondimenti tematici.pdf": {
        "profile": "structured_parent_child",
        "skip_start": 10,
        "skip_end": 8,
        "gate_min_length": 300,
        "chunk_max": 60000,  # Parent
        "child_max": 3000,
        "notes": "Sezioni lunghe, PARENT-CHILD obbligatorio, cap parent 60k",
    },

    "Volume III_2017_Approfond_Tematici.pdf": {
        "profile": "structured_parent_child",
        "skip_start": 8,
        "skip_end": 5,
        "gate_min_length": 300,
        "chunk_max": 60000,
        "child_max": 3000,
        "notes": "Rischio collasso in 1 unit, split per sezioni, cap rigido + parent-child",
    },
}


def get_diagnosed_config(filename: str) -> dict | None:
    """Ritorna config diagnosticata per un filename."""
    return DIAGNOSED_CONFIGS.get(filename)


if __name__ == "__main__":
    # Test
    print("Profili disponibili:")
    for name, profile in PROFILES.items():
        print(f"  {name}: {profile.description}")

    print("\nConfig diagnosticate:")
    for fname, cfg in DIAGNOSED_CONFIGS.items():
        print(f"  {fname[:40]:40} -> {cfg['profile']}")
