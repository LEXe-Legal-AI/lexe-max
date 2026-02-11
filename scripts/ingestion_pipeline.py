#!/usr/bin/env python
"""
Ingestion Pipeline - 4 Source Architecture

Architettura a 4 fonti per massima qualità e resilienza:

1. PDF ALTALEX (primaria) - 69 documenti estratti
2. STUDIO CATALDI (offline) - 24 codici
3. BROCARDI (offline) - CC, CP, CPC, CPP, etc.
4. BROCARDI/NORMATTIVA (online) - fallback su incertezza

Flusso:
┌─────────────────────────────────────────────────────────────────┐
│                    PDF ALTALEX (primaria)                        │
│                    text_primary, rubrica                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONFRONTO OFFLINE (parallelo)                       │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │ STUDIO CATALDI  │         │ BROCARDI LOCAL  │               │
│   │   (24 codici)   │         │ (CC,CP,CPC,CPP) │               │
│   └────────┬────────┘         └────────┬────────┘               │
│            └──────────┬────────────────┘                        │
│                       ▼                                          │
│              SIMILARITY CHECK                                    │
│         ┌─────────────────────────┐                             │
│         │ sim >= 0.70 → CONFIRM   │                             │
│         │ 0.40 <= sim < 0.70 →    │                             │
│         │    PARTIAL (usa migliore)│                             │
│         │ sim < 0.40 → CONFLICT   │─────┐                       │
│         └─────────────────────────┘     │                       │
└─────────────────────────────────────────│───────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FALLBACK ONLINE                               │
│   ┌─────────────────┐         ┌─────────────────┐               │
│   │ BROCARDI ONLINE │         │   NORMATTIVA    │               │
│   │  (rate limited) │         │  (API ufficiale)│               │
│   └─────────────────┘         └─────────────────┘               │
│         Solo per: CONFLICT, EMPTY, watchlist critica            │
└─────────────────────────────────────────────────────────────────┘

Usage:
    uv run python scripts/ingestion_pipeline.py --doc CC --dry-run
    uv run python scripts/ingestion_pipeline.py --all --batch-size 5
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
ALTALEX_ROOT = Path("C:/PROJECTS/lexe-genesis/altalex pdf")
STUDIO_CATALDI_ROOT = Path("C:/Mie pagine Web/giur e cod/www.studiocataldi.it/normativa")
BROCARDI_ROOT = Path("C:/Mie pagine Web/broc-civ/www.brocardi.it")
NEXUSCODE_ROOT = Path("C:/PROJECTS/lexe-genesis/nexuscode")
OUTPUT_ROOT = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/ingested")

# Mapping codici
BROCARDI_LOCAL_CODES = {
    'CC': 'codice-civile',
    # Altri codici quando disponibili
}

STUDIO_CATALDI_CODES = {
    'TUB': 'testo-unico-bancario',
    'TUF': 'testo-unico-intermediazione-finanziaria',
    'TUIR': 'testo-unico-imposte-sui-redditi',
    'TUE': 'testo-unico-edilizia',
    'TUEL': 'testo-unico-enti-locali',
    'TUI': 'testo-unico-immigrazione',
    'TUSL': 'testo-unico-sicurezza-sul-lavoro',
    'TUSG': 'testo-unico-spese-giustizia',
    'DIVA': 'testo-unico-iva',
    'CAPP': 'codice-degli-appalti',
    'CBCP': 'codice-dei-beni-culturali',
    'CCONS': 'codice-del-consumo',
    'CAMB': 'codice-dell-ambiente',
    'CPRIV': 'codice-della-privacy',
    'CPI': 'codice-della-proprieta-industriale',
    'CDS': 'codicedellastrada',
    'CAP': 'codice-delle-assicurazioni-private',
    'CNAV': 'codice-della-navigazione',
    'SL': 'statuto-dei-lavoratori',
    'OP': 'ordinamento-penitenziario',
    'LF': 'legge-fallimentare',
    'LDIP': 'diritto-internazionale-privato',
    'LDIV': 'legge-divorzio',
    'LLOC': 'legge-locazioni-abitative',
}

# Soglie
SIMILARITY_CONFIRM = 0.70
SIMILARITY_PARTIAL = 0.40

# Rate limits (online fallback)
BROCARDI_RATE_LIMIT = 3.0  # secondi tra richieste
NORMATTIVA_RATE_LIMIT = 1.0


# ==============================================================================
# ENUMS
# ==============================================================================

class SourceType(Enum):
    """Tipo di fonte."""
    PDF_ALTALEX = "pdf_altalex"
    STUDIO_CATALDI = "studio_cataldi"
    BROCARDI_LOCAL = "brocardi_local"
    BROCARDI_ONLINE = "brocardi_online"
    NORMATTIVA = "normattiva"


class ComparisonResult(Enum):
    """Risultato del confronto."""
    CONFIRM = "confirm"           # sim >= 0.70
    PARTIAL = "partial"           # 0.40 <= sim < 0.70
    CONFLICT = "conflict"         # sim < 0.40
    NOT_FOUND = "not_found"       # Fonte non disponibile
    ERROR = "error"               # Errore di parsing


class IngestionStatus(Enum):
    """Stato finale dell'articolo."""
    READY = "ready"               # Pronto per embedding
    NEEDS_REVIEW = "needs_review" # Conflitto, richiede review
    SKIPPED = "skipped"           # Saltato (invalid)
    ERROR = "error"               # Errore


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SourceData:
    """Dati da una fonte."""
    source: SourceType
    text: Optional[str] = None
    rubrica: Optional[str] = None
    available: bool = False
    error: str = ""


@dataclass
class ComparisonData:
    """Risultato del confronto tra fonti."""
    primary_vs_cataldi: Optional[ComparisonResult] = None
    primary_vs_brocardi: Optional[ComparisonResult] = None
    cataldi_vs_brocardi: Optional[ComparisonResult] = None
    similarity_cataldi: float = 0.0
    similarity_brocardi: float = 0.0
    best_match: Optional[SourceType] = None


@dataclass
class IngestedArticle:
    """Articolo pronto per ingestion."""
    # Identificazione
    article_key: str           # es. "CC:2043"
    doc_id: str                # es. "CC"
    base_num: int
    suffix: Optional[str]

    # Testi per fonte
    text_primary: str          # Da PDF
    text_cataldi: Optional[str] = None
    text_brocardi: Optional[str] = None

    # Testo finale
    text_final: str = ""       # Il testo scelto per embedding
    final_source: SourceType = SourceType.PDF_ALTALEX

    # Rubriche
    rubrica_primary: Optional[str] = None
    rubrica_cataldi: Optional[str] = None
    rubrica_brocardi: Optional[str] = None
    rubrica_final: Optional[str] = None

    # Confronto
    comparison: Optional[ComparisonData] = None

    # Status
    status: IngestionStatus = IngestionStatus.READY
    quality_class: str = "VALID"
    warnings: list = field(default_factory=list)
    used_online_fallback: bool = False

    # Metadata
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['final_source'] = self.final_source.value
        d['status'] = self.status.value
        if self.comparison:
            d['comparison'] = {
                'primary_vs_cataldi': self.comparison.primary_vs_cataldi.value if self.comparison.primary_vs_cataldi else None,
                'primary_vs_brocardi': self.comparison.primary_vs_brocardi.value if self.comparison.primary_vs_brocardi else None,
                'similarity_cataldi': self.comparison.similarity_cataldi,
                'similarity_brocardi': self.comparison.similarity_brocardi,
                'best_match': self.comparison.best_match.value if self.comparison.best_match else None,
            }
        return d


@dataclass
class IngestionStats:
    """Statistiche di ingestion per documento."""
    doc_id: str
    total: int = 0
    ready: int = 0
    needs_review: int = 0
    skipped: int = 0
    error: int = 0

    # Fonti usate per text_final
    used_primary: int = 0
    used_cataldi: int = 0
    used_brocardi: int = 0
    used_online: int = 0

    # Fonti disponibili (per confronto)
    found_cataldi: int = 0
    found_brocardi: int = 0

    # Confronti
    confirmed: int = 0          # sim >= 0.70
    partial: int = 0            # 0.40 <= sim < 0.70
    conflicts: int = 0          # sim < 0.40

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================================================================
# SIMILARITY (placeholder - usa difflib per ora)
# ==============================================================================

import difflib
import re

def calculate_similarity(text1: str, text2: str) -> float:
    """Calcola similarity tra due testi."""
    if not text1 or not text2:
        return 0.0

    # Normalizza
    t1 = re.sub(r'\s+', ' ', text1.lower().strip())
    t2 = re.sub(r'\s+', ' ', text2.lower().strip())

    return difflib.SequenceMatcher(None, t1, t2).ratio()


def classify_similarity(sim: float) -> ComparisonResult:
    """Classifica similarity in CONFIRM/PARTIAL/CONFLICT."""
    if sim >= SIMILARITY_CONFIRM:
        return ComparisonResult.CONFIRM
    elif sim >= SIMILARITY_PARTIAL:
        return ComparisonResult.PARTIAL
    else:
        return ComparisonResult.CONFLICT


# ==============================================================================
# SOURCE LOADERS
# ==============================================================================

# Lazy load parsers
_cataldi_parser = None
_brocardi_parser = None


def get_cataldi_parser():
    """Lazy load Studio Cataldi parser."""
    global _cataldi_parser
    if _cataldi_parser is None:
        try:
            from studio_cataldi_parser import StudioCataldiParser
            _cataldi_parser = StudioCataldiParser(STUDIO_CATALDI_ROOT)
        except ImportError:
            logger.warning("studio_cataldi_parser not found")
    return _cataldi_parser


def get_brocardi_parser():
    """Lazy load Brocardi parser."""
    global _brocardi_parser
    if _brocardi_parser is None:
        try:
            from brocardi_parser import BrocardiParser
            _brocardi_parser = BrocardiParser(BROCARDI_ROOT)
        except ImportError:
            logger.warning("brocardi_parser not found")
    return _brocardi_parser


def load_altalex_article(doc_id: str, base_num: int, suffix: Optional[str] = None) -> SourceData:
    """Carica articolo da JSON Altalex estratto."""
    # Altalex è già caricato come primary nel flusso principale
    return SourceData(source=SourceType.PDF_ALTALEX, available=False)


def load_cataldi_article(doc_id: str, base_num: int, suffix: Optional[str] = None) -> SourceData:
    """Carica articolo da Studio Cataldi locale."""
    if doc_id not in STUDIO_CATALDI_CODES:
        return SourceData(source=SourceType.STUDIO_CATALDI, available=False)

    parser = get_cataldi_parser()
    if not parser:
        return SourceData(source=SourceType.STUDIO_CATALDI, available=False, error="parser_not_found")

    try:
        article = parser.get_article(doc_id, base_num, suffix)
        if article:
            return SourceData(
                source=SourceType.STUDIO_CATALDI,
                text=article.testo,
                rubrica=article.rubrica,
                available=True,
            )
        return SourceData(source=SourceType.STUDIO_CATALDI, available=False)
    except Exception as e:
        return SourceData(source=SourceType.STUDIO_CATALDI, available=False, error=str(e))


def load_brocardi_local_article(doc_id: str, base_num: int, suffix: Optional[str] = None) -> SourceData:
    """Carica articolo da Brocardi locale."""
    if doc_id not in BROCARDI_LOCAL_CODES:
        return SourceData(source=SourceType.BROCARDI_LOCAL, available=False)

    parser = get_brocardi_parser()
    if not parser:
        return SourceData(source=SourceType.BROCARDI_LOCAL, available=False, error="parser_not_found")

    try:
        article = parser.get_article(doc_id, base_num, suffix)
        if article:
            return SourceData(
                source=SourceType.BROCARDI_LOCAL,
                text=article.testo,
                rubrica=article.rubrica,
                available=True,
            )
        return SourceData(source=SourceType.BROCARDI_LOCAL, available=False)
    except Exception as e:
        return SourceData(source=SourceType.BROCARDI_LOCAL, available=False, error=str(e))


def load_brocardi_online_article(doc_id: str, base_num: int, suffix: Optional[str] = None) -> SourceData:
    """Carica articolo da Brocardi online (rate limited)."""
    # TODO: Implementare client HTTP con rate limiting
    time.sleep(BROCARDI_RATE_LIMIT)
    return SourceData(source=SourceType.BROCARDI_ONLINE, available=False)


def load_normattiva_article(doc_id: str, base_num: int, suffix: Optional[str] = None) -> SourceData:
    """Carica articolo da Normattiva API."""
    # TODO: Implementare client Normattiva
    time.sleep(NORMATTIVA_RATE_LIMIT)
    return SourceData(source=SourceType.NORMATTIVA, available=False)


# ==============================================================================
# INGESTION LOGIC
# ==============================================================================

def ingest_article(
    doc_id: str,
    base_num: int,
    suffix: Optional[str],
    text_primary: str,
    rubrica_primary: Optional[str],
    quality_class: str,
    use_online_fallback: bool = True,
) -> IngestedArticle:
    """
    Processa un singolo articolo attraverso la pipeline.

    Args:
        doc_id: Codice documento (CC, TUB, etc.)
        base_num: Numero articolo
        suffix: Suffisso opzionale (bis, ter, etc.)
        text_primary: Testo da PDF
        rubrica_primary: Rubrica da PDF
        quality_class: VALID/WEAK/EMPTY/INVALID
        use_online_fallback: Se usare fallback online

    Returns:
        IngestedArticle con tutti i dati
    """
    article_key = f"{doc_id}:{base_num}" + (f"-{suffix}" if suffix else "")

    article = IngestedArticle(
        article_key=article_key,
        doc_id=doc_id,
        base_num=base_num,
        suffix=suffix,
        text_primary=text_primary,
        rubrica_primary=rubrica_primary,
        quality_class=quality_class,
    )

    # Se INVALID, salta
    if quality_class == "INVALID":
        article.status = IngestionStatus.SKIPPED
        article.warnings.append("quality_invalid")
        return article

    # Step 1: Carica fonti offline
    cataldi = load_cataldi_article(doc_id, base_num, suffix)
    brocardi = load_brocardi_local_article(doc_id, base_num, suffix)

    if cataldi.available:
        article.text_cataldi = cataldi.text
        article.rubrica_cataldi = cataldi.rubrica

    if brocardi.available:
        article.text_brocardi = brocardi.text
        article.rubrica_brocardi = brocardi.rubrica

    # Step 2: Calcola similarity
    comparison = ComparisonData()

    if cataldi.available and cataldi.text:
        sim = calculate_similarity(text_primary, cataldi.text)
        comparison.similarity_cataldi = sim
        comparison.primary_vs_cataldi = classify_similarity(sim)

    if brocardi.available and brocardi.text:
        sim = calculate_similarity(text_primary, brocardi.text)
        comparison.similarity_brocardi = sim
        comparison.primary_vs_brocardi = classify_similarity(sim)

    article.comparison = comparison

    # Step 3: Decidi testo finale
    # Logica: preferisci primary se VALID, altrimenti usa la fonte con similarity migliore
    if quality_class == "VALID":
        article.text_final = text_primary
        article.final_source = SourceType.PDF_ALTALEX
        article.status = IngestionStatus.READY

    elif quality_class in ("WEAK", "EMPTY"):
        # Cerca la fonte migliore
        best_sim = 0.0
        best_source = SourceType.PDF_ALTALEX
        best_text = text_primary

        if comparison.similarity_cataldi > best_sim and cataldi.text:
            best_sim = comparison.similarity_cataldi
            best_source = SourceType.STUDIO_CATALDI
            best_text = cataldi.text

        if comparison.similarity_brocardi > best_sim and brocardi.text:
            best_sim = comparison.similarity_brocardi
            best_source = SourceType.BROCARDI_LOCAL
            best_text = brocardi.text

        # Se sim < 0.40 e use_online_fallback, prova online
        if best_sim < SIMILARITY_PARTIAL and use_online_fallback:
            online = load_brocardi_online_article(doc_id, base_num, suffix)
            if online.available and online.text:
                sim_online = calculate_similarity(text_primary, online.text)
                if sim_online > best_sim:
                    best_sim = sim_online
                    best_source = SourceType.BROCARDI_ONLINE
                    best_text = online.text
                    article.used_online_fallback = True

        article.text_final = best_text
        article.final_source = best_source
        comparison.best_match = best_source

        if best_sim >= SIMILARITY_PARTIAL:
            article.status = IngestionStatus.READY
        else:
            article.status = IngestionStatus.NEEDS_REVIEW
            article.warnings.append(f"low_similarity_{best_sim:.2f}")

    # Step 4: Scegli rubrica migliore
    # Preferenza: brocardi > cataldi > primary (Brocardi ha rubriche più pulite)
    if article.rubrica_brocardi:
        article.rubrica_final = article.rubrica_brocardi
    elif article.rubrica_cataldi:
        article.rubrica_final = article.rubrica_cataldi
    else:
        article.rubrica_final = rubrica_primary

    # Provenance
    article.provenance = {
        "primary_source": "pdf_altalex",
        "cataldi_available": cataldi.available,
        "brocardi_available": brocardi.available,
        "used_online": article.used_online_fallback,
        "final_source": article.final_source.value,
    }

    return article


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def ingest_document(
    doc_id: str,
    articles: list[dict],
    use_online_fallback: bool = True,
    dry_run: bool = False,
) -> tuple[list[IngestedArticle], IngestionStats]:
    """
    Processa tutti gli articoli di un documento.

    Args:
        doc_id: Codice documento
        articles: Lista di articoli dal JSON estratto
        use_online_fallback: Se usare fallback online
        dry_run: Se True, non salva nulla

    Returns:
        (lista articoli processati, statistiche)
    """
    stats = IngestionStats(doc_id=doc_id, total=len(articles))
    results = []

    for a in articles:
        quality = a.get('quality', {})
        quality_class = quality.get('quality', 'VALID')

        ingested = ingest_article(
            doc_id=doc_id,
            base_num=a.get('articolo_num_norm', 0),
            suffix=a.get('articolo_suffix'),
            text_primary=a.get('testo', ''),
            rubrica_primary=a.get('rubrica'),
            quality_class=quality_class,
            use_online_fallback=use_online_fallback,
        )

        results.append(ingested)

        # Update stats
        if ingested.status == IngestionStatus.READY:
            stats.ready += 1
        elif ingested.status == IngestionStatus.NEEDS_REVIEW:
            stats.needs_review += 1
        elif ingested.status == IngestionStatus.SKIPPED:
            stats.skipped += 1
        else:
            stats.error += 1

        # Source stats - final source
        if ingested.final_source == SourceType.PDF_ALTALEX:
            stats.used_primary += 1
        elif ingested.final_source == SourceType.STUDIO_CATALDI:
            stats.used_cataldi += 1
        elif ingested.final_source in (SourceType.BROCARDI_LOCAL, SourceType.BROCARDI_ONLINE):
            stats.used_brocardi += 1

        if ingested.used_online_fallback:
            stats.used_online += 1

        # Source stats - found sources
        if ingested.text_cataldi:
            stats.found_cataldi += 1
        if ingested.text_brocardi:
            stats.found_brocardi += 1

        # Comparison stats - considera sia Cataldi che Brocardi
        if ingested.comparison:
            # Prendi il miglior risultato tra le due fonti
            best_result = None
            if ingested.comparison.primary_vs_brocardi:
                best_result = ingested.comparison.primary_vs_brocardi
            if ingested.comparison.primary_vs_cataldi:
                # Cataldi ha priorità se entrambi disponibili
                best_result = ingested.comparison.primary_vs_cataldi

            if best_result == ComparisonResult.CONFIRM:
                stats.confirmed += 1
            elif best_result == ComparisonResult.PARTIAL:
                stats.partial += 1
            elif best_result == ComparisonResult.CONFLICT:
                stats.conflicts += 1

    if not dry_run:
        # Salva output
        output_dir = OUTPUT_ROOT / doc_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "ingested.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "doc_id": doc_id,
                "stats": stats.to_dict(),
                "articles": [a.to_dict() for a in results],
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved: {output_file}")

    return results, stats


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingestion Pipeline")
    parser.add_argument("--doc", help="Codice documento (es. CC, TUB)")
    parser.add_argument("--all", action="store_true", help="Processa tutti i documenti")
    parser.add_argument("--no-online", action="store_true", help="Disabilita fallback online")
    parser.add_argument("--dry-run", action="store_true", help="Non salva output")

    args = parser.parse_args()

    print("=" * 60)
    print("INGESTION PIPELINE - 4 SOURCE ARCHITECTURE")
    print("=" * 60)
    print(f"Altalex root: {ALTALEX_ROOT}")
    print(f"Studio Cataldi: {STUDIO_CATALDI_ROOT}")
    print(f"Brocardi local: {BROCARDI_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print()

    # Trova JSON estratti
    json_files = list(ALTALEX_ROOT.rglob("*.llm_extracted.json"))
    json_files = [f for f in json_files if '.old.' not in f.name]
    print(f"JSON estratti disponibili: {len(json_files)}")

    if args.doc:
        # Processa singolo documento - cerca per codice nel JSON
        target_code = args.doc.upper()
        json_file = None
        data = None

        for jf in json_files:
            try:
                with open(jf, encoding='utf-8') as f:
                    jdata = json.load(f)
                if jdata.get('codice', '').upper() == target_code:
                    json_file = jf
                    data = jdata
                    break
            except Exception as e:
                logger.warning(f"Error reading {jf}: {e}")
                continue

        if not json_file:
            logger.error(f"JSON non trovato per codice {target_code}")
            return

        doc_id = data.get('codice', args.doc.upper())
        articles = data.get('articles', [])

        results, stats = ingest_document(
            doc_id=doc_id,
            articles=articles,
            use_online_fallback=not args.no_online,
            dry_run=args.dry_run,
        )

        print(f"\n=== {doc_id} ===")
        print(f"Totale: {stats.total}")
        print(f"Ready: {stats.ready}")
        print(f"Needs review: {stats.needs_review}")
        print(f"Skipped: {stats.skipped}")
        print(f"\nFonti trovate per confronto:")
        print(f"  Studio Cataldi: {stats.found_cataldi}")
        print(f"  Brocardi local: {stats.found_brocardi}")
        print(f"\nFonti usate per testo finale:")
        print(f"  Primary (PDF): {stats.used_primary}")
        print(f"  Studio Cataldi: {stats.used_cataldi}")
        print(f"  Brocardi: {stats.used_brocardi}")
        print(f"  Online fallback: {stats.used_online}")
        print(f"\nRisultati confronto:")
        print(f"  Confirmed (>=0.70): {stats.confirmed}")
        print(f"  Partial (0.40-0.70): {stats.partial}")
        print(f"  Conflicts (<0.40): {stats.conflicts}")

    elif args.all:
        # Processa tutti i documenti
        all_stats = []
        total_ready = 0
        total_review = 0
        total_skipped = 0
        total_articles = 0

        print(f"\n{'='*60}")
        print(f"PROCESSING ALL {len(json_files)} DOCUMENTS")
        print(f"{'='*60}\n")

        for jf in json_files:
            try:
                with open(jf, encoding='utf-8') as f:
                    data = json.load(f)

                doc_id = data.get('codice', jf.stem.split('.')[0].upper())
                articles = data.get('articles', [])

                if not articles:
                    logger.warning(f"No articles in {jf.name}, skipping")
                    continue

                print(f"\n--- Processing {doc_id} ({len(articles)} articles) ---")

                results, stats = ingest_document(
                    doc_id=doc_id,
                    articles=articles,
                    use_online_fallback=not args.no_online,
                    dry_run=args.dry_run,
                )

                all_stats.append(stats)
                total_ready += stats.ready
                total_review += stats.needs_review
                total_skipped += stats.skipped
                total_articles += stats.total

                print(f"  Ready: {stats.ready} | Review: {stats.needs_review} | Skipped: {stats.skipped}")
                print(f"  Sources found - Cataldi: {stats.found_cataldi} | Brocardi: {stats.found_brocardi}")

            except Exception as e:
                logger.error(f"Error processing {jf.name}: {e}")
                continue

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - ALL DOCUMENTS")
        print(f"{'='*60}")
        print(f"Documents processed: {len(all_stats)}")
        print(f"Total articles: {total_articles}")
        print(f"Total ready: {total_ready}")
        print(f"Total needs review: {total_review}")
        print(f"Total skipped: {total_skipped}")

        if not args.dry_run:
            # Save global summary
            summary_file = OUTPUT_ROOT / "ingestion_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents_processed": len(all_stats),
                    "total_articles": total_articles,
                    "total_ready": total_ready,
                    "total_needs_review": total_review,
                    "total_skipped": total_skipped,
                    "per_document": [s.to_dict() for s in all_stats],
                }, f, ensure_ascii=False, indent=2)
            print(f"\nSummary saved: {summary_file}")

    else:
        print("\nUsa --doc CODICE per processare un documento")
        print("Oppure --all per processare tutti")


if __name__ == "__main__":
    main()
