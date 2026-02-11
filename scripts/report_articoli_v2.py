#!/usr/bin/env python
"""
Report Articoli Estratti v2.0

Analisi e validazione delle estrazioni JSON con:
- Rilevamento falsi positivi migliorato
- Validazione coverage
- Cross-reference con metadata NexusCode
- Output CSV + JSON dettagliato

Migliorie rispetto a v1:
1. Integrazione con metadata.json di NexusCode per validazione articoli attesi
2. Euristica falsi positivi più robusta (posizione + contenuto)
3. Rilevamento articoli fuori sequenza
4. Report anomalie con severity levels
5. Output JSON oltre a CSV per analisi programmatica
6. Multiprocessing per elaborazione parallela
7. Logging strutturato

Usage:
    uv run --no-sync python scripts/report_articoli_v2.py [--parallel] [--output-dir DIR]
"""

import csv
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# ==============================================================================
# CONFIG
# ==============================================================================

SUFFIX_RE = re.compile(r"^(?P<num>\d+)(?:[-\s]?(?P<suf>[a-z]+))?$", re.IGNORECASE)

# Suffissi latini validi (per validazione)
VALID_SUFFIXES = {
    "bis", "ter", "quater", "quinquies", "sexies", "septies", "octies",
    "novies", "nonies", "decies", "undecies", "duodecies", "terdecies",
    "quaterdecies", "quinquiesdecies", "sexiesdecies", "septiesdecies",
}

# Pattern che indicano falsi positivi nel testo dell'articolo
# NOTA: Rimosso "^\d+\)\s+" perché articoli legittimi iniziano con elenchi numerati
# es. "1) se l'enfiteuta deteriora..." è un articolo valido
FALSE_POSITIVE_PATTERNS = [
    r"^-\s*[a-z]\)",                    # "- a)", "- b)" - liste modifiche in note
    r"^all[''']articolo\s+\d+",          # "all'articolo 70" - riferimento
    r"^decreto\s+legislativo",           # "decreto legislativo..." - citazione
    r"^d\.?lgs\.?\s+\d+",               # "D.Lgs. 14" - citazione
    r"^legge\s+\d+",                    # "legge 241" - citazione
    r"^comma\s+\d+",                    # "comma 1" - riferimento interno
    r"^[a-z]\)\s+al\s",                 # "a) al comma..." - modifica, non "a) testo articolo"
    r"^luglio\s+\d{4}",                 # "luglio 1989" - data
    r"^gennaio|febbraio|marzo|aprile|maggio|giugno|agosto|settembre|ottobre|novembre|dicembre",
]
FALSE_POSITIVE_RE = re.compile("|".join(FALSE_POSITIVE_PATTERNS), re.IGNORECASE)

# Soglia posizione per articoli "in coda" (potenziali falsi positivi)
TAIL_POSITION_THRESHOLD = 0.9  # ultimo 10% del documento

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class ArticleAnalysis:
    """Analisi di un singolo articolo."""
    articolo_num: str
    base_num: int
    suffix: Optional[str]
    position: int
    text_length: int
    is_suspect: bool = False
    suspect_reason: str = ""


@dataclass
class Anomaly:
    """Anomalia rilevata."""
    type: str
    severity: str  # "warning", "error"
    description: str
    articles: list = field(default_factory=list)


@dataclass
class DocumentReport:
    """Report completo per un documento."""
    json_path: str
    pdf_name: str
    codice: str = ""

    # Range
    articolo_min: int = 0
    articolo_max: int = 0

    # Conteggi
    articoli_totali: int = 0
    articoli_base_unici: int = 0
    articoli_con_suffisso: int = 0

    # Coverage
    coverage_pct: float = 0.0
    buchi_count: int = 0
    buchi_preview: str = ""

    # Suffissi
    suffissi_trovati: str = ""
    suffissi_invalidi: str = ""

    # Falsi positivi
    sospetti_count: int = 0
    sospetti_preview: str = ""

    # Fuori sequenza
    fuori_sequenza_count: int = 0

    # Anomalie
    anomalie_count: int = 0
    anomalie: list = field(default_factory=list)

    # Validazione vs metadata
    articoli_attesi: int = 0
    delta_attesi: int = 0


# ==============================================================================
# PARSING
# ==============================================================================

def parse_article_id(articolo_num: str) -> tuple[Optional[int], Optional[str]]:
    """Parse numero articolo in (base, suffix)."""
    if not articolo_num:
        return None, None

    s = str(articolo_num).strip()
    m = SUFFIX_RE.match(s)
    if not m:
        return None, None

    base = int(m.group("num"))
    suffix = (m.group("suf") or "").lower() or None

    return base, suffix


def is_valid_suffix(suffix: str) -> bool:
    """Verifica se il suffisso è un ordinale latino valido."""
    if not suffix:
        return True
    return suffix.lower() in VALID_SUFFIXES


def is_false_positive_text(text: str) -> tuple[bool, str]:
    """
    Verifica se il testo dell'articolo indica un falso positivo.

    Returns:
        (is_false_positive, reason)
    """
    if not text:
        return True, "empty_text"

    t = text.strip()

    # Testo troppo corto (< 20 chars)
    if len(t) < 20:
        return True, "too_short"

    # Match pattern falsi positivi (riferimenti, citazioni, date)
    if FALSE_POSITIVE_RE.match(t):
        return True, "pattern_match"

    # Inizia con minuscola E testo breve (< 100 chars) → probabilmente frammento
    # Ma se il testo è lungo, potrebbe essere un articolo che continua da comma precedente
    if t and t[0].islower() and len(t) < 100:
        return True, "lowercase_start"

    return False, ""


# ==============================================================================
# ANALYSIS
# ==============================================================================

def load_articles(json_path: Path) -> tuple[list, dict]:
    """Carica articoli e metadata dal JSON."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles") or data.get("articoli") or []
    if not isinstance(articles, list):
        raise ValueError(f"Formato inatteso in {json_path}")

    metadata = {k: v for k, v in data.items() if k not in ("articles", "articoli")}

    return articles, metadata


def load_nexuscode_metadata(codice: str) -> Optional[dict]:
    """Carica metadata da NexusCode per validazione."""
    index_path = Path("C:/PROJECTS/lexe-genesis/nexuscode/index.json")
    if not index_path.exists():
        return None

    try:
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)

        codice_to_path = index.get("codice_to_path", {})
        if codice.upper() not in codice_to_path:
            return None

        doc_path = Path("C:/PROJECTS/lexe-genesis/nexuscode") / codice_to_path[codice.upper()]
        metadata_path = doc_path / "metadata.json"

        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Errore caricamento metadata NexusCode per {codice}: {e}")

    return None


def analyze_document(json_path: Path) -> DocumentReport:
    """Analizza un documento JSON estratto."""
    articles, metadata = load_articles(json_path)

    report = DocumentReport(
        json_path=str(json_path),
        pdf_name=json_path.name.replace(".llm_extracted.json", ".pdf"),
        codice=metadata.get("codice", "")
    )

    if not articles:
        report.anomalie.append(Anomaly(
            type="no_articles",
            severity="error",
            description="Nessun articolo estratto"
        ))
        return report

    # Analisi articoli
    base_nums = set()
    full_ids = set()
    suffixes_found = set()
    invalid_suffixes = set()
    suspect_articles = []
    positions = []

    # Trova posizione max per calcolo % posizione
    max_position = max((a.get("position", 0) for a in articles), default=1) or 1

    for a in articles:
        art_id = a.get("articolo_num") or a.get("articolo") or a.get("id")
        text = a.get("testo") or a.get("text") or ""
        position = a.get("position", 0)

        if not art_id:
            continue

        base, suffix = parse_article_id(str(art_id))
        if base is None:
            continue

        base_nums.add(base)
        positions.append((base, position))

        if suffix:
            suffixes_found.add(suffix)
            full_ids.add(f"{base}-{suffix}")

            if not is_valid_suffix(suffix):
                invalid_suffixes.add(suffix)
        else:
            full_ids.add(str(base))

        # Check falso positivo
        is_fp, reason = is_false_positive_text(text)

        # Check posizione anomala (articolo appare in coda ma numero basso)
        # NOTA: Escludi articoli con suffisso (bis, ter, etc.) perché sono spesso
        # nelle Disposizioni Transitorie alla fine del documento ma legittimi
        position_pct = position / max_position if max_position > 0 else 0
        is_tail = position_pct > TAIL_POSITION_THRESHOLD
        is_out_of_order = (
            is_tail and
            base < (max(base_nums) * 0.5) and
            suffix is None  # Solo articoli senza suffisso
        ) if base_nums else False

        if is_fp or is_out_of_order:
            suspect_articles.append({
                "articolo": str(art_id),
                "reason": reason if is_fp else "out_of_order",
                "position_pct": round(position_pct, 2)
            })

    # Calcola range e coverage
    if base_nums:
        report.articolo_min = min(base_nums)
        report.articolo_max = max(base_nums)

        expected_range = set(range(report.articolo_min, report.articolo_max + 1))
        gaps = sorted(expected_range - base_nums)

        report.buchi_count = len(gaps)
        report.buchi_preview = ",".join(map(str, gaps[:12]))
        report.coverage_pct = round((len(base_nums) / len(expected_range)) * 100, 1) if expected_range else 0

    # Conteggi
    report.articoli_totali = len(full_ids)
    report.articoli_base_unici = len(base_nums)
    report.articoli_con_suffisso = sum(1 for fid in full_ids if "-" in fid)

    # Suffissi
    report.suffissi_trovati = ",".join(sorted(suffixes_found))
    report.suffissi_invalidi = ",".join(sorted(invalid_suffixes))

    # Sospetti
    report.sospetti_count = len(suspect_articles)
    report.sospetti_preview = ",".join(s["articolo"] for s in suspect_articles[:5])

    # Fuori sequenza
    report.fuori_sequenza_count = sum(1 for s in suspect_articles if s["reason"] == "out_of_order")

    # Validazione vs NexusCode
    if report.codice:
        nc_meta = load_nexuscode_metadata(report.codice)
        if nc_meta:
            struttura = nc_meta.get("struttura", {})
            report.articoli_attesi = struttura.get("articoli_base_attesi", 0)
            report.delta_attesi = report.articoli_base_unici - report.articoli_attesi

    # Anomalie
    if report.coverage_pct < 95:
        report.anomalie.append(Anomaly(
            type="low_coverage",
            severity="warning",
            description=f"Coverage bassa: {report.coverage_pct}%"
        ))

    if invalid_suffixes:
        report.anomalie.append(Anomaly(
            type="invalid_suffix",
            severity="warning",
            description=f"Suffissi non standard: {','.join(invalid_suffixes)}"
        ))

    if report.sospetti_count > 0:
        report.anomalie.append(Anomaly(
            type="suspected_false_positives",
            severity="warning" if report.sospetti_count < 5 else "error",
            description=f"{report.sospetti_count} articoli sospetti",
            articles=[s["articolo"] for s in suspect_articles]
        ))

    if report.delta_attesi != 0 and report.articoli_attesi > 0:
        severity = "warning" if abs(report.delta_attesi) < 10 else "error"
        report.anomalie.append(Anomaly(
            type="count_mismatch",
            severity=severity,
            description=f"Delta vs attesi: {report.delta_attesi:+d} (attesi: {report.articoli_attesi})"
        ))

    report.anomalie_count = len(report.anomalie)

    return report


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Report Articoli Estratti v2")
    parser.add_argument("--root", default=r"C:\PROJECTS\lexe-genesis\altalex pdf",
                       help="Directory root con i JSON estratti")
    parser.add_argument("--output-dir", default=None,
                       help="Directory output (default: stesso di root)")
    parser.add_argument("--parallel", action="store_true",
                       help="Elaborazione parallela")

    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root

    json_files = list(root.rglob("*.llm_extracted.json"))
    logger.info(f"Trovati {len(json_files)} file JSON")

    if not json_files:
        logger.warning("Nessun file trovato")
        return

    # Analisi
    reports = []

    if args.parallel and len(json_files) > 4:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(analyze_document, p): p for p in json_files}
            for future in as_completed(futures):
                try:
                    reports.append(future.result())
                except Exception as e:
                    logger.error(f"Errore analisi {futures[future]}: {e}")
    else:
        for json_path in json_files:
            try:
                reports.append(analyze_document(json_path))
            except Exception as e:
                logger.error(f"Errore analisi {json_path}: {e}")

    # Output CSV
    csv_path = output_dir / "lexe_report_articoli_v2.csv"
    fieldnames = [
        "pdf_name", "codice", "articolo_min", "articolo_max",
        "articoli_totali", "articoli_base_unici", "articoli_con_suffisso",
        "coverage_pct", "buchi_count", "buchi_preview",
        "suffissi_trovati", "suffissi_invalidi",
        "sospetti_count", "sospetti_preview", "fuori_sequenza_count",
        "articoli_attesi", "delta_attesi", "anomalie_count"
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", extrasaction='ignore')
        w.writeheader()
        for r in reports:
            row = asdict(r)
            row.pop("anomalie", None)
            row.pop("json_path", None)
            w.writerow(row)

    logger.info(f"CSV: {csv_path}")

    # Output JSON dettagliato
    json_path = output_dir / "lexe_report_articoli_v2.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "generated": "2026-02-05",
            "total_documents": len(reports),
            "reports": [asdict(r) for r in reports]
        }, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"JSON: {json_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_articles = sum(r.articoli_totali for r in reports)
    avg_coverage = sum(r.coverage_pct for r in reports) / len(reports) if reports else 0
    docs_with_anomalies = sum(1 for r in reports if r.anomalie_count > 0)
    total_suspects = sum(r.sospetti_count for r in reports)

    print(f"Documenti analizzati: {len(reports)}")
    print(f"Articoli totali: {total_articles:,}")
    print(f"Coverage media: {avg_coverage:.1f}%")
    print(f"Documenti con anomalie: {docs_with_anomalies}")
    print(f"Articoli sospetti totali: {total_suspects}")

    # Top anomalie
    if docs_with_anomalies > 0:
        print(f"\n--- DOCUMENTI CON ANOMALIE ---")
        for r in sorted(reports, key=lambda x: x.anomalie_count, reverse=True)[:10]:
            if r.anomalie_count > 0:
                print(f"  {r.pdf_name}: {r.anomalie_count} anomalie")
                for a in r.anomalie[:3]:
                    print(f"    - [{a.severity}] {a.description}")


if __name__ == "__main__":
    main()
