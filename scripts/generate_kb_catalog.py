#!/usr/bin/env python
"""
Generate KB Normativa Catalog v0.1

Genera il catalogo normalizzato della Knowledge Base con:
- Conteggi separati (ingestiti, validati, in_review)
- Delta vs fonti mirror (Cataldi, Brocardi)
- Quality gates automatici
- Flag per anomalie
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Paths
INGESTED_ROOT = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/ingested")
OUTPUT_FILE = Path("C:/PROJECTS/lexe-genesis/lexe-max/data/kb_catalog_v0.1.json")

# Reference data - articoli attesi per documento (da fonti ufficiali)
ARTICOLI_ATTESI = {
    "CC": 2969,      # Codice Civile (senza suffix)
    "CPC": 840,      # Codice Procedura Civile
    "CPP": 746,      # Codice Procedura Penale
    "CP": 734,       # Codice Penale
    "COST": 139,     # Costituzione
    "CCI": 391,      # Codice Crisi Impresa
    "TUB": 162,      # TU Bancario
    "TUF": 214,      # TU Finanza
    "GDPR": 99,      # Regolamento UE 2016/679
    "DUDU": 30,      # Dichiarazione Universale
    "L241": 31,      # L. 241/1990
    "SLAV": 41,      # Statuto Lavoratori
    # Aggiungi altri quando noti
}

# Atti di riferimento
ATTI_RIFERIMENTO = {
    "CC": {"tipo": "REGIO_DECRETO", "numero": 262, "anno": 1942},
    "CPC": {"tipo": "REGIO_DECRETO", "numero": 1443, "anno": 1940},
    "CPP": {"tipo": "DLGS", "numero": 447, "anno": 1988},
    "CP": {"tipo": "REGIO_DECRETO", "numero": 1398, "anno": 1930},
    "COST": {"tipo": "COSTITUZIONE", "numero": None, "anno": 1948},
    "CCI": {"tipo": "DLGS", "numero": 14, "anno": 2019},
    "TUB": {"tipo": "DLGS", "numero": 385, "anno": 1993},
    "TUF": {"tipo": "DLGS", "numero": 58, "anno": 1998},
    "GDPR": {"tipo": "REGOLAMENTO_UE", "numero": 679, "anno": 2016},
    "L241": {"tipo": "LEGGE", "numero": 241, "anno": 1990},
    "D231": {"tipo": "DLGS", "numero": 231, "anno": 2001},
    "CPRIV": {"tipo": "DLGS", "numero": 196, "anno": 2003},
    "CCONS": {"tipo": "DLGS", "numero": 206, "anno": 2005},
    "CAD": {"tipo": "DLGS", "numero": 82, "anno": 2005},
    "TUDA": {"tipo": "DPR", "numero": 445, "anno": 2000},
    "TUDOC": {"tipo": "DPR", "numero": 445, "anno": 2000},  # Same as TUDA!
    "SLAV": {"tipo": "LEGGE", "numero": 300, "anno": 1970},
    "DUDU": {"tipo": "ALTRO", "numero": None, "anno": 1948},
    "CAPP": {"tipo": "DLGS", "numero": 36, "anno": 2023},
    "TUSL": {"tipo": "DLGS", "numero": 81, "anno": 2008},
    "TUIR": {"tipo": "DPR", "numero": 917, "anno": 1986},
    "TUEL": {"tipo": "DLGS", "numero": 267, "anno": 2000},
}

# Documenti da escludere da READY
EXCLUDED_PENDING_VERIFICATION = ["AUTO"]

# Soglie quality gates
MAX_NUM_THRESHOLD_RATIO = 5  # Se MAX > COUNT * 5, è anomalo


@dataclass
class DocumentCatalogEntry:
    sigla: str
    nome: str
    atto_riferimento: Optional[dict] = None
    fonte_primaria: str = "PDF_ALTALEX"
    fonti_mirror: list = field(default_factory=list)
    data_aggiornamento_fonte: Optional[str] = None
    data_ingestione: str = ""
    articoli_attesi: Optional[int] = None
    articoli_ingestiti: int = 0
    articoli_validati: int = 0
    articoli_in_review: int = 0
    articoli_skipped: int = 0
    delta_vs_riferimento: Optional[int] = None
    delta_vs_mirror: dict = field(default_factory=dict)
    range_articoli: dict = field(default_factory=dict)
    status: str = "PENDING_VALIDATION"
    status_reason: Optional[str] = None
    flags: list = field(default_factory=list)
    parent_document: Optional[str] = None
    variant_type: Optional[str] = None
    note: Optional[str] = None
    url_fonte: Optional[str] = None


def analyze_document(code: str, ing_file: Path) -> DocumentCatalogEntry:
    """Analizza un documento e genera l'entry del catalogo."""

    with open(ing_file, encoding='utf-8') as f:
        data = json.load(f)

    stats = data.get('stats', {})
    articles = data.get('articles', [])

    entry = DocumentCatalogEntry(
        sigla=code,
        nome=code,  # Will be enriched later
        data_ingestione=datetime.now().isoformat(),
    )

    # Conteggi
    entry.articoli_ingestiti = stats.get('total', len(articles))
    entry.articoli_validati = stats.get('ready', 0)
    entry.articoli_in_review = stats.get('needs_review', 0)
    entry.articoli_skipped = stats.get('skipped', 0)

    # Quality gate: validati + in_review = ingestiti - skipped
    expected = entry.articoli_ingestiti - entry.articoli_skipped
    actual = entry.articoli_validati + entry.articoli_in_review
    if actual != expected:
        entry.flags.append("CONTEGGIO_MISMATCH")

    # Delta vs mirror
    entry.delta_vs_mirror = {
        "cataldi_found": stats.get('found_cataldi', 0),
        "cataldi_confirmed": stats.get('confirmed', 0),
        "brocardi_found": stats.get('found_brocardi', 0),
    }
    if stats.get('found_cataldi', 0) > 0:
        entry.fonti_mirror.append("STUDIO_CATALDI")
    if stats.get('found_brocardi', 0) > 0:
        entry.fonti_mirror.append("BROCARDI")

    # Range articoli
    nums = [a.get('base_num', 0) for a in articles if a.get('base_num')]
    suffixes = [a for a in articles if a.get('suffix')]
    if nums:
        min_num = min(nums)
        max_num = max(nums)
        gaps = len(set(range(min_num, max_num + 1)) - set(nums))
        entry.range_articoli = {
            "min": min_num,
            "max": max_num,
            "con_suffix": len(suffixes),
            "gaps": gaps
        }

        # Quality gate: MAX anomalo
        if max_num > entry.articoli_ingestiti * MAX_NUM_THRESHOLD_RATIO:
            entry.flags.append("MAX_NUM_ANOMALY")

    # Articoli attesi
    if code in ARTICOLI_ATTESI:
        entry.articoli_attesi = ARTICOLI_ATTESI[code]
        entry.delta_vs_riferimento = entry.articoli_attesi - entry.articoli_validati

    # Atto di riferimento
    if code in ATTI_RIFERIMENTO:
        entry.atto_riferimento = ATTI_RIFERIMENTO[code]

    # Check for manual QC override
    qc_action = stats.get('qc_action')
    qc_note = stats.get('qc_note', '')

    if qc_action == 'ACCEPT':
        # QC manuale: documento approvato nonostante anomalie
        entry.status = "READY"
        entry.status_reason = f"QC manuale: {qc_note}"
        entry.flags = [f for f in entry.flags if f != "MAX_NUM_ANOMALY"]
    elif qc_action == 'PARTIAL_EXTRACT':
        # QC manuale: classificato come estratto parziale
        entry.status = "PARTIAL_EXTRACT"
        entry.status_reason = qc_note or "Fonte parziale"
        entry.variant_type = "EXTRACT"
    elif qc_action == 'FIX':
        # QC manuale: fix applicato
        entry.status = "READY"
        entry.status_reason = f"QC fix: {qc_note}"
    elif code in EXCLUDED_PENDING_VERIFICATION:
        entry.status = "EXCLUDED"
        entry.status_reason = "Pending source verification"
        entry.flags.append("MISSING_SOURCE_VERIFICATION")
    elif entry.articoli_in_review > 0:
        entry.status = "NEEDS_REVIEW"
        entry.status_reason = f"{entry.articoli_in_review} articoli da revisionare"
        entry.flags.append("HAS_REVIEW_ARTICLES")
    elif "MAX_NUM_ANOMALY" in entry.flags:
        entry.status = "NEEDS_REVIEW"
        entry.status_reason = "MAX numero articolo anomalo, possibili errori parsing"
        entry.flags.append("HAS_PARSING_ERRORS")
    elif entry.articoli_ingestiti < 10:
        entry.status = "PARTIAL_EXTRACT"
        entry.status_reason = f"Solo {entry.articoli_ingestiti} articoli estratti"
        entry.flags.append("PARTIAL_EXTRACTION")
    else:
        entry.status = "READY"

    return entry


def detect_duplicates(entries: list[DocumentCatalogEntry]) -> list[DocumentCatalogEntry]:
    """Rileva documenti con stesso atto di riferimento."""

    atto_to_codes = {}
    for e in entries:
        if e.atto_riferimento:
            key = f"{e.atto_riferimento.get('tipo')}_{e.atto_riferimento.get('numero')}_{e.atto_riferimento.get('anno')}"
            if key not in atto_to_codes:
                atto_to_codes[key] = []
            atto_to_codes[key].append(e.sigla)

    # Mark duplicates
    for key, codes in atto_to_codes.items():
        if len(codes) > 1:
            for e in entries:
                if e.sigla in codes:
                    e.flags.append("DUPLICATE_CANDIDATE")
                    e.note = f"Possibile duplicato con: {', '.join(c for c in codes if c != e.sigla)}"

    return entries


def generate_catalog():
    """Genera il catalogo completo."""

    entries = []

    for d in sorted(INGESTED_ROOT.iterdir()):
        if not d.is_dir():
            continue

        ing_file = d / "ingested.json"
        if not ing_file.exists():
            continue

        try:
            entry = analyze_document(d.name, ing_file)
            entries.append(entry)
        except Exception as e:
            print(f"ERROR processing {d.name}: {e}")

    # Detect duplicates
    entries = detect_duplicates(entries)

    # Generate output
    catalog = {
        "version": "0.1",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_documents": len(entries),
            "total_articles_ingested": sum(e.articoli_ingestiti for e in entries),
            "total_articles_validated": sum(e.articoli_validati for e in entries),
            "total_articles_in_review": sum(e.articoli_in_review for e in entries),
            "status_breakdown": {
                "READY": len([e for e in entries if e.status == "READY"]),
                "NEEDS_REVIEW": len([e for e in entries if e.status == "NEEDS_REVIEW"]),
                "PARTIAL_EXTRACT": len([e for e in entries if e.status == "PARTIAL_EXTRACT"]),
                "EXCLUDED": len([e for e in entries if e.status == "EXCLUDED"]),
            }
        },
        "quality_gates": {
            "check_conteggi": "validati + in_review = ingestiti - skipped",
            "check_duplicati": "stesso atto_riferimento → DUPLICATE_CANDIDATE",
            "check_max_anomaly": f"MAX > COUNT * {MAX_NUM_THRESHOLD_RATIO} → MAX_NUM_ANOMALY",
            "check_partial": "articoli < 10 → PARTIAL_EXTRACT",
        },
        "documents": [asdict(e) for e in entries]
    }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"Catalog saved to: {OUTPUT_FILE}")
    print()
    print("=== SUMMARY ===")
    print(f"Documents: {catalog['summary']['total_documents']}")
    print(f"Articles ingested: {catalog['summary']['total_articles_ingested']}")
    print(f"Articles validated: {catalog['summary']['total_articles_validated']}")
    print(f"Articles in review: {catalog['summary']['total_articles_in_review']}")
    print()
    print("Status breakdown:")
    for status, count in catalog['summary']['status_breakdown'].items():
        print(f"  {status}: {count}")

    # Print issues
    print()
    print("=== ISSUES DETECTED ===")
    for e in entries:
        if e.flags:
            print(f"{e.sigla}: {', '.join(e.flags)}")
            if e.status_reason:
                print(f"  → {e.status_reason}")


if __name__ == "__main__":
    generate_catalog()
