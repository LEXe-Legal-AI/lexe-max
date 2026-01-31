#!/usr/bin/env python3
"""
Generate Golden Set for Norm Query Evaluation

Creates 100 test queries:
- 60 pure_norm: Only norm reference (art. 2043 c.c., dlgs 165 2001)
- 40 mixed: Semantic + norm (danno ingiusto art 2043 c.c.)

Distribution for pure_norm:
- 25 codes (CC, CPC, CP, CPP, COST)
- 25 laws (LEGGE, DLGS, DPR, DL, TUB, TUF)
- 10 dirty variants (no dots, wrong spacing)

Usage:
    uv run python scripts/qa/generate_golden_norm_set.py
    uv run python scripts/qa/generate_golden_norm_set.py --dry-run
"""

import argparse
import asyncio
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings


# ============================================================
# TEMPLATES
# ============================================================

# Pure norm templates - codes (CC, CPC, etc.)
CODE_TEMPLATES = [
    "art. {art} {code_lit}",
    "art {art} {code_lit}",
    "artt. {art} {code_lit}",
    "{art} {code_short}",
    "articolo {art} {code_lit}",
]

# Pure norm templates - laws (LEGGE, DLGS, etc.)
LAW_TEMPLATES = [
    "{code_lit} {num}/{year}",
    "{code_lit} n. {num}/{year}",
    "{code_short} {num}/{year}",
    "{code_short} {num} {year}",
    "{code_lit} n. {num} del {year}",
]

# Mixed templates - semantic + norm
MIXED_CODE_TEMPLATES = [
    "{phrase} art. {art} {code_lit}",
    "{phrase} {art} {code_short}",
    "{phrase} ex art. {art} {code_lit}",
    "art. {art} {code_lit} {phrase}",
    "{phrase} ai sensi dell'art. {art} {code_lit}",
]

MIXED_LAW_TEMPLATES = [
    "{phrase} {code_lit} {num}/{year}",
    "{phrase} {code_short} {num}/{year}",
    "{code_lit} {num}/{year} {phrase}",
    "{phrase} ex {code_lit} {num}/{year}",
]

# Semantic phrases for mixed queries
PHRASES_BY_CODE = {
    "CC": [
        "danno ingiusto",
        "responsabilita extracontrattuale",
        "nesso causale",
        "inadempimento contrattuale",
        "buona fede",
        "risarcimento danni",
        "obbligazioni",
        "proprieta",
    ],
    "CPC": [
        "termini impugnazione",
        "ricorso cassazione",
        "competenza territoriale",
        "litisconsorzio",
        "prova testimoniale",
        "esecuzione forzata",
        "provvisoria esecutorieta",
    ],
    "CP": [
        "dolo eventuale",
        "colpa grave",
        "concorso di reati",
        "circostanze attenuanti",
        "pena pecuniaria",
    ],
    "CPP": [
        "misure cautelari",
        "custodia cautelare",
        "intercettazioni",
        "patteggiamento",
        "giudizio abbreviato",
    ],
    "COST": [
        "giusto processo",
        "diritto di difesa",
        "uguaglianza formale",
        "liberta personale",
        "principio di legalita",
    ],
    "LEGGE": [
        "procedimento amministrativo",
        "accesso agli atti",
        "silenzio assenso",
        "sanzioni amministrative",
    ],
    "DLGS": [
        "pubblico impiego",
        "contratti pubblici",
        "processo tributario",
        "immigrazione",
        "licenziamento",
    ],
    "DPR": [
        "autocertificazione",
        "documento amministrativo",
    ],
    "DL": [
        "decreto urgente",
        "conversione",
    ],
}

# Code literals and shorts
CODE_LITERAL = {
    "CC": "c.c.",
    "CPC": "c.p.c.",
    "CP": "c.p.",
    "CPP": "c.p.p.",
    "COST": "Cost.",
    "TUB": "t.u.b.",
    "TUF": "t.u.f.",
    "CAD": "c.a.d.",
}

CODE_SHORT = {
    "CC": "cc",
    "CPC": "cpc",
    "CP": "cp",
    "CPP": "cpp",
    "COST": "cost",
    "TUB": "tub",
    "TUF": "tuf",
    "CAD": "cad",
}

LAW_LITERAL = {
    "LEGGE": "L.",
    "DLGS": "D.Lgs.",
    "DPR": "D.P.R.",
    "DL": "D.L.",
}

LAW_SHORT = {
    "LEGGE": "l",
    "DLGS": "dlgs",
    "DPR": "dpr",
    "DL": "dl",
}


# ============================================================
# HELPERS
# ============================================================

def dirtyify(query: str) -> str:
    """Make query dirty: remove dots, compress spaces."""
    q = query.replace(".", "")
    q = re.sub(r"\s+", " ", q).strip()
    # Random case changes
    if random.random() < 0.3:
        q = q.lower()
    elif random.random() < 0.3:
        q = q.upper()
    return q


def get_phrase(code: str) -> str:
    """Get a random phrase for a code type."""
    # Map law codes to generic phrases
    if code in PHRASES_BY_CODE:
        return random.choice(PHRASES_BY_CODE[code])
    # Fallback for laws
    if code in ("LEGGE", "DL"):
        return random.choice(PHRASES_BY_CODE["LEGGE"])
    if code in ("DLGS", "DPR"):
        return random.choice(PHRASES_BY_CODE["DLGS"])
    return "giurisprudenza"


def is_code(code: str) -> bool:
    """Check if code is a code (vs law)."""
    return code in ("CC", "CPC", "CP", "CPP", "COST", "TUB", "TUF", "CAD")


def render_query(norm: dict, template: str, phrase: str = None) -> str:
    """Render a query from norm and template."""
    code = norm["code"]

    if is_code(code):
        return template.format(
            art=norm["art"] or "",
            code_lit=CODE_LITERAL.get(code, code.lower()),
            code_short=CODE_SHORT.get(code, code.lower()),
            phrase=phrase or "",
        ).strip()
    else:
        return template.format(
            num=norm["number"] or "",
            year=norm["year"] or "",
            code_lit=LAW_LITERAL.get(code, code),
            code_short=LAW_SHORT.get(code, code.lower()),
            phrase=phrase or "",
        ).strip()


# ============================================================
# MAIN
# ============================================================

async def generate_golden_set(dry_run: bool = False):
    """Generate 100 golden queries."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    # Use shorter format for int32 compatibility
    batch_id = int(datetime.now().strftime("%m%d%H%M"))

    # Fetch top norms by citation count
    rows = await conn.fetch("""
        SELECT id, code, article, suffix, number, year, citation_count
        FROM kb.norms
        WHERE citation_count >= 5
        ORDER BY citation_count DESC
        LIMIT 300
    """)

    # Separate codes and laws
    codes = []
    laws = []
    for r in rows:
        norm = {
            "norm_id": r["id"],
            "code": r["code"],
            "art": r["article"],
            "suffix": r["suffix"],
            "number": r["number"],
            "year": r["year"],
            "citations": r["citation_count"],
        }
        if is_code(r["code"]):
            codes.append(norm)
        else:
            laws.append(norm)

    print(f"Found {len(codes)} codes, {len(laws)} laws with >= 5 citations")

    queries = []
    used_norm_ids = set()

    # --- 60 PURE NORM QUERIES ---

    # 25 codes (clean)
    random.shuffle(codes)
    for norm in codes[:25]:
        if norm["norm_id"] in used_norm_ids:
            continue
        template = random.choice(CODE_TEMPLATES)
        q = render_query(norm, template)
        queries.append(("pure_norm", q, norm["norm_id"], "code_clean"))
        used_norm_ids.add(norm["norm_id"])

    # 25 laws (clean)
    random.shuffle(laws)
    for norm in laws[:25]:
        if norm["norm_id"] in used_norm_ids:
            continue
        template = random.choice(LAW_TEMPLATES)
        q = render_query(norm, template)
        queries.append(("pure_norm", q, norm["norm_id"], "law_clean"))
        used_norm_ids.add(norm["norm_id"])

    # 10 dirty variants (mix of codes and laws)
    dirty_pool = codes[25:35] + laws[25:35]
    random.shuffle(dirty_pool)
    for norm in dirty_pool[:10]:
        if norm["norm_id"] in used_norm_ids:
            continue
        if is_code(norm["code"]):
            template = random.choice(CODE_TEMPLATES)
        else:
            template = random.choice(LAW_TEMPLATES)
        q = dirtyify(render_query(norm, template))
        queries.append(("pure_norm", q, norm["norm_id"], "dirty"))
        used_norm_ids.add(norm["norm_id"])

    # --- 40 MIXED QUERIES ---

    # 20 codes + semantic
    remaining_codes = [n for n in codes if n["norm_id"] not in used_norm_ids][:20]
    for norm in remaining_codes:
        template = random.choice(MIXED_CODE_TEMPLATES)
        phrase = get_phrase(norm["code"])
        q = render_query(norm, template, phrase)
        if random.random() < 0.2:
            q = dirtyify(q)
        queries.append(("mixed", q, norm["norm_id"], f"mixed_code:{phrase}"))
        used_norm_ids.add(norm["norm_id"])

    # 20 laws + semantic
    remaining_laws = [n for n in laws if n["norm_id"] not in used_norm_ids][:20]
    for norm in remaining_laws:
        template = random.choice(MIXED_LAW_TEMPLATES)
        phrase = get_phrase(norm["code"])
        q = render_query(norm, template, phrase)
        if random.random() < 0.2:
            q = dirtyify(q)
        queries.append(("mixed", q, norm["norm_id"], f"mixed_law:{phrase}"))
        used_norm_ids.add(norm["norm_id"])

    # Count by class
    pure_count = sum(1 for q in queries if q[0] == "pure_norm")
    mixed_count = sum(1 for q in queries if q[0] == "mixed")

    print(f"\nGenerated {len(queries)} queries:")
    print(f"  pure_norm: {pure_count}")
    print(f"  mixed: {mixed_count}")

    if dry_run:
        print("\n[DRY RUN] Sample queries:")
        for cls, q, norm_id, note in queries[:10]:
            print(f"  {cls:10} | {norm_id:20} | {q}")
        await conn.close()
        return

    # Deactivate old queries
    await conn.execute("""
        UPDATE kb.golden_norm_queries
        SET is_active = FALSE
        WHERE is_active = TRUE
    """)

    # Insert new queries
    await conn.executemany(
        """
        INSERT INTO kb.golden_norm_queries(batch_id, query_text, query_class, expected_norm_id, notes)
        VALUES($1, $2, $3, $4, $5)
        ON CONFLICT (batch_id, query_text) DO NOTHING
        """,
        [(batch_id, q, cls, norm_id, note) for (cls, q, norm_id, note) in queries],
    )

    print(f"\nInserted batch_id={batch_id}")

    # Verify
    counts = await conn.fetch("""
        SELECT query_class, COUNT(*) as cnt
        FROM kb.golden_norm_queries
        WHERE is_active = TRUE AND batch_id = $1
        GROUP BY query_class
    """, batch_id)

    print("\nVerification:")
    for row in counts:
        print(f"  {row['query_class']}: {row['cnt']}")

    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Golden Norm Set")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    args = parser.parse_args()

    asyncio.run(generate_golden_set(args.dry_run))
