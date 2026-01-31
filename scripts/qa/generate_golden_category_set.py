#!/usr/bin/env python3
"""
Generate Golden Category Queries

Creates 50 test queries for category-based retrieval evaluation:
- 30 topic_only: Pure topic queries (e.g., "responsabilità civile")
- 20 topic_semantic: Topic + semantic mix (e.g., "risarcimento danno da incidente")

Usage:
    uv run python scripts/qa/generate_golden_category_set.py
    uv run python scripts/qa/generate_golden_category_set.py --count 50 --commit
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings


# ============================================================
# GOLDEN CATEGORY QUERIES
# ============================================================

# Topic-only queries (pure topic, expect router to detect topic)
TOPIC_ONLY_QUERIES = [
    # CIVILE L1
    ("contratto di compravendita", "CIVILE", 1),
    ("responsabilita extracontrattuale", "CIVILE", 1),
    ("usucapione", "CIVILE", 1),
    ("inadempimento contrattuale", "CIVILE", 1),
    # CIVILE L2
    ("danno biologico risarcimento", "CIVILE_RESP_CIVILE", 2),
    ("comunione e condominio", "CIVILE_PROPRIETA", 2),
    ("separazione dei coniugi", "CIVILE_FAMIGLIA", 2),
    ("successione testamentaria", "CIVILE_SUCCESSIONI", 2),

    # LAVORO L1
    ("rapporto di lavoro subordinato", "LAVORO", 1),
    ("licenziamento per giusta causa", "LAVORO", 1),
    ("contrattazione collettiva", "LAVORO", 1),
    # LAVORO L2
    ("reintegra licenziamento illegittimo", "LAVORO_LICENZIAMENTO", 2),
    ("pubblico impiego concorso", "LAVORO_PUBBLICO", 2),
    ("pensione anzianita contributi", "LAVORO_PREVIDENZA", 2),

    # PROCESSUALE_CIVILE L1
    ("ricorso per cassazione civile", "PROCESSUALE_CIVILE", 1),
    ("appello sentenza", "PROCESSUALE_CIVILE", 1),
    ("esecuzione forzata pignoramento", "PROCESSUALE_CIVILE", 1),
    ("decreto ingiuntivo opposizione", "PROCESSUALE_CIVILE", 1),
    # PROCESSUALE_CIVILE L2
    ("motivi di ricorso cassazione", "PROC_CIV_IMPUGNAZIONI", 2),
    ("sequestro conservativo cautelare", "PROC_CIV_CAUTELARE", 2),

    # PENALE L1
    ("reato di truffa", "PENALE", 1),
    ("omicidio colposo", "PENALE", 1),
    ("corruzione pubblico ufficiale", "PENALE", 1),
    # PENALE L2
    ("rapina aggravata", "PENALE_PATRIMONIO", 2),
    ("spaccio stupefacenti", "PENALE_STUPEFACENTI", 2),

    # PROCESSUALE_PENALE L1
    ("custodia cautelare carcere", "PROCESSUALE_PENALE", 1),
    ("patteggiamento rito abbreviato", "PROCESSUALE_PENALE", 1),
    # PROCESSUALE_PENALE L2
    ("misura cautelare esigenze", "PROC_PEN_CAUTELARI", 2),

    # TRIBUTARIO L1
    ("accertamento tributario agenzia entrate", "TRIBUTARIO", 1),
    ("contenzioso tributario ricorso", "TRIBUTARIO", 1),
]

# Topic + semantic queries (topic with more context)
TOPIC_SEMANTIC_QUERIES = [
    # CIVILE
    ("danno da incidente stradale risarcimento", "CIVILE", 1),
    ("contratto di locazione abitativa morosita", "CIVILE", 1),
    ("responsabilita del custode art 2051", "CIVILE_RESP_CIVILE", 2),
    ("vendita immobile vizi occulti", "CIVILE_OBBLIGAZIONI", 2),

    # LAVORO
    ("licenziamento collettivo mobilita azienda", "LAVORO", 1),
    ("demansionamento risarcimento danno professionale", "LAVORO", 1),
    ("tfr trattamento fine rapporto calcolo", "LAVORO_RAPPORTO", 2),

    # PROCESSUALE_CIVILE
    ("violazione art 360 cpc motivazione", "PROCESSUALE_CIVILE", 1),
    ("giudicato cosa giudicata preclusione", "PROC_CIV_IMPUGNAZIONI", 2),
    ("provvedimento urgenza art 700 periculum", "PROC_CIV_CAUTELARE", 2),

    # PENALE
    ("concorso di persone nel reato", "PENALE", 1),
    ("appropriazione indebita gestione denaro", "PENALE_PATRIMONIO", 2),
    ("peculato pubblico ufficiale condanna", "PENALE_PA", 2),

    # PROCESSUALE_PENALE
    ("intercettazioni telefoniche inutilizzabilita", "PROCESSUALE_PENALE", 1),
    ("arresti domiciliari misura cautelare", "PROC_PEN_CAUTELARI", 2),

    # AMMINISTRATIVO
    ("appalto pubblico aggiudicazione gara", "AMMINISTRATIVO", 1),
    ("silenzio rifiuto pubblica amministrazione", "AMM_PROCEDIMENTO", 2),

    # TRIBUTARIO
    ("cartella pagamento prescrizione tributi", "TRIBUTARIO", 1),
    ("iva detrazione indebita rimborso", "TRIB_IVA", 2),

    # FALLIMENTARE
    ("revocatoria fallimentare pagamenti", "FALLIMENTARE_CRISI", 1),
]


async def generate_golden_set(count: int = 50, commit: bool = False):
    """Generate golden category queries."""

    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Generate batch ID
        batch_id = int(datetime.now().strftime("%m%d%H%M"))

        # Combine queries
        topic_only = TOPIC_ONLY_QUERIES[:30]
        topic_semantic = TOPIC_SEMANTIC_QUERIES[:20]

        print(f"Generating golden set batch_id={batch_id}")
        print(f"  topic_only: {len(topic_only)}")
        print(f"  topic_semantic: {len(topic_semantic)}")

        # Validate category IDs exist
        for query_text, cat_id, level in topic_only + topic_semantic:
            row = await conn.fetchrow(
                "SELECT id, level FROM kb.categories WHERE id = $1",
                cat_id
            )
            if not row:
                print(f"  WARNING: Category {cat_id} not found!")
            elif row["level"] != level:
                print(f"  WARNING: Category {cat_id} level mismatch: {row['level']} != {level}")

        if not commit:
            print("\n[DRY RUN] Would insert:")
            for query_text, cat_id, level in (topic_only + topic_semantic)[:10]:
                print(f"  {query_text[:40]:40} -> {cat_id}")
            print(f"  ... and {len(topic_only) + len(topic_semantic) - 10} more")
            return

        # Clear old queries
        await conn.execute("DELETE FROM kb.golden_category_queries")
        print("Cleared old queries")

        # Insert topic_only
        for query_text, cat_id, level in topic_only:
            await conn.execute("""
                INSERT INTO kb.golden_category_queries
                    (batch_id, query_text, query_class, expected_category_id, expected_level)
                VALUES ($1, $2, 'topic_only', $3, $4)
            """, batch_id, query_text, cat_id, level)

        # Insert topic_semantic
        for query_text, cat_id, level in topic_semantic:
            await conn.execute("""
                INSERT INTO kb.golden_category_queries
                    (batch_id, query_text, query_class, expected_category_id, expected_level)
                VALUES ($1, $2, 'topic_semantic', $3, $4)
            """, batch_id, query_text, cat_id, level)

        total = len(topic_only) + len(topic_semantic)
        print(f"\n[OK] Inserted {total} golden category queries")

        # Verify
        counts = await conn.fetch("""
            SELECT query_class, COUNT(*) as cnt
            FROM kb.golden_category_queries
            WHERE batch_id = $1
            GROUP BY query_class
        """, batch_id)
        for row in counts:
            print(f"  {row['query_class']}: {row['cnt']}")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden category set")
    parser.add_argument("--count", type=int, default=50, help="Number of queries")
    parser.add_argument("--commit", action="store_true", help="Actually insert data")
    args = parser.parse_args()

    asyncio.run(generate_golden_set(args.count, args.commit))
