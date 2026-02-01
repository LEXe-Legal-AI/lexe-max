"""
QA Protocol - Phase 7: Generate Query Set

Generates 200+ evaluation queries from four sources:
1. Self-retrieval: 63 queries (1 per doc, random massima, first 50 words)
2. Citation retrieval: ~100 queries from massime with cross-doc citations
3. Benchmark: 30 queries from BENCHMARK_QUERIES
4. Adversarial: 30 queries (negations, boundary cases)

Usage (on staging server):
    cd /opt/lexe-platform/lexe-max
    uv run python scripts/qa/s7_generate_query_set.py
"""

import asyncio
import random

import asyncpg

from qa_config import DB_URL

# From run_retrieval_benchmark.py
BENCHMARK_QUERIES = [
    {"type": "istituto", "query": "responsabilita civile per danni", "keywords": ["responsabil", "dann", "civil"]},
    {"type": "istituto", "query": "risarcimento del danno patrimoniale", "keywords": ["risarciment", "dann", "patrimonial"]},
    {"type": "istituto", "query": "nullita del contratto", "keywords": ["null", "contratt"]},
    {"type": "istituto", "query": "prescrizione del diritto", "keywords": ["prescrizion", "diritt"]},
    {"type": "istituto", "query": "successione ereditaria legittima", "keywords": ["succession", "ereditar", "legittim"]},
    {"type": "istituto", "query": "fallimento e procedure concorsuali", "keywords": ["falliment", "concorsual", "procedur"]},
    {"type": "istituto", "query": "licenziamento per giusta causa", "keywords": ["licenziament", "giusta", "causa"]},
    {"type": "istituto", "query": "reato di truffa elementi costitutivi", "keywords": ["truff", "reat", "element"]},
    {"type": "istituto", "query": "omicidio colposo presupposti", "keywords": ["omicid", "colpos"]},
    {"type": "istituto", "query": "concorso di persone nel reato", "keywords": ["concors", "person", "reat"]},
    {"type": "istituto", "query": "misure cautelari personali", "keywords": ["misur", "cautelar", "personal"]},
    {"type": "istituto", "query": "appello nel processo civile", "keywords": ["appell", "process", "civil"]},
    {"type": "avversaria", "query": "quando NON sussiste responsabilita", "keywords": ["responsabil", "sussist"]},
    {"type": "avversaria", "query": "esclusione del dolo nel reato", "keywords": ["dol", "reat", "esclus"]},
    {"type": "avversaria", "query": "inammissibilita del ricorso cassazione", "keywords": ["inammissibil", "ricors", "cassazion"]},
    {"type": "avversaria", "query": "rigetto della domanda risarcitoria", "keywords": ["rigett", "risarcitor"]},
    {"type": "avversaria", "query": "mancanza di legittimazione attiva", "keywords": ["legittimaz", "attiv", "mancanz"]},
    {"type": "avversaria", "query": "improcedibilita della querela", "keywords": ["improcedibil", "querel"]},
    {"type": "avversaria", "query": "assenza di nesso causale", "keywords": ["nesso", "causal", "assenz"]},
    {"type": "avversaria", "query": "insussistenza del fatto contestato", "keywords": ["insussistenz", "fatt", "contestat"]},
    {"type": "avversaria", "query": "non punibilita per particolare tenuita", "keywords": ["punibil", "tenu"]},
    {"type": "avversaria", "query": "difetto di motivazione sentenza", "keywords": ["difett", "motivazion", "sentenz"]},
    {"type": "avversaria", "query": "prescrizione maturata durante processo", "keywords": ["prescrizion", "maturat", "process"]},
    {"type": "avversaria", "query": "incompetenza territoriale del giudice", "keywords": ["incompetenz", "territorial", "giudic"]},
    {"type": "citazione", "query": "art. 2043 codice civile", "keywords": ["2043", "civil"]},
    {"type": "citazione", "query": "art. 640 codice penale truffa", "keywords": ["640", "penal", "truff"]},
    {"type": "citazione", "query": "art. 575 codice penale omicidio", "keywords": ["575", "penal", "omicid"]},
    {"type": "citazione", "query": "art. 1218 responsabilita contrattuale", "keywords": ["1218", "contrattual"]},
    {"type": "citazione", "query": "art. 337 cpp resistenza pubblico ufficiale", "keywords": ["337", "resistenz", "pubblic"]},
    {"type": "citazione", "query": "legge fallimentare art. 67", "keywords": ["67", "fallimentar"]},
]


async def main():
    print("=" * 70)
    print("QA PROTOCOL - PHASE 7: GENERATE QUERY SET")
    print("=" * 70)

    conn = await asyncpg.connect(DB_URL)
    print("[OK] Database connected")

    qa_run_id = await conn.fetchval(
        "SELECT id FROM kb.qa_runs ORDER BY started_at DESC LIMIT 1"
    )
    print(f"[OK] qa_run_id={qa_run_id}")

    # Check existing queries
    existing = await conn.fetchval(
        "SELECT count(*) FROM kb.retrieval_eval_queries WHERE qa_run_id = $1",
        qa_run_id,
    )
    if existing > 0:
        print(f"[SKIP] {existing} queries already exist for this qa_run")
        await conn.close()
        return

    total = 0

    # ── 1. Self-retrieval (1 per document) ────────────────────────
    docs = await conn.fetch(
        """
        SELECT pm.doc_id, pm.filename
        FROM kb.pdf_manifest pm
        WHERE pm.qa_run_id = $1
        """,
        qa_run_id,
    )

    for doc in docs:
        # Get a random massima from this doc
        massima = await conn.fetchrow(
            """
            SELECT id, testo FROM kb.massime
            WHERE document_id = $1
            ORDER BY random()
            LIMIT 1
            """,
            doc["doc_id"],
        )
        if not massima:
            continue

        # First 50 words as query
        words = (massima["testo"] or "").split()[:50]
        query_text = " ".join(words)

        await conn.execute(
            """
            INSERT INTO kb.retrieval_eval_queries
              (qa_run_id, query_text, query_type, source_massima_id, ground_truth_ids)
            VALUES ($1, $2, $3, $4, ARRAY[$4]::uuid[])
            """,
            qa_run_id, query_text, "self_retrieval", massima["id"],
        )
        total += 1

    print(f"[OK] Self-retrieval queries: {total}")

    # ── 2. Citation retrieval ─────────────────────────────────────
    citation_count = 0
    # Find massime that reference other massime (via Sez. pattern matching)
    massime_with_cit = await conn.fetch(
        """
        SELECT id, testo FROM kb.massime
        WHERE testo ~* 'Sez\.?\s*(U|L|[0-9]+)[,\s]+n\.?\s*[0-9]+'
        ORDER BY random()
        LIMIT 100
        """,
    )

    for ms in massime_with_cit:
        words = (ms["testo"] or "").split()[:40]
        query_text = " ".join(words)

        await conn.execute(
            """
            INSERT INTO kb.retrieval_eval_queries
              (qa_run_id, query_text, query_type, source_massima_id, ground_truth_ids)
            VALUES ($1, $2, $3, $4, ARRAY[$4]::uuid[])
            """,
            qa_run_id, query_text, "citation_retrieval", ms["id"],
        )
        citation_count += 1

    total += citation_count
    print(f"[OK] Citation retrieval queries: {citation_count}")

    # ── 3. Benchmark queries ──────────────────────────────────────
    for bq in BENCHMARK_QUERIES:
        await conn.execute(
            """
            INSERT INTO kb.retrieval_eval_queries
              (qa_run_id, query_text, query_type, keywords)
            VALUES ($1, $2, $3, $4)
            """,
            qa_run_id, bq["query"], f"benchmark_{bq['type']}", bq["keywords"],
        )
        total += 1

    print(f"[OK] Benchmark queries: {len(BENCHMARK_QUERIES)}")

    # ── 4. Adversarial queries ────────────────────────────────────
    adversarial = [
        "questo testo non esiste nel massimario",
        "ricetta per la pasta alla carbonara",
        "previsioni meteo per domani roma",
        "art. 99999 codice civile inesistente",
        "dinosauri nel periodo giurassico",
        "formula chimica dell'acqua H2O",
        "bitcoin criptovaluta blockchain",
        "responsabilita civile extraterrestre aliena",
        "il gatto sul tetto che scotta",
        "analisi quantistica delle particelle subatomiche",
        "reato di magia e stregoneria",
        "procedura concorsuale per aziende marziane",
        "successione ereditaria dei faraoni egizi",
        "cassazione sezione 999 numero 0000001",
        "diritto alla felicita nella costituzione italiana",
        "contravvenzione per eccesso di gentilezza",
        "articolo 42 guida galattica autostoppisti",
        "prescrizione del diritto di volare",
        "nullita contratto con extraterrestri",
        "licenziamento per giusta causa aliena",
        "risarcimento danni da meteorite",
        "appello processo penale sottomarino",
        "reato di furto di nuvole",
        "causalita omissiva nel multiverse",
        "giurisdizione tribunale antartico",
        "diritto internazionale privato lunare",
        "competenza giudice di pace cosmico",
        "abuso del diritto di teletrasporto",
        "concorso formale in universi paralleli",
        "recidiva specifica interdimensionale",
    ]

    for adv in adversarial:
        await conn.execute(
            """
            INSERT INTO kb.retrieval_eval_queries
              (qa_run_id, query_text, query_type)
            VALUES ($1, $2, $3)
            """,
            qa_run_id, adv, "adversarial",
        )
        total += 1

    print(f"[OK] Adversarial queries: {len(adversarial)}")

    print(f"\n{'=' * 70}")
    print(f"QUERY SET COMPLETE: {total} total queries")
    print(f"{'=' * 70}")

    await conn.close()
    print("[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
