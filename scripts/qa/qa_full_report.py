#!/usr/bin/env python3
"""
QA Protocol - Report Esaustivo Completo

Genera un report dettagliato con tutti i risultati del QA Protocol.
"""

import asyncio
from collections import Counter
from datetime import datetime

import asyncpg

from qa_config import DB_URL


async def main():
    conn = await asyncpg.connect(DB_URL)

    print("=" * 80)
    print("QA PROTOCOL - REPORT ESAUSTIVO KB MASSIMARI")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. INVENTARIO
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("1. INVENTARIO DATI")
    print("=" * 80)

    stats = {
        "manifest": await conn.fetchval("SELECT count(*) FROM kb.pdf_manifest"),
        "ref_units": await conn.fetchval("SELECT count(*) FROM kb.qa_reference_units"),
        "massime": await conn.fetchval("SELECT count(*) FROM kb.massime"),
        "documents": await conn.fetchval("SELECT count(*) FROM kb.documents"),
        "ab_results_a": await conn.fetchval(
            "SELECT count(*) FROM kb.doc_intel_ab_results WHERE run_name='A'"
        ),
        "ab_results_b": await conn.fetchval(
            "SELECT count(*) FROM kb.doc_intel_ab_results WHERE run_name='B'"
        ),
        "alignment": await conn.fetchval(
            "SELECT count(*) FROM kb.reference_alignment_summary"
        ),
        "windows": await conn.fetchval("SELECT count(*) FROM kb.qa_sample_windows"),
    }

    print(f"""
   PDF nel manifest:              {stats["manifest"]:>6}
   Reference units (Phase 0):     {stats["ref_units"]:>6}
   Massime pipeline:              {stats["massime"]:>6}
   Documenti elaborati:           {stats["documents"]:>6}
   Doc Intel Run A:               {stats["ab_results_a"]:>6}
   Doc Intel Run B:               {stats["ab_results_b"]:>6}
   Sample windows (5x15):         {stats["windows"]:>6}
   Alignment summaries:           {stats["alignment"]:>6}
""")

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. DOCUMENT INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("2. DOCUMENT INTELLIGENCE (LLM Classification)")
    print("=" * 80)

    run_a = await conn.fetch(
        "SELECT doc_type, profile, confidence FROM kb.doc_intel_ab_results WHERE run_name='A'"
    )

    print(f"\n   Documenti classificati: {len(run_a)}/63")

    # doc_type distribution
    doc_types = Counter(r["doc_type"] for r in run_a)
    print("\n   doc_type distribution:")
    print("   " + "-" * 60)
    for dt, cnt in doc_types.most_common():
        bar = "#" * int(cnt * 40 / len(run_a))
        print(f"   {dt:28} {cnt:>3} ({100*cnt/len(run_a):>5.1f}%) {bar}")

    # profile distribution
    profiles = Counter(r["profile"] for r in run_a)
    print("\n   profile distribution:")
    print("   " + "-" * 60)
    for p, cnt in profiles.most_common():
        bar = "#" * int(cnt * 40 / len(run_a))
        print(f"   {p:28} {cnt:>3} ({100*cnt/len(run_a):>5.1f}%) {bar}")

    # A/B Comparison
    comparison = await conn.fetch(
        """
        SELECT a.filename, a.doc_type as a_type, b.doc_type as b_type,
               a.profile as a_prof, b.profile as b_prof,
               a.confidence as a_conf, b.confidence as b_conf
        FROM kb.doc_intel_ab_results a
        JOIN kb.doc_intel_ab_results b ON a.manifest_id = b.manifest_id
        WHERE a.run_name = 'A' AND b.run_name = 'B'
    """
    )

    if comparison:
        type_flips = [r for r in comparison if r["a_type"] != r["b_type"]]
        prof_flips = [r for r in comparison if r["a_prof"] != r["b_prof"]]

        print(f"\n   A/B Test Comparison (temp 0.0 vs 0.1):")
        print("   " + "-" * 60)
        print(f"   Documenti confrontati:      {len(comparison)}")
        print(
            f"   doc_type flip rate:         {len(type_flips)} ({100*len(type_flips)/len(comparison):.1f}%)"
        )
        print(
            f"   profile flip rate:          {len(prof_flips)} ({100*len(prof_flips)/len(comparison):.1f}%)"
        )

        if type_flips:
            print("\n   doc_type flips:")
            for r in type_flips:
                print(f"     {r['filename'][:40]:40} {r['a_type']} -> {r['b_type']}")

        if prof_flips:
            print("\n   profile flips:")
            for r in prof_flips[:5]:
                print(f"     {r['filename'][:40]:40} {r['a_prof']} -> {r['b_prof']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. INGESTION STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("3. INGESTION STATISTICS")
    print("=" * 80)

    # Massime per anno
    by_anno = await conn.fetch(
        """
        SELECT d.anno, count(m.id) as cnt
        FROM kb.documents d
        LEFT JOIN kb.massime m ON m.document_id = d.id
        WHERE d.anno IS NOT NULL
        GROUP BY d.anno
        ORDER BY d.anno
    """
    )

    print("\n   Massime per anno:")
    print("   " + "-" * 60)
    for r in by_anno:
        bar = "#" * int(r["cnt"] / 30)
        print(f"   {r['anno']}: {r['cnt']:>5} {bar}")

    # Massime per tipo
    by_tipo = await conn.fetch(
        """
        SELECT d.tipo, count(m.id) as cnt
        FROM kb.documents d
        LEFT JOIN kb.massime m ON m.document_id = d.id
        GROUP BY d.tipo
        ORDER BY cnt DESC
    """
    )

    print("\n   Massime per tipo documento:")
    print("   " + "-" * 60)
    for r in by_tipo:
        print(f"   {r['tipo'] or 'N/A':20} {r['cnt']:>6}")

    # Top 10 documenti
    top_docs = await conn.fetch(
        """
        SELECT d.source_path, d.anno, count(m.id) as cnt
        FROM kb.documents d
        LEFT JOIN kb.massime m ON m.document_id = d.id
        GROUP BY d.id, d.source_path, d.anno
        ORDER BY cnt DESC
        LIMIT 10
    """
    )

    print("\n   Top 10 documenti per numero massime:")
    print("   " + "-" * 60)
    for i, r in enumerate(top_docs, 1):
        print(f"   {i:>2}. {r['source_path'][:45]:45} ({r['anno']}) {r['cnt']:>5}")

    # Bottom 10
    bottom_docs = await conn.fetch(
        """
        SELECT d.source_path, d.anno, count(m.id) as cnt
        FROM kb.documents d
        LEFT JOIN kb.massime m ON m.document_id = d.id
        GROUP BY d.id, d.source_path, d.anno
        ORDER BY cnt ASC
        LIMIT 10
    """
    )

    print("\n   Bottom 10 documenti (meno massime):")
    print("   " + "-" * 60)
    for i, r in enumerate(bottom_docs, 1):
        print(f"   {i:>2}. {r['source_path'][:45]:45} ({r['anno']}) {r['cnt']:>5}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. REFERENCE ALIGNMENT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("4. REFERENCE ALIGNMENT ANALYSIS")
    print("=" * 80)

    alignment = await conn.fetch("SELECT * FROM kb.reference_alignment_summary")

    if alignment:
        total_ref = sum(r["total_ref_units"] for r in alignment)
        total_matched = sum(r["matched_count"] for r in alignment)
        total_unmatched = sum(r["unmatched_count"] for r in alignment)
        avg_coverage = sum(r["coverage_pct"] for r in alignment) / len(alignment)
        avg_frag = sum(r["fragmentation_score"] for r in alignment) / len(alignment)

        print(f"""
   Documenti analizzati:         {len(alignment)}
   Reference units totali:       {total_ref}
   Matched:                      {total_matched} ({100*total_matched/total_ref:.2f}%)
   Unmatched:                    {total_unmatched} ({100*total_unmatched/total_ref:.2f}%)
   Coverage media:               {avg_coverage:.2f}%
   Fragmentation media:          {avg_frag:.2f}
""")

        # Coverage distribution
        buckets = {"0%": 0, "1-5%": 0, "6-25%": 0, "26-50%": 0, "51-100%": 0}
        for a in alignment:
            cov = a["coverage_pct"]
            if cov == 0:
                buckets["0%"] += 1
            elif cov <= 5:
                buckets["1-5%"] += 1
            elif cov <= 25:
                buckets["6-25%"] += 1
            elif cov <= 50:
                buckets["26-50%"] += 1
            else:
                buckets["51-100%"] += 1

        print("   Coverage distribution:")
        print("   " + "-" * 60)
        for bucket, cnt in buckets.items():
            bar = "#" * (cnt)
            pct = 100 * cnt / len(alignment)
            print(f"   {bucket:>10}: {cnt:>3} docs ({pct:>5.1f}%) {bar}")

        # Best and worst coverage
        sorted_align = sorted(alignment, key=lambda x: x["coverage_pct"], reverse=True)

        non_zero = [a for a in sorted_align if a["coverage_pct"] > 0]
        if non_zero:
            print("\n   Documenti con coverage > 0%:")
            print("   " + "-" * 60)
            for a in non_zero[:10]:
                print(
                    f"   {a['manifest_id']:>3}: coverage={a['coverage_pct']:.1f}%"
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. REFERENCE UNITS QUALITY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("5. REFERENCE UNITS QUALITY ANALYSIS")
    print("=" * 80)

    ref_stats = await conn.fetchrow(
        """
        SELECT
            count(*) as total,
            avg(char_count) as avg_chars,
            min(char_count) as min_chars,
            max(char_count) as max_chars,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY char_count) as median_chars,
            sum(case when has_citation then 1 else 0 end) as with_citation
        FROM kb.qa_reference_units
    """
    )

    if ref_stats:
        print(f"""
   Totale units:                 {ref_stats["total"]}
   Chars medio:                  {ref_stats["avg_chars"]:.0f}
   Chars mediano:                {ref_stats["median_chars"]:.0f}
   Chars min/max:                {ref_stats["min_chars"]} / {ref_stats["max_chars"]}
   Con citazioni:                {ref_stats["with_citation"]} ({100*ref_stats["with_citation"]/ref_stats["total"]:.1f}%)
""")

    # Check for spacing issues
    sample = await conn.fetchrow(
        "SELECT testo_norm FROM kb.qa_reference_units WHERE char_count > 100 LIMIT 1"
    )
    if sample:
        text = sample["testo_norm"][:150]
        space_ratio = text.count(" ") / len(text) if text else 0

        print("   Sample reference unit text:")
        print("   " + "-" * 60)
        print(f'   "{text[:70]}..."')
        print()
        if space_ratio > 0.3:
            print(
                "   [!] PROBLEMA RILEVATO: Alta densita di spazi nel testo"
            )
            print(
                f"       Space ratio: {space_ratio:.1%} (atteso <15%)"
            )
            print(
                "       Causa: Estrazione Unstructured con spaziatura carattere-per-carattere"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. PIPELINE MASSIME QUALITY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("6. PIPELINE MASSIME QUALITY")
    print("=" * 80)

    massime_stats = await conn.fetchrow(
        """
        SELECT
            count(*) as total,
            avg(length(testo)) as avg_chars,
            min(length(testo)) as min_chars,
            max(length(testo)) as max_chars,
            sum(case when citation_extracted then 1 else 0 end) as with_citation
        FROM kb.massime
    """
    )

    if massime_stats:
        print(f"""
   Totale massime:               {massime_stats["total"]}
   Chars medio:                  {massime_stats["avg_chars"]:.0f}
   Chars min/max:                {massime_stats["min_chars"]} / {massime_stats["max_chars"]}
   Con citazioni:                {massime_stats["with_citation"] or 0}
""")

    # Sample massima
    massima_sample = await conn.fetchrow(
        "SELECT testo FROM kb.massime WHERE length(testo) > 200 LIMIT 1"
    )
    if massima_sample:
        text = massima_sample["testo"][:150]
        print("   Sample massima text:")
        print("   " + "-" * 60)
        print(f'   "{text[:70]}..."')

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. PROBLEMI IDENTIFICATI
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("7. PROBLEMI IDENTIFICATI")
    print("=" * 80)

    problems = []

    # Problem 1: Low coverage
    if alignment:
        avg_cov = sum(r["coverage_pct"] for r in alignment) / len(alignment)
        if avg_cov < 10:
            problems.append(
                {
                    "severity": "CRITICO",
                    "area": "Reference Alignment",
                    "problema": f"Coverage molto bassa ({avg_cov:.1f}%)",
                    "causa": "Metodi di estrazione diversi (Unstructured vs PyMuPDF)",
                    "impatto": "Impossibile validare qualita pipeline rispetto a ground truth",
                }
            )

    # Problem 2: Spacing in reference units
    if sample and space_ratio > 0.3:
        problems.append(
            {
                "severity": "ALTO",
                "area": "Reference Units",
                "problema": "Spaziatura anomala nel testo estratto",
                "causa": "Unstructured estrae carattere-per-carattere da alcuni PDF",
                "impatto": "Jaccard similarity fallisce, matching impossibile",
            }
        )

    # Problem 3: Missing documents
    missing_docs = 63 - stats["documents"]
    if missing_docs > 0:
        problems.append(
            {
                "severity": "MEDIO",
                "area": "Ingestion",
                "problema": f"{missing_docs} documenti non elaborati",
                "causa": "Errori durante guided ingestion (constraint violations)",
                "impatto": "Coverage incompleta del corpus",
            }
        )

    # Problem 4: A/B flips
    if comparison and len(prof_flips) > 3:
        problems.append(
            {
                "severity": "BASSO",
                "area": "Document Intelligence",
                "problema": f"{len(prof_flips)} profile flip tra Run A e B",
                "causa": "Variabilita LLM anche con temperature bassa",
                "impatto": "Leggera incertezza su profili borderline",
            }
        )

    print()
    for i, p in enumerate(problems, 1):
        print(f"   [{p['severity']}] Problema #{i}: {p['area']}")
        print(f"   " + "-" * 60)
        print(f"   Problema:  {p['problema']}")
        print(f"   Causa:     {p['causa']}")
        print(f"   Impatto:   {p['impatto']}")
        print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. AREE DI MIGLIORAMENTO
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("8. AREE DI MIGLIORAMENTO")
    print("=" * 80)

    improvements = [
        {
            "priorita": 1,
            "area": "Estrazione Unificata",
            "azione": "Usare stesso metodo (PyMuPDF o Unstructured) per reference units e pipeline",
            "beneficio": "Coverage alignment realistica, validazione ground truth",
        },
        {
            "priorita": 2,
            "area": "Text Normalization",
            "azione": "Migliorare normalizzazione per gestire spaziatura anomala",
            "beneficio": "Matching robusto indipendentemente da estrazione",
        },
        {
            "priorita": 3,
            "area": "TOC Detection",
            "azione": "Implementare skip automatico pagine TOC per profili baseline_toc_filter",
            "beneficio": "Riduzione rumore, massime piu pulite",
        },
        {
            "priorita": 4,
            "area": "Chunking Strategy",
            "azione": "Implementare parent-child chunking per profili structured_parent_child",
            "beneficio": "Context retrieval migliore per massime con commento",
        },
        {
            "priorita": 5,
            "area": "Gate Policy Logging",
            "azione": "Aggiungere logging dettagliato decisioni gate",
            "beneficio": "Debugging e tuning soglie piu facile",
        },
    ]

    print()
    for imp in improvements:
        print(f"   #{imp['priorita']} [{imp['area']}]")
        print(f"   " + "-" * 60)
        print(f"   Azione:    {imp['azione']}")
        print(f"   Beneficio: {imp['beneficio']}")
        print()

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. RACCOMANDAZIONI FINALI
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("9. RACCOMANDAZIONI OPERATIVE")
    print("=" * 80)

    print("""
   1. DOCUMENT INTELLIGENCE
      - Usa temperature=0.0 per massima riproducibilita
      - Profilo default: structured_parent_child (63% dei documenti)
      - Attiva baseline_toc_filter per documenti con >10% pagine TOC

   2. EXTRACTION PIPELINE
      - PyMuPDF: veloce, testo pulito, ma perde struttura
      - Unstructured: piu lento, preserva struttura, ma spaziatura problematica
      - Raccomandazione: PyMuPDF per produzione, Unstructured per analisi

   3. CHUNKING
      - mixed (62%): chunking ibrido by_title + by_similarity
      - massima_plus_commentary (29%): parent-child chunking
      - list_only (5%): small chunks, no parent needed
      - toc_heavy (5%): skip TOC pages, aggressive filtering

   4. QUALITY GATES
      - min_length: 150 chars (default), 120 per legacy
      - citation_ratio: max 3% (default), 5% per citation_dense
      - Skip prime 10-15 pagine per documenti toc_heavy
""")

    # ═══════════════════════════════════════════════════════════════════════════
    # 10. METRICHE FINALI
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("10. METRICHE FINALI")
    print("=" * 80)

    print(f"""
   +------------------------------------------+--------+
   | Metrica                                  | Valore |
   +------------------------------------------+--------+
   | PDF processati                           | {stats["manifest"]:>6} |
   | Documenti classificati                   | {stats["ab_results_a"]:>6} |
   | Massime estratte                         | {stats["massime"]:>6} |
   | Reference units                          | {stats["ref_units"]:>6} |
   | Coverage media alignment                 | {avg_coverage if alignment else 0:>5.1f}% |
   | doc_type stability (A/B)                 | {100-100*len(type_flips)/len(comparison) if comparison else 0:>5.1f}% |
   | profile stability (A/B)                  | {100-100*len(prof_flips)/len(comparison) if comparison else 0:>5.1f}% |
   +------------------------------------------+--------+
""")

    print("=" * 80)
    print("[REPORT COMPLETATO]")
    print("=" * 80)

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
