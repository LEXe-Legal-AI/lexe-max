#!/usr/bin/env python
"""
Batch Altalex Embedding Pipeline

Processa tutti i JSON in altalex-md/batch e genera embeddings.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexe_api.kb.ingestion.altalex_pipeline import AltalexPipeline, PipelineConfig


# Mapping filename pattern -> codice
CODICE_MAPPING = {
    # Codici principali
    "codice-civile": "CC",
    "codice-penale": "CP",
    "codice-procedura-civile": "CPC",
    "codice-procedura-penale": "CPP",
    "codice-processo-penale-minorile": "CPPM",
    "costituzione-italiana": "COST",

    # GDPR e Privacy
    "gdpr": "GDPR",
    "codice-privacy": "PRIVACY",

    # Amministrativo
    "codice-amministrazione-digitale": "CAD",
    "codice-giustizia-contabile": "CGC",
    "codice-antimafia": "ANTIMAFIA",
    "codice-appalti": "APPALTI",
    "testo-unico-edilizia": "TUE",
    "testo-unico-enti-locali": "TUEL",
    "testo-unico-societa-partecipate": "TUSP",
    "legge-procedimento-amministrativo": "LPA",
    "responsabilita-amministrativa-societa": "DLGS231",
    "testo-unico-espropriazioni": "TUE",
    "testo-unico-documentazione-amministrativa": "DPR445",
    "sicurezza-urbana": "SICURB",

    # Commerciale e Impresa
    "codice-crisi-impresa": "CCI",
    "legge-fallimentare": "LFAL",
    "codice-proprieta-industriale": "CPI",
    "codice-consumo": "CCONS",
    "codice-del-turismo": "CTUR",

    # Lavoro
    "statuto-lavoratori": "STATLAV",
    "riforma-lavoro-biagi": "BIAGI",
    "riforma-lavoro-fornero": "FORNERO",
    "testo-unico-maternita-paternita": "TUMAT",
    "testo-unico-pubblico-impiego": "TUPI",
    "testo-unico-sicurezza-lavoro": "TUSL",
    "codice-pari-opportunita": "CPO",
    "testo-unico-previdenza-complementare": "TUPC",
    "legge-sciopero-servizi-pubblici": "LSCIO",

    # Penale
    "ordinamento-penitenziario": "ORDPEN",
    "testo-unico-stupefacenti": "TUSTUP",
    "testo-unico-casellario-giudiziale": "TUCAS",
    "legge-depenalizzazione": "LDEPEN",
    "legge-reati-tributari": "LREAT",

    # Fisco
    "testo-unico-imposte-redditi": "TUIR",
    "testo-unico-iva": "TUIVA",
    "statuto-contribuente": "STATCONTR",

    # Banche e Finanza
    "testo-unico-bancario": "TUB",
    "testo-unico-finanza": "TUF",

    # Ambiente
    "codice-ambiente": "CAMB",
    "codice-beni-culturali": "CBC",
    "testo-unico-foreste": "TUF",

    # Trasporti e Comunicazioni
    "codice-della-strada": "CDS",
    "codice-nautica-diporto": "CNAUT",
    "regolamento-esecuzione-attuazione-codice-strada": "REGCDS",
    "codice-comunicazioni-elettroniche": "CCE",
    "tu-radiotelevisione": "TURTV",
    "codice-assicurazioni-private": "CAP",

    # Sanità
    "codice-medicinali": "CMED",

    # Professioni
    "legge-professionale-forense": "LPFOR",
    "codice-deontologico-forense": "CDFOR",

    # Sport
    "codice-giustizia-sportiva": "CGSPORT",

    # Civile vario
    "legge-locazioni-abitative": "LLOC",
    "legge-divorzio": "LDIV",
    "mediazione-civile": "MEDCIV",
    "codice-terzo-settore": "CTS",

    # Immigrazione
    "testo-unico-immigrazione": "TUI",

    # Internazionale
    "legge-diritto-internazionale-privato": "LDIP",
    "dichiarazione-universale-diritti-uomo": "DUDU",
}


def extract_codice(filename: str) -> str:
    """Extract codice from filename using mapping."""
    filename_lower = filename.lower()

    # Try exact matches first
    for pattern, codice in CODICE_MAPPING.items():
        if pattern in filename_lower:
            return codice

    # Fallback: extract first meaningful word
    # Remove date patterns and common suffixes
    cleaned = re.sub(r'\d{1,2}[-_]?\w+[-_]?\d{4}', '', filename_lower)
    cleaned = re.sub(r'[-_]def[-_]?pdf', '', cleaned)
    cleaned = re.sub(r'[-_]pdf', '', cleaned)
    cleaned = cleaned.strip('-_ ')

    # Take first 2-3 words and create acronym
    words = cleaned.split('-')[:3]
    if words:
        acronym = ''.join(w[0].upper() for w in words if w)
        return acronym if acronym else "UNK"

    return "UNK"


async def run_batch(
    json_dir: str,
    skip_embed: bool = False,
    limit: int | None = None,
) -> dict:
    """
    Run batch embedding on all JSON files.

    Args:
        json_dir: Directory containing JSON files
        skip_embed: Skip embedding (validation only)
        limit: Limit number of files to process

    Returns:
        Summary dict
    """
    json_dir = Path(json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"Directory not found: {json_dir}")

    # Find all JSON files (excluding meta)
    json_files = sorted(
        [f for f in json_dir.rglob("*.json") if not f.name.endswith("_meta.json")],
        key=lambda f: f.stat().st_size  # Process smallest first
    )

    if limit:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} JSON files to process")
    print("=" * 60)

    # Initialize pipeline
    config = PipelineConfig(
        embed_batch_size=50,
        quarantine_on_error=True,
    )
    pipeline = AltalexPipeline(config=config)

    results = []
    total_articles = 0
    total_embedded = 0
    total_quarantine = 0
    start_time = time.time()

    try:
        for i, json_file in enumerate(json_files, 1):
            codice = extract_codice(json_file.stem)
            file_size_kb = json_file.stat().st_size // 1024

            print(f"\n[{i}/{len(json_files)}] {json_file.name[:50]}... ({file_size_kb}KB)")
            print(f"         Codice: {codice}")

            try:
                result = await pipeline.run(
                    json_path=json_file,
                    codice=codice,
                    skip_embed=skip_embed,
                    skip_store=True,
                )

                total_articles += result.stats.total_articles
                total_embedded += result.stats.embedded_articles
                total_quarantine += result.stats.quarantine_articles

                print(f"         -> {result.stats.valid_articles}/{result.stats.total_articles} valid, "
                      f"{result.stats.embedded_articles} embedded, "
                      f"{result.stats.total_time_ms:.0f}ms")

                results.append({
                    "file": json_file.name,
                    "codice": codice,
                    "total": result.stats.total_articles,
                    "valid": result.stats.valid_articles,
                    "embedded": result.stats.embedded_articles,
                    "quarantine": result.stats.quarantine_articles,
                    "time_ms": result.stats.total_time_ms,
                    "tokens": result.stats.total_tokens,
                    "status": "OK" if result.stage.value == "complete" else result.stage.value,
                })

            except Exception as e:
                print(f"         -> ERROR: {e}")
                results.append({
                    "file": json_file.name,
                    "codice": codice,
                    "status": "ERROR",
                    "error": str(e),
                })

    finally:
        await pipeline.close()

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(json_files)}")
    print(f"Total articles: {total_articles}")
    print(f"Total embedded: {total_embedded}")
    print(f"Total quarantine: {total_quarantine}")
    print(f"Success rate: {total_embedded / total_articles * 100:.1f}%" if total_articles > 0 else "N/A")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time per file: {total_time / len(json_files):.1f}s" if json_files else "N/A")

    # Get cache stats from pipeline
    if pipeline._embedder:
        cache_stats = pipeline._embedder.get_cache_stats()
        print(f"Cache hits: {cache_stats['hits']}")
        print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")

    return {
        "files_processed": len(json_files),
        "total_articles": total_articles,
        "total_embedded": total_embedded,
        "total_quarantine": total_quarantine,
        "total_time_s": total_time,
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch Altalex Embedding")
    parser.add_argument(
        "json_dir",
        nargs="?",
        default="../altalex-md/batch",
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding (validation only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_embed_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Check API key
    if not args.skip_embed and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Set it with: export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    # Run batch
    summary = asyncio.run(run_batch(
        json_dir=args.json_dir,
        skip_embed=args.skip_embed,
        limit=args.limit,
    ))

    # Save results
    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
