#!/usr/bin/env python3
"""
Fetch Codice Civile from Brocardi.it

Scarica tutti gli articoli del CC da Brocardi.it (fonte editoriale).
Rate limited: 1 req/sec per rispettare ToS.

Usage:
    cd lexe-max
    uv run python scripts/fetch_cc_brocardi.py

    # Solo primi N articoli (test)
    uv run python scripts/fetch_cc_brocardi.py --limit 10

    # Altro codice
    uv run python scripts/fetch_cc_brocardi.py --codice CP
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    parser = argparse.ArgumentParser(description="Fetch codice from Brocardi.it")
    parser.add_argument("--codice", default="CC", help="Codice to fetch (CC, CP, CPC, CPP, COST)")
    parser.add_argument("--limit", type=int, default=0, help="Limit articles (0 = all)")
    parser.add_argument("--rps", type=float, default=1.0, help="Requests per second")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")
    args = parser.parse_args()

    from lexe_api.kb.sources.brocardi_adapter import BrocardiAdapter

    print(f"\n{'='*60}")
    print(f"  FETCH {args.codice} FROM BROCARDI.IT")
    print(f"  {datetime.now().isoformat()}")
    print(f"  Rate limit: {args.rps} req/sec")
    print(f"{'='*60}\n")

    start = datetime.now()
    articles = []
    total_citations = set()

    def progress(current: int, total: int, msg: str | None):
        elapsed = (datetime.now() - start).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        print(f"  [{current}/{total}] {rate:.2f} art/sec | ETA: {eta:.0f}s | {msg}")

        if args.limit > 0 and current >= args.limit:
            raise StopIteration("Limit reached")

    async with BrocardiAdapter(requests_per_second=args.rps) as adapter:
        # List available codici
        available = await adapter.list_codici()
        print(f"Available codici: {', '.join(available)}")

        if args.codice.upper() not in available:
            print(f"ERROR: {args.codice} not available!")
            return

        print(f"\nFetching {args.codice}...")
        print("(This will take a while due to rate limiting)\n")

        try:
            # Use streaming for memory efficiency
            async for article in adapter.stream_codice(args.codice, progress_callback=progress):
                articles.append({
                    "articolo": article.articolo,
                    "rubrica": article.rubrica,
                    "urn": article.urn_nir,
                    "testo": article.testo,  # Full text, no truncation!
                    "testo_length": len(article.testo),
                    "citations": article.citations_raw or [],
                    "libro": article.libro,
                    "titolo": article.titolo,
                    "source_url": article.source_url,
                })

                if article.citations_raw:
                    total_citations.update(article.citations_raw)

                if args.limit > 0 and len(articles) >= args.limit:
                    print(f"\n  Limit of {args.limit} reached, stopping...")
                    break

        except StopIteration:
            pass
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (datetime.now() - start).total_seconds()

    # Stats
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    print(f"  Codice: {args.codice}")
    print(f"  Articles fetched: {len(articles)}")
    print(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Rate: {len(articles)/elapsed:.2f} articles/sec")
    print(f"  Unique citations: {len(total_citations)}")

    if articles:
        avg_len = sum(a["testo_length"] for a in articles) / len(articles)
        print(f"  Avg text length: {avg_len:.0f} chars")

    # Sample
    print(f"\n  Sample articles:")
    for art in articles[:3]:
        print(f"    Art. {art['articolo']}: {art['rubrica'] or '(no rubrica)'}")
        print(f"      URN: {art['urn']}")
        print(f"      Citations: {len(art['citations'])}")

    # Save results
    output_file = args.output or Path(__file__).parent / f"{args.codice.lower()}_brocardi_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "codice": args.codice,
            "source": "brocardi",
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "articles": len(articles),
                "elapsed_seconds": elapsed,
                "unique_citations": len(total_citations),
            },
            "citations": sorted(total_citations),
            "articles": articles,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
