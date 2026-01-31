#!/usr/bin/env python3
"""
Backfill RV Column from Text

Cerca massime con rv NULL ma con pattern RV nel testo e popola la colonna.
Miglioria v3.2.1 #4: Eseguire PRIMA della costruzione del grafo per alzare
il resolution_rate e ridurre i fallback testuali.

Usage:
    # Dry run (default)
    uv run python scripts/graph/backfill_rv_column.py

    # With commit
    uv run python scripts/graph/backfill_rv_column.py --commit

    # Stats only
    uv run python scripts/graph/backfill_rv_column.py --stats
"""

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import asyncpg

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lexe_api.kb.config import KBSettings

# Pattern per RV nel testo
# Match: Rv. 639966, Rv 639966-01, Rv.639966
RV_IN_TEXT_PATTERN = re.compile(r"Rv\.?\s*(\d{5,7})(?:-\d+)?", re.IGNORECASE)

# Pattern per estrarre RV multipli dallo stesso testo
RV_ALL_PATTERN = re.compile(r"Rv\.?\s*(\d{5,7})(?:-\d+)?", re.IGNORECASE)


@dataclass
class BackfillResult:
    """Risultato backfill operazione."""

    massima_id: str
    rv_extracted: str
    rv_position: int  # Posizione nel testo (per audit)
    context: str  # 50 chars intorno al match


@dataclass
class BackfillStats:
    """Statistiche backfill."""

    total_massime: int
    rv_populated_before: int
    rv_null_before: int
    candidates_found: int
    updates_applied: int
    rv_populated_after: int
    rv_coverage_before: float
    rv_coverage_after: float


def normalize_rv(rv: str) -> str:
    """
    Normalizza RV: estrai core numerico, rimuovi suffissi.
    "639966-01" -> "639966"
    "0639966" -> "639966" (lstrip zeri)
    """
    # Prendi solo la parte numerica core
    match = re.match(r"0*(\d{5,7})", rv)
    if match:
        return match.group(1)
    return rv


def extract_first_rv(testo: str) -> tuple[str | None, int, str]:
    """
    Estrae il primo RV dal testo.

    Returns:
        (rv_normalizzato, posizione, contesto_50_chars)
    """
    match = RV_IN_TEXT_PATTERN.search(testo)
    if match:
        rv_raw = match.group(1)
        rv_norm = normalize_rv(rv_raw)
        pos = match.start()
        # Context: 25 chars before, match, 25 chars after
        start = max(0, pos - 25)
        end = min(len(testo), match.end() + 25)
        context = testo[start:end].replace("\n", " ").strip()
        return rv_norm, pos, context
    return None, -1, ""


async def get_current_stats(conn: asyncpg.Connection) -> dict:
    """Ottiene statistiche correnti RV coverage."""
    stats = await conn.fetchrow("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE rv IS NOT NULL) as rv_populated,
            COUNT(*) FILTER (WHERE rv IS NULL) as rv_null,
            ROUND(100.0 * COUNT(*) FILTER (WHERE rv IS NOT NULL) / NULLIF(COUNT(*), 0), 2) as rv_pct
        FROM kb.massime
        WHERE is_active = TRUE OR is_active IS NULL
    """)
    return dict(stats) if stats else {}


async def find_backfill_candidates(conn: asyncpg.Connection) -> list[BackfillResult]:
    """
    Trova massime con rv NULL ma con pattern RV nel testo.
    """
    # Query candidati: rv NULL e pattern RV presente nel testo
    candidates = await conn.fetch("""
        SELECT id, testo
        FROM kb.massime
        WHERE (rv IS NULL OR rv = '')
        AND (is_active = TRUE OR is_active IS NULL)
        AND testo ~ 'Rv\\.?\\s*\\d{5,7}'
    """)

    results = []
    for row in candidates:
        rv, pos, context = extract_first_rv(row["testo"])
        if rv:
            results.append(
                BackfillResult(
                    massima_id=str(row["id"]),
                    rv_extracted=rv,
                    rv_position=pos,
                    context=context,
                )
            )

    return results


async def apply_backfill(
    conn: asyncpg.Connection, candidates: list[BackfillResult]
) -> int:
    """Applica backfill updates."""
    if not candidates:
        return 0

    # Batch update
    values = [(c.massima_id, c.rv_extracted) for c in candidates]
    await conn.executemany(
        "UPDATE kb.massime SET rv = $2, updated_at = NOW() WHERE id = $1::uuid",
        values,
    )
    return len(values)


async def run_backfill(
    commit: bool = False, stats_only: bool = False, verbose: bool = True
) -> BackfillStats:
    """
    Esegue backfill RV column.

    Args:
        commit: Se True, applica le modifiche. Altrimenti dry-run.
        stats_only: Se True, mostra solo statistiche senza cercare candidati.
        verbose: Se True, stampa output dettagliato.
    """
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)

    try:
        # Stats before
        before = await get_current_stats(conn)

        if verbose:
            print("=" * 60)
            print("LEXE KB - Backfill RV Column")
            print("=" * 60)
            print(f"\nStatistiche PRIMA del backfill:")
            print(f"  Massime totali:     {before.get('total', 0):,}")
            print(f"  RV popolati:        {before.get('rv_populated', 0):,}")
            print(f"  RV NULL:            {before.get('rv_null', 0):,}")
            print(f"  Coverage:           {before.get('rv_pct', 0):.1f}%")

        if stats_only:
            return BackfillStats(
                total_massime=before.get("total", 0),
                rv_populated_before=before.get("rv_populated", 0),
                rv_null_before=before.get("rv_null", 0),
                candidates_found=0,
                updates_applied=0,
                rv_populated_after=before.get("rv_populated", 0),
                rv_coverage_before=before.get("rv_pct", 0),
                rv_coverage_after=before.get("rv_pct", 0),
            )

        # Find candidates
        if verbose:
            print("\nCercando candidati per backfill...")

        candidates = await find_backfill_candidates(conn)

        if verbose:
            print(f"  Candidati trovati:  {len(candidates):,}")

        if candidates and verbose:
            print("\nEsempi candidati (primi 5):")
            for c in candidates[:5]:
                print(f"  - {c.massima_id[:8]}... -> Rv. {c.rv_extracted}")
                print(f"    Context: ...{c.context}...")

        updates_applied = 0
        if commit and candidates:
            if verbose:
                print(f"\nApplicando {len(candidates)} updates...")
            updates_applied = await apply_backfill(conn, candidates)
            if verbose:
                print(f"  Updates applicati: {updates_applied}")
        elif not commit and candidates:
            if verbose:
                print("\n[DRY-RUN] Nessuna modifica applicata. Usa --commit per applicare.")

        # Stats after
        after = await get_current_stats(conn)

        if verbose and commit:
            print(f"\nStatistiche DOPO il backfill:")
            print(f"  RV popolati:        {after.get('rv_populated', 0):,}")
            print(f"  Coverage:           {after.get('rv_pct', 0):.1f}%")
            delta = after.get("rv_pct", 0) - before.get("rv_pct", 0)
            print(f"  Delta coverage:     +{delta:.1f}%")

        return BackfillStats(
            total_massime=before.get("total", 0),
            rv_populated_before=before.get("rv_populated", 0),
            rv_null_before=before.get("rv_null", 0),
            candidates_found=len(candidates),
            updates_applied=updates_applied,
            rv_populated_after=after.get("rv_populated", 0) if commit else before.get("rv_populated", 0),
            rv_coverage_before=before.get("rv_pct", 0),
            rv_coverage_after=after.get("rv_pct", 0) if commit else before.get("rv_pct", 0),
        )

    finally:
        await conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill RV column from text patterns"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Apply changes (default: dry-run)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show stats only, no candidate search",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_backfill(
            commit=args.commit,
            stats_only=args.stats,
            verbose=not args.quiet,
        )
    )

    # Exit code: 0 if candidates found or stats mode
    sys.exit(0 if result.candidates_found >= 0 else 1)


if __name__ == "__main__":
    main()
