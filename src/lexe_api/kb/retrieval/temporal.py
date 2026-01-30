"""
LEXE Knowledge Base - Temporal Ranking

Ranking temporale con data_decisione per boost recency.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TemporalBoostConfig:
    """Configurazione boost temporale."""

    recency_weight: float = 0.1
    decay_years: float = 10.0  # Anni per decadimento 50%
    exact_year_boost: float = 1.5
    exact_date_boost: float = 2.0


def calculate_recency_score(
    decision_date: date | None,
    decision_year: int | None,
    reference_date: date | None = None,
) -> float:
    """
    Calcola score recency per massima.

    Usa data_decisione se disponibile, altrimenti anno.

    Args:
        decision_date: Data decisione (se nota)
        decision_year: Anno (fallback)
        reference_date: Data di riferimento (default oggi)

    Returns:
        Score 0-1 (1 = recentissimo)
    """
    if reference_date is None:
        reference_date = date.today()

    if decision_date:
        # Calcolo preciso con data
        days_old = (reference_date - decision_date).days
        # Decay esponenziale: 50% dopo 10 anni (~3650 giorni)
        recency = 1.0 / (1 + days_old / 3650)
    elif decision_year:
        # Fallback a anno
        years_old = reference_date.year - decision_year
        recency = 1.0 / (1 + years_old * 0.1)
    else:
        # Nessuna info temporale
        recency = 0.5

    return min(max(recency, 0.0), 1.0)


def apply_temporal_boost(
    results: list[Any],
    massima_dates: dict[UUID, tuple[date | None, int | None]],
    config: TemporalBoostConfig,
    query_year: int | None = None,
    query_date: date | None = None,
) -> list[Any]:
    """
    Applica boost temporale ai risultati.

    Args:
        results: Lista risultati (con attributi massima_id e score/rrf_score)
        massima_dates: Dict massima_id -> (data_decisione, anno)
        config: Configurazione boost
        query_year: Anno nella query (per exact match)
        query_date: Data nella query (per exact match)

    Returns:
        Risultati con score aggiornato e riordinati
    """
    if not results:
        return results

    today = date.today()

    for result in results:
        massima_id = result.massima_id
        dates = massima_dates.get(massima_id, (None, None))
        decision_date, decision_year = dates

        # Calcola recency base
        recency = calculate_recency_score(decision_date, decision_year, today)

        # Boost per match esatto
        boost_multiplier = 1.0
        if query_date and decision_date and decision_date == query_date:
            boost_multiplier = config.exact_date_boost
        elif query_year and decision_year and decision_year == query_year:
            boost_multiplier = config.exact_year_boost

        # Applica boost
        original_score = getattr(result, "rrf_score", None) or getattr(result, "score", 0)
        temporal_contribution = config.recency_weight * recency * boost_multiplier
        new_score = original_score + temporal_contribution

        # Aggiorna score
        if hasattr(result, "final_score"):
            result.final_score = new_score
        elif hasattr(result, "rrf_score"):
            result.rrf_score = new_score
        else:
            result.score = new_score

    # Riordina per nuovo score
    score_attr = "final_score" if hasattr(results[0], "final_score") else (
        "rrf_score" if hasattr(results[0], "rrf_score") else "score"
    )
    results.sort(key=lambda x: getattr(x, score_attr), reverse=True)

    # Aggiorna rank se presente
    for i, r in enumerate(results):
        if hasattr(r, "final_rank"):
            r.final_rank = i + 1
        elif hasattr(r, "new_rank"):
            r.new_rank = i + 1

    return results


async def fetch_massima_dates(
    massima_ids: list[UUID],
    db_pool: Any,
) -> dict[UUID, tuple[date | None, int | None]]:
    """
    Fetch date decisione dal database.

    Args:
        massima_ids: Lista ID massime
        db_pool: Connection pool

    Returns:
        Dict massima_id -> (data_decisione, anno)
    """
    if not massima_ids:
        return {}

    query = """
    SELECT id, data_decisione, anno
    FROM kb.massime
    WHERE id = ANY($1)
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, massima_ids)

    return {
        row["id"]: (row["data_decisione"], row["anno"])
        for row in rows
    }


def extract_temporal_from_query(query: str) -> tuple[int | None, date | None]:
    """
    Estrai riferimenti temporali dalla query.

    Args:
        query: Query testuale

    Returns:
        (anno, data) estratti
    """
    import re

    anno = None
    data_decisione = None

    # Pattern per anno (es. "2020", "nel 2019")
    anno_match = re.search(r"\b(20[0-2][0-9]|201[0-9])\b", query)
    if anno_match:
        anno = int(anno_match.group(1))

    # Pattern per data (es. "15/03/2020", "15-03-2020")
    data_match = re.search(
        r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})",
        query,
    )
    if data_match:
        try:
            data_decisione = date(
                int(data_match.group(3)),
                int(data_match.group(2)),
                int(data_match.group(1)),
            )
            anno = data_decisione.year
        except ValueError:
            pass

    return anno, data_decisione


async def apply_temporal_to_results(
    results: list[Any],
    query: str,
    db_pool: Any,
    config: TemporalBoostConfig | None = None,
) -> list[Any]:
    """
    Convenience function: estrai temporali da query e applica boost.

    Args:
        results: Risultati ricerca
        query: Query originale
        db_pool: Connection pool
        config: Configurazione (default se None)

    Returns:
        Risultati con boost temporale
    """
    if not results:
        return results

    if config is None:
        config = TemporalBoostConfig()

    # Estrai riferimenti temporali dalla query
    query_year, query_date = extract_temporal_from_query(query)

    # Fetch date massime
    massima_ids = [r.massima_id for r in results]
    massima_dates = await fetch_massima_dates(massima_ids, db_pool)

    # Applica boost
    return apply_temporal_boost(
        results,
        massima_dates,
        config,
        query_year,
        query_date,
    )
