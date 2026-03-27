"""Normattiva OpenData fetch activities for KB nightly sync.

Fetches recently modified acts and their article content from the
Normattiva OpenData API (dati.normattiva.it).

API docs: https://dati.normattiva.it/assets/come_fare_per/API_Normattiva_OpenData.pdf
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from temporalio import activity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NORMATTIVA_API_BASE = os.getenv(
    "NORMATTIVA_API_BASE_URL",
    "https://api.normattiva.it/t/normattiva.api",
)

_TIMEOUT_S = 30.0
_MAX_RETRIES = 3
_PAGE_SIZE = 50  # max useful page size for bulk scanning


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class UpdatedAct:
    """An act identified as recently modified."""

    urn: str
    codice_redazionale: str
    data_gu: str | None  # YYYY-MM-DD
    title: str
    act_type: str


@dataclass
class ArticleContent:
    """Article content fetched from a single act."""

    codice_redazionale: str
    article: str  # article identifier (e.g. "1", "2043")
    text: str
    vigenza_inizio: str | None  # YYYYMMDD
    vigenza_fine: str | None  # YYYYMMDD
    html: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_urn(item: dict[str, Any]) -> str:
    """Construct URN:NIR from API search result item.

    The API response does not include a URN directly; we construct one
    from denominazioneAtto + anno/mese/giorno + numero.
    """
    denom = (item.get("denominazioneAtto") or "").strip().lower()
    anno = item.get("annoProvvedimento")
    mese = item.get("meseProvvedimento")
    giorno = item.get("giornoProvvedimento")
    numero = item.get("numeroProvvedimento")

    # Map denominazione to URN act type slug
    denom_map: dict[str, str] = {
        "legge": "legge",
        "decreto-legge": "decreto.legge",
        "decreto legislativo": "decreto.legislativo",
        "decreto del presidente della repubblica": "decreto.del.presidente.della.repubblica",
        "regio decreto": "regio.decreto",
        "decreto": "decreto",
        "costituzione": "costituzione",
    }
    act_slug = denom_map.get(denom, denom.replace(" ", "."))

    if anno and mese and giorno and numero:
        date_part = f"{int(anno):04d}-{int(mese):02d}-{int(giorno):02d}"
        return f"urn:nir:stato:{act_slug}:{date_part};{numero}"
    if anno and numero:
        return f"urn:nir:stato:{act_slug}:{anno};{numero}"
    return f"urn:nir:stato:{act_slug}:{item.get('codiceRedazionale', 'unknown')}"


def _convert_data_gu_for_api(date_str: str) -> str:
    """Convert YYYY-MM-DD to the bizarre YYYY-DD-MM format the API expects."""
    if not date_str or len(date_str) != 10:
        return date_str or ""
    parts = date_str.split("-")
    if len(parts) != 3:
        return date_str
    # Swap month and day: YYYY-MM-DD -> YYYY-DD-MM
    return f"{parts[0]}-{parts[2]}-{parts[1]}"


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    max_retries: int = _MAX_RETRIES,
) -> httpx.Response:
    """Execute an HTTP request with retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.request(method, url, json=json)
            if response.status_code >= 500 and attempt < max_retries - 1:
                logger.warning(
                    "Normattiva API %d on attempt %d for %s, retrying",
                    response.status_code,
                    attempt + 1,
                    url,
                )
                continue
            return response
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                logger.warning(
                    "Normattiva API timeout on attempt %d for %s, retrying",
                    attempt + 1,
                    url,
                )
                continue
            raise
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                logger.warning(
                    "Normattiva API transport error on attempt %d for %s: %s",
                    attempt + 1,
                    url,
                    exc,
                )
                continue
            raise
    # Should not reach here, but satisfy type checker
    raise last_exc or RuntimeError("Retry exhausted")  # pragma: no cover


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------


@activity.defn
async def fetch_updated_acts(sync_date: str, lookback_days: int) -> list[dict]:
    """Fetch acts modified within the lookback window via Normattiva OpenData API.

    Uses POST /bff-opendata/v1/api/v1/ricerca/avanzata with order_type="recente"
    and date filters covering [sync_date - lookback_days, sync_date].

    Args:
        sync_date: ISO date string (YYYY-MM-DD) for the sync reference point.
        lookback_days: How many days back to scan for modifications.

    Returns:
        List of dicts with keys: urn, codice_redazionale, data_gu, title, act_type.
    """
    ref_date = date.fromisoformat(sync_date)
    start_date = ref_date - timedelta(days=lookback_days)

    logger.info(
        "fetch_updated_acts: scanning %s to %s (lookback=%d days)",
        start_date.isoformat(),
        ref_date.isoformat(),
        lookback_days,
    )

    acts_seen: dict[str, UpdatedAct] = {}  # dedup by codice_redazionale

    async with httpx.AsyncClient(
        base_url=NORMATTIVA_API_BASE,
        timeout=_TIMEOUT_S,
        headers={
            "Accept": "application/json",
            "User-Agent": "LEXE-KB-Sync/1.0",
        },
    ) as client:
        page = 1
        exhausted = False

        while not exhausted:
            activity.heartbeat()

            payload: dict[str, Any] = {
                "orderType": "recente",
                "dataInizioPubProvvedimento": start_date.isoformat(),
                "dataFinePubProvvedimento": ref_date.isoformat(),
                "paginazione": {
                    "paginaCorrente": page,
                    "numeroElementiPerPagina": _PAGE_SIZE,
                },
            }

            response = await _request_with_retry(
                client,
                "POST",
                "/bff-opendata/v1/api/v1/ricerca/avanzata",
                json=payload,
            )

            if response.status_code == 404:
                logger.info("fetch_updated_acts: no results at page %d, stopping", page)
                break

            response.raise_for_status()
            data = response.json()

            total = data.get("numeroAttiTrovati", 0)
            items = data.get("listaAtti", [])

            if not items:
                break

            for item in items:
                cod_red = item.get("codiceRedazionale", "")
                if not cod_red or cod_red in acts_seen:
                    continue

                # Extract GU date — use dataGU directly from search result (YYYY-MM-DD)
                data_gu_raw: str | None = None
                if item.get("dataGU"):
                    data_gu_raw = str(item["dataGU"])[:10]
                elif item.get("annoGU") and item.get("meseGU") and item.get("giornoGU"):
                    data_gu_raw = (
                        f"{int(item['annoGU']):04d}-{int(item['meseGU']):02d}-{int(item['giornoGU']):02d}"
                    )

                act = UpdatedAct(
                    urn=_build_urn(item),
                    codice_redazionale=cod_red,
                    data_gu=data_gu_raw,
                    title=(item.get("titoloAtto") or item.get("descrizioneAtto") or ""),
                    act_type=(item.get("denominazioneAtto") or "UNKNOWN"),
                )
                acts_seen[cod_red] = act

            # Check if we've exhausted all pages
            fetched_so_far = page * _PAGE_SIZE
            if fetched_so_far >= total or len(items) < _PAGE_SIZE:
                exhausted = True
            else:
                page += 1

            logger.info(
                "fetch_updated_acts: page %d done, %d acts collected so far (total=%d)",
                page - (0 if exhausted else 1),
                len(acts_seen),
                total,
            )

    result = [asdict(act) for act in acts_seen.values()]
    logger.info("fetch_updated_acts: returning %d acts", len(result))
    return result


@activity.defn
async def fetch_act_articles(act: dict) -> list[dict]:
    """Fetch all articles for a given act via Normattiva dettaglio-atto endpoint.

    Calls POST /bff-opendata/v1/api/v1/atto/dettaglio-atto WITHOUT idArticolo
    to retrieve the full act, then parses article sections.

    Args:
        act: Dict with keys from UpdatedAct (urn, codice_redazionale, data_gu, ...).

    Returns:
        List of dicts with keys: codice_redazionale, article, text, vigenza_inizio,
        vigenza_fine, html.
    """
    cod_red = act.get("codice_redazionale", "")
    data_gu = act.get("data_gu")

    if not cod_red:
        logger.warning("fetch_act_articles: missing codice_redazionale, skipping")
        return []

    if not data_gu:
        logger.warning(
            "fetch_act_articles: no data_gu for %s, skipping", cod_red
        )
        return []

    logger.info("fetch_act_articles: fetching %s", cod_red)

    async with httpx.AsyncClient(
        base_url=NORMATTIVA_API_BASE,
        timeout=_TIMEOUT_S,
        headers={
            "Accept": "application/json",
            "User-Agent": "LEXE-KB-Sync/1.0",
        },
    ) as client:
        activity.heartbeat()

        # Request full act detail (no idArticolo = full act)
        # NOTE: dettaglio-atto accepts dataGU as YYYY-MM-DD (NOT inverted)
        payload: dict[str, Any] = {
            "dataGU": data_gu,
            "codiceRedazionale": cod_red,
        }

        response = await _request_with_retry(
            client,
            "POST",
            "/bff-opendata/v1/api/v1/atto/dettaglio-atto",
            json=payload,
        )

        if response.status_code == 404:
            logger.info("fetch_act_articles: act %s not found (404)", cod_red)
            return []

        response.raise_for_status()
        raw = response.json()

        # API wraps response in {"code":..., "data":{"atto":{...}}, "success":...}
        data = raw.get("data", raw)
        if isinstance(data, dict) and "atto" in data:
            data = data["atto"]

    # Parse article list from response.
    # The dettaglio-atto response for a full act contains "listaArticoli" or
    # the entire act HTML depending on the act structure.
    articles: list[dict] = []

    # Case 1: structured article list
    lista_articoli = data.get("listaArticoli", [])
    if lista_articoli:
        for art_item in lista_articoli:
            activity.heartbeat()

            art_num = str(art_item.get("numeroArticolo") or art_item.get("idArticolo") or "")
            art_html = art_item.get("testoArticolo") or art_item.get("articoloHtml") or ""
            art_text = _strip_html(art_html)

            if not art_text.strip():
                continue

            articles.append(
                asdict(
                    ArticleContent(
                        codice_redazionale=cod_red,
                        article=art_num,
                        text=art_text,
                        vigenza_inizio=art_item.get("vigenzaInizio"),
                        vigenza_fine=art_item.get("vigenzaFine"),
                        html=art_html,
                    )
                )
            )
    else:
        # Case 2: single HTML body for entire act (typical for short acts)
        full_html = (
            data.get("articoloHtml")
            or data.get("testoArticolo")
            or data.get("testo")
            or ""
        )
        full_text = _strip_html(full_html)

        vigenza_inizio = data.get("vigenzaInizio")
        vigenza_fine = data.get("vigenzaFine")

        if full_text.strip():
            articles.append(
                asdict(
                    ArticleContent(
                        codice_redazionale=cod_red,
                        article="full",
                        text=full_text,
                        vigenza_inizio=vigenza_inizio,
                        vigenza_fine=vigenza_fine,
                        html=full_html,
                    )
                )
            )

    logger.info(
        "fetch_act_articles: %s returned %d articles", cod_red, len(articles)
    )
    return articles


def _strip_html(html: str) -> str:
    """Remove HTML tags to get plain text. Lightweight, no external deps."""
    import re

    if not html:
        return ""
    # Replace <br>, <p>, <div> with newlines before stripping
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|tr)>", "\n", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
