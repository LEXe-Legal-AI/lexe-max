"""Normattiva Tool.

Searches Italian legislation on normattiva.it.
"""

from datetime import datetime
from typing import Any

import structlog
from bs4 import BeautifulSoup

from lexe_api.scrapers.http_client import ScrapingError, ThrottledHttpClient
from lexe_api.scrapers.selectors import Selectors
from lexe_api.tools.base import BaseLegalTool

logger = structlog.get_logger(__name__)


class NormattivaTool(BaseLegalTool):
    """Tool for searching Italian legislation on Normattiva.it."""

    TOOL_NAME = "normattiva"
    CACHE_PREFIX = "norm"

    def __init__(self):
        super().__init__()
        self.http = ThrottledHttpClient(source="normattiva")

    def _build_cache_key(self, **kwargs: Any) -> str:
        """Build cache key from request parameters."""
        parts = [kwargs.get("act_type", "")]
        if kwargs.get("date"):
            parts.append(kwargs["date"])
        if kwargs.get("act_number"):
            parts.append(kwargs["act_number"])
        if kwargs.get("article"):
            parts.append(f"art{kwargs['article']}")
        parts.append(kwargs.get("version", "vigente"))
        return ":".join(str(p) for p in parts if p)

    def _build_urn(
        self,
        act_type: str,
        date: str | None = None,
        act_number: str | None = None,
        article: str | None = None,
    ) -> str:
        """Build Normattiva URN from parameters.

        URN format: urn:nir:stato:legge:1990-08-07;241
        """
        # Normalize act type
        act_map = {
            "legge": "legge",
            "decreto legge": "decreto.legge",
            "decreto legislativo": "decreto.legislativo",
            "d.lgs": "decreto.legislativo",
            "d.l.": "decreto.legge",
            "d.p.r.": "decreto.presidente.repubblica",
            "costituzione": "costituzione",
            "codice civile": "codice.civile",
            "codice penale": "codice.penale",
            "cc": "codice.civile",
            "cp": "codice.penale",
        }
        norm_type = act_map.get(act_type.lower(), act_type.lower().replace(" ", "."))

        urn = f"urn:nir:stato:{norm_type}"

        if date:
            urn += f":{date}"
        if act_number:
            urn += f";{act_number}"
        if article:
            urn += f"~art{article}"

        return urn

    async def _fetch(self, **kwargs: Any) -> dict:
        """Fetch document from Normattiva."""
        act_type = kwargs.get("act_type", "")
        date = kwargs.get("date")
        act_number = kwargs.get("act_number")
        article = kwargs.get("article")
        version = kwargs.get("version", "vigente")

        urn = self._build_urn(act_type, date, act_number, article)
        url = f"https://www.normattiva.it/uri-res/N2Ls?{urn}"

        logger.info("Fetching from Normattiva", urn=urn, url=url)

        try:
            html = await self.http.get_html(url)
            return self._parse_html(html, urn, act_type, date, act_number, article, version)
        except Exception as e:
            logger.error("Normattiva fetch failed", urn=urn, error=str(e))
            raise ScrapingError(str(e), source="normattiva") from e

    def _parse_html(
        self,
        html: str,
        urn: str,
        act_type: str,
        date: str | None,
        act_number: str | None,
        article: str | None,
        version: str,
    ) -> dict:
        """Parse Normattiva HTML response."""
        soup = BeautifulSoup(html, "lxml")

        # Check for error
        error = soup.select_one(Selectors.Normattiva.ERROR_NOT_FOUND)
        if error:
            raise ScrapingError("Document not found", source="normattiva", status_code=404)

        # Extract content
        text = ""
        text_elem = soup.select_one(Selectors.Normattiva.ARTICLE_TEXT)
        if text_elem:
            text = text_elem.get_text(strip=True)

        # Extract title/rubrica
        title = ""
        rubrica = soup.select_one(Selectors.Normattiva.ARTICLE_RUBRICA)
        if rubrica:
            title = rubrica.get_text(strip=True)

        # Extract metadata
        meta_urn = soup.select_one(Selectors.Normattiva.METADATA_URN)
        if meta_urn:
            urn = meta_urn.get("content", urn)

        # Check vigenza
        vigente = True
        abrogato_da = None
        note_modifiche = None

        vigenza_box = soup.select_one(Selectors.Normattiva.VIGENZA_BOX)
        if vigenza_box:
            abrogato = soup.select_one(Selectors.Normattiva.ABROGATO_INFO)
            if abrogato:
                vigente = False
                abrogato_da = abrogato.get_text(strip=True)

            modifiche = soup.select(Selectors.Normattiva.MODIFICHE_INFO)
            if modifiche:
                note_modifiche = "; ".join(m.get_text(strip=True) for m in modifiche[:5])

        # Extract GU date from metadata
        data_gu = None
        date_meta = soup.select_one(Selectors.Normattiva.METADATA_DATE)
        if date_meta:
            data_gu = date_meta.get("content")

        return {
            "success": True,
            "urn": urn,
            "codice_redazionale": None,  # Would need additional parsing
            "data_gu": data_gu,
            "title": title,
            "text": text,
            "vigente": vigente,
            "abrogato_da": abrogato_da,
            "note_modifiche": note_modifiche,
            "act_type": act_type,
            "act_number": act_number,
            "act_date": date,
            "article": article,
            "version": version,
            "source": "normattiva",
        }

    async def verify_vigenza(
        self,
        act_type: str,
        date: str | None = None,
        act_number: str | None = None,
        article: str | None = None,
    ) -> dict:
        """Quick check if article is still in force.

        Returns minimal data, faster than full search.
        """
        from lexe_api.cache import cache

        urn = self._build_urn(act_type, date, act_number, article)

        # Check vigenza cache (shorter TTL)
        cached = await cache.get_vigenza(urn)
        if cached:
            return cached

        # Fetch and extract only vigenza info
        result = await self.search(
            act_type=act_type,
            date=date,
            act_number=act_number,
            article=article,
        )

        vigenza = {
            "urn": urn,
            "is_vigente": result.get("vigente", True),
            "abrogato_da": result.get("abrogato_da"),
            "modificato_da": [],
            "data_verifica": datetime.utcnow().isoformat(),
        }

        await cache.set_vigenza(urn, vigenza)
        return vigenza


# Singleton instance
normattiva_tool = NormattivaTool()
