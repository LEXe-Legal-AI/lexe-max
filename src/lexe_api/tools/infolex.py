"""InfoLex Tool (Brocardi).

Searches Italian case law and commentary on brocardi.it.
"""

import re
from typing import Any

import structlog
from bs4 import BeautifulSoup

from lexe_api.scrapers.http_client import ScrapingError, ThrottledHttpClient
from lexe_api.scrapers.selectors import Selectors
from lexe_api.tools.base import BaseLegalTool

logger = structlog.get_logger(__name__)


class InfoLexTool(BaseLegalTool):
    """Tool for searching Italian case law on Brocardi.it."""

    TOOL_NAME = "infolex"
    CACHE_PREFIX = "brocardi"

    def __init__(self):
        super().__init__()
        self.http = ThrottledHttpClient(source="brocardi")

    def _build_cache_key(self, **kwargs: Any) -> str:
        """Build cache key from act type and article."""
        act_type = kwargs.get("act_type", "").lower().replace(" ", "_")
        article = kwargs.get("article", "")
        return f"{act_type}:art{article}"

    async def _fetch(self, **kwargs: Any) -> dict:
        """Fetch case law from Brocardi."""
        act_type = kwargs.get("act_type", "")
        article = kwargs.get("article", "")
        include_massime = kwargs.get("include_massime", True)
        include_relazioni = kwargs.get("include_relazioni", False)
        include_footnotes = kwargs.get("include_footnotes", False)

        url = Selectors.URLs.brocardi_url(act_type, article)

        logger.info("Fetching from Brocardi", act_type=act_type, article=article, url=url)

        try:
            html = await self.http.get_html(url)
            return self._parse_html(
                html,
                act_type,
                article,
                url,
                include_massime,
                include_relazioni,
                include_footnotes,
            )
        except Exception as e:
            logger.error("Brocardi fetch failed", article=article, error=str(e))
            raise ScrapingError(str(e), source="brocardi") from e

    def _parse_html(
        self,
        html: str,
        act_type: str,
        article: str,
        url: str,
        include_massime: bool,
        include_relazioni: bool,
        include_footnotes: bool,
    ) -> dict:
        """Parse Brocardi HTML response."""
        soup = BeautifulSoup(html, "lxml")

        # Extract article info
        article_title = ""
        article_text = ""

        rubrica = soup.select_one(Selectors.Brocardi.ARTICLE_RUBRICA)
        if rubrica:
            article_title = rubrica.get_text(strip=True)

        text_elem = soup.select_one(Selectors.Brocardi.ARTICLE_TEXT)
        if text_elem:
            article_text = text_elem.get_text(strip=True)

        # Extract spiegazione (Brocardi commentary)
        spiegazione = None
        spieg_elem = soup.select_one(Selectors.Brocardi.SPIEGAZIONE)
        if spieg_elem:
            spiegazione = spieg_elem.get_text(strip=True)

        # Extract massime
        massime = []
        if include_massime:
            massime = self._parse_massime(soup)

        # Extract relazioni
        relazioni = []
        if include_relazioni:
            relazioni = self._parse_relazioni(soup)

        # Extract footnotes
        footnotes = []
        if include_footnotes:
            footnotes = self._parse_footnotes(soup)

        return {
            "success": True,
            "act_type": act_type,
            "article": article,
            "article_title": article_title,
            "article_text": article_text,
            "massime": massime,
            "relazioni": relazioni,
            "footnotes": footnotes,
            "spiegazione": spiegazione,
            "brocardi_url": url,
            "source": "brocardi",
        }

    def _parse_massime(self, soup: BeautifulSoup) -> list[dict]:
        """Parse case law summaries (massime)."""
        massime = []

        for item in soup.select(Selectors.Brocardi.MASSIMA_ITEM):
            massima = {}

            # Autorita (e.g., "Cass. civ.")
            autorita = item.select_one(Selectors.Brocardi.MASSIMA_AUTORITA)
            if autorita:
                massima["autorita"] = autorita.get_text(strip=True)
            else:
                massima["autorita"] = "Non specificata"

            # Data sentenza
            data = item.select_one(Selectors.Brocardi.MASSIMA_DATA)
            if data:
                date_str = data.get_text(strip=True)
                massima["data"] = self._parse_date(date_str)

            # Numero sentenza
            numero = item.select_one(Selectors.Brocardi.MASSIMA_NUMERO)
            if numero:
                massima["numero"] = numero.get_text(strip=True)

            # Testo
            testo = item.select_one(Selectors.Brocardi.MASSIMA_TESTO)
            if testo:
                massima["testo"] = testo.get_text(strip=True)
            else:
                continue  # Skip if no text

            # Extract keywords from text
            massima["keywords"] = self._extract_keywords(massima.get("testo", ""))

            massime.append(massima)

        return massime

    def _parse_relazioni(self, soup: BeautifulSoup) -> list[dict]:
        """Parse related articles and norms."""
        relazioni = []

        for item in soup.select(Selectors.Brocardi.RELAZIONE_ITEM):
            text = item.get_text(strip=True)
            link = item.select_one("a")
            relazioni.append({
                "text": text,
                "url": link.get("href") if link else None,
            })

        return relazioni

    def _parse_footnotes(self, soup: BeautifulSoup) -> list[str]:
        """Parse footnotes."""
        footnotes = []

        for item in soup.select(Selectors.Brocardi.FOOTNOTE_ITEM):
            footnotes.append(item.get_text(strip=True))

        return footnotes

    def _parse_date(self, date_str: str) -> str | None:
        """Parse Italian date string to ISO format."""
        # Common patterns: "15 gennaio 2024", "15/01/2024"
        months = {
            "gennaio": "01",
            "febbraio": "02",
            "marzo": "03",
            "aprile": "04",
            "maggio": "05",
            "giugno": "06",
            "luglio": "07",
            "agosto": "08",
            "settembre": "09",
            "ottobre": "10",
            "novembre": "11",
            "dicembre": "12",
        }

        # Try DD/MM/YYYY
        match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
        if match:
            d, m, y = match.groups()
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        # Try DD month YYYY
        match = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", date_str.lower())
        if match:
            d, m, y = match.groups()
            if m in months:
                return f"{y}-{months[m]}-{d.zfill(2)}"

        return None

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract legal keywords from massima text."""
        keywords = []

        # Common legal terms to look for
        legal_terms = [
            "contratto",
            "obbligazione",
            "responsabilità",
            "risarcimento",
            "danno",
            "inadempimento",
            "nullità",
            "annullabilità",
            "prescrizione",
            "decadenza",
            "proprietà",
            "possesso",
            "servitù",
            "usufrutto",
            "ipoteca",
            "pegno",
            "fideiussione",
            "mandato",
            "locazione",
            "compravendita",
            "successione",
            "donazione",
            "testamento",
            "legitima",
            "collazione",
            "divorzio",
            "separazione",
            "affidamento",
            "alimenti",
            "reato",
            "dolo",
            "colpa",
            "tentativo",
            "concorso",
            "aggravante",
            "attenuante",
        ]

        text_lower = text.lower()
        for term in legal_terms:
            if term in text_lower:
                keywords.append(term)

        return keywords[:10]  # Limit to 10 keywords


# Singleton instance
infolex_tool = InfoLexTool()
