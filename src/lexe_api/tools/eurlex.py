"""EUR-Lex Tool.

Searches European legislation on EUR-Lex using SPARQL + REST APIs.
"""

from typing import Any

import structlog
from bs4 import BeautifulSoup

from lexe_api.scrapers.http_client import ScrapingError, SparqlClient, ThrottledHttpClient
from lexe_api.scrapers.selectors import Selectors
from lexe_api.tools.base import BaseLegalTool

logger = structlog.get_logger(__name__)


class EurLexTool(BaseLegalTool):
    """Tool for searching European legislation on EUR-Lex."""

    TOOL_NAME = "eurlex"
    CACHE_PREFIX = "eurlex"

    # Act type to CELEX prefix mapping
    ACT_TYPE_MAP = {
        "regolamento": "R",
        "direttiva": "L",
        "decisione": "D",
        "raccomandazione": "H",
        "trattato": "C",
    }

    def __init__(self):
        super().__init__()
        self.sparql = SparqlClient()
        self.http = ThrottledHttpClient(source="eurlex")

    def _build_cache_key(self, **kwargs: Any) -> str:
        """Build cache key from CELEX-like identifier."""
        act_type = kwargs.get("act_type", "")
        year = kwargs.get("year", "")
        number = kwargs.get("number", "")
        article = kwargs.get("article", "")

        prefix = self.ACT_TYPE_MAP.get(act_type.lower(), "R")
        celex = f"3{year}{prefix}{str(number).zfill(4)}"

        if article:
            celex += f":art{article}"

        return celex

    def _build_celex(self, act_type: str, year: int, number: int) -> str:
        """Build CELEX identifier.

        CELEX format: 3YYYYTNNN
        - 3 = EU legislation
        - YYYY = year
        - T = type (R=regulation, L=directive, D=decision)
        - NNNN = number (4 digits, zero-padded)
        """
        prefix = self.ACT_TYPE_MAP.get(act_type.lower(), "R")
        return f"3{year}{prefix}{str(number).zfill(4)}"

    async def _fetch(self, **kwargs: Any) -> dict:
        """Fetch document from EUR-Lex.

        Strategy:
        1. Try SPARQL query for metadata
        2. Fetch HTML content
        3. Fallback to direct scraping if SPARQL fails
        """
        act_type = kwargs.get("act_type", "")
        year = kwargs.get("year", 2020)
        number = kwargs.get("number", 1)
        article = kwargs.get("article")
        language = kwargs.get("language", "ita")

        celex = self._build_celex(act_type, year, number)

        logger.info("Fetching from EUR-Lex", celex=celex, act_type=act_type)

        # Try SPARQL first for metadata
        metadata = {}
        try:
            metadata = await self._sparql_metadata(celex)
        except Exception as e:
            logger.warning("SPARQL query failed, using fallback", error=str(e))

        # Fetch HTML content
        lang_map = {"ita": "IT", "eng": "EN", "fra": "FR", "deu": "DE", "spa": "ES"}
        lang_code = lang_map.get(language, "IT")
        url = Selectors.URLs.eurlex_celex_url(celex, lang_code)

        try:
            html = await self.http.get_html(url)
            result = self._parse_html(html, celex, act_type, year, number, article, language)
            result.update(metadata)
            return result
        except Exception as e:
            logger.error("EUR-Lex fetch failed", celex=celex, error=str(e))
            raise ScrapingError(str(e), source="eurlex") from e

    async def _sparql_metadata(self, celex: str) -> dict:
        """Query SPARQL endpoint for document metadata."""
        query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        PREFIX eli: <http://data.europa.eu/eli/ontology#>

        SELECT ?title ?date_document ?in_force
        WHERE {{
            ?work cdm:resource_legal_id_celex "{celex}" ;
                  cdm:resource_legal_date_document ?date_document .
            OPTIONAL {{ ?work cdm:resource_legal_in-force ?in_force }}
            OPTIONAL {{
                ?work cdm:work_has_expression ?expr .
                ?expr cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ITA> ;
                      cdm:expression_title ?title .
            }}
        }}
        LIMIT 1
        """

        result = await self.sparql.query(query)

        bindings = result.get("results", {}).get("bindings", [])
        if bindings:
            b = bindings[0]
            return {
                "title": b.get("title", {}).get("value"),
                "publication_date": b.get("date_document", {}).get("value"),
                "in_force": b.get("in_force", {}).get("value", "true") == "true",
            }
        return {}

    def _parse_html(
        self,
        html: str,
        celex: str,
        act_type: str,
        year: int,
        number: int,
        article: str | None,
        language: str,
    ) -> dict:
        """Parse EUR-Lex HTML response."""
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = ""
        title_elem = soup.select_one(Selectors.EurLex.DOCUMENT_TITLE)
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Extract body
        text = ""
        if article:
            # Try to find specific article
            for art in soup.select(Selectors.EurLex.ARTICLE_CONTAINER):
                art_num = art.select_one(Selectors.EurLex.ARTICLE_NUMBER)
                if art_num and article in art_num.get_text():
                    content = art.select_one(Selectors.EurLex.ARTICLE_TEXT)
                    if content:
                        text = content.get_text(strip=True)
                    break
        else:
            # Get full document body
            body = soup.select_one(Selectors.EurLex.DOCUMENT_BODY)
            if body:
                text = body.get_text(strip=True)[:10000]  # Limit size

        # Extract preamble if present
        preamble = None
        preamble_elem = soup.select_one(Selectors.EurLex.DOCUMENT_PREAMBLE)
        if preamble_elem:
            preamble = preamble_elem.get_text(strip=True)[:2000]

        # Build ELI
        eli = f"http://data.europa.eu/eli/{act_type.lower()}/{year}/{number}/oj"

        return {
            "success": True,
            "celex": celex,
            "eli": eli,
            "title": title,
            "text": text,
            "preamble": preamble,
            "act_type": act_type,
            "year": year,
            "number": number,
            "article": article,
            "in_force": True,  # Will be overwritten by SPARQL if available
            "source": "eurlex",
            "language": language,
        }


# Singleton instance
eurlex_tool = EurLexTool()
