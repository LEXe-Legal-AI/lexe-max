"""CSS Selectors for legal document scraping.

Centralized selectors for maintainability when sources change their HTML structure.
"""


class Selectors:
    """CSS selectors for different sources."""

    # =========================================================================
    # Normattiva Selectors
    # =========================================================================

    class Normattiva:
        """Selectors for normattiva.it."""

        # Article content
        ARTICLE_BODY = "#articolo_body"
        ARTICLE_TEXT = "#articolo_corpo"
        ARTICLE_RUBRICA = "#articolo_rubrica"

        # Metadata
        METADATA_URN = "meta[name='DC.identifier']"
        METADATA_TITLE = "meta[name='DC.title']"
        METADATA_DATE = "meta[name='DC.date']"

        # Vigenza info
        VIGENZA_BOX = ".vigenza-box"
        VIGENZA_STATUS = ".vigenza-status"
        ABROGATO_INFO = ".abrogato-info"
        MODIFICHE_INFO = ".modifiche-elenco li"

        # Tree navigation
        TREE_CONTAINER = "#indice-atto"
        TREE_ITEMS = ".indice-item"
        TREE_ARTICLE_LINK = "a[href*='articolo']"

        # Error messages
        ERROR_NOT_FOUND = ".messaggio-errore"

    # =========================================================================
    # Brocardi Selectors
    # =========================================================================

    class Brocardi:
        """Selectors for brocardi.it."""

        # Article content
        ARTICLE_CONTAINER = ".articolo-container"
        ARTICLE_TEXT = ".testo-articolo"
        ARTICLE_RUBRICA = ".rubrica-articolo"

        # Brocardi commentary
        SPIEGAZIONE = ".spiegazione-container"
        RATIO = ".ratio-articolo"

        # Massime
        MASSIME_CONTAINER = ".massime-container"
        MASSIMA_ITEM = ".massima-item"
        MASSIMA_AUTORITA = ".massima-autorita"
        MASSIMA_DATA = ".massima-data"
        MASSIMA_NUMERO = ".massima-numero"
        MASSIMA_TESTO = ".massima-testo"

        # Related content
        RELAZIONI_CONTAINER = ".relazioni-container"
        RELAZIONE_ITEM = ".relazione-item"

        # Footnotes
        FOOTNOTES_CONTAINER = ".note-container"
        FOOTNOTE_ITEM = ".nota-item"

    # =========================================================================
    # EUR-Lex Selectors (for fallback scraping)
    # =========================================================================

    class EurLex:
        """Selectors for eur-lex.europa.eu (fallback scraping)."""

        # Document content
        DOCUMENT_BODY = "#document1"
        DOCUMENT_TITLE = ".doc-ti"
        DOCUMENT_PREAMBLE = ".eli-preamble"

        # Article content
        ARTICLE_CONTAINER = ".eli-subdivision"
        ARTICLE_NUMBER = ".eli-title"
        ARTICLE_TEXT = ".eli-content"

        # Metadata
        CELEX_ID = ".celex-number"
        IN_FORCE_STATUS = ".in-force"
        PUBLICATION_DATE = ".pub-date"

        # Navigation
        TOC_CONTAINER = "#toc"
        TOC_ITEM = ".toc-item"

    # =========================================================================
    # URL Patterns
    # =========================================================================

    class URLs:
        """URL templates for different sources."""

        # Normattiva
        NORMATTIVA_BASE = "https://www.normattiva.it"
        NORMATTIVA_URN = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:{urn}"
        NORMATTIVA_ARTICLE = (
            "https://www.normattiva.it/uri-res/N2Ls?"
            "urn:nir:{act_type}:{date};{number}~art{article}"
        )
        NORMATTIVA_SEARCH = "https://www.normattiva.it/ricerca/semplice"

        # Brocardi
        BROCARDI_BASE = "https://www.brocardi.it"
        BROCARDI_SEARCH = "https://www.brocardi.it/cerca?q={query}"

        # EUR-Lex
        EURLEX_BASE = "https://eur-lex.europa.eu"
        EURLEX_CELEX = (
            "https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/"
            "?uri=CELEX:{celex}"
        )
        EURLEX_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"

        # Brocardi Codice Civile structure mapping
        # Article ranges -> (libro, titolo)
        # See: https://www.brocardi.it/codice-civile/
        CODICE_CIVILE_STRUCTURE = [
            # Libro Primo - Delle persone e della famiglia
            (1, 10, "libro-primo", "titolo-i"),  # Persone fisiche
            (11, 35, "libro-primo", "titolo-ii"),  # Persone giuridiche
            (36, 42, "libro-primo", "titolo-iii"),  # Domicilio e residenza
            (43, 78, "libro-primo", "titolo-iv"),  # Assenza e morte presunta
            (79, 142, "libro-primo", "titolo-v"),  # Parentela e affinità
            (143, 230, "libro-primo", "titolo-vi"),  # Del matrimonio
            (231, 314, "libro-primo", "titolo-vii"),  # Della filiazione
            (315, 342, "libro-primo", "titolo-viii"),  # Dell'adozione
            (343, 389, "libro-primo", "titolo-ix"),  # Della responsabilità genitoriale
            (390, 413, "libro-primo", "titolo-x"),  # Della tutela e dell'emancipazione
            (414, 432, "libro-primo", "titolo-xi"),  # Dell'affiliazione
            (433, 455, "libro-primo", "titolo-xii"),  # Degli alimenti
            # Libro Secondo - Delle successioni
            (456, 564, "libro-secondo", "titolo-i"),  # Successioni in generale
            (565, 586, "libro-secondo", "titolo-ii"),  # Delle successioni legittime
            (587, 712, "libro-secondo", "titolo-iii"),  # Delle successioni testamentarie
            (713, 768, "libro-secondo", "titolo-iv"),  # Della divisione
            (769, 809, "libro-secondo", "titolo-v"),  # Delle donazioni
            # Libro Terzo - Della proprietà
            (810, 831, "libro-terzo", "titolo-i"),  # Dei beni
            (832, 951, "libro-terzo", "titolo-ii"),  # Della proprietà
            (952, 1026, "libro-terzo", "titolo-iii"),  # Della superficie
            (1027, 1099, "libro-terzo", "titolo-iv"),  # Dell'enfiteusi
            (1100, 1139, "libro-terzo", "titolo-v"),  # Dell'usufrutto
            (1140, 1172, "libro-terzo", "titolo-vi"),  # Delle servitù
            # Libro Quarto - Delle obbligazioni
            (1173, 1320, "libro-quarto", "titolo-i"),  # Obbligazioni in generale
            (1321, 1469, "libro-quarto", "titolo-ii"),  # Dei contratti in generale
            (1470, 1547, "libro-quarto", "titolo-iii"),  # Dei singoli contratti
            (1548, 1654, "libro-quarto", "titolo-iv"),  # Locazione
            (1655, 1702, "libro-quarto", "titolo-v"),  # Appalto e trasporto
            (1703, 1765, "libro-quarto", "titolo-vi"),  # Mandato, agenzia
            (1766, 1860, "libro-quarto", "titolo-vii"),  # Deposito e sequestro
            (1861, 1935, "libro-quarto", "titolo-viii"),  # Società
            (1936, 1959, "libro-quarto", "titolo-viii"),  # Associazione in partecipazione
            (1960, 2027, "libro-quarto", "titolo-viii"),  # Mutuo, c/c, rendita
            (2028, 2042, "libro-quarto", "titolo-viii"),  # Fideiussione
            (2043, 2059, "libro-quarto", "titolo-ix"),  # Fatti illeciti
            # Libro Quinto - Del lavoro
            (2060, 2081, "libro-quinto", "titolo-i"),  # Disciplina delle attività
            (2082, 2221, "libro-quinto", "titolo-ii"),  # Del lavoro nell'impresa
            (2222, 2238, "libro-quinto", "titolo-iii"),  # Del lavoro autonomo
            (2239, 2246, "libro-quinto", "titolo-iv"),  # Del lavoro subordinato
            (2247, 2510, "libro-quinto", "titolo-v"),  # Delle società
            (2511, 2548, "libro-quinto", "titolo-vi"),  # Cooperative
            (2549, 2554, "libro-quinto", "titolo-vii"),  # Associazione in partecipazione
            (2555, 2574, "libro-quinto", "titolo-viii"),  # Dell'azienda
            (2575, 2594, "libro-quinto", "titolo-ix"),  # Diritti sulle opere
            (2595, 2642, "libro-quinto", "titolo-x"),  # Disciplina concorrenza
            # Libro Sesto - Della tutela dei diritti
            (2643, 2696, "libro-sesto", "titolo-i"),  # Della trascrizione
            (2697, 2739, "libro-sesto", "titolo-ii"),  # Delle prove
            (2740, 2744, "libro-sesto", "titolo-iii"),  # Responsabilità patrimoniale
            (2745, 2783, "libro-sesto", "titolo-iv"),  # Cause di prelazione
            (2784, 2807, "libro-sesto", "titolo-iv"),  # Del pegno
            (2808, 2899, "libro-sesto", "titolo-iv"),  # Dell'ipoteca
            (2900, 2906, "libro-sesto", "titolo-v"),  # Mezzi di conservazione
            (2907, 2933, "libro-sesto", "titolo-vi"),  # Esecuzione forzata
            (2934, 2969, "libro-sesto", "titolo-vii"),  # Prescrizione e decadenza
        ]

        # Codice Penale structure mapping
        CODICE_PENALE_STRUCTURE = [
            # Libro Primo - Dei reati in generale
            (1, 16, "libro-primo", "titolo-i"),  # Della legge penale
            (17, 38, "libro-primo", "titolo-ii"),  # Delle pene
            (39, 58, "libro-primo", "titolo-iii"),  # Del reato
            (59, 84, "libro-primo", "titolo-iv"),  # Del reo e della persona offesa
            (85, 98, "libro-primo", "titolo-v"),  # Modificazione, applicazione
            (99, 108, "libro-primo", "titolo-vi"),  # Estinzione reato e pena
            (109, 126, "libro-primo", "titolo-vii"),  # Misure di sicurezza
            # Libro Secondo - Dei delitti in particolare
            (241, 313, "libro-secondo", "titolo-i"),  # Delitti contro la personalità
            (314, 360, "libro-secondo", "titolo-ii"),  # Delitti contro P.A.
            (361, 401, "libro-secondo", "titolo-iii"),  # Delitti contro amministrazione
            (402, 413, "libro-secondo", "titolo-iv"),  # Delitti contro sentimento religioso
            (414, 421, "libro-secondo", "titolo-v"),  # Delitti contro ordine pubblico
            (422, 452, "libro-secondo", "titolo-vi"),  # Delitti contro incolumità pubblica
            (453, 475, "libro-secondo", "titolo-vii"),  # Delitti contro fede pubblica
            (476, 498, "libro-secondo", "titolo-viii"),  # Delitti contro economia pubblica
            (499, 518, "libro-secondo", "titolo-ix"),  # Delitti contro moralità pubblica
            (519, 574, "libro-secondo", "titolo-xi"),  # Delitti contro la famiglia
            (575, 623, "libro-secondo", "titolo-xii"),  # Delitti contro la persona
            (624, 649, "libro-secondo", "titolo-xiii"),  # Delitti contro il patrimonio
            # Libro Terzo - Delle contravvenzioni
            (650, 659, "libro-terzo", "titolo-i"),  # Contravvenzioni di polizia
            (660, 678, "libro-terzo", "titolo-ii"),  # Contravvenzioni concernenti attività
        ]

        @classmethod
        def _get_brocardi_path(cls, article_num: int, structure: list) -> tuple[str, str] | None:
            """Find libro and titolo for an article number."""
            for start, end, libro, titolo in structure:
                if start <= article_num <= end:
                    return libro, titolo
            return None

        @classmethod
        def brocardi_url(cls, act_type: str, article: str) -> str:
            """Build Brocardi URL for an article.

            Brocardi uses hierarchical URLs with libro/titolo:
            https://www.brocardi.it/codice-civile/libro-quarto/titolo-ix/art2043.html
            """
            act_type_lower = act_type.lower()

            # Extract numeric part of article (e.g., "2043" from "2043" or "2043-bis")
            import re
            article_match = re.match(r"(\d+)", str(article))
            if not article_match:
                return cls.BROCARDI_SEARCH.format(query=f"{act_type}+art+{article}")

            article_num = int(article_match.group(1))

            # Determine which code and structure to use
            if act_type_lower in ("codice civile", "cc"):
                code_path = "codice-civile"
                path_info = cls._get_brocardi_path(article_num, cls.CODICE_CIVILE_STRUCTURE)
            elif act_type_lower in ("codice penale", "cp"):
                code_path = "codice-penale"
                path_info = cls._get_brocardi_path(article_num, cls.CODICE_PENALE_STRUCTURE)
            else:
                # For other codes, use search
                return cls.BROCARDI_SEARCH.format(query=f"{act_type}+art+{article}")

            if path_info:
                libro, titolo = path_info
                return f"{cls.BROCARDI_BASE}/{code_path}/{libro}/{titolo}/art{article}.html"
            else:
                # Fallback to search if article not in known ranges
                return cls.BROCARDI_SEARCH.format(query=f"{act_type}+art+{article}")

        @classmethod
        def eurlex_celex_url(cls, celex: str, lang: str = "IT") -> str:
            """Build EUR-Lex URL for a CELEX ID."""
            return cls.EURLEX_CELEX.format(celex=celex, lang=lang)
