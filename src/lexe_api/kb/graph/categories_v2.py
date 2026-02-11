"""
Category Definitions for Category Graph v2.4

Three-Axis Taxonomy:
- Axis A: Materia (Subject Matter) - 6 values
- Axis B: Natura (Legal Nature) - 2 values
- Axis C: Ambito (Procedural Scope) - 4 values (only if PROCESSUALE)

Plus Topic L2 for granular classification (with abstain option).
"""

from dataclasses import dataclass


@dataclass
class CategoryDef:
    """Definition of a category with metadata."""

    id: str
    name: str
    description: str
    keywords: list[str]
    parent_id: str | None = None


# =============================================================================
# AXIS A: MATERIA (Subject Matter) - 6 values
# =============================================================================

MATERIE: dict[str, CategoryDef] = {
    "CIVILE": CategoryDef(
        id="CIVILE",
        name="Diritto Civile",
        description="Obbligazioni, contratti, proprietà, famiglia, successioni",
        keywords=[
            "contratto",
            "obbligazione",
            "responsabilità civile",
            "proprietà",
            "possesso",
            "servitù",
            "usufrutto",
            "locazione",
            "compravendita",
            "donazione",
            "successione",
            "testamento",
            "famiglia",
            "matrimonio",
            "separazione",
            "divorzio",
            "affidamento",
            "mantenimento",
            "alimenti",
            "risarcimento",
            "danno",
            "inadempimento",
            "risoluzione",
            "rescissione",
        ],
    ),
    "PENALE": CategoryDef(
        id="PENALE",
        name="Diritto Penale",
        description="Reati, pene, circostanze, imputabilità",
        keywords=[
            "reato",
            "delitto",
            "contravvenzione",
            "pena",
            "reclusione",
            "arresto",
            "multa",
            "ammenda",
            "dolo",
            "colpa",
            "preterintenzione",
            "tentativo",
            "concorso",
            "circostanze",
            "aggravanti",
            "attenuanti",
            "imputabilità",
            "prescrizione",
            "estinzione",
            "sospensione condizionale",
            "omicidio",
            "lesioni",
            "furto",
            "rapina",
            "truffa",
            "appropriazione",
        ],
    ),
    "LAVORO": CategoryDef(
        id="LAVORO",
        name="Diritto del Lavoro",
        description="Rapporto di lavoro, licenziamento, previdenza",
        keywords=[
            "lavoro",
            "lavoratore",
            "datore",
            "licenziamento",
            "dimissioni",
            "retribuzione",
            "tfr",
            "ferie",
            "permessi",
            "straordinario",
            "contratto collettivo",
            "sindacato",
            "sciopero",
            "inps",
            "inail",
            "previdenza",
            "pensione",
            "contributi",
            "subordinazione",
            "parasubordinazione",
            "somministrazione",
            "appalto",
            "distacco",
            "trasferimento d'azienda",
        ],
    ),
    "TRIBUTARIO": CategoryDef(
        id="TRIBUTARIO",
        name="Diritto Tributario",
        description="Imposte, accertamento, riscossione, contenzioso tributario",
        keywords=[
            "imposta",
            "tributo",
            "iva",
            "irpef",
            "ires",
            "irap",
            "imu",
            "accertamento",
            "avviso",
            "cartella esattoriale",
            "riscossione",
            "agenzia delle entrate",
            "equitalia",
            "agenzia riscossione",
            "evasione",
            "elusione",
            "sanzioni tributarie",
            "contenzioso tributario",
            "commissione tributaria",
            "ricorso tributario",
            "studi di settore",
        ],
    ),
    "AMMINISTRATIVO": CategoryDef(
        id="AMMINISTRATIVO",
        name="Diritto Amministrativo",
        description="PA, atto amministrativo, appalti, urbanistica",
        keywords=[
            "pubblica amministrazione",
            "atto amministrativo",
            "provvedimento",
            "autorizzazione",
            "concessione",
            "permesso di costruire",
            "scia",
            "silenzio assenso",
            "accesso agli atti",
            "trasparenza",
            "appalto pubblico",
            "gara",
            "bando",
            "aggiudicazione",
            "urbanistica",
            "edilizia",
            "espropriazione per pubblica utilità",
            "tar",
            "consiglio di stato",
            "giurisdizione amministrativa",
            "risarcimento danni da pa",
        ],
    ),
    "CRISI": CategoryDef(
        id="CRISI",
        name="Crisi d'Impresa e Insolvenza",
        description="Fallimento, concordato, liquidazione giudiziale",
        keywords=[
            "fallimento",
            "liquidazione giudiziale",
            "concordato preventivo",
            "concordato fallimentare",
            "amministrazione straordinaria",
            "stato passivo",
            "ammissione al passivo",
            "insinuazione",
            "revocatoria fallimentare",
            "azione revocatoria",
            "curatore",
            "giudice delegato",
            "comitato creditori",
            "piano attestato",
            "accordi di ristrutturazione",
            "sovraindebitamento",
            "esdebitazione",
        ],
    ),
}

# Ordered list for consistent iteration
MATERIA_ORDER = ["CIVILE", "PENALE", "LAVORO", "TRIBUTARIO", "AMMINISTRATIVO", "CRISI"]


# =============================================================================
# AXIS B: NATURA (Legal Nature) - 2 values
# =============================================================================

NATURE: dict[str, CategoryDef] = {
    "SOSTANZIALE": CategoryDef(
        id="SOSTANZIALE",
        name="Diritto Sostanziale",
        description="Diritti, obblighi, responsabilità, elementi costitutivi",
        keywords=[
            "diritto",
            "obbligo",
            "responsabilità",
            "elemento costitutivo",
            "presupposti",
            "requisiti",
            "fattispecie",
            "titolarità",
            "legittimazione",
            "danno",
            "nesso causale",
            "colpa",
            "dolo",
            "buona fede",
            "diligenza",
            "interpretazione",
            "qualificazione",
            "natura giuridica",
            "effetti",
        ],
    ),
    "PROCESSUALE": CategoryDef(
        id="PROCESSUALE",
        name="Diritto Processuale",
        description="Procedura, ammissibilità, competenza, termini, impugnazioni",
        keywords=[
            "procedimento",
            "processo",
            "giudizio",
            "competenza",
            "giurisdizione",
            "legittimazione processuale",
            "interesse ad agire",
            "ammissibilità",
            "procedibilità",
            "ricorso",
            "appello",
            "impugnazione",
            "termine",
            "decadenza",
            "preclusione",
            "notifica",
            "nullità",
            "sanatoria",
            "onere della prova",
            "prova",
            "istruttoria",
            "motivazione",
            "sentenza",
        ],
    ),
}

NATURA_ORDER = ["SOSTANZIALE", "PROCESSUALE"]


# =============================================================================
# AXIS C: AMBITO (Procedural Scope) - 4 values + UNKNOWN
# Only applicable when natura=PROCESSUALE
# =============================================================================

AMBITI: dict[str, CategoryDef] = {
    "GIUDIZIO": CategoryDef(
        id="GIUDIZIO",
        name="Giudizio di Cognizione",
        description="Istruttoria, prove, competenza, notifiche, contraddittorio",
        keywords=[
            "competenza",
            "giurisdizione",
            "notifica",
            "contraddittorio",
            "istruttoria",
            "prova",
            "onere della prova",
            "ctu",
            "testimone",
            "giudice istruttore",
            "udienza",
            "comparsa",
            "memoria",
            "termine",
            "decadenza",
            "preclusione",
            "costituzione",
            "contumacia",
            "nullità",
        ],
    ),
    "IMPUGNAZIONI": CategoryDef(
        id="IMPUGNAZIONI",
        name="Impugnazioni",
        description="Appello, ricorso per cassazione, revocazione, opposizione di terzo",
        keywords=[
            "impugnazione",
            "appello",
            "ricorso",
            "cassazione",
            "revocazione",
            "opposizione di terzo",
            "termine breve",
            "termine lungo",
            "motivi",
            "violazione di legge",
            "vizio di motivazione",
            "error in iudicando",
            "error in procedendo",
            "inammissibilità",
            "improcedibilità",
            "controricorso",
            "ricorso incidentale",
            "rinuncia",
            "desistenza",
        ],
    ),
    "ESECUZIONE": CategoryDef(
        id="ESECUZIONE",
        name="Esecuzione Forzata",
        description="Pignoramento, espropriazione, precetto, opposizioni esecutive",
        keywords=[
            "esecuzione",
            "titolo esecutivo",
            "precetto",
            "pignoramento",
            "espropriazione",
            "vendita forzata",
            "assegnazione",
            "distribuzione",
            "opposizione all'esecuzione",
            "opposizione agli atti esecutivi",
            "sospensione dell'esecuzione",
            "terzo pignorato",
            "custodia",
        ],
    ),
    "MISURE": CategoryDef(
        id="MISURE",
        name="Misure Cautelari",
        description="Provvedimenti d'urgenza, sequestri, sospensione",
        keywords=[
            "cautelare",
            "cautela",
            "urgenza",
            "periculum in mora",
            "fumus boni iuris",
            "sequestro conservativo",
            "sequestro giudiziario",
            "provvedimento d'urgenza",
            "700",
            "inibitoria",
            "sospensione",
            "revoca",
            "modifica",
            "reclamo cautelare",
        ],
    ),
    "UNKNOWN": CategoryDef(
        id="UNKNOWN",
        name="Ambito Non Determinato",
        description="Ambito processuale non classificabile con certezza",
        keywords=[],
    ),
}

AMBITO_ORDER = ["GIUDIZIO", "IMPUGNAZIONI", "ESECUZIONE", "MISURE", "UNKNOWN"]


# =============================================================================
# TOPIC L2 (Granular topics under each Materia)
# =============================================================================

TOPICS_L2: dict[str, dict[str, CategoryDef]] = {
    "CIVILE": {
        "CIVILE_OBBLIGAZIONI": CategoryDef(
            id="CIVILE_OBBLIGAZIONI",
            name="Obbligazioni e Contratti",
            description="Obbligazioni, contratti tipici e atipici, inadempimento",
            keywords=["obbligazione", "contratto", "inadempimento", "risoluzione", "rescissione"],
            parent_id="CIVILE",
        ),
        "CIVILE_RESP_CIVILE": CategoryDef(
            id="CIVILE_RESP_CIVILE",
            name="Responsabilità Civile",
            description="Responsabilità contrattuale ed extracontrattuale",
            keywords=["responsabilità", "risarcimento", "danno", "2043", "custodia"],
            parent_id="CIVILE",
        ),
        "CIVILE_PROPRIETA": CategoryDef(
            id="CIVILE_PROPRIETA",
            name="Proprietà e Diritti Reali",
            description="Proprietà, possesso, servitù, usufrutto",
            keywords=["proprietà", "possesso", "servitù", "usufrutto", "usucapione"],
            parent_id="CIVILE",
        ),
        "CIVILE_FAMIGLIA": CategoryDef(
            id="CIVILE_FAMIGLIA",
            name="Diritto di Famiglia",
            description="Matrimonio, separazione, divorzio, filiazione",
            keywords=["famiglia", "matrimonio", "separazione", "divorzio", "filiazione"],
            parent_id="CIVILE",
        ),
        "CIVILE_SUCCESSIONI": CategoryDef(
            id="CIVILE_SUCCESSIONI",
            name="Successioni",
            description="Successioni legittime e testamentarie, donazioni",
            keywords=["successione", "testamento", "eredità", "legato", "donazione"],
            parent_id="CIVILE",
        ),
        "CIVILE_SOCIETA": CategoryDef(
            id="CIVILE_SOCIETA",
            name="Diritto Societario",
            description="Società di persone e capitali, governance",
            keywords=["società", "srl", "spa", "socio", "assemblea", "amministratore"],
            parent_id="CIVILE",
        ),
        "CIVILE_ASSICURAZIONE": CategoryDef(
            id="CIVILE_ASSICURAZIONE",
            name="Assicurazioni",
            description="Contratto assicurativo, sinistri, indennizzi",
            keywords=["assicurazione", "polizza", "sinistro", "indennizzo", "premio"],
            parent_id="CIVILE",
        ),
        "CIVILE_LOCAZIONI": CategoryDef(
            id="CIVILE_LOCAZIONI",
            name="Locazioni",
            description="Locazione abitativa e commerciale",
            keywords=["locazione", "affitto", "conduttore", "locatore", "canone"],
            parent_id="CIVILE",
        ),
    },
    "PENALE": {
        "PENALE_PERSONA": CategoryDef(
            id="PENALE_PERSONA",
            name="Reati contro la Persona",
            description="Omicidio, lesioni, violenza, minaccia",
            keywords=["omicidio", "lesioni", "violenza", "minaccia", "stalking"],
            parent_id="PENALE",
        ),
        "PENALE_PATRIMONIO": CategoryDef(
            id="PENALE_PATRIMONIO",
            name="Reati contro il Patrimonio",
            description="Furto, rapina, truffa, appropriazione indebita",
            keywords=["furto", "rapina", "truffa", "appropriazione", "ricettazione"],
            parent_id="PENALE",
        ),
        "PENALE_PA": CategoryDef(
            id="PENALE_PA",
            name="Reati contro la PA",
            description="Corruzione, concussione, abuso d'ufficio",
            keywords=["corruzione", "concussione", "peculato", "abuso d'ufficio"],
            parent_id="PENALE",
        ),
        "PENALE_STUPEFACENTI": CategoryDef(
            id="PENALE_STUPEFACENTI",
            name="Stupefacenti",
            description="Reati in materia di stupefacenti",
            keywords=["stupefacenti", "droga", "spaccio", "detenzione", "dpr 309"],
            parent_id="PENALE",
        ),
    },
    "LAVORO": {
        "LAVORO_RAPPORTO": CategoryDef(
            id="LAVORO_RAPPORTO",
            name="Rapporto di Lavoro",
            description="Costituzione, svolgimento, tipologie contrattuali",
            keywords=["rapporto", "assunzione", "contratto", "subordinazione"],
            parent_id="LAVORO",
        ),
        "LAVORO_LICENZIAMENTO": CategoryDef(
            id="LAVORO_LICENZIAMENTO",
            name="Licenziamento",
            description="Licenziamento individuale e collettivo, tutele",
            keywords=["licenziamento", "giusta causa", "giustificato motivo", "reintegra"],
            parent_id="LAVORO",
        ),
        "LAVORO_PREVIDENZA": CategoryDef(
            id="LAVORO_PREVIDENZA",
            name="Previdenza e Assistenza",
            description="Pensioni, contributi, prestazioni INPS/INAIL",
            keywords=["pensione", "inps", "inail", "contributi", "invalidità"],
            parent_id="LAVORO",
        ),
    },
    "TRIBUTARIO": {
        "TRIB_ACCERTAMENTO": CategoryDef(
            id="TRIB_ACCERTAMENTO",
            name="Accertamento",
            description="Accertamento tributario, avvisi, studi di settore",
            keywords=["accertamento", "avviso", "studi di settore", "presunzioni"],
            parent_id="TRIBUTARIO",
        ),
        "TRIB_RISCOSSIONE": CategoryDef(
            id="TRIB_RISCOSSIONE",
            name="Riscossione",
            description="Cartelle, iscrizione a ruolo, fermo, ipoteca",
            keywords=["cartella", "riscossione", "ruolo", "fermo", "ipoteca"],
            parent_id="TRIBUTARIO",
        ),
        "TRIB_CONTENZIOSO": CategoryDef(
            id="TRIB_CONTENZIOSO",
            name="Contenzioso Tributario",
            description="Processo tributario, commissioni, ricorso",
            keywords=["ricorso tributario", "commissione tributaria", "dlgs 546"],
            parent_id="TRIBUTARIO",
        ),
    },
    "AMMINISTRATIVO": {
        "AMM_ATTO": CategoryDef(
            id="AMM_ATTO",
            name="Atto Amministrativo",
            description="Provvedimenti, validità, vizi, annullamento",
            keywords=["provvedimento", "atto", "vizio", "annullamento", "revoca"],
            parent_id="AMMINISTRATIVO",
        ),
        "AMM_APPALTI": CategoryDef(
            id="AMM_APPALTI",
            name="Appalti Pubblici",
            description="Gare, bandi, aggiudicazione, codice appalti",
            keywords=["appalto", "gara", "bando", "aggiudicazione", "offerta"],
            parent_id="AMMINISTRATIVO",
        ),
        "AMM_CONTENZIOSO": CategoryDef(
            id="AMM_CONTENZIOSO",
            name="Contenzioso Amministrativo",
            description="TAR, Consiglio di Stato, giurisdizione",
            keywords=["tar", "consiglio di stato", "ricorso", "giurisdizione"],
            parent_id="AMMINISTRATIVO",
        ),
    },
    "CRISI": {
        "FALL_FALLIMENTO": CategoryDef(
            id="FALL_FALLIMENTO",
            name="Fallimento/Liquidazione",
            description="Dichiarazione, effetti, procedimento",
            keywords=["fallimento", "liquidazione giudiziale", "sentenza dichiarativa"],
            parent_id="CRISI",
        ),
        "FALL_STATO_PASSIVO": CategoryDef(
            id="FALL_STATO_PASSIVO",
            name="Stato Passivo",
            description="Domande di ammissione, verifiche, contestazioni",
            keywords=["passivo", "ammissione", "insinuazione", "privilegio"],
            parent_id="CRISI",
        ),
        "FALL_REVOCATORIA": CategoryDef(
            id="FALL_REVOCATORIA",
            name="Revocatoria Fallimentare",
            description="Azioni revocatorie, presupposti, effetti",
            keywords=["revocatoria", "azione revocatoria", "scientia decoctionis"],
            parent_id="CRISI",
        ),
        "FALL_CONCORDATO": CategoryDef(
            id="FALL_CONCORDATO",
            name="Concordato",
            description="Concordato preventivo e fallimentare",
            keywords=["concordato", "piano", "omologazione", "creditori"],
            parent_id="CRISI",
        ),
    },
}


def get_all_topic_l2_ids() -> list[str]:
    """Return flat list of all Topic L2 IDs."""
    result = []
    for materia_topics in TOPICS_L2.values():
        result.extend(materia_topics.keys())
    return result


def get_topics_for_materia(materia: str) -> dict[str, CategoryDef]:
    """Return Topic L2 definitions for a given Materia."""
    return TOPICS_L2.get(materia, {})


def get_all_materia_keywords() -> dict[str, set[str]]:
    """Return materia -> keywords mapping for classifier."""
    return {materia: set(cat.keywords) for materia, cat in MATERIE.items()}


def get_all_natura_keywords() -> dict[str, set[str]]:
    """Return natura -> keywords mapping for classifier."""
    return {natura: set(cat.keywords) for natura, cat in NATURE.items()}


def get_all_ambito_keywords() -> dict[str, set[str]]:
    """Return ambito -> keywords mapping for classifier (excluding UNKNOWN)."""
    return {ambito: set(cat.keywords) for ambito, cat in AMBITI.items() if ambito != "UNKNOWN"}
