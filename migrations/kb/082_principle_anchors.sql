-- Migration 082: NOVA RATIO — Principle Anchors for Zero Norm Engine
BEGIN;

CREATE TABLE IF NOT EXISTS kb.principle_anchors (
    id VARCHAR(80) PRIMARY KEY,
    principle_name TEXT NOT NULL,
    principle_text TEXT NOT NULL,
    source_norm_id VARCHAR(50),
    source_type VARCHAR(30) NOT NULL,  -- constitutional, general_clause, eu_principle
    domain TEXT[],
    keywords TEXT[],
    embedding vector(1536),
    citation_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pa_source_type ON kb.principle_anchors(source_type);
CREATE INDEX IF NOT EXISTS idx_pa_domain ON kb.principle_anchors USING gin(domain);
CREATE INDEX IF NOT EXISTS idx_pa_keywords ON kb.principle_anchors USING gin(keywords);
CREATE INDEX IF NOT EXISTS idx_pa_embedding ON kb.principle_anchors
    USING hnsw(embedding vector_cosine_ops) WHERE embedding IS NOT NULL;

-- Link principio -> massima (supporto giurisprudenziale)
CREATE TABLE IF NOT EXISTS kb.principle_massima_links (
    principle_id VARCHAR(80) NOT NULL REFERENCES kb.principle_anchors(id) ON DELETE CASCADE,
    massima_id UUID NOT NULL,
    relevance_score FLOAT NOT NULL DEFAULT 0.5,
    link_type VARCHAR(30) DEFAULT 'embodies',
    run_id INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (principle_id, massima_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_pml_principle ON kb.principle_massima_links(principle_id);
CREATE INDEX IF NOT EXISTS idx_pml_massima ON kb.principle_massima_links(massima_id);

-- Seed: 24 principi fondamentali del diritto italiano
INSERT INTO kb.principle_anchors (id, principle_name, principle_text, source_norm_id, source_type, domain, keywords) VALUES
-- COSTITUZIONALI
('COST:1:democrazia', 'Principio democratico', 'L''Italia e'' una Repubblica democratica, fondata sul lavoro.', 'COST:1', 'constitutional', ARRAY['pubblico','costituzionale'], ARRAY['democrazia','lavoro','repubblica']),
('COST:2:solidarieta', 'Principio di solidarieta'' sociale', 'La Repubblica riconosce e garantisce i diritti inviolabili dell''uomo e richiede l''adempimento dei doveri inderogabili di solidarieta'' politica, economica e sociale.', 'COST:2', 'constitutional', ARRAY['civile','penale','amministrativo'], ARRAY['solidarieta','diritti inviolabili','doveri inderogabili']),
('COST:3:uguaglianza', 'Principio di uguaglianza', 'Tutti i cittadini hanno pari dignita'' sociale e sono eguali davanti alla legge.', 'COST:3', 'constitutional', ARRAY['civile','penale','amministrativo','lavoro'], ARRAY['uguaglianza','pari dignita','discriminazione','ragionevolezza']),
('COST:13:liberta_personale', 'Principio di liberta'' personale', 'La liberta'' personale e'' inviolabile.', 'COST:13', 'constitutional', ARRAY['penale','costituzionale'], ARRAY['liberta personale','habeas corpus','inviolabilita']),
('COST:24:diritto_difesa', 'Diritto di difesa', 'Tutti possono agire in giudizio per la tutela dei propri diritti e interessi legittimi.', 'COST:24', 'constitutional', ARRAY['processuale','civile','penale','amministrativo'], ARRAY['diritto difesa','azione giudizio','contraddittorio']),
('COST:25:giudice_naturale', 'Principio del giudice naturale', 'Nessuno puo'' essere distolto dal giudice naturale precostituito per legge.', 'COST:25', 'constitutional', ARRAY['processuale','penale'], ARRAY['giudice naturale','legalita penale','irretroattivita']),
('COST:27:personalita_pena', 'Principio di personalita'' della pena', 'La responsabilita'' penale e'' personale.', 'COST:27', 'constitutional', ARRAY['penale'], ARRAY['personalita pena','responsabilita personale','rieducazione']),
('COST:41:iniziativa_economica', 'Liberta'' di iniziativa economica', 'L''iniziativa economica privata e'' libera.', 'COST:41', 'constitutional', ARRAY['civile','societario','tributario'], ARRAY['iniziativa economica','utilita sociale','liberta impresa']),
('COST:42:proprieta', 'Diritto di proprieta''', 'La proprieta'' e'' riconosciuta e garantita dalla legge.', 'COST:42', 'constitutional', ARRAY['civile','tributario'], ARRAY['proprieta','espropriazione','funzione sociale']),
('COST:97:buon_andamento', 'Principio di buon andamento e imparzialita''', 'I pubblici uffici sono organizzati secondo disposizioni di legge, in modo che siano assicurati il buon andamento e l''imparzialita'' dell''amministrazione.', 'COST:97', 'constitutional', ARRAY['amministrativo'], ARRAY['buon andamento','imparzialita','amministrazione']),
-- CLAUSOLE GENERALI C.C.
('CC:1175:buona_fede', 'Principio di buona fede (correttezza)', 'Il debitore e il creditore devono comportarsi secondo le regole della correttezza.', 'CC:1175', 'general_clause', ARRAY['civile','obbligazioni'], ARRAY['buona fede','correttezza','lealta']),
('CC:1337:buona_fede_trattative', 'Buona fede nelle trattative', 'Le parti, nello svolgimento delle trattative e nella formazione del contratto, devono comportarsi secondo buona fede.', 'CC:1337', 'general_clause', ARRAY['civile','contratti'], ARRAY['buona fede','trattative','responsabilita precontrattuale']),
('CC:1375:buona_fede_esecuzione', 'Buona fede nell''esecuzione del contratto', 'Il contratto deve essere eseguito secondo buona fede.', 'CC:1375', 'general_clause', ARRAY['civile','contratti'], ARRAY['buona fede','esecuzione contratto','integrazione']),
('CC:2043:neminem_laedere', 'Principio del neminem laedere', 'Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto a risarcire il danno.', 'CC:2043', 'general_clause', ARRAY['civile','responsabilita'], ARRAY['danno ingiusto','responsabilita extracontrattuale','neminem laedere','risarcimento']),
('CC:2059:danno_non_patrimoniale', 'Danno non patrimoniale', 'Il danno non patrimoniale deve essere risarcito solo nei casi determinati dalla legge.', 'CC:2059', 'general_clause', ARRAY['civile','responsabilita'], ARRAY['danno non patrimoniale','danno morale','danno biologico']),
('CC:1218:responsabilita_debitore', 'Responsabilita'' del debitore', 'Il debitore che non esegue esattamente la prestazione dovuta e'' tenuto al risarcimento del danno.', 'CC:1218', 'general_clause', ARRAY['civile','obbligazioni'], ARRAY['inadempimento','responsabilita contrattuale','risarcimento']),
('CC:1372:pacta_sunt_servanda', 'Pacta sunt servanda', 'Il contratto ha forza di legge tra le parti.', 'CC:1372', 'general_clause', ARRAY['civile','contratti'], ARRAY['pacta sunt servanda','forza di legge','contratto']),
('CC:1374:equita', 'Principio di equita''', 'Il contratto obbliga le parti non solo a quanto e'' nel medesimo espresso, ma anche a tutte le conseguenze che ne derivano secondo la legge, o, in mancanza, secondo gli usi e l''equita''.', 'CC:1374', 'general_clause', ARRAY['civile','contratti'], ARRAY['equita','integrazione contratto','buona fede']),
-- PENALE
('CP:2:favor_rei', 'Principio del favor rei', 'Nessuno puo'' essere punito per un fatto che, secondo la legge del tempo in cui fu commesso, non costituiva reato.', 'CP:2', 'general_clause', ARRAY['penale'], ARRAY['favor rei','irretroattivita','successione leggi']),
('CPP:649:ne_bis_in_idem', 'Principio del ne bis in idem', 'L''imputato prosciolto o condannato con sentenza o decreto penale divenuti irrevocabili non puo'' essere di nuovo sottoposto a procedimento penale per il medesimo fatto.', 'CPP:649', 'general_clause', ARRAY['penale','processuale'], ARRAY['ne bis in idem','giudicato','cosa giudicata']),
-- EU
('EU:PROPORZIONALITA', 'Principio di proporzionalita'' UE', 'Il contenuto e la forma dell''azione dell''Unione si limitano a quanto necessario per il conseguimento degli obiettivi dei trattati.', NULL, 'eu_principle', ARRAY['amministrativo','civile','eu'], ARRAY['proporzionalita','necessita','adeguatezza']),
('EU:PRECAUZIONE', 'Principio di precauzione', 'In caso di rischio di danno grave o irreversibile, l''assenza di certezza scientifica non deve servire da pretesto per rinviare l''adozione di misure.', NULL, 'eu_principle', ARRAY['amministrativo','ambiente','consumatore'], ARRAY['precauzione','rischio','incertezza scientifica']),
('EU:LEGITTIMO_AFFIDAMENTO', 'Principio del legittimo affidamento', 'I privati hanno diritto di fare affidamento sulle situazioni giuridiche create dall''azione dell''amministrazione.', NULL, 'eu_principle', ARRAY['amministrativo','eu'], ARRAY['legittimo affidamento','affidamento','tutela aspettativa']),
('EU:NON_DISCRIMINAZIONE', 'Principio di non discriminazione', 'E'' vietata qualsiasi discriminazione fondata sulla nazionalita''.', NULL, 'eu_principle', ARRAY['civile','lavoro','eu'], ARRAY['non discriminazione','parita trattamento','nazionalita'])
ON CONFLICT (id) DO UPDATE SET
    principle_text = EXCLUDED.principle_text,
    keywords = EXCLUDED.keywords,
    updated_at = NOW();

COMMIT;
