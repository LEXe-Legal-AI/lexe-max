-- ============================================================================
-- KB Normativa V3 Seed Data
-- Migration: 052_seed_data.sql
-- Date: 2026-02-06
-- Description: Seed data for 69 Italian legal documents
-- ============================================================================

BEGIN;

-- ============================================
-- SEED: SOURCE_TYPE (gerarchico)
-- ============================================
INSERT INTO kb.source_type(code, name, parent_id) VALUES
('COSTITUZIONE', 'Costituzione', NULL),
('CODICE', 'Codici', NULL),
('CODICE_PROCEDURA', 'Codici di Procedura', (SELECT id FROM kb.source_type WHERE code='CODICE')),
('TESTO_UNICO', 'Testi Unici', NULL),
('LEGGE', 'Leggi', NULL),
('DECRETO', 'Decreti', NULL),
('REGOLAMENTO', 'Regolamenti', NULL),
('UE', 'Normativa UE', NULL),
('INTERNAZIONALE', 'Normativa Internazionale', NULL),
('DEONTOLOGIA', 'Deontologia Professionale', NULL)
ON CONFLICT (code) DO NOTHING;

-- ============================================
-- SEED: TOPIC (gerarchico)
-- ============================================
INSERT INTO kb.topic(slug, name, parent_id) VALUES
-- Level 1
('civile', 'Diritto Civile', NULL),
('penale', 'Diritto Penale', NULL),
('amministrativo', 'Diritto Amministrativo', NULL),
('commerciale', 'Diritto Commerciale', NULL),
('lavoro', 'Diritto del Lavoro', NULL),
('tributario', 'Diritto Tributario', NULL),
('ambiente', 'Diritto Ambientale', NULL),
('comunicazioni', 'Comunicazioni', NULL),
('internazionale', 'Diritto Internazionale', NULL),
('professioni', 'Professioni', NULL),
-- Level 2
('civile-procedura', 'Procedura Civile', (SELECT id FROM kb.topic WHERE slug='civile')),
('civile-obbligazioni', 'Obbligazioni', (SELECT id FROM kb.topic WHERE slug='civile')),
('civile-famiglia', 'Famiglia e Successioni', (SELECT id FROM kb.topic WHERE slug='civile')),
('civile-privacy', 'Privacy e Dati', (SELECT id FROM kb.topic WHERE slug='civile')),
('penale-procedura', 'Procedura Penale', (SELECT id FROM kb.topic WHERE slug='penale')),
('penale-esecuzione', 'Esecuzione Penale', (SELECT id FROM kb.topic WHERE slug='penale')),
('commerciale-crisi', 'Crisi d''Impresa', (SELECT id FROM kb.topic WHERE slug='commerciale')),
('commerciale-bancario', 'Bancario e Finanziario', (SELECT id FROM kb.topic WHERE slug='commerciale')),
('amministrativo-appalti', 'Appalti', (SELECT id FROM kb.topic WHERE slug='amministrativo')),
('amministrativo-enti', 'Enti Locali', (SELECT id FROM kb.topic WHERE slug='amministrativo')),
('lavoro-sicurezza', 'Sicurezza sul Lavoro', (SELECT id FROM kb.topic WHERE slug='lavoro'))
ON CONFLICT (slug) DO NOTHING;

-- ============================================
-- SEED: NIR_MAPPING (URN:NIR verificati)
-- normattiva_resolver = NULL qui, verrà generato con UPDATE dopo
-- ============================================
INSERT INTO kb.nir_mapping(code, nir_base, canonical_title) VALUES
-- Costituzione e 4 codici (URN verificati)
('COST', 'urn:nir:stato:costituzione:1947-12-27', 'Costituzione della Repubblica Italiana'),
('CC', 'urn:nir:stato:regio.decreto:1942-03-16;262', 'Codice Civile'),
('CP', 'urn:nir:stato:regio.decreto:1930-10-19;1398', 'Codice Penale'),
('CPC', 'urn:nir:stato:regio.decreto:1940-10-28;1443', 'Codice di Procedura Civile'),
('CPP', 'urn:nir:presidente.repubblica:decreto:1988-09-22;447', 'Codice di Procedura Penale'),
-- Commerciale
('CCII', 'urn:nir:stato:decreto.legislativo:2019-01-12;14', 'Codice della Crisi d''Impresa'),
('CPI', 'urn:nir:stato:decreto.legislativo:2005-02-10;30', 'Codice Proprietà Industriale'),
('LF', 'urn:nir:stato:regio.decreto:1942-03-16;267', 'Legge Fallimentare'),
('TUB', 'urn:nir:stato:decreto.legislativo:1993-09-01;385', 'Testo Unico Bancario'),
('TUF', 'urn:nir:stato:decreto.legislativo:1998-02-24;58', 'Testo Unico Finanza'),
-- Civile
('CCONS', 'urn:nir:stato:decreto.legislativo:2005-09-06;206', 'Codice del Consumo'),
('CTUR', 'urn:nir:stato:decreto.legislativo:2011-05-23;79', 'Codice del Turismo'),
('CGS', 'urn:nir:figc:regolamento:2019', 'Codice Giustizia Sportiva'),
('CPRIV', 'urn:nir:stato:decreto.legislativo:2003-06-30;196', 'Codice Privacy'),
('CTS', 'urn:nir:stato:decreto.legislativo:2017-07-03;117', 'Codice Terzo Settore'),
('GDPR', 'urn:nir:unione.europea:regolamento:2016-04-27;2016-679', 'GDPR'),
('LDA', 'urn:nir:stato:legge:1941-04-22;633', 'Legge Diritto Autore'),
('LDIV', 'urn:nir:stato:legge:1970-12-01;898', 'Legge Divorzio'),
('LLOC', 'urn:nir:stato:legge:1998-12-09;431', 'Legge Locazioni'),
('DMED', 'urn:nir:stato:decreto.legislativo:2010-03-04;28', 'Mediazione Civile'),
('TUSG', 'urn:nir:stato:decreto.presidente.repubblica:2002-05-30;115', 'TU Spese Giustizia'),
-- Amministrativo
('CAD', 'urn:nir:stato:decreto.legislativo:2005-03-07;82', 'Codice Amministrazione Digitale'),
('CAM', 'urn:nir:stato:decreto.legislativo:2011-09-06;159', 'Codice Antimafia'),
('CAPP', 'urn:nir:stato:decreto.legislativo:2023-03-31;36', 'Codice Appalti'),
('CGC', 'urn:nir:stato:decreto.legislativo:2016-08-26;174', 'Codice Giustizia Contabile'),
('CMED', 'urn:nir:stato:decreto.legislativo:2006-04-24;219', 'Codice Medicinali'),
('CPA', 'urn:nir:stato:decreto.legislativo:2010-07-02;104', 'Codice Processo Amministrativo'),
('LPA', 'urn:nir:stato:legge:1990-08-07;241', 'Legge Procedimento Amministrativo'),
('DLAS', 'urn:nir:stato:decreto.legislativo:2001-06-08;231', 'D.Lgs. 231/2001'),
('DSIC', 'urn:nir:stato:decreto.legge:2017-02-20;14', 'Sicurezza Urbana'),
('TUE', 'urn:nir:stato:decreto.presidente.repubblica:2001-06-06;380', 'TU Edilizia'),
('TUEL', 'urn:nir:stato:decreto.legislativo:2000-08-18;267', 'TU Enti Locali'),
('TUESP', 'urn:nir:stato:decreto.presidente.repubblica:2001-06-08;327', 'TU Espropriazioni'),
('TUI', 'urn:nir:stato:decreto.legislativo:1998-07-25;286', 'TU Immigrazione'),
('TUIST', 'urn:nir:stato:decreto.legislativo:1994-04-16;297', 'TU Istruzione'),
('TUPI', 'urn:nir:stato:decreto.legislativo:2001-03-30;165', 'TU Pubblico Impiego'),
('TUSP', 'urn:nir:stato:decreto.legislativo:2016-08-19;175', 'TU Società Partecipate'),
('TUDA', 'urn:nir:stato:decreto.presidente.repubblica:2000-12-28;445', 'TU Documentazione Amministrativa'),
-- Ambiente
('CAMB', 'urn:nir:stato:decreto.legislativo:2006-04-03;152', 'Codice Ambiente'),
('CBCP', 'urn:nir:stato:decreto.legislativo:2004-01-22;42', 'Codice Beni Culturali'),
('TUFOR', 'urn:nir:stato:decreto.legislativo:2018-04-03;34', 'TU Foreste'),
-- Circolazione
('CAP', 'urn:nir:stato:decreto.legislativo:2005-09-07;209', 'Codice Assicurazioni'),
('CDS', 'urn:nir:stato:decreto.legislativo:1992-04-30;285', 'Codice della Strada'),
('CND', 'urn:nir:stato:decreto.legislativo:2005-07-18;171', 'Codice Nautica'),
('REGCDS', 'urn:nir:stato:decreto.presidente.repubblica:1992-12-16;495', 'Regolamento CdS'),
-- Comunicazioni
('CCE', 'urn:nir:stato:decreto.legislativo:2003-08-01;259', 'Codice Comunicazioni Elettroniche'),
('TUSMAR', 'urn:nir:stato:decreto.legislativo:2005-07-31;177', 'TU Radiotelevisione'),
-- Fisco
('CPT', 'urn:nir:stato:decreto.legislativo:1992-12-31;546', 'Codice Processo Tributario'),
('LRT', 'urn:nir:stato:decreto.legislativo:2000-03-10;74', 'Legge Reati Tributari'),
('STATC', 'urn:nir:stato:legge:2000-07-27;212', 'Statuto Contribuente'),
('TUIR', 'urn:nir:stato:decreto.presidente.repubblica:1986-12-22;917', 'TU Imposte Redditi'),
('DIVA', 'urn:nir:stato:decreto.presidente.repubblica:1972-10-26;633', 'DPR IVA'),
-- Internazionale
('DUDU', 'urn:nir:onu:dichiarazione:1948-12-10', 'Dichiarazione Universale Diritti Uomo'),
('LDIP', 'urn:nir:stato:legge:1995-05-31;218', 'Legge DIP'),
-- Lavoro
('CPO', 'urn:nir:stato:decreto.legislativo:2006-04-11;198', 'Codice Pari Opportunità'),
('LSSP', 'urn:nir:stato:legge:1990-06-12;146', 'Legge Sciopero SP'),
('DBIAGI', 'urn:nir:stato:decreto.legislativo:2003-09-10;276', 'Riforma Biagi'),
('LFORN', 'urn:nir:stato:legge:2012-06-28;92', 'Riforma Fornero'),
('SL', 'urn:nir:stato:legge:1970-05-20;300', 'Statuto Lavoratori'),
('TUPC', 'urn:nir:stato:decreto.legislativo:2005-12-05;252', 'TU Previdenza Complementare'),
('TUMP', 'urn:nir:stato:decreto.legislativo:2001-03-26;151', 'TU Maternità Paternità'),
('TUSL', 'urn:nir:stato:decreto.legislativo:2008-04-09;81', 'TU Sicurezza Lavoro'),
-- Penale
('CPPM', 'urn:nir:stato:decreto.presidente.repubblica:1988-09-22;448', 'CPP Minorile'),
('LDEP', 'urn:nir:stato:decreto.legislativo:2016-01-15;8', 'Legge Depenalizzazione'),
('OP', 'urn:nir:stato:legge:1975-07-26;354', 'Ordinamento Penitenziario'),
('TUCG', 'urn:nir:stato:decreto.presidente.repubblica:2002-11-14;313', 'TU Casellario Giudiziale'),
('TUS', 'urn:nir:stato:decreto.presidente.repubblica:1990-10-09;309', 'TU Stupefacenti'),
-- Professioni
('CDF', 'urn:nir:cnf:codice:2014-01-31', 'Codice Deontologico Forense'),
('LPF', 'urn:nir:stato:legge:2012-12-31;247', 'Legge Professionale Forense')
ON CONFLICT (code) DO NOTHING;

-- SET source_system_id per gold mappings (NORMATTIVA_ONLINE)
UPDATE kb.nir_mapping SET source_system_id = (SELECT id FROM kb.source_system WHERE code='NORMATTIVA_ONLINE')
WHERE code IN ('COST','CC','CP','CPC','CPP','CCII','TUB','TUF');

-- ============================================
-- SEED: WORK (69 documenti)
-- ============================================
INSERT INTO kb.work(code, title, notes) VALUES
-- Costituzione e 4 codici
('COST', 'Costituzione della Repubblica Italiana', '139 articoli'),
('CC', 'Codice Civile', '2969 articoli'),
('CP', 'Codice Penale', '734 articoli'),
('CPC', 'Codice di Procedura Civile', '840 articoli'),
('CPP', 'Codice di Procedura Penale', '746 articoli'),
-- Commerciale
('CCII', 'Codice della Crisi d''Impresa e dell''Insolvenza', 'D.Lgs. 14/2019, 391 articoli'),
('CPI', 'Codice della Proprietà Industriale', 'D.Lgs. 30/2005, 245 articoli'),
('LF', 'Legge Fallimentare', 'R.D. 267/1942 - abrogata, 264 articoli'),
('TUB', 'Testo Unico Bancario', 'D.Lgs. 385/1993, 162 articoli'),
('TUF', 'Testo Unico della Finanza', 'D.Lgs. 58/1998, 214 articoli'),
-- Civile
('CCONS', 'Codice del Consumo', 'D.Lgs. 206/2005, 170 articoli'),
('CTUR', 'Codice del Turismo', 'D.Lgs. 79/2011, 74 articoli'),
('CGS', 'Codice Giustizia Sportiva', 'FIGC, 152 articoli'),
('CPRIV', 'Codice Privacy', 'D.Lgs. 196/2003, 186 articoli'),
('CTS', 'Codice del Terzo Settore', 'D.Lgs. 117/2017, 104 articoli'),
('GDPR', 'Regolamento Generale Protezione Dati', 'Reg. UE 2016/679, 99 articoli'),
('LDA', 'Legge Diritto d''Autore', 'L. 633/1941, 206 articoli'),
('LDIV', 'Legge Divorzio', 'L. 898/1970, 12 articoli'),
('LLOC', 'Legge Locazioni Abitative', 'L. 431/1998, 14 articoli'),
('DMED', 'Mediazione Civile', 'D.Lgs. 28/2010, 24 articoli'),
('TUSG', 'Testo Unico Spese di Giustizia', 'D.P.R. 115/2002, 302 articoli'),
-- Amministrativo
('CAD', 'Codice Amministrazione Digitale', 'D.Lgs. 82/2005, 92 articoli'),
('CAM', 'Codice Antimafia', 'D.Lgs. 159/2011, 144 articoli'),
('CAPP', 'Codice degli Appalti', 'D.Lgs. 36/2023, 229 articoli'),
('CGC', 'Codice Giustizia Contabile', 'D.Lgs. 174/2016, 219 articoli'),
('CMED', 'Codice dei Medicinali', 'D.Lgs. 219/2006, 158 articoli'),
('CPA', 'Codice Processo Amministrativo', 'D.Lgs. 104/2010, 138 articoli'),
('LPA', 'Legge Procedimento Amministrativo', 'L. 241/1990, 31 articoli'),
('DLAS', 'Responsabilità Amministrativa Società', 'D.Lgs. 231/2001, 83 articoli'),
('DSIC', 'Sicurezza Urbana', 'D.L. 14/2017, 17 articoli'),
('TUE', 'Testo Unico Edilizia', 'D.P.R. 380/2001, 138 articoli'),
('TUEL', 'Testo Unico Enti Locali', 'D.Lgs. 267/2000, 275 articoli'),
('TUESP', 'Testo Unico Espropriazioni', 'D.P.R. 327/2001, 59 articoli'),
('TUI', 'Testo Unico Immigrazione', 'D.Lgs. 286/1998, 46 articoli'),
('TUIST', 'Testo Unico Istruzione', 'D.Lgs. 297/1994, 676 articoli'),
('TUPI', 'Testo Unico Pubblico Impiego', 'D.Lgs. 165/2001, 73 articoli'),
('TUSP', 'Testo Unico Società Partecipate', 'D.Lgs. 175/2016, 26 articoli'),
('TUDA', 'Testo Unico Documentazione Amministrativa', 'D.P.R. 445/2000, 78 articoli'),
-- Ambiente
('CAMB', 'Codice dell''Ambiente', 'D.Lgs. 152/2006, 318 articoli'),
('CBCP', 'Codice Beni Culturali e Paesaggio', 'D.Lgs. 42/2004, 184 articoli'),
('TUFOR', 'Testo Unico Foreste', 'D.Lgs. 34/2018, 20 articoli'),
-- Circolazione
('CAP', 'Codice Assicurazioni Private', 'D.Lgs. 209/2005, 355 articoli'),
('CDS', 'Codice della Strada', 'D.Lgs. 285/1992, 245 articoli'),
('CND', 'Codice Nautica da Diporto', 'D.Lgs. 171/2005, 66 articoli'),
('REGCDS', 'Regolamento CdS', 'D.P.R. 495/1992, 408 articoli'),
-- Comunicazioni
('CCE', 'Codice Comunicazioni Elettroniche', 'D.Lgs. 259/2003, 98 articoli'),
('TUSMAR', 'Testo Unico Radiotelevisione', 'D.Lgs. 177/2005, 63 articoli'),
-- Fisco
('CPT', 'Codice Processo Tributario', 'D.Lgs. 546/1992, 72 articoli'),
('LRT', 'Legge Reati Tributari', 'D.Lgs. 74/2000, 25 articoli'),
('STATC', 'Statuto del Contribuente', 'L. 212/2000, 21 articoli'),
('TUIR', 'Testo Unico Imposte sui Redditi', 'D.P.R. 917/1986, 188 articoli'),
('DIVA', 'DPR IVA', 'D.P.R. 633/1972, 74 articoli'),
-- Internazionale
('DUDU', 'Dichiarazione Universale Diritti Uomo', 'ONU 1948, 30 articoli'),
('LDIP', 'Legge Diritto Internazionale Privato', 'L. 218/1995, 74 articoli'),
-- Lavoro
('CPO', 'Codice Pari Opportunità', 'D.Lgs. 198/2006, 60 articoli'),
('LSSP', 'Legge Sciopero Servizi Pubblici', 'L. 146/1990, 13 articoli'),
('DBIAGI', 'Riforma Lavoro Biagi', 'D.Lgs. 276/2003, 86 articoli'),
('LFORN', 'Riforma Lavoro Fornero', 'L. 92/2012, 4 articoli'),
('SL', 'Statuto dei Lavoratori', 'L. 300/1970, 41 articoli'),
('TUPC', 'Testo Unico Previdenza Complementare', 'D.Lgs. 252/2005, 23 articoli'),
('TUMP', 'Testo Unico Maternità Paternità', 'D.Lgs. 151/2001, 88 articoli'),
('TUSL', 'Testo Unico Sicurezza Lavoro', 'D.Lgs. 81/2008, 306 articoli'),
-- Penale
('CPPM', 'Codice Procedura Penale Minorile', 'D.P.R. 448/1988, 41 articoli'),
('LDEP', 'Legge Depenalizzazione', 'D.Lgs. 8/2016, 42 articoli'),
('OP', 'Ordinamento Penitenziario', 'L. 354/1975, 91 articoli'),
('TUCG', 'Testo Unico Casellario Giudiziale', 'D.P.R. 313/2002, 48 articoli'),
('TUS', 'Testo Unico Stupefacenti', 'D.P.R. 309/1990, 127 articoli'),
-- Professioni
('CDF', 'Codice Deontologico Forense', 'CNF 2014, 73 articoli'),
('LPF', 'Legge Professionale Forense', 'L. 247/2012, 67 articoli')
ON CONFLICT (code) DO NOTHING;

-- ============================================
-- Link work -> nir_mapping
-- ============================================
UPDATE kb.work w SET nir_mapping_id = (SELECT id FROM kb.nir_mapping WHERE code = w.code);

-- ============================================
-- Assign source_type_id per categoria
-- ============================================
UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'COSTITUZIONE')
WHERE code = 'COST';

UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'CODICE')
WHERE code IN ('CC','CP','CPC','CPP','CCII','CPI','CCONS','CTUR','CGS','CPRIV','CTS','CAD','CAM','CAPP','CGC','CMED','CPA','CAMB','CBCP','CAP','CDS','CND','CCE','CPT','CPO','CPPM','CDF');

UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'TESTO_UNICO')
WHERE code IN ('TUB','TUF','TUSG','TUE','TUEL','TUESP','TUI','TUIST','TUPI','TUSP','TUDA','TUFOR','TUSMAR','TUIR','TUPC','TUMP','TUSL','TUCG','TUS');

UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'LEGGE')
WHERE code IN ('LF','LDA','LDIV','LLOC','DMED','LPA','DSIC','LRT','STATC','DUDU','LDIP','LSSP','DBIAGI','LFORN','SL','LDEP','OP','LPF');

UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'REGOLAMENTO')
WHERE code IN ('REGCDS');

UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'UE')
WHERE code = 'GDPR';

UPDATE kb.work SET source_type_id = (SELECT id FROM kb.source_type WHERE code = 'DECRETO')
WHERE code = 'DLAS';

-- ============================================
-- Fix normattiva_resolver: costruisci dinamicamente da nir_base
-- NOTA: Per URL encoding robusto conviene fare in applicazione Python
-- Questo encoding minimale funziona per i nostri URN:NIR
-- ============================================
UPDATE kb.nir_mapping
SET normattiva_resolver = 'https://www.normattiva.it/uri-res/N2Ls?' ||
    replace(replace(nir_base, ':', '%3A'), ';', '%3B')
WHERE code IN ('COST','CC','CP','CPC','CPP','CCII','TUB','TUF');

-- ============================================
-- SEED: WORK_ALIAS (CCI -> CCII, etc.)
-- ============================================
INSERT INTO kb.work_alias(work_id, alias) VALUES
((SELECT id FROM kb.work WHERE code='CCII'), 'CCI'),
((SELECT id FROM kb.work WHERE code='CC'), 'c.c.'),
((SELECT id FROM kb.work WHERE code='CC'), 'cod. civ.'),
((SELECT id FROM kb.work WHERE code='CP'), 'c.p.'),
((SELECT id FROM kb.work WHERE code='CP'), 'cod. pen.'),
((SELECT id FROM kb.work WHERE code='CPC'), 'c.p.c.'),
((SELECT id FROM kb.work WHERE code='CPP'), 'c.p.p.')
ON CONFLICT DO NOTHING;

COMMIT;
