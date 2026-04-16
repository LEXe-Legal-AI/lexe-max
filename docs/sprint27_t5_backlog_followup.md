# Sprint 27 T5 — Backlog Follow-up

> **Origine**: Sprint 27 T5 (metadata massime recovery) — Gate B ridefinito per ceiling strutturale IPZS.
> **Documento correlato**: [sprint27_t5_gate_b_redefine.md](sprint27_t5_gate_b_redefine.md)
> **Owner proposto**: backlog Sprint 28+, non bloccante

---

## Stato di partenza (post T5)

- `anno IS NULL` = 8,097 (17.3% delle 46,767 massime staging)
- `numero IS NULL` = 9,656 (20.6%)
- 7,688 massime "below_threshold" post-LLM = testi narrativi senza citation strutturata

Recupero addizionale oltre questi livelli richiede strategie diverse dalla cascade regex+LLM attuale.

---

## P3 — Mini-bench agreement su 200 sample below_threshold

**Obiettivo**: capire se abbassare la soglia `--min-confidence` da 0.80 a 0.65-0.70 introdurrebbe errori o permetterebbe recupero addizionale accettabile.

**Effort**: ~6h (4h annotazione + 2h tuning + rerun)

**Procedura**:
1. Query 200 massime a caso da below_threshold post-LLM (staging):
   ```sql
   SELECT id, testo
   FROM kb.massime
   WHERE anno IS NULL
     AND citation_extracted = TRUE -- già passate via regex OR LLM
   ORDER BY random()
   LIMIT 200;
   ```
2. Re-run `massima_metadata_recovery.py --only-llm --execute=FALSE --min-confidence 0.0` catturando TUTTI i risultati LLM (incluso below_threshold).
3. Annotazione manuale: per ciascuna delle 200 massime, verdict umano "sufficientemente corretta" vs "errata/ambigua" sui campi restituiti dall'LLM.
4. Calcola F1 agreement per threshold ∈ {0.50, 0.60, 0.65, 0.70, 0.75}. Scegli F1-max.
5. Se F1-max threshold < 0.80 e agreement ≥ 90% → rerun production con threshold dinamico, update DB.

**Gate**: se F1-max < 90%, NON abbassare threshold. Accettare ceiling attuale.

**Rischio**: update con info parziale errata può rompere citation graph (italgiure resolver). Richiede rollback script.

---

## P3 — Matching massima→sentenza via ItalGiure resolver (strategia alternativa)

**Obiettivo**: recuperare metadata non dal `testo` della massima ma dalla sentenza sottostante, tramite lookup inverso su ItalGiure (già implementato in Sprint 21 SRA).

**Effort**: ~10-15h (dipende dal recall di ItalGiure su Corpus senza metadata parziale)

**Rationale**: il ceiling IPZS è sul formato della massima. Ma molte di queste massime DERIVANO da sentenze che sono note e indicizzate altrove. Un match testo-based (embedding similarity + section heading) potrebbe:
- Recuperare `anno`/`numero`/`sezione`/`rv` dalla sentenza matchata
- Richiede però un text match con `confidence ≥ 0.85` per evitare falsi positivi

**Procedura (outline)**:
1. Embedding-based retrieval sulla massima `testo` contro `kb.sentenze_cassazione` (se popolato) o ItalGiure API.
2. Top-1 match con cosine ≥ 0.85 → estrai metadata dalla sentenza.
3. Gate: verifica che testo massima sia subset/paraphrase della sentenza matchata (via NLI o span containment).
4. UPDATE solo se gate superato.

**Dipendenze**: Sprint 21 SRA POC chiuso (30/30 resolved), batch 38K staging ready (vedi `project_sra_poc.md`).

**Output atteso**: recupero aggiuntivo 2-4K massime (rough estimate, da validare POC).

---

## P4 — Reference permanente: ceiling canale IPZS

**Obiettivo**: documentare il ceiling strutturale come reference memory/docs permanente, per evitare che futuri stream riscoprano lo stesso limite.

**Effort**: ~1h

**Deliverable**: creare `reference_kb_massime_structure.md` in auto-memory con:
- Schema `kb.massime` e semantica campi
- Tasso "narrative vs strutturata" osservato post-T5 (~30-40% narrative)
- `testo_con_contesto` NULL ovunque (regressione da investigare o limite design)
- Cascade regex+LLM hit rate reale (43% recovery)
- Warning: non provare bench/target più aggressivi senza P3 sopra

**Link da aggiungere**:
- In `MEMORY.md` sotto "Reference Docs"
- In `docs/KB-MASSIMARI-ARCHITECTURE.md` come appendice ceiling

---

## P3 — Investigazione `testo_con_contesto=NULL` ovunque

**Osservazione**: tutte 46,767 massime staging hanno `testo_con_contesto=NULL`. Schema prevede "Chunk B: con blocchi OCR attigui" per dedup e contesto esteso.

**Effort**: ~2h (triage)

**Ipotesi**:
1. Regressione pipeline ingest (il campo non viene più popolato)
2. Design evolution: il campo è stato deprecato silenziosamente
3. Popolamento demandato a processo batch mai eseguito su staging

**Procedura**:
1. `git log --all -- src/lexe_api/kb/ingestion/ | grep -i contesto` per capire history
2. Check script ingestion per ritrovare logica popolamento
3. Se regressione: aprire ticket Sprint 28
4. Se deprecato: rimuovere campo dallo schema + dalle query dello script T5

**Impatto su T5 re-run**: se `testo_con_contesto` venisse popolato (più testo = più contesto), regex+LLM hit rate salirebbe. Stima: recupero addizionale 10-15% → potenziale `anno_null` < 6,000. Vale P3 sotto P4 reference.

---

## Riepilogo priorità

| # | Item | Effort | Expected gain | Priority |
|---|---|---|---|---|
| 1 | P3 — Mini-bench agreement 200 sample | 6h | 5-15% recupero aggiuntivo | Medio |
| 2 | P3 — Matching via ItalGiure resolver | 10-15h | 10-20% recupero aggiuntivo | Basso (dipende da SRA batch) |
| 3 | P4 — Reference ceiling IPZS | 1h | 0 (documentale) | Alto (knowledge capture) |
| 4 | P3 — Investigazione testo_con_contesto | 2h | 10-15% se sistemato | Medio |

Tutti gli item sono **non bloccanti** per qualsiasi Sprint 28+. T5 è chiuso.

---

*Documento generato 2026-04-16, T5 closure.*
