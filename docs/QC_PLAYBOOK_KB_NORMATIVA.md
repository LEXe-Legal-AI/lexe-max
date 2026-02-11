# QC Playbook - KB Normativa v1.0

## KPI Ufficiali Post-Fix

| Metric | Value | Note |
|--------|-------|------|
| Documenti | 76 | |
| Articoli ingestiti | 10,556 | |
| Articoli validati | 10,555 | |
| In review | 1 | CCI:373 |
| READY | 58 | |
| NEEDS_REVIEW | 10 | |
| PARTIAL_EXTRACT | 7 | |
| EXCLUDED | 1 | AUTO |

---

## Regole di Classificazione

### 1. Quarantine Automatica
Numeri articolo con pattern sospetti → **QUARANTINE** (mai READY diretto):
- `>= 1000` senza suffix → probabile data (2024, 1990)
- Pattern `NNNN` dove `N > count * 5` → anomalo
- Pattern protocollo/pagina: `/`, `pag.`, `n.`

### 2. Suffix Ammessi
```
bis, ter, quater, quinquies, sexies, septies, octies,
novies, decies, undecies, duodecies, terdecies, quaterdecies
```
Articoli con suffix valido → MAX alto ammesso (es. 416-bis)

### 3. Eccezione CAMAFIA
- MAX 416 ammesso per presenza 416-bis, 275-bis, 146-bis
- Sono riferimenti CP integrati nel codice antimafia

---

## Criteri Promozione a READY

### Da NEEDS_REVIEW → READY
- [ ] Root cause identificata
- [ ] Articoli anomali rimossi o giustificati
- [ ] `MAX <= EXPECTED_MAX` per il documento
- [ ] `count(review) == 0`
- [ ] Nessun duplicato interno

### Da PARTIAL_EXTRACT → READY
- [ ] Fonte dichiarata come "estratto intenzionale" **OPPURE**
- [ ] Completamento con fonte aggiuntiva
- [ ] `variant_type: EXTRACT` se parziale

### EXCLUDED → Non promuovere
- Richiede: fonte primaria verificata + perimetro chiaro

---

## Checklist QC per Documento

```
DOCUMENTO: ___________
DATA QC: ___________

1. ROOT CAUSE
   [ ] Parsing error (date/riferimenti come art)
   [ ] Duplicati (stesso articolo, chiavi diverse)
   [ ] Fonte corrotta
   [ ] Numbering non standard
   [ ] Altro: ___________

2. AZIONE
   [ ] Fix - rimuovi articoli anomali
   [ ] Merge - unisci con altro documento
   [ ] Split - separa in sotto-documenti
   [ ] Exclude - escludere da produzione
   [ ] Accept - anomalia giustificata

3. DELTA
   Prima: ___ articoli | MAX: ___
   Dopo:  ___ articoli | MAX: ___

4. STATUS FINALE
   [ ] READY
   [ ] PARTIAL_EXTRACT (con giustificazione)
   [ ] EXCLUDED (con motivazione)

5. NOTE
   ___________________________________
```

---

## Ordine QC Consigliato

| # | Doc | Priorità | Rischio | Note |
|---|-----|----------|---------|------|
| 1 | CCI | ALTA | 1 art review | Compliance enterprise |
| 2 | TUB | ALTA | Rinvii | Parsing aggressivo |
| 3 | REGCDS | MEDIA | Allegati | Falsi positivi |
| 4 | CAMAFIA | BASSA | Solo verifica | 416-bis OK |
| 5 | CGS | BASSA | Batch | |
| 6 | CDF | BASSA | Batch | |
| 7 | CGC | BASSA | Batch | |
| 8 | TUESP | BASSA | Batch | |
| 9 | LDEP | BASSA | Batch | |
| 10 | LDIP | BASSA | Batch | |

---

## PARTIAL_EXTRACT - Decisioni

| Doc | Articoli | Decisione | Azione |
|-----|----------|-----------|--------|
| DUDU | 2 | ESTRATTO | Marcare `variant_type: EXTRACT`, nota "solo preambolo" |
| LDIV | 1 | ESTRATTO | Fonte parziale, accettare |
| LSCIO | 4 | COMPLETARE | Cercare fonte completa |
| LSIC | 8 | COMPLETARE | Cercare fonte completa |
| RFORN | 2 | ESTRATTO | Legge breve, accettare |
| TUDA | 5 | MERGE | Unire con TUDOC |
| TUDOC | 3 | MERGE | Unire con TUDA → TUDA-DPR445 |

---

## Edge Cases

### Date come articoli
```
2024 → parsing "2024" come art. 2024
Fix: quarantine se base_num > expected_max
```

### Riferimenti incrociati
```
"v. art. 416 c.p." → parsing "416" come articolo
Fix: context check, se contiene "c.p.", "c.c." → quarantine
```

### Commi alti
```
"comma 12-bis" → parsing come articolo separato
Fix: suffix detection, unire a articolo padre
```

### Allegati
```
"Allegato A, art. 1" → duplicato art.1
Fix: chiave composta `doc:allegato:art`
```

---

## Soglie per Documento

| Doc | Expected MAX | Soglia Anomalia |
|-----|--------------|-----------------|
| CC | 2969 | > 3000 |
| CPC | 840 | > 900 |
| CPP | 746 | > 800 |
| CP | 734 | > 800 |
| COST | 139 | > 150 |
| CCI | 391 | > 400 |
| TUB | 165 | > 200 |
| TUF | 214 | > 250 |
| CAMAFIA | 500* | > 500 |
| ... | ... | ... |

*CAMAFIA: MAX alto per 416-bis

---

*Versione: 1.0 | Data: 2026-02-08 | Owner: Frisco*
