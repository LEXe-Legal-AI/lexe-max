# Confronto Cloud Chunking vs Local Extraction

**Data:** 2026-01-29
**Test reale su 5 PDF eterogenei**

---

## Risultati Dettagliati

| PDF | Pagine | Cloud Chunks | Cloud Avg | Local Units | Local Avg | Note |
|-----|--------|--------------|-----------|-------------|-----------|------|
| 2014 Mass civile Vol 1 | 408 | **ERROR** | - | 1 | 1,519,399 | >300 pag + segmentazione rotta |
| 2015 principi Vol 2 | 250 | **430** | 2,289 | 165 | 5,901 | Cloud 2.6x chunks |
| Rassegna Penale 2011 | 359 | **ERROR** | - | 39 | 21,948 | >300 pag |
| Rassegna Civile 2012 II | ~200 | **206** | 2,558 | 17 | 31,096 | Cloud 12x chunks |
| Rassegna Penale 2012 | 149 | **436** | 1,639 | 117 | 6,079 | Cloud 3.7x chunks |

---

## Summary Aggregato

| Metrica | Cloud by_title | Local Reference |
|---------|----------------|-----------------|
| **Total chunks** | 1,072 | 339 |
| **Avg chars/chunk** | 2,162 | 316,885 |
| **% con pattern massima** | 87.9% | 95.4% |
| **% short (<150 chars)** | 1.5% | 2.0% |
| **Tempo totale** | ~50 min | 0.3s |
| **Score euristico** | **94.7** | 67.6 |

---

## Pattern Massima Usati

```python
MASSIMA_PATTERNS = [
    r"Sez\.\s*[IVX\d]+",           # Sez. I, Sez. II, Sez. Un.
    r"Cass\.",                      # Cassazione
    r"sent\.\s*n\.\s*\d+",          # sent. n. 1234
    r"ord\.\s*n\.\s*\d+",           # ord. n. 1234
    r"Rv\.\s*\d+",                  # Rv. 123456
    r"n\.\s*\d+/\d{2,4}",           # n. 1234/2020
    r"art\.\s*\d+",                 # art. 123
    r"c\.p\.c\.|c\.p\.p\.|c\.c\.",  # codici
]
```

**% con pattern massima** = percentuale di chunk che contengono almeno uno di questi pattern.
- **95.4% local** = chunk più "puri", quasi tutti contengono citazioni legali
- **87.9% cloud** = più chunk "rumore" (TOC, header, frammenti)

---

## Chunking Strategies Testate

| Strategy | Chunks | Avg Chars | Qualità | Raccomandato |
|----------|--------|-----------|---------|--------------|
| **by_title** | 206-436 | 1,639-2,558 | Ottimo | **SI** |
| by_similarity | 2,922 | 209 | Pessimo | NO |
| basic | N/A | N/A | Non testato | - |

**by_similarity** frammenta troppo - spezza massime a metà. Non usare per documenti legali strutturati.

---

## Problemi Identificati

### 1. Limite 300 pagine cloud
2 PDF su 5 hanno fallito per limite pagine:
- `2014 Mass civile Vol 1` (408 pag)
- `Rassegna Penale 2011` (359 pag)

**Soluzione:** Usare `smart_pdf_split.py` per dividere a chapter boundaries.

### 2. Under-segmentation locale
Local produce chunk enormi:
- `2014 Mass civile Vol 1`: **1 chunk da 1.5M chars** (tutto il doc!)
- `Rassegna Civile 2012 II`: 31K chars medi
- `Rassegna Penale 2011`: 22K chars medi

**Causa:** Pattern segmentazione (`Sez.`) non presente in alcuni documenti.
**Soluzione:** Aggiungere pattern `La Corte`, `In tema di` per 2014 Mass civile.

---

## Raccomandazioni

### Opzione A: Cloud by_title (consigliato per RAG)
- Chunk size ideale 2-3K chars
- Richiede split preventivo per PDF >300 pag
- Costo API + tempo (~15-20 min/PDF)

### Opzione B: Local + Post-chunking
- Usare local extraction (alta precisione 95.4%)
- Applicare post-chunking per dividere chunk >3000 chars
- Fix pattern segmentazione per documenti problematici

### Opzione C: Ibrida
- Local per identificazione massime (alta precisione)
- Cloud by_title per chunking finale (size ideale)

---

## Prossimi Passi

1. [x] Confronto cloud vs local completato
2. [ ] Fix segmentation 2014 Mass civile (pattern `La Corte`, `In tema di`)
3. [ ] Completare Phase 0 extraction (42/63 PDF)
4. [ ] Phase 6: Reference Alignment
5. [ ] Benchmark retrieval con diversi chunk size

---

*Report generato da compare_cloud_vs_local.py - 2026-01-29*
