# Dry-Run Statistico - Citation-Anchored

**Date:** 2026-01-30T19:26:51.116445

## Guardrail

| Parameter | Value |
|-----------|-------|
| min_char | 180 |
| max_char | 1400 |
| max_massime_per_page | 25 |
| window_before | 2 |
| window_after | 1 |
| toc_skip_pages | 25 |

## Soglie

| Threshold | Value |
|-----------|-------|
| pct_short_fail | 8.0 |
| massime_per_page_p95_fail | 25 |
| duplicates_pct_fail | 3.0 |
| pct_complete_citation_warn | 50.0 |
| pct_toc_like_fail | 5.0 |
| pct_citation_list_fail | 7.0 |

## Summary

| Status | Count |
|--------|-------|
| PASS | 2 |
| WARNING | 0 |
| FAIL | 0 |

## Per-Document Results

| Document | Status | Current | New | p50 | p90 | pct_short | dup% | mpp_p95 | cit% |
|----------|--------|---------|-----|-----|-----|-----------|------|---------|------|
| Volume I_2016_Massimario_Civile_1_3 | PASS | 43 | 1501 | 1400 | 1404 | 0.0% | 0.0% | 7.0 | 81.1% |
| Volume II_2024_Massimario_Civile(vo | PASS | 28 | 864 | 1395 | 1401 | 0.0% | 0.0% | 6.0 | 87.3% |
