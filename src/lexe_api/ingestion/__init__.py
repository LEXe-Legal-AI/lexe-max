"""Top-level external-source ingestion modules.

Distinct from `lexe_api.kb.ingestion` which handles PDF → KB processing.
This package hosts nightly/scheduled ingesters for external APIs:

- cgue_cellar: EUR-Lex CELLAR (Sprint 30 P1.1) — CGUE judgments.

Future (planned, Sprint 30 P1.x):

- garante_crawler: Garante Privacy doc.web crawler.
- edpb_guidelines: EDPB Guidelines ingester.
"""

from __future__ import annotations
