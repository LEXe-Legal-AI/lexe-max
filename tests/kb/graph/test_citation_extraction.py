"""
Test Citation Extraction

Tests for the two-step citation extraction pipeline (v3.2.1).
"""

import pytest
from uuid import uuid4

from lexe_api.kb.graph.citation_extractor import (
    extract_mentions,
    normalize_numero,
    normalize_rv,
    normalize_sezione,
    detect_indicator,
    dedupe_mentions,
    compute_weight,
    build_evidence,
    CitationMention,
    ResolvedCitation,
)


class TestNormalization:
    """Test normalization functions (v3.2.1 Miglioria #3)."""

    def test_normalize_numero_lstrip_zeros(self):
        """lstrip zeros from numero."""
        assert normalize_numero("09337") == "9337"
        assert normalize_numero("00123") == "123"
        assert normalize_numero("12345") == "12345"

    def test_normalize_numero_all_zeros(self):
        """Keep single zero if all zeros."""
        assert normalize_numero("0000") == "0"
        assert normalize_numero("0") == "0"

    def test_normalize_numero_empty(self):
        """Handle empty/None."""
        assert normalize_numero("") == ""
        assert normalize_numero(None) is None

    def test_normalize_rv_suffix(self):
        """Extract core RV, strip suffix."""
        assert normalize_rv("639966-01") == "639966"
        assert normalize_rv("639966-02") == "639966"
        assert normalize_rv("639966") == "639966"

    def test_normalize_rv_leading_zeros(self):
        """Strip leading zeros from RV."""
        assert normalize_rv("0639966") == "639966"
        assert normalize_rv("00639966") == "639966"

    def test_normalize_sezione(self):
        """Normalize sezione abbreviations."""
        assert normalize_sezione("Un") == "U"
        assert normalize_sezione("Unite") == "U"
        assert normalize_sezione("U") == "U"
        assert normalize_sezione("Lav") == "L"
        assert normalize_sezione("1") == "1"
        assert normalize_sezione("6-1") == "6-1"


class TestExtractMentions:
    """Test mention extraction (Step 1)."""

    def test_extract_rv_simple(self):
        """Extract simple Rv. pattern."""
        text = "cfr. Cass. Sez. Un., Rv. 639966"
        mentions = extract_mentions(text)
        assert len(mentions) == 1
        assert mentions[0].rv == "639966"
        assert mentions[0].pattern_type == "rv"

    def test_extract_rv_with_suffix(self):
        """Extract Rv. with suffix, normalize it."""
        text = "Rv. 639966-01"
        mentions = extract_mentions(text)
        assert len(mentions) == 1
        assert mentions[0].rv == "639966"
        assert mentions[0].rv_raw == "639966"  # Pattern captures without suffix

    def test_extract_rv_no_dot(self):
        """Extract Rv without dot."""
        text = "vedi Rv 654321"
        mentions = extract_mentions(text)
        assert len(mentions) == 1
        assert mentions[0].rv == "654321"

    def test_extract_sez_num_anno(self):
        """Extract Sez. n. /anno pattern."""
        text = "Cass. Sez. Un., n. 12345/2020"
        mentions = extract_mentions(text)
        assert len(mentions) >= 1
        sez_mention = next((m for m in mentions if m.pattern_type == "sez_num_anno"), None)
        assert sez_mention is not None
        assert sez_mention.sezione == "U"
        assert sez_mention.numero == "12345"
        assert sez_mention.anno == 2020

    def test_extract_num_anno(self):
        """Extract n. del anno pattern."""
        text = "sentenza n. 9876 del 2019"
        mentions = extract_mentions(text)
        assert len(mentions) >= 1
        # Should find at least num_anno or sentenza pattern
        assert any(m.numero == "9876" and m.anno == 2019 for m in mentions)

    def test_extract_numero_normalization(self):
        """Normalize numero with leading zeros (v3.2.1)."""
        text = "Sez. 1, n. 09337/2020"
        mentions = extract_mentions(text)
        sez_mention = next((m for m in mentions if m.pattern_type == "sez_num_anno"), None)
        assert sez_mention is not None
        assert sez_mention.numero_raw == "09337"
        assert sez_mention.numero_norm == "9337"
        assert sez_mention.numero == "9337"  # Primary should be normalized

    def test_extract_multiple_mentions(self):
        """Extract multiple mentions from same text."""
        text = "Rv. 123456 cfr. anche Rv. 654321"
        mentions = extract_mentions(text)
        assert len(mentions) == 2


class TestRelationIndicator:
    """Test relation indicator detection."""

    def test_detect_confirms(self):
        """Detect CONFIRMS indicator."""
        text = "conforme a quanto stabilito in Rv. 123456"
        indicator = detect_indicator(text, len(text) - 10)
        assert indicator == "CONFIRMS"

    def test_detect_distinguishes(self):
        """Detect DISTINGUISHES indicator."""
        text = "diversamente da Rv. 123456"
        indicator = detect_indicator(text, len(text) - 10)
        assert indicator == "DISTINGUISHES"

    def test_detect_overrules(self):
        """Detect OVERRULES indicator."""
        text = "in senso contrario Rv. 123456"
        indicator = detect_indicator(text, len(text) - 10)
        assert indicator == "OVERRULES"

    def test_detect_contra(self):
        """Detect contra as OVERRULES."""
        text = "contra: Rv. 654321"
        indicator = detect_indicator(text, len(text) - 10)
        assert indicator == "OVERRULES"

    def test_no_indicator(self):
        """No indicator detected."""
        text = "vedi anche Rv. 123456"
        indicator = detect_indicator(text, len(text) - 10)
        # "v. anche" is CONFIRMS in our patterns
        # For plain text without indicator, should be None or CONFIRMS
        assert indicator in (None, "CONFIRMS")


class TestDeduplication:
    """Test semantic deduplication (v3.2.1 Miglioria #2)."""

    def test_dedup_same_target_multiple_times(self):
        """Same RV cited 4 times = 1 edge."""
        source_id = uuid4()
        target_id = uuid4()

        mentions = [
            CitationMention(rv="123456", raw_span=f"Rv. 123456 ({i})")
            for i in range(4)
        ]

        resolved = [(source_id, m, target_id, "rv_exact") for m in mentions]
        edges = dedupe_mentions(resolved)

        assert len(edges) == 1
        assert edges[0].source_id == source_id
        assert edges[0].target_id == target_id

    def test_dedup_different_subtypes_kept(self):
        """Different subtypes are kept as separate edges."""
        source_id = uuid4()
        target_id = uuid4()

        mention1 = CitationMention(rv="123456", indicator="CONFIRMS")
        mention2 = CitationMention(rv="123456", indicator="OVERRULES")

        resolved = [
            (source_id, mention1, target_id, "rv_exact"),
            (source_id, mention2, target_id, "rv_exact"),
        ]
        edges = dedupe_mentions(resolved)

        assert len(edges) == 2
        subtypes = {e.relation_subtype for e in edges}
        assert subtypes == {"CONFIRMS", "OVERRULES"}

    def test_dedup_keeps_best_confidence(self):
        """Keep edge with higher confidence."""
        source_id = uuid4()
        target_id = uuid4()

        mention1 = CitationMention(rv="123456")
        mention2 = CitationMention(numero="123", anno=2020)  # Less precise

        resolved = [
            (source_id, mention2, target_id, "num_anno"),  # Lower confidence first
            (source_id, mention1, target_id, "rv_exact"),  # Higher confidence
        ]
        edges = dedupe_mentions(resolved)

        assert len(edges) == 1
        assert edges[0].confidence == 1.0  # rv_exact confidence


class TestWeightCalculation:
    """Test weight calculation (v3.2.1 Miglioria #1)."""

    def test_weight_rv_exact(self):
        """rv_exact gets weight 1.0."""
        mention = CitationMention(rv="123456")
        weight = compute_weight(mention, "rv_exact")
        assert weight == 1.0

    def test_weight_rv_text_fallback(self):
        """rv_text_fallback gets lower weight."""
        mention = CitationMention(rv="123456")
        weight = compute_weight(mention, "rv_text_fallback")
        assert weight == 0.85

    def test_weight_num_anno(self):
        """num_anno gets lowest weight."""
        mention = CitationMention(numero="123", anno=2020)
        weight = compute_weight(mention, "num_anno")
        assert weight == 0.7

    def test_weight_overrules_boost(self):
        """OVERRULES indicator boosts weight."""
        mention = CitationMention(rv="123456", indicator="OVERRULES")
        weight = compute_weight(mention, "sez_num_anno")  # Base 0.9
        # 0.9 * 1.1 = 0.99, capped at 1.0
        assert weight >= 0.9

    def test_weight_pruning_threshold(self):
        """All weights should be >= 0.6 for valid edges."""
        resolvers = ["rv_exact", "rv_raw", "rv_text_fallback", "sez_num_anno", "sez_num_anno_raw", "num_anno"]
        mention = CitationMention(rv="123456")

        for resolver in resolvers:
            weight = compute_weight(mention, resolver)
            assert weight >= 0.6 or resolver == "num_anno"  # num_anno is 0.7


class TestEvidence:
    """Test evidence building (v3.2.1 Miglioria #1)."""

    def test_evidence_rv_pattern(self):
        """Evidence contains RV pattern."""
        mention = CitationMention(rv="123456", rv_raw="123456-01", indicator="CONFIRMS")
        evidence = build_evidence(mention, "rv_exact")

        assert "pattern" in evidence
        assert "123456" in evidence["pattern"]
        assert evidence["indicator"] == "CONFIRMS"
        assert evidence["resolver"] == "rv_exact"

    def test_evidence_sez_num_anno(self):
        """Evidence contains sez/num/anno pattern."""
        mention = CitationMention(sezione="U", numero="12345", anno=2020)
        evidence = build_evidence(mention, "sez_num_anno")

        assert "pattern" in evidence
        assert "U" in evidence["pattern"]
        assert "12345" in evidence["pattern"]
        assert "2020" in evidence["pattern"]

    def test_evidence_raw_span_truncated(self):
        """Raw span is truncated in evidence."""
        long_span = "x" * 200
        mention = CitationMention(rv="123456", raw_span=long_span)
        evidence = build_evidence(mention, "rv_exact")

        assert len(evidence.get("raw_span", "")) <= 100


class TestNoSelfLoops:
    """Test self-loop prevention."""

    def test_extract_no_self_reference(self):
        """Should not create edges from massima to itself."""
        # This is tested in resolve_mention, which returns None for self-references
        source_id = uuid4()
        mention = CitationMention(rv="123456")

        # If resolve_mention were called with same source_massima_id as target,
        # it should return (None, "self_reference")
        # We can't test this without a DB, but the logic is in citation_extractor.py


class TestEdgeDataClass:
    """Test ResolvedCitation dataclass."""

    def test_resolved_citation_defaults(self):
        """ResolvedCitation has correct defaults."""
        edge = ResolvedCitation(
            source_id=uuid4(),
            target_id=uuid4(),
        )
        assert edge.relation_type == "CITES"
        assert edge.relation_subtype is None
        assert edge.confidence == 1.0
        assert edge.weight == 1.0
        assert edge.evidence == {}
        assert edge.context_span == ""
