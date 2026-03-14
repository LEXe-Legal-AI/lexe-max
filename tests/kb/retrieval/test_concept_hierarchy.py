"""Tests for concept hierarchy expansion."""
import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from lexe_api.kb.retrieval.concept_expansion import (
    get_concept_path,
    get_parent_concepts,
    expand_concept_path,
    hierarchical_search,
    MAX_SIBLINGS,
    ADJACENCY_BOOST,
)


@pytest.fixture
def mock_conn():
    return AsyncMock()


class TestGetConceptPath:
    @pytest.mark.asyncio
    async def test_returns_path(self, mock_conn):
        nid = uuid4()
        mock_conn.fetchrow.return_value = {"concept_path": ["CC", "Libro IV", "Titolo IX"]}
        path = await get_concept_path(mock_conn, nid)
        assert path == ["CC", "Libro IV", "Titolo IX"]

    @pytest.mark.asyncio
    async def test_returns_none_when_missing(self, mock_conn):
        mock_conn.fetchrow.return_value = {"concept_path": None}
        path = await get_concept_path(mock_conn, uuid4())
        assert path is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_row(self, mock_conn):
        mock_conn.fetchrow.return_value = None
        path = await get_concept_path(mock_conn, uuid4())
        assert path is None


class TestGetParentConcepts:
    @pytest.mark.asyncio
    async def test_returns_ascending_chain(self, mock_conn):
        nid = uuid4()
        mock_conn.fetch.return_value = [
            {"id": uuid4(), "level": 1, "label": "CC"},
            {"id": uuid4(), "level": 2, "label": "Libro IV"},
            {"id": uuid4(), "level": 3, "label": "Titolo IX"},
        ]
        parents = await get_parent_concepts(mock_conn, nid)
        assert len(parents) == 3
        assert parents[0]["label"] == "CC"
        assert parents[-1]["label"] == "Titolo IX"


class TestExpandConceptPath:
    @pytest.mark.asyncio
    async def test_finds_siblings(self, mock_conn):
        sibling1, sibling2 = uuid4(), uuid4()
        mock_conn.fetch.return_value = [
            {"normativa_id": sibling1},
            {"normativa_id": sibling2},
        ]
        path = ["CC", "Libro IV", "Titolo IX", "Capo I"]
        result = await expand_concept_path(mock_conn, path)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_respects_max_siblings(self, mock_conn):
        ids = [uuid4() for _ in range(MAX_SIBLINGS + 5)]
        mock_conn.fetch.return_value = [{"normativa_id": uid} for uid in ids]
        path = ["CC", "Libro IV", "Titolo IX"]
        result = await expand_concept_path(mock_conn, path)
        assert len(result) <= MAX_SIBLINGS

    @pytest.mark.asyncio
    async def test_excludes_existing(self, mock_conn):
        existing = uuid4()
        other = uuid4()
        mock_conn.fetch.return_value = [
            {"normativa_id": existing},
            {"normativa_id": other},
        ]
        path = ["CC", "Libro IV"]
        result = await expand_concept_path(mock_conn, path, exclude_ids={existing})
        assert existing not in result
        assert other in result

    @pytest.mark.asyncio
    async def test_empty_path(self, mock_conn):
        result = await expand_concept_path(mock_conn, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_single_element_path(self, mock_conn):
        result = await expand_concept_path(mock_conn, ["CC"])
        assert result == []


class TestHierarchicalSearch:
    @pytest.mark.asyncio
    async def test_no_expansion_when_disabled(self, mock_conn):
        results = [{"id": uuid4(), "score": 0.9}]
        enhanced = await hierarchical_search(mock_conn, results, expand_concepts=False)
        assert enhanced == results

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_conn):
        enhanced = await hierarchical_search(mock_conn, [])
        assert enhanced == []

    @pytest.mark.asyncio
    async def test_adds_concept_path(self, mock_conn):
        nid = uuid4()
        mock_conn.fetchrow.return_value = {"concept_path": ["CC", "Libro IV"]}
        # No siblings found
        mock_conn.fetch.return_value = []

        results = [{"id": nid, "score": 0.9}]
        enhanced = await hierarchical_search(mock_conn, results)
        assert enhanced[0].get("concept_path") == ["CC", "Libro IV"]
