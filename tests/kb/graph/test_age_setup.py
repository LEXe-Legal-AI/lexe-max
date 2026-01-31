"""
Test AGE Extension Setup

Verifica che Apache AGE sia correttamente installato e configurato.
"""

import pytest
import asyncpg

from lexe_api.kb.config import KBSettings


@pytest.fixture
async def db_conn():
    """Create database connection."""
    settings = KBSettings()
    conn = await asyncpg.connect(settings.kb_database_url)
    yield conn
    await conn.close()


class TestAGESetup:
    """Test Apache AGE extension setup."""

    @pytest.mark.asyncio
    async def test_age_extension_loaded(self, db_conn):
        """Verify AGE extension is loaded."""
        result = await db_conn.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'age'"
        )
        assert result == "age", "Apache AGE extension not installed"

    @pytest.mark.asyncio
    async def test_shared_preload_libraries(self, db_conn):
        """Verify age is in shared_preload_libraries."""
        result = await db_conn.fetchval("SHOW shared_preload_libraries")
        assert "age" in result, f"age not in shared_preload_libraries: {result}"

    @pytest.mark.asyncio
    async def test_graph_exists(self, db_conn):
        """Verify lexe_jurisprudence graph exists."""
        result = await db_conn.fetchrow(
            "SELECT * FROM ag_catalog.ag_graph WHERE name = 'lexe_jurisprudence'"
        )
        assert result is not None, "Graph 'lexe_jurisprudence' not found"

    @pytest.mark.asyncio
    async def test_can_query_graph(self, db_conn):
        """Verify we can execute cypher queries."""
        # Set search path for AGE
        await db_conn.execute("SET search_path TO ag_catalog, kb, public")
        await db_conn.execute("LOAD 'age'")

        # Simple count query
        result = await db_conn.fetchval("""
            SELECT * FROM cypher('lexe_jurisprudence', $$
                MATCH (n)
                RETURN count(n) as node_count
            $$) as (node_count agtype)
        """)
        # Result will be agtype, just verify no error
        assert result is not None or result == 0


class TestGraphSchemaSetup:
    """Test graph schema tables exist."""

    @pytest.mark.asyncio
    async def test_graph_runs_table_exists(self, db_conn):
        """Verify kb.graph_runs table exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'kb' AND table_name = 'graph_runs'
            )
        """)
        assert result is True, "Table kb.graph_runs not found"

    @pytest.mark.asyncio
    async def test_graph_edges_table_exists(self, db_conn):
        """Verify kb.graph_edges table exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'kb' AND table_name = 'graph_edges'
            )
        """)
        assert result is True, "Table kb.graph_edges not found"

    @pytest.mark.asyncio
    async def test_categories_table_exists(self, db_conn):
        """Verify kb.categories table exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'kb' AND table_name = 'categories'
            )
        """)
        assert result is True, "Table kb.categories not found"

    @pytest.mark.asyncio
    async def test_categories_seeded(self, db_conn):
        """Verify L1 categories are seeded."""
        result = await db_conn.fetchval("""
            SELECT COUNT(*) FROM kb.categories WHERE level = 1
        """)
        assert result >= 6, f"Expected >= 6 L1 categories, got {result}"

    @pytest.mark.asyncio
    async def test_turning_points_table_exists(self, db_conn):
        """Verify kb.turning_points table exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'kb' AND table_name = 'turning_points'
            )
        """)
        assert result is True, "Table kb.turning_points not found"

    @pytest.mark.asyncio
    async def test_norms_table_exists(self, db_conn):
        """Verify kb.norms table exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'kb' AND table_name = 'norms'
            )
        """)
        assert result is True, "Table kb.norms not found"


class TestGraphEdgesSchema:
    """Test graph_edges table schema (v3.2.1 requirements)."""

    @pytest.mark.asyncio
    async def test_weight_column_exists(self, db_conn):
        """Verify weight column exists (Miglioria #1 v3.2.1)."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'kb'
                AND table_name = 'graph_edges'
                AND column_name = 'weight'
            )
        """)
        assert result is True, "Column 'weight' not found in kb.graph_edges"

    @pytest.mark.asyncio
    async def test_evidence_column_exists(self, db_conn):
        """Verify evidence column exists (Miglioria #1 v3.2.1)."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'kb'
                AND table_name = 'graph_edges'
                AND column_name = 'evidence'
            )
        """)
        assert result is True, "Column 'evidence' not found in kb.graph_edges"

    @pytest.mark.asyncio
    async def test_relation_subtype_column_exists(self, db_conn):
        """Verify relation_subtype column exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'kb'
                AND table_name = 'graph_edges'
                AND column_name = 'relation_subtype'
            )
        """)
        assert result is True, "Column 'relation_subtype' not found in kb.graph_edges"


class TestHelperFunctions:
    """Test helper SQL functions."""

    @pytest.mark.asyncio
    async def test_set_active_run_function_exists(self, db_conn):
        """Verify kb.set_active_run function exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'kb' AND p.proname = 'set_active_run'
            )
        """)
        assert result is True, "Function kb.set_active_run not found"

    @pytest.mark.asyncio
    async def test_get_neighbors_function_exists(self, db_conn):
        """Verify kb.get_neighbors function exists."""
        result = await db_conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = 'kb' AND p.proname = 'get_neighbors'
            )
        """)
        assert result is True, "Function kb.get_neighbors not found"
