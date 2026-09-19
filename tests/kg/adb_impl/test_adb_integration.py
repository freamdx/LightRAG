"""End-to-end smoke tests for ADB storage backends.

Requires a live AnalyticDB MySQL instance.  The entire module is skipped when
the ``ADB_HOST`` / ``ADB_USER`` / ``ADB_PASSWORD`` / ``ADB_DATABASE``
environment variables are not all set.

Run with::

    ADB_HOST=… ADB_USER=… ADB_PASSWORD=… ADB_DATABASE=… \\
        pytest tests/kg/adb_impl/test_adb_integration.py --run-integration
"""

from __future__ import annotations

import os
import uuid

import pytest

from lightrag.base import DocStatus

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


def _adb_env_complete() -> bool:
    return all(
        os.getenv(v) for v in ("ADB_HOST", "ADB_USER", "ADB_PASSWORD", "ADB_DATABASE")
    )


pytest.skip(
    reason="ADB env vars not configured",
    allow_module_level=not _adb_env_complete(),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_workspace() -> str:
    return f"adb_test_{uuid.uuid4().hex[:10]}"


async def _build_doc_status(workspace: str):
    from lightrag.kg.adb_mysql_impl import ADBDocStatusStorage
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    storage = ADBDocStatusStorage(
        namespace="doc_status",
        global_config={"embedding_batch_num": 8},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    await storage.initialize()
    return storage


async def _build_graph_storage(workspace: str):
    from lightrag.kg.adb_mysql_impl import ADBGraphStorage
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()
    storage = ADBGraphStorage(
        namespace="chunk_entity_relation",
        global_config={"embedding_batch_num": 8},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    await storage.initialize()
    return storage


class _DummyEmbeddingFunc:
    embedding_dim = 4

    async def __call__(self, texts):
        return [[0.0] * self.embedding_dim for _ in texts]


# ===================================================================
# DocStatusStorage end-to-end
# ===================================================================


@pytest.mark.asyncio
async def test_doc_status_e2e():
    ws = _unique_workspace()
    storage = await _build_doc_status(ws)
    try:
        # Upsert
        await storage.upsert(
            {
                "doc-e2e-1": {
                    "content_summary": "E2E test document",
                    "content_length": 100,
                    "chunks_count": 2,
                    "status": DocStatus.PENDING,
                    "file_path": "e2e_test.pdf",
                    "chunks_list": ["c1", "c2"],
                    "metadata": {"source": "test"},
                    "content_hash": "e2e_hash_001",
                    "created_at": "2024-01-15 12:00:00",
                }
            }
        )

        # get_by_id
        doc = await storage.get_by_id("doc-e2e-1")
        assert doc is not None
        assert doc["content_length"] == 100
        assert doc["content_hash"] == "e2e_hash_001"

        # get_by_id_strict
        doc_strict = await storage.get_by_id_strict("doc-e2e-1")
        assert doc_strict is not None

        # get_doc_by_content_hash
        result = await storage.get_doc_by_content_hash("e2e_hash_001")
        assert result is not None
        assert result[0] == "doc-e2e-1"

        # update_doc_status_fields
        await storage.update_doc_status_fields(
            "doc-e2e-1", {"status": DocStatus.PROCESSED.value}
        )

        # get_docs_by_statuses_page
        page = await storage.get_docs_by_statuses_page([DocStatus.PROCESSED], limit=10)
        assert "doc-e2e-1" in page.docs

        # resolve_doc_source_strict
        resolution = await storage.resolve_doc_source_strict("e2e_test.pdf")
        from lightrag.base import SourceUnique

        assert isinstance(resolution, SourceUnique)
        assert resolution.doc_id == "doc-e2e-1"

        # count_docs_by_statuses
        count = await storage.count_docs_by_statuses([DocStatus.PROCESSED])
        assert count >= 1

    finally:
        await storage.drop()
        await storage.finalize()


# ===================================================================
# GraphStorage end-to-end
# ===================================================================


@pytest.mark.asyncio
async def test_graph_storage_e2e():
    ws = _unique_workspace()
    storage = await _build_graph_storage(ws)
    try:
        # upsert_node
        await storage.upsert_node("entity_a", {"entity_id": "entity_a", "desc": "A"})
        await storage.upsert_node("entity_b", {"entity_id": "entity_b", "desc": "B"})

        # has_node
        assert await storage.has_node("entity_a") is True
        assert await storage.has_node("nonexistent") is False

        # get_node
        node = await storage.get_node("entity_a")
        assert node is not None
        assert node["entity_id"] == "entity_a"
        assert node["desc"] == "A"

        # upsert_edge
        await storage.upsert_edge("entity_a", "entity_b", {"relationship": "knows"})

        # has_edge
        assert await storage.has_edge("entity_a", "entity_b") is True

        # get_edge
        edge = await storage.get_edge("entity_a", "entity_b")
        assert edge is not None
        assert edge["relationship"] == "knows"

        # get_all_labels
        labels = await storage.get_all_labels()
        assert "entity_a" in labels
        assert "entity_b" in labels

        # search_labels
        results = await storage.search_labels("entity")
        assert "entity_a" in results
        assert "entity_b" in results

        # get_popular_labels
        popular = await storage.get_popular_labels(10)
        assert len(popular) >= 2

        # get_knowledge_graph
        kg = await storage.get_knowledge_graph("*")
        assert len(kg.nodes) >= 2
        assert len(kg.edges) >= 1

        # drop
        result = await storage.drop()
        assert result["status"] == "success"

        # Verify data is gone
        assert await storage.has_node("entity_a") is False

    finally:
        await storage.finalize()


# ===================================================================
# content_hash round-trip
# ===================================================================


@pytest.mark.asyncio
async def test_content_hash_roundtrip():
    ws = _unique_workspace()
    storage = await _build_doc_status(ws)
    try:
        await storage.upsert(
            {
                "doc-hash-1": {
                    "content_summary": "Hash test",
                    "content_length": 50,
                    "chunks_count": 1,
                    "status": DocStatus.PROCESSED,
                    "file_path": "hash_test.pdf",
                    "content_hash": "unique_hash_value",
                    "created_at": "2024-01-15 12:00:00",
                }
            }
        )

        result = await storage.get_doc_by_content_hash("unique_hash_value")
        assert result is not None
        doc_id, doc = result
        assert doc_id == "doc-hash-1"
        assert doc["content_hash"] == "unique_hash_value"

        # Non-existent hash
        assert await storage.get_doc_by_content_hash("no_such_hash") is None

    finally:
        await storage.drop()
        await storage.finalize()
