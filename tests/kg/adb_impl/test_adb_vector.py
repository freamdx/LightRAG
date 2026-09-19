"""Mock-based tests for ADBVectorStorage.

All tests mock ``db.query`` / ``db.execute`` to verify SQL generation and
result shaping without a live AnalyticDB connection.
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag.kg import adb_mysql_impl
from lightrag.kg.adb_mysql_impl import SQL_TEMPLATES, ADBVectorStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.namespace import NameSpace

DIM = 4

# Naive UTC datetimes mirror aiomysql TIMESTAMP rows (MySQL form, no tzinfo).
_RAW_STAMP_A = datetime.datetime(2023, 11, 14, 22, 13, 20)  # noqa: DTZ001
_RAW_STAMP_B = datetime.datetime(2023, 11, 14, 22, 15, 0)  # noqa: DTZ001


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_storage(namespace: str = NameSpace.VECTOR_STORE_CHUNKS) -> ADBVectorStorage:
    storage = ADBVectorStorage.__new__(ADBVectorStorage)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.cosine_better_than_threshold = 0.2
    storage._max_batch_size = 10
    storage.db = MagicMock()
    storage.db.query = AsyncMock()
    storage.db.execute = AsyncMock()
    storage.db.execute_transaction = AsyncMock()

    async def _fake_embed(texts, **kwargs):
        return np.ones((len(texts), DIM), dtype=np.float32)

    storage.embedding_func = AsyncMock(side_effect=_fake_embed)
    storage.embedding_func.embedding_dim = DIM
    return storage


def _chunk_items(n: int) -> dict[str, dict]:
    return {
        f"chunk-{i}": {
            "tokens": 10 + i,
            "chunk_order_index": i,
            "full_doc_id": f"doc-{i}",
            "content": f"content {i}",
            "file_path": f"file{i}.txt",
        }
        for i in range(n)
    }


# ===================================================================
# query — cosine similarity semantics (P0)
# ===================================================================


@pytest.mark.asyncio
async def test_query_uses_cosine_similarity_and_direct_threshold():
    storage = _make_storage()
    storage.db.query.return_value = []
    await storage.query("what is rag?", top_k=5)

    sql = storage.db.query.call_args.args[0]
    params = storage.db.query.call_args.kwargs["params"]
    assert "cosine_similarity" in sql
    assert "l2_distance" not in sql
    assert "ORDER BY similarity DESC" in sql
    # Threshold is a cosine-similarity floor: passed through, NOT inverted.
    assert params["cosine_better_than_threshold"] == 0.2
    assert "closer_than_threshold" not in params
    assert params["top_k"] == 5
    assert params["workspace"] == "test_ws"


@pytest.mark.asyncio
async def test_query_passes_query_context_and_priority():
    storage = _make_storage()
    storage.db.query.return_value = []
    await storage.query("q", top_k=3)
    kwargs = storage.embedding_func.call_args.kwargs
    assert kwargs.get("context") == "query"
    assert "_priority" in kwargs


@pytest.mark.asyncio
async def test_query_precomputed_embedding_skips_embed_call():
    storage = _make_storage()
    storage.db.query.return_value = []
    await storage.query("q", top_k=3, query_embedding=[0.1, 0.2, 0.3, 0.4])
    storage.embedding_func.assert_not_called()
    sql = storage.db.query.call_args.args[0]
    assert "0.1,0.2,0.3,0.4" in sql


@pytest.mark.asyncio
async def test_query_templates_all_use_cosine_similarity():
    for key in ("chunks", "entities", "relationships"):
        tpl = SQL_TEMPLATES[key]
        assert "cosine_similarity" in tpl, key
        assert "l2_distance" not in tpl, key
        assert "ORDER BY similarity DESC" in tpl, key


# ===================================================================
# upsert — document context + REPLACE INTO preserved
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_passes_document_context():
    storage = _make_storage()
    await storage.upsert(_chunk_items(2))
    kwargs = storage.embedding_func.call_args.kwargs
    assert kwargs.get("context") == "document"


@pytest.mark.asyncio
async def test_upsert_uses_replace_into_and_batches():
    storage = _make_storage()
    await storage.upsert(_chunk_items(3))
    storage.db.execute.assert_called_once()
    sql = storage.db.execute.call_args.args[0]
    datas = storage.db.execute.call_args.args[1]
    assert "REPLACE INTO LIGHTRAG_VDB_CHUNKS" in sql
    assert len(datas) == 3
    # Vector serialized as JSON array string
    assert datas[0]["content_vector"].startswith("[")


@pytest.mark.asyncio
async def test_upsert_entity_and_relationship_namespaces():
    entity_storage = _make_storage(NameSpace.VECTOR_STORE_ENTITIES)
    await entity_storage.upsert(
        {
            "ent-1": {
                "entity_name": "ALICE",
                "content": "desc",
                "source_id": "c1<SEP>c2",
            }
        }
    )
    sql = entity_storage.db.execute.call_args.args[0]
    data = entity_storage.db.execute.call_args.args[1][0]
    assert "REPLACE INTO LIGHTRAG_VDB_ENTITY" in sql
    assert data["chunk_ids"] == '["c1", "c2"]'

    rel_storage = _make_storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    await rel_storage.upsert(
        {
            "rel-1": {
                "src_id": "ALICE",
                "tgt_id": "BOB",
                "content": "knows",
                "source_id": "c1",
            }
        }
    )
    sql = rel_storage.db.execute.call_args.args[0]
    data = rel_storage.db.execute.call_args.args[1][0]
    assert "REPLACE INTO LIGHTRAG_VDB_RELATION" in sql
    assert data["source_id"] == "ALICE"
    assert data["target_id"] == "BOB"


# ---------------------------------------------------------------------------
# upsert — create_time bound from the app clock, update_time server-side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_payload_binds_create_time_only():
    storage = _make_storage()
    before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    await storage.upsert(_chunk_items(2))

    datas = storage.db.execute.call_args.args[1]
    stamps = [row["create_time"] for row in datas]
    # One application-clock reading per upsert batch (KV parity).
    assert len(set(stamps)) == 1
    assert stamps[0] >= before
    assert stamps[0].tzinfo is None  # naive UTC — MySQL TIMESTAMP form
    # update_time is server-generated; never bound from Python.
    for row in datas:
        assert "update_time" not in row


@pytest.mark.asyncio
async def test_vector_upsert_templates_bind_create_time():
    for key in ("upsert_chunk", "upsert_entity", "upsert_relationship"):
        tpl = SQL_TEMPLATES[key]
        assert "create_time" in tpl, key
        # create_time is a bound param, update_time is server-generated
        assert "%(create_time)s" in tpl, key
        assert "CURRENT_TIMESTAMP" in tpl, key
        assert "%(update_time)s" not in tpl, key


@pytest.mark.asyncio
async def test_vector_table_ddls_declare_time_columns():
    for name, spec in adb_mysql_impl.VECTOR_TABLES.items():
        ddl = spec["ddl"]
        assert "create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in ddl, name
        assert "update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in ddl, name


# ===================================================================
# deletes — failures must re-raise (P1)
# ===================================================================


@pytest.mark.asyncio
async def test_delete_reraises_on_failure():
    storage = _make_storage()
    storage.db.execute.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await storage.delete(["chunk-1"])


@pytest.mark.asyncio
async def test_delete_entity_reraises_on_failure():
    storage = _make_storage(NameSpace.VECTOR_STORE_ENTITIES)
    storage.db.execute.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await storage.delete_entity("ALICE")


@pytest.mark.asyncio
async def test_delete_entity_relation_reraises_on_failure():
    storage = _make_storage(NameSpace.VECTOR_STORE_RELATIONSHIPS)
    storage.db.execute.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        await storage.delete_entity_relation("ALICE")
    sql = storage.db.execute.call_args.args[0]
    assert "source_id=%(entity_name)s OR target_id=%(entity_name)s" in sql


@pytest.mark.asyncio
async def test_delete_empty_ids_noop():
    storage = _make_storage()
    await storage.delete([])
    storage.db.execute.assert_not_called()


# ===================================================================
# get_by_id / get_by_ids — embedding column stripped (P2)
# ===================================================================


@pytest.mark.asyncio
async def test_get_by_id_strips_content_vector():
    storage = _make_storage()
    # Real row shape from `SELECT *, UNIX_TIMESTAMP(create_time) as created_at`:
    # SELECT * leaks the raw TIMESTAMP physical columns (aiomysql datetimes)
    # alongside the created_at epoch alias; only the vector column is stripped.
    storage.db.query.return_value = {
        "id": "chunk-1",
        "content": "text",
        "content_vector": "[0.1,0.2,0.3,0.4]",
        "create_time": _RAW_STAMP_A,
        "update_time": _RAW_STAMP_A,
        "created_at": 1700000000,
    }
    result = await storage.get_by_id("chunk-1")
    assert result is not None
    assert "content_vector" not in result
    assert result["content"] == "text"
    # Contract: the epoch alias is preserved; raw physical columns pass through.
    assert result["created_at"] == 1700000000
    assert result["create_time"] == _RAW_STAMP_A


@pytest.mark.asyncio
async def test_get_by_id_not_found():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.get_by_id("missing") is None


@pytest.mark.asyncio
async def test_get_by_ids_strips_vector_and_preserves_order():
    storage = _make_storage()
    # Same SELECT * row shape as get_by_id: raw create_time/update_time
    # datetimes plus the created_at epoch alias ride along untouched.
    storage.db.query.return_value = [
        {
            "id": "b",
            "content": "B",
            "content_vector": "[1.0]",
            "create_time": _RAW_STAMP_B,
            "update_time": _RAW_STAMP_B,
            "created_at": 1700000100,
        },
        {
            "id": "a",
            "content": "A",
            "content_vector": "[2.0]",
            "create_time": _RAW_STAMP_A,
            "update_time": _RAW_STAMP_A,
            "created_at": 1700000000,
        },
    ]
    results = await storage.get_by_ids(["a", "b", "missing"])
    assert len(results) == 3
    # Caller-requested order; missing id -> None slot
    assert results[0]["id"] == "a"
    assert results[1]["id"] == "b"
    assert results[2] is None
    for row in results[:2]:
        assert "content_vector" not in row
    assert results[0]["created_at"] == 1700000000
    assert results[1]["created_at"] == 1700000100


@pytest.mark.asyncio
async def test_get_vectors_by_ids_parses_vectors():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"id": "a", "content_vector": "[0.1, 0.2]"},
    ]
    result = await storage.get_vectors_by_ids(["a"])
    assert result == {"a": [0.1, 0.2]}


# ===================================================================
# initialize — dimension from embedding_dim, no dummy embed (P2)
# ===================================================================


@pytest.mark.asyncio
async def test_initialize_uses_embedding_dim_without_embed_call():
    initialize_share_data(1)
    storage = _make_storage()
    storage.db = None  # force the create-db branch

    mock_db = MagicMock()
    mock_db.initdb = AsyncMock()
    mock_db.query = AsyncMock(return_value=None)  # every table missing
    mock_db.execute = AsyncMock()
    mock_db.workspace = ""

    with (
        patch.object(adb_mysql_impl, "AnalyticDB", return_value=mock_db),
        # XUANWU_V2 tables provision asynchronously
        patch.object(adb_mysql_impl.asyncio, "sleep", new=AsyncMock()),
    ):
        await storage.initialize()

    # No dummy embedding call; dim comes from embedding_func.embedding_dim
    storage.embedding_func.assert_not_called()
    calls = [c.args[0] for c in mock_db.execute.call_args_list]
    assert len(calls) == 2 * len(adb_mysql_impl.VECTOR_TABLES)
    create_ddls, ann_ddls = calls[0::2], calls[1::2]
    for ddl in create_ddls:
        assert f"ARRAY<FLOAT>({DIM})" in ddl
        assert "EMBEDDING_DIM" not in ddl
        assert "ENGINE='XUANWU_V2'" in ddl
        assert "distancemeasure" not in ddl  # index is a separate DDL now
    for ddl in ann_ddls:
        assert "ANN INDEX idx_content_vector" in ddl
        assert "distancemeasure=CosineSimilarity" in ddl


# ===================================================================
# drop
# ===================================================================


@pytest.mark.asyncio
async def test_drop_workspace_scoped_delete():
    storage = _make_storage()
    result = await storage.drop()
    assert result["status"] == "success"
    sql = storage.db.execute.call_args.args[0]
    assert "DELETE FROM LIGHTRAG_VDB_CHUNKS" in sql
    assert "workspace=%(workspace)s" in sql
