"""Mock-based tests for ADBKVStorage.

All tests mock ``db.query`` / ``db.execute`` to verify SQL generation,
field coverage (PG parity: pipeline fields on full_docs, heading/sidecar on
text_chunks), and result shaping without a live AnalyticDB connection.
"""

from __future__ import annotations

import datetime
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.adb_mysql_impl import SQL_TEMPLATES, TABLES, ADBKVStorage
from lightrag.namespace import NameSpace

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_kv(namespace: str = NameSpace.KV_STORE_TEXT_CHUNKS) -> ADBKVStorage:
    storage = ADBKVStorage.__new__(ADBKVStorage)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage._max_batch_size = 200
    storage._max_delete_records_per_batch = 2
    storage.db = MagicMock()
    storage.db.query = AsyncMock()
    storage.db.execute = AsyncMock()
    storage.db.execute_transaction = AsyncMock()
    return storage


def _chunk_row(**overrides):
    """Real row shape of the get_by_id(s)_text_chunks explicit column list.

    The read template COALESCEs content/llm_cache_list/heading/sidecar, so a
    live row always carries those keys; create_time/update_time arrive as
    UNIX_TIMESTAMP epoch ints under their physical column names.
    """
    base = {
        "id": "chunk-1",
        "tokens": 10,
        "content": "text",
        "chunk_order_index": 0,
        "full_doc_id": "doc-1",
        "file_path": "f.txt",
        "llm_cache_list": "[]",
        "heading": "{}",
        "sidecar": "{}",
        "create_time": 100,
        "update_time": 200,
    }
    base.update(overrides)
    return base


def _full_doc_row(**overrides):
    """Real row shape of the get_by_id(s)_full_docs explicit column list."""
    base = {
        "id": "doc-1",
        "content": "body",
        "file_path": "f.pdf",
        "sidecar_location": None,
        "parse_format": None,
        "content_hash": None,
        "process_options": None,
        "chunk_options": "{}",
        "parse_engine": None,
    }
    base.update(overrides)
    return base


# ===================================================================
# SQL templates — field coverage vs PGKVStorage (P1)
# ===================================================================


def test_full_docs_templates_cover_pipeline_fields():
    pipeline_fields = (
        "sidecar_location",
        "parse_format",
        "content_hash",
        "process_options",
        "chunk_options",
        "parse_engine",
    )
    for key in ("get_by_id_full_docs", "get_by_ids_full_docs", "upsert_doc_full"):
        tpl = SQL_TEMPLATES[key]
        for field_name in pipeline_fields:
            assert field_name in tpl, f"{key} missing {field_name}"


def test_text_chunks_templates_cover_heading_sidecar():
    for key in (
        "get_by_id_text_chunks",
        "get_by_ids_text_chunks",
        "upsert_text_chunk",
    ):
        tpl = SQL_TEMPLATES[key]
        assert "heading" in tpl, key
        assert "sidecar" in tpl, key


def test_doc_full_ddl_declares_pipeline_fields():
    ddl = TABLES["LIGHTRAG_DOC_FULL"]["ddl"]
    for field_name in (
        "sidecar_location",
        "parse_format",
        "content_hash",
        "process_options",
        "chunk_options",
        "parse_engine",
    ):
        assert field_name in ddl
    chunks_ddl = TABLES["LIGHTRAG_DOC_CHUNKS"]["ddl"]
    assert "heading" in chunks_ddl
    assert "sidecar" in chunks_ddl


# ===================================================================
# get_by_id — JSON parsing / normalization
# ===================================================================


@pytest.mark.asyncio
async def test_get_by_id_text_chunks_parses_json_fields():
    storage = _make_kv()
    storage.db.query.return_value = _chunk_row(
        llm_cache_list='["cache-a"]',
        heading='{"heading": "Intro", "level": 1}',
        sidecar='{"type": "table", "id": "tbl-1"}',
        create_time=100,
        update_time=0,
    )
    row = await storage.get_by_id("chunk-1")
    assert row["llm_cache_list"] == ["cache-a"]
    assert row["heading"] == {"heading": "Intro", "level": 1}
    assert row["sidecar"] == {"type": "table", "id": "tbl-1"}
    # update_time falls back to create_time when stored as 0
    assert row["update_time"] == 100


@pytest.mark.asyncio
async def test_get_by_id_text_chunks_normalizes_empty_json_to_empty_dict():
    storage = _make_kv()
    # COALESCE in the read template guarantees heading/sidecar arrive as the
    # '{}' string when stored NULL; parsing must normalize them to dicts.
    storage.db.query.return_value = _chunk_row()
    row = await storage.get_by_id("chunk-1")
    assert row["heading"] == {}
    assert row["sidecar"] == {}


@pytest.mark.asyncio
async def test_get_by_id_full_docs_parses_chunk_options():
    storage = _make_kv(NameSpace.KV_STORE_FULL_DOCS)
    storage.db.query.return_value = _full_doc_row(
        sidecar_location="/data/f.sidecar.json",
        parse_format="lightrag",
        content_hash="abc123",
        process_options="i",
        chunk_options='{"chunk_method": "paragraph"}',
        parse_engine="native",
    )
    row = await storage.get_by_id("doc-1")
    assert row["chunk_options"] == {"chunk_method": "paragraph"}
    # Pipeline fields pass through for resume/reprocess decisions
    assert row["parse_format"] == "lightrag"
    assert row["sidecar_location"] == "/data/f.sidecar.json"
    assert row["content_hash"] == "abc123"
    assert row["parse_engine"] == "native"


@pytest.mark.asyncio
async def test_get_by_id_llm_cache_reshapes_fields():
    storage = _make_kv(NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
    storage.db.query.return_value = {
        "id": "cache-1",
        "original_prompt": "prompt",
        "return_value": "answer",
        "chunk_id": "chunk-1",
        "cache_type": "extract",
        "queryparam": '{"mode": "hybrid"}',
        "create_time": 100,
        "update_time": 0,
    }
    row = await storage.get_by_id("cache-1")
    assert row["return"] == "answer"
    assert row["queryparam"] == {"mode": "hybrid"}
    assert row["update_time"] == 100


@pytest.mark.asyncio
async def test_get_by_id_missing_returns_none():
    storage = _make_kv()
    storage.db.query.return_value = None
    assert await storage.get_by_id("nope") is None


@pytest.mark.asyncio
async def test_get_by_id_strict_delegates_to_get_by_id():
    storage = _make_kv()
    storage.db.query.return_value = _chunk_row(create_time=1, update_time=2)
    row = await storage.get_by_id_strict("chunk-1")
    assert row["id"] == "chunk-1"


# ===================================================================
# get_by_ids — ordering + normalization
# ===================================================================


@pytest.mark.asyncio
async def test_get_by_ids_text_chunks_orders_and_normalizes():
    storage = _make_kv()
    storage.db.query.return_value = [
        _chunk_row(
            id="chunk-b",
            heading='{"heading": "B"}',
            create_time=1,
            update_time=2,
        ),
        _chunk_row(
            id="chunk-a",
            sidecar='{"type": "drawing", "id": "d-1"}',
            create_time=3,
            update_time=4,
        ),
    ]
    rows = await storage.get_by_ids(["chunk-a", "chunk-b", "chunk-missing"])
    assert [r["id"] if r else None for r in rows] == [
        "chunk-a",
        "chunk-b",
        None,
    ]
    assert rows[0]["heading"] == {}
    assert rows[0]["sidecar"] == {"type": "drawing", "id": "d-1"}
    assert rows[1]["heading"] == {"heading": "B"}
    assert rows[1]["sidecar"] == {}


@pytest.mark.asyncio
async def test_get_by_ids_full_docs_chunk_options_normalized():
    storage = _make_kv(NameSpace.KV_STORE_FULL_DOCS)
    storage.db.query.return_value = [_full_doc_row(chunk_options="{}")]
    rows = await storage.get_by_ids(["doc-1"])
    assert rows[0]["chunk_options"] == {}


@pytest.mark.asyncio
async def test_get_by_ids_empty_input_returns_empty():
    storage = _make_kv()
    assert await storage.get_by_ids([]) == []
    storage.db.query.assert_not_called()


# ===================================================================
# upsert — new fields + batching + unknown namespace (P1/P2)
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_text_chunk_writes_heading_sidecar():
    storage = _make_kv()
    await storage.upsert(
        {
            "chunk-1": {
                "tokens": 5,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "content": "text",
                "file_path": "f.txt",
                "heading": {"heading": "Intro", "level": 1},
                "sidecar": {"type": "table", "id": "tbl-1"},
            }
        }
    )
    sql = storage.db.execute.call_args.args[0]
    data = storage.db.execute.call_args.args[1][0]
    assert "REPLACE INTO LIGHTRAG_DOC_CHUNKS" in sql
    assert json.loads(data["heading"]) == {"heading": "Intro", "level": 1}
    assert json.loads(data["sidecar"]) == {"type": "table", "id": "tbl-1"}


@pytest.mark.asyncio
async def test_upsert_text_chunk_defaults_heading_sidecar_to_empty():
    storage = _make_kv()
    await storage.upsert(
        {
            "chunk-1": {
                "tokens": 5,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "content": "text",
                "file_path": "f.txt",
            }
        }
    )
    data = storage.db.execute.call_args.args[1][0]
    assert data["heading"] == "{}"
    assert data["sidecar"] == "{}"


@pytest.mark.asyncio
async def test_upsert_full_docs_writes_pipeline_fields():
    storage = _make_kv(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert(
        {
            "doc-1": {
                "content": "body",
                "file_path": "f.pdf",
                "sidecar_location": "/data/f.sidecar.json",
                "parse_format": "lightrag",
                "content_hash": "abc123",
                "process_options": "i",
                "chunk_options": {"chunk_method": "paragraph"},
                "parse_engine": "native",
            }
        }
    )
    sql = storage.db.execute.call_args.args[0]
    data = storage.db.execute.call_args.args[1][0]
    assert "REPLACE INTO LIGHTRAG_DOC_FULL" in sql
    assert data["sidecar_location"] == "/data/f.sidecar.json"
    assert data["parse_format"] == "lightrag"
    assert data["content_hash"] == "abc123"
    assert data["process_options"] == "i"
    assert json.loads(data["chunk_options"]) == {"chunk_method": "paragraph"}
    assert data["parse_engine"] == "native"


@pytest.mark.asyncio
async def test_upsert_full_docs_defaults_missing_meta():
    storage = _make_kv(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert({"doc-1": {"content": "body"}})
    data = storage.db.execute.call_args.args[1][0]
    assert data["sidecar_location"] is None
    assert data["parse_format"] is None
    assert data["content_hash"] is None
    assert data["process_options"] is None
    assert data["parse_engine"] is None
    assert data["chunk_options"] == "{}"
    assert data["doc_name"] == ""


@pytest.mark.asyncio
async def test_upsert_unknown_namespace_raises():
    storage = _make_kv("bogus_namespace")
    with pytest.raises(ValueError, match="Unknown namespace"):
        await storage.upsert({"k": {"content": "v"}})


@pytest.mark.asyncio
async def test_upsert_batches_at_max_batch_size():
    storage = _make_kv(NameSpace.KV_STORE_FULL_DOCS)
    storage._max_batch_size = 2
    data = {f"doc-{i}": {"content": f"body-{i}"} for i in range(5)}
    await storage.upsert(data)
    # 5 records / batch 2 -> executes of size 2, 2, 1
    assert storage.db.execute.await_count == 3
    sizes = [len(call.args[1]) for call in storage.db.execute.call_args_list]
    assert sizes == [2, 2, 1]


# ===================================================================
# delete — bounded IN clauses (P2)
# ===================================================================


@pytest.mark.asyncio
async def test_delete_single_chunk_uses_execute():
    storage = _make_kv()
    await storage.delete(["chunk-1", "chunk-2"])
    storage.db.execute.assert_called_once()
    storage.db.execute_transaction.assert_not_called()
    sql = storage.db.execute.call_args.args[0]
    assert "DELETE FROM LIGHTRAG_DOC_CHUNKS" in sql
    assert "workspace=%(workspace)s" in sql


@pytest.mark.asyncio
async def test_delete_large_id_list_uses_transaction_chunks():
    storage = _make_kv()
    ids = [f"chunk-{i}" for i in range(5)]
    await storage.delete(ids)
    # 5 ids / batch 2 -> 3 chunked statements in ONE transaction
    storage.db.execute.assert_not_called()
    storage.db.execute_transaction.assert_called_once()
    statements = storage.db.execute_transaction.call_args.args[0]
    assert len(statements) == 3
    total_placeholders = sum(sql.count("%(id_") for sql, _params in statements)
    assert total_placeholders == 5


@pytest.mark.asyncio
async def test_delete_accepts_set_input():
    storage = _make_kv()
    await storage.delete({"chunk-1"})
    storage.db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_empty_is_noop():
    storage = _make_kv()
    await storage.delete([])
    storage.db.execute.assert_not_called()
    storage.db.execute_transaction.assert_not_called()


# ===================================================================
# upsert — create_time bound from the app clock, update_time server-side
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_payload_binds_create_time_only():
    storage = _make_kv()
    before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    await storage.upsert(
        {
            "chunk-1": {
                "tokens": 5,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "content": "text",
                "file_path": "f.txt",
            }
        }
    )
    # No pre-fetch round-trip
    storage.db.query.assert_not_called()
    data = storage.db.execute.call_args.args[1][0]
    # create_time comes from the application clock; update_time is left to
    # the server-side CURRENT_TIMESTAMP
    assert data["create_time"] >= before
    assert "update_time" not in data


def test_kv_upsert_templates_bind_create_time():
    for key in (
        "upsert_doc_full",
        "upsert_text_chunk",
        "upsert_llm_response_cache",
        "upsert_full_entities",
        "upsert_full_relations",
        "upsert_entity_chunks",
        "upsert_relation_chunks",
    ):
        tpl = SQL_TEMPLATES[key]
        assert "create_time" in tpl, key
        assert "update_time" in tpl, key
        # create_time is a bound param, update_time is server-generated
        assert "%(create_time)s" in tpl, key
        assert "CURRENT_TIMESTAMP" in tpl, key
        assert "%(update_time)s" not in tpl, key


# ===================================================================
# filter_keys / drop
# ===================================================================


@pytest.mark.asyncio
async def test_filter_keys_returns_missing_keys():
    storage = _make_kv(NameSpace.KV_STORE_FULL_DOCS)
    storage.db.query.return_value = [{"id": "doc-1"}]
    missing = await storage.filter_keys({"doc-1", "doc-2"})
    assert missing == {"doc-2"}


@pytest.mark.asyncio
async def test_drop_workspace_scoped_delete():
    storage = _make_kv()
    result = await storage.drop()
    assert result["status"] == "success"
    sql = storage.db.execute.call_args.args[0]
    params = storage.db.execute.call_args.args[1]
    assert "DELETE FROM LIGHTRAG_DOC_CHUNKS" in sql
    assert params == {"workspace": "test_ws"}
