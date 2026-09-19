"""Unit tests for ADBDocStatusStorage database-interacting methods (mock).

All tests mock ``db.query`` / ``db.execute`` to verify SQL generation and
result parsing without a live AnalyticDB connection.
"""

from __future__ import annotations

import datetime
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.base import (
    CURSOR_END,
    CursorAfter,
    DocSchedulingRecord,
    DocStatus,
)
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.kg.adb_mysql_impl import ADBDocStatusStorage
from lightrag.namespace import NameSpace

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_storage() -> ADBDocStatusStorage:
    storage = ADBDocStatusStorage.__new__(ADBDocStatusStorage)
    storage.namespace = NameSpace.DOC_STATUS
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage._max_batch_size = 200
    # Small delete chunk so chunking behaviour is observable in tests.
    storage._max_delete_records_per_batch = 2
    storage.db = MagicMock()
    storage.db.query = AsyncMock()
    storage.db.execute = AsyncMock()
    storage.db.execute_transaction = AsyncMock()
    return storage


def _row(**overrides):
    base = {
        "id": "doc-1",
        "content_summary": "summary",
        "content_length": 42,
        "chunks_count": 3,
        "status": "processed",
        "file_path": "report.pdf",
        "chunks_list": "[]",
        "metadata": "{}",
        "error_msg": None,
        "track_id": None,
        "content_hash": "abc123",
        "created_at": datetime.datetime(2024, 1, 15, 12, 0, 0),
        "updated_at": datetime.datetime(2024, 1, 15, 12, 0, 0),
    }
    base.update(overrides)
    return base


# ===================================================================
# get_by_id_strict
# ===================================================================


@pytest.mark.asyncio
async def test_get_by_id_strict_delegates_to_get_by_id():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]
    result = await storage.get_by_id_strict("doc-1")
    assert result is not None
    assert result["content_length"] == 42


@pytest.mark.asyncio
async def test_get_by_id_strict_none_when_absent():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_by_id_strict("missing") is None


# ===================================================================
# get_doc_by_file_basename
# ===================================================================


@pytest.mark.asyncio
async def test_basename_empty_returns_none():
    storage = _make_storage()
    assert await storage.get_doc_by_file_basename("") is None
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_basename_unknown_source_returns_none():
    storage = _make_storage()
    assert await storage.get_doc_by_file_basename("unknown_source") is None


@pytest.mark.asyncio
async def test_basename_exact_match():
    storage = _make_storage()
    storage.db.query.return_value = [_row(file_path="report.pdf")]
    result = await storage.get_doc_by_file_basename("report.pdf")
    assert result is not None
    doc_id, doc = result
    assert doc_id == "doc-1"
    assert doc["file_path"] == "report.pdf"
    sql = storage.db.query.call_args.args[0]
    assert "LIGHTRAG_DOC_STATUS" in sql
    assert "JSON_EXTRACT" in sql


@pytest.mark.asyncio
async def test_basename_no_match_returns_none():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_doc_by_file_basename("missing.pdf") is None


# ===================================================================
# get_doc_by_content_hash
# ===================================================================


@pytest.mark.asyncio
async def test_content_hash_empty_returns_none():
    storage = _make_storage()
    assert await storage.get_doc_by_content_hash("") is None


@pytest.mark.asyncio
async def test_content_hash_match():
    storage = _make_storage()
    storage.db.query.return_value = [_row(content_hash="hash-abc")]
    result = await storage.get_doc_by_content_hash("hash-abc")
    assert result is not None
    doc_id, doc = result
    assert doc_id == "doc-1"
    sql = storage.db.query.call_args.args[0]
    assert "content_hash=%(content_hash)s" in sql


@pytest.mark.asyncio
async def test_content_hash_no_match():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_doc_by_content_hash("nope") is None


@pytest.mark.asyncio
async def test_content_hash_exclude_doc_id():
    storage = _make_storage()
    storage.db.query.return_value = [_row(id="doc-2", content_hash="hash-abc")]
    result = await storage.get_doc_by_content_hash("hash-abc", exclude_doc_id="doc-1")
    assert result is not None and result[0] == "doc-2"
    sql = storage.db.query.call_args.args[0]
    assert "id <> %(exclude_id)s" in sql


@pytest.mark.asyncio
async def test_content_hash_filters_pointer_rows():
    """Rows marked is_duplicate pointing at exclude_doc_id are skipped."""
    storage = _make_storage()
    pointer_row = _row(
        id="doc-dup",
        content_hash="hash-abc",
        metadata='{"is_duplicate": true, "original_doc_id": "doc-1"}',
    )
    storage.db.query.return_value = [pointer_row]
    result = await storage.get_doc_by_content_hash("hash-abc", exclude_doc_id="doc-1")
    assert result is None


# ===================================================================
# resolve_doc_source_strict
# ===================================================================


@pytest.mark.asyncio
async def test_resolve_unknown_source():
    storage = _make_storage()
    from lightrag.base import SourceAbsent

    result = await storage.resolve_doc_source_strict("unknown_source")
    assert isinstance(result, SourceAbsent)


@pytest.mark.asyncio
async def test_resolve_empty_string():
    storage = _make_storage()
    from lightrag.base import SourceAbsent

    result = await storage.resolve_doc_source_strict("")
    assert isinstance(result, SourceAbsent)


@pytest.mark.asyncio
async def test_resolve_no_rows():
    storage = _make_storage()
    storage.db.query.return_value = []
    from lightrag.base import SourceAbsent

    result = await storage.resolve_doc_source_strict("file.pdf")
    assert isinstance(result, SourceAbsent)


@pytest.mark.asyncio
async def test_resolve_single_row():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]
    from lightrag.base import SourceUnique

    result = await storage.resolve_doc_source_strict("report.pdf")
    assert isinstance(result, SourceUnique)
    assert result.doc_id == "doc-1"


@pytest.mark.asyncio
async def test_resolve_conflict():
    storage = _make_storage()
    storage.db.query.side_effect = [
        [_row(id="doc-1"), _row(id="doc-2")],  # initial LIMIT 2
        {"c": 2},  # COUNT query
    ]
    from lightrag.base import SourceConflict

    result = await storage.resolve_doc_source_strict("report.pdf")
    assert isinstance(result, SourceConflict)
    assert result.candidate_count == 2


# ===================================================================
# get_docs_by_statuses_page
# ===================================================================


@pytest.mark.asyncio
async def test_page_empty_statuses():
    storage = _make_storage()
    page = await storage.get_docs_by_statuses_page([], limit=10)
    assert page.docs == {}
    assert page.next_position is CURSOR_END


@pytest.mark.asyncio
async def test_page_cursor_end():
    storage = _make_storage()
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PROCESSED], limit=10, position=CURSOR_END
    )
    assert page.docs == {}


@pytest.mark.asyncio
async def test_page_single_status():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]
    page = await storage.get_docs_by_statuses_page([DocStatus.PROCESSED], limit=10)
    assert "doc-1" in page.docs
    assert isinstance(page.docs["doc-1"], DocSchedulingRecord)


@pytest.mark.asyncio
async def test_page_keyset_cursor_advance():
    storage = _make_storage()
    rows = [_row(id=f"doc-{i}") for i in range(10)]
    storage.db.query.return_value = rows
    page = await storage.get_docs_by_statuses_page([DocStatus.PROCESSED], limit=10)
    assert len(page.docs) == 10
    assert isinstance(page.next_position, CursorAfter)


@pytest.mark.asyncio
async def test_page_partial_returns_cursor_end():
    storage = _make_storage()
    storage.db.query.return_value = [_row(id="doc-1"), _row(id="doc-2")]
    page = await storage.get_docs_by_statuses_page([DocStatus.PROCESSED], limit=10)
    assert len(page.docs) == 2
    assert page.next_position is CURSOR_END


@pytest.mark.asyncio
async def test_page_invalid_limit():
    storage = _make_storage()
    with pytest.raises(ValueError, match="positive"):
        await storage.get_docs_by_statuses_page([DocStatus.PROCESSED], limit=0)


# ===================================================================
# count_docs_by_statuses
# ===================================================================


@pytest.mark.asyncio
async def test_count_empty_list():
    storage = _make_storage()
    assert await storage.count_docs_by_statuses([]) == 0


@pytest.mark.asyncio
async def test_count_normal():
    storage = _make_storage()
    storage.db.query.return_value = {"cnt": 5}
    count = await storage.count_docs_by_statuses([DocStatus.PROCESSED])
    assert count == 5


@pytest.mark.asyncio
async def test_count_fail_closed():
    storage = _make_storage()
    storage.db.query.return_value = None
    with pytest.raises(StorageControlPlaneError):
        await storage.count_docs_by_statuses([DocStatus.PROCESSED])


# ===================================================================
# update_doc_status_fields
# ===================================================================


@pytest.mark.asyncio
async def test_update_rejects_created_at():
    storage = _make_storage()
    with pytest.raises(ValueError, match="immutable"):
        await storage.update_doc_status_fields("doc-1", {"created_at": "now"})


@pytest.mark.asyncio
async def test_update_rejects_unknown_column():
    storage = _make_storage()
    with pytest.raises(ValueError, match="unknown"):
        await storage.update_doc_status_fields("doc-1", {"bogus_col": "val"})


@pytest.mark.asyncio
async def test_update_empty_fields_exists_check():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]  # full-row read (SELECT *)
    await storage.update_doc_status_fields("doc-1", {})
    storage.db.query.assert_called_once()


@pytest.mark.asyncio
async def test_update_empty_fields_missing_raises():
    storage = _make_storage()
    storage.db.query.return_value = []
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("doc-1", {})


@pytest.mark.asyncio
async def test_update_json_column_serialized():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]  # full-row read (SELECT *)
    await storage.update_doc_status_fields("doc-1", {"metadata": {"key": "value"}})
    execute_call = storage.db.execute.call_args
    params = execute_call.args[1]
    assert isinstance(params["metadata"], str)
    assert json.loads(params["metadata"]) == {"key": "value"}


@pytest.mark.asyncio
async def test_update_stamps_updated_at_server_side():
    """updated_at is always server-stamped via CURRENT_TIMESTAMP.

    Even when the caller passes updated_at, the write relies on the server
    stamp rather than an explicit binding — an explicit ``+00:00`` ISO value
    could never reach a MySQL TIMESTAMP column, which rejects the offset
    suffix (strict mode error 1292).
    """
    storage = _make_storage()
    storage.db.query.return_value = [_row()]  # full-row read (SELECT *)
    await storage.update_doc_status_fields(
        "doc-1", {"updated_at": "2026-08-16T04:00:00+00:00"}
    )
    sql = storage.db.execute.call_args.args[0]
    assert "REPLACE INTO LIGHTRAG_DOC_STATUS" in sql
    assert "%(created_at)s, CURRENT_TIMESTAMP)" in sql
    assert "%(updated_at)s" not in sql


@pytest.mark.asyncio
async def test_update_preserves_untouched_columns():
    """REPLACE INTO rewrites the whole row; every column not named in the
    update must round-trip from the stored row — created_at (the immutable
    keyset sort key) and content_hash most of all."""
    storage = _make_storage()
    storage.db.query.return_value = [_row()]  # full-row read (existence + merge)
    await storage.update_doc_status_fields("doc-1", {"status": "failed"})
    sql = storage.db.execute.call_args.args[0]
    params = storage.db.execute.call_args.args[1]
    assert "REPLACE INTO LIGHTRAG_DOC_STATUS" in sql
    assert params["status"] == "failed"
    assert params["content_hash"] == "abc123"
    assert params["created_at"] == "2024-01-15 12:00:00"
    assert json.loads(params["chunks_list"]) == []


@pytest.mark.asyncio
async def test_update_missing_ok_skips_write():
    storage = _make_storage()
    storage.db.query.return_value = []
    await storage.update_doc_status_fields(
        "doc-x", {"status": "failed"}, missing_ok=True
    )
    storage.db.execute.assert_not_called()


# ===================================================================
# get_docs_by_ids
# ===================================================================


@pytest.mark.asyncio
async def test_docs_by_ids_empty():
    storage = _make_storage()
    result = await storage.get_docs_by_ids([])
    assert result == {}


@pytest.mark.asyncio
async def test_docs_by_ids_strict():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]
    result = await storage.get_docs_by_ids(["doc-1"], strict=True)
    assert "doc-1" in result
    assert isinstance(result["doc-1"], DocSchedulingRecord)


@pytest.mark.asyncio
async def test_docs_by_ids_relaxed_skips_bad_rows():
    storage = _make_storage()
    storage.db.query.return_value = [_row(id=None)]
    result = await storage.get_docs_by_ids(["bad-id"], strict=False)
    assert result == {}


# ===================================================================
# get_full_docs_by_ids
# ===================================================================


@pytest.mark.asyncio
async def test_full_docs_by_ids_empty():
    storage = _make_storage()
    result = await storage.get_full_docs_by_ids([])
    assert result == {}


@pytest.mark.asyncio
async def test_full_docs_by_ids_hydration():
    storage = _make_storage()
    storage.db.query.return_value = [_row()]
    result = await storage.get_full_docs_by_ids(["doc-1"])
    assert "doc-1" in result
    assert result["doc-1"].content_length == 42


@pytest.mark.asyncio
async def test_full_docs_by_ids_strict_raises_on_bad_row():
    storage = _make_storage()
    # A row missing a required field (content_length) should cause KeyError in _parse_row
    bad_row = _row()
    del bad_row["content_length"]
    storage.db.query.return_value = [bad_row]
    with pytest.raises(KeyError):
        await storage.get_full_docs_by_ids(["doc-1"], strict=True)


# ===================================================================
# list_source_conflicts_page
# ===================================================================


@pytest.mark.asyncio
async def test_conflicts_empty():
    storage = _make_storage()
    storage.db.query.return_value = []
    page = await storage.list_source_conflicts_page(limit=10)
    assert page.conflicts == ()
    assert page.next_position is CURSOR_END


@pytest.mark.asyncio
async def test_conflicts_with_results():
    storage = _make_storage()
    storage.db.query.side_effect = [
        [{"file_path": "dup.pdf", "c": 2}],  # GROUP BY query
        [{"id": "doc-1"}, {"id": "doc-2"}],  # sample query
    ]
    page = await storage.list_source_conflicts_page(limit=10)
    assert len(page.conflicts) == 1
    assert page.conflicts[0].canonical_source_key == "dup.pdf"
    assert page.conflicts[0].candidate_count == 2


@pytest.mark.asyncio
async def test_conflicts_cursor_end():
    storage = _make_storage()
    page = await storage.list_source_conflicts_page(limit=10, position=CURSOR_END)
    assert page.conflicts == ()


@pytest.mark.asyncio
async def test_conflicts_invalid_limit():
    storage = _make_storage()
    with pytest.raises(ValueError, match="positive"):
        await storage.list_source_conflicts_page(limit=0)


# ===================================================================
# repair_source_conflict
# ===================================================================


@pytest.mark.asyncio
async def test_repair_dry_run():
    storage = _make_storage()
    storage.db.query.return_value = [{"id": "doc-1"}, {"id": "doc-2"}]
    fp = ADBDocStatusStorage._conflict_fingerprint(["doc-1", "doc-2"])
    result = await storage.repair_source_conflict(
        "report.pdf",
        primary_doc_id="doc-1",
        expected_candidate_count=2,
        expected_candidate_fingerprint=fp,
        dry_run=True,
    )
    assert result.committed is False
    assert result.primary_doc_id == "doc-1"


@pytest.mark.asyncio
async def test_repair_cas_failure():
    storage = _make_storage()
    # Simulate candidate set changing between list and repair
    storage.db.query.return_value = [{"id": "doc-1"}, {"id": "doc-3"}]
    wrong_fp = ADBDocStatusStorage._conflict_fingerprint(["doc-1", "doc-2"])
    with pytest.raises(SourceConflictRepairCASError):
        await storage.repair_source_conflict(
            "report.pdf",
            primary_doc_id="doc-1",
            expected_candidate_count=2,
            expected_candidate_fingerprint=wrong_fp,
            dry_run=False,
        )


@pytest.mark.asyncio
async def test_repair_committed():
    storage = _make_storage()
    fp = ADBDocStatusStorage._conflict_fingerprint(["doc-1", "doc-2"])
    # query calls: (1) re-read candidates, (2) one batched full-row read of
    # the demoted docs (REPLACE INTO needs every column to rewrite the row)
    storage.db.query.side_effect = [
        [{"id": "doc-1"}, {"id": "doc-2"}],  # re-read
        [_row(id="doc-2")],  # demoted rows in full
    ]
    result = await storage.repair_source_conflict(
        "report.pdf",
        primary_doc_id="doc-1",
        expected_candidate_count=2,
        expected_candidate_fingerprint=fp,
        dry_run=False,
    )
    assert result.committed is True
    assert result.demoted_sample_doc_ids == ("doc-2",)
    # Demotions land atomically via the single-connection transaction path,
    # one REPLACE INTO per demoted doc (never per-statement db.execute).
    storage.db.execute_transaction.assert_awaited_once()
    updates = storage.db.execute_transaction.call_args.args[0]
    assert len(updates) == 1
    sql, params = updates[0]
    assert "REPLACE INTO LIGHTRAG_DOC_STATUS" in sql
    assert params["id"] == "doc-2"
    meta = json.loads(params["metadata"])
    assert meta["is_duplicate"] is True
    assert meta["original_doc_id"] == "doc-1"
    # Untouched columns are carried over from the read row, not reset.
    assert params["content_hash"] == "abc123"
    assert params["created_at"] == "2024-01-15 12:00:00"
    storage.db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_repair_primary_not_in_candidates():
    storage = _make_storage()
    storage.db.query.return_value = [{"id": "doc-1"}, {"id": "doc-2"}]
    with pytest.raises(ValueError, match="not a current primary candidate"):
        await storage.repair_source_conflict(
            "report.pdf",
            primary_doc_id="doc-999",
            expected_candidate_count=2,
            expected_candidate_fingerprint="whatever",
            dry_run=True,
        )


# ===================================================================
# DDL regression
# ===================================================================


def test_doc_status_ddl_declares_content_hash():
    """Regression: the code reads/writes content_hash (upsert binding and
    get_doc_by_content_hash filtering) — the DDL must declare the column."""
    from lightrag.kg.adb_mysql_impl import TABLES

    ddl = TABLES["LIGHTRAG_DOC_STATUS"]["ddl"]
    assert "content_hash" in ddl


# ===================================================================
# upsert: created_at written from payload regression
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_writes_created_at_from_payload():
    """Regression: upsert must write created_at from the payload, not the DDL default.

    created_at is the immutable scheduling sort key behind ORDER BY created_at
    keyset pages. All pipeline callers carry it (status resets pass the
    original status_doc.created_at); the SQL must include the column and the
    params must forward the payload value — a REPLACE INTO omitting it would
    silently reset it to now() on every re-enqueue, corrupting FIFO ordering.
    """
    storage = _make_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "s",
                "content_length": 1,
                "status": "pending",
                "file_path": "a.pdf",
                "created_at": "2024-01-15T12:00:00+00:00",
            }
        }
    )
    sql = storage.db.execute.call_args.args[0]
    assert "REPLACE INTO LIGHTRAG_DOC_STATUS" in sql
    # created_at is an explicit column bound to a parameter, never the DDL default
    assert "created_at, updated_at)" in sql
    assert "%(content_hash)s, %(created_at)s, CURRENT_TIMESTAMP)" in sql
    # upsert flushes an executemany list of payload dicts
    batch = storage.db.execute.call_args.args[1]
    assert isinstance(batch, list) and len(batch) == 1
    # tz-aware ISO payload is normalized for the MySQL TIMESTAMP column
    assert batch[0]["created_at"] == "2024-01-15 12:00:00"


@pytest.mark.asyncio
async def test_upsert_empty_data_no_execute():
    storage = _make_storage()
    await storage.upsert({})
    storage.db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_batches_at_max_batch_size():
    """5 records with _max_batch_size=200 flush in ONE executemany call;
    shrinking the cap splits into ceil(n/batch) calls (PG parity)."""
    storage = _make_storage()
    storage._max_batch_size = 2
    payload = {
        f"doc-{i}": {
            "content_summary": "s",
            "content_length": 1,
            "status": "pending",
            "file_path": "a.pdf",
        }
        for i in range(5)
    }
    await storage.upsert(payload)
    assert storage.db.execute.await_count == 3
    sizes = [len(call.args[1]) for call in storage.db.execute.call_args_list]
    assert sizes == [2, 2, 1]


# ===================================================================
# P1-1: timezone-aware datetime emission (PG parity)
# ===================================================================


def test_format_datetime_naive_gets_utc_suffix():
    storage = _make_storage()
    naive = datetime.datetime(2024, 1, 15, 12, 0, 0)  # noqa: DTZ001 (naive is the test input)
    assert storage._format_datetime(naive) == "2024-01-15T12:00:00+00:00"


def test_format_datetime_aware_kept_as_is():
    storage = _make_storage()
    aware = datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
    assert storage._format_datetime(aware) == "2024-01-15T12:00:00+00:00"


def test_format_datetime_none_and_passthrough():
    storage = _make_storage()
    assert storage._format_datetime(None) == ""
    assert storage._format_datetime("already-a-string") == "already-a-string"


@pytest.mark.asyncio
async def test_get_by_id_emits_timezone_aware_timestamps():
    """Regression: aiomysql returns naive datetimes for TIMESTAMP columns;
    read paths must attach UTC so consumers (JS frontend parses naive ISO
    as LOCAL time) see the same ``...+00:00`` form as PGDocStatusStorage."""
    storage = _make_storage()
    storage.db.query.return_value = [_row()]
    result = await storage.get_by_id("doc-1")
    assert result is not None
    assert result["created_at"] == "2024-01-15T12:00:00+00:00"
    assert result["updated_at"] == "2024-01-15T12:00:00+00:00"


# ===================================================================
# P2-2: delete chunking + transaction + set input (PG parity)
# ===================================================================


@pytest.mark.asyncio
async def test_delete_single_chunk_uses_execute():
    storage = _make_storage()
    await storage.delete(["doc-1"])
    storage.db.execute.assert_awaited_once()
    storage.db.execute_transaction.assert_not_called()
    sql = storage.db.execute.call_args.args[0]
    assert "DELETE FROM LIGHTRAG_DOC_STATUS" in sql


@pytest.mark.asyncio
async def test_delete_large_id_list_uses_transaction_chunks():
    """5 ids with chunk=2 -> 3 statements in ONE transaction."""
    storage = _make_storage()
    ids = [f"doc-{i}" for i in range(5)]
    await storage.delete(ids)
    storage.db.execute.assert_not_called()
    storage.db.execute_transaction.assert_awaited_once()
    statements = storage.db.execute_transaction.call_args.args[0]
    assert len(statements) == 3
    placeholder_counts = [stmt[0].count("%(id_") for stmt in statements]
    assert placeholder_counts == [2, 2, 1]


@pytest.mark.asyncio
async def test_delete_accepts_set_input():
    storage = _make_storage()
    await storage.delete({"doc-1", "doc-2"})
    storage.db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_swallows_backend_error():
    storage = _make_storage()
    storage.db.execute.side_effect = RuntimeError("boom")
    await storage.delete(["doc-1"])  # must not raise (PG parity)


# ===================================================================
# P2-3: get_docs_paginated excludes chunks_list (PG parity)
# ===================================================================


@pytest.mark.asyncio
async def test_get_docs_paginated_excludes_chunks_list():
    storage = _make_storage()
    # paged rows carry no chunks_list column at all
    paged_row = _row()
    del paged_row["chunks_list"]
    storage.db.query.side_effect = [{"total": 1}, [paged_row]]
    documents, total = await storage.get_docs_paginated(page=1, page_size=10)
    assert total == 1
    assert len(documents) == 1
    doc_id, doc = documents[0]
    assert doc_id == "doc-1"
    assert doc.chunks_list == []
    data_sql = storage.db.query.call_args_list[1].args[0]
    assert "chunks_list" not in data_sql
    assert "SELECT id, content_summary" in data_sql


# ===================================================================
# P2-4: get_docs_by_track_id relaxed skip (PG parity)
# ===================================================================


@pytest.mark.asyncio
async def test_get_docs_by_track_id_skips_bad_rows():
    storage = _make_storage()
    bad_row = _row(id="doc-bad")
    del bad_row["content_length"]  # KeyError during parse
    storage.db.query.return_value = [bad_row, _row(id="doc-ok")]
    result = await storage.get_docs_by_track_id("track-1")
    assert list(result) == ["doc-ok"]


@pytest.mark.asyncio
async def test_get_docs_by_track_id_empty_result():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.get_docs_by_track_id("track-1") == {}


# ===================================================================
# get_doc_by_file_path
# ===================================================================


@pytest.mark.asyncio
async def test_get_doc_by_file_path_returns_first_row():
    storage = _make_storage()
    storage.db.query.return_value = [_row(file_path="report.pdf")]
    result = await storage.get_doc_by_file_path("report.pdf")
    assert result is not None
    assert result["file_path"] == "report.pdf"
    params = storage.db.query.call_args.args[1]
    assert params["file_path"] == "report.pdf"
    assert params["workspace"] == "test_ws"


@pytest.mark.asyncio
async def test_get_doc_by_file_path_not_found():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_doc_by_file_path("missing.pdf") is None


# ===================================================================
# get_status_counts / get_all_status_counts
# ===================================================================


@pytest.mark.asyncio
async def test_get_status_counts():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"status": "pending", "count": 2},
        {"status": "processed", "count": 5},
    ]
    result = await storage.get_status_counts()
    assert result == {"pending": 2, "processed": 5}
    sql = storage.db.query.call_args.args[0]
    assert "GROUP BY status" in sql


@pytest.mark.asyncio
async def test_get_all_status_counts_includes_all_total():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"status": "pending", "count": 2},
        {"status": "failed", "count": 1},
    ]
    result = await storage.get_all_status_counts()
    assert result == {"pending": 2, "failed": 1, "all": 3}


# ===================================================================
# get_docs_by_statuses relaxed/strict row parsing
# ===================================================================


@pytest.mark.asyncio
async def test_get_docs_by_statuses_empty_list_returns_empty():
    storage = _make_storage()
    assert await storage.get_docs_by_statuses([]) == {}
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_docs_by_statuses_relaxed_skips_bad_rows():
    storage = _make_storage()
    bad_row = _row(id="doc-bad")
    del bad_row["content_length"]  # KeyError during parse
    storage.db.query.return_value = [bad_row, _row(id="doc-ok")]
    result = await storage.get_docs_by_statuses([DocStatus.PROCESSED])
    assert list(result) == ["doc-ok"]
    assert result["doc-ok"].content_summary == "summary"


@pytest.mark.asyncio
async def test_get_docs_by_statuses_strict_raises_on_bad_row():
    storage = _make_storage()
    bad_row = _row(id="doc-bad")
    del bad_row["content_length"]
    storage.db.query.return_value = [bad_row]
    with pytest.raises((KeyError, TypeError)):
        await storage.get_docs_by_statuses([DocStatus.PROCESSED], strict=True)


# ===================================================================
# is_empty
# ===================================================================


@pytest.mark.asyncio
async def test_is_empty_false_when_rows_exist():
    storage = _make_storage()
    storage.db.query.return_value = {"has_data": True}
    assert await storage.is_empty() is False


@pytest.mark.asyncio
async def test_is_empty_true_when_no_rows():
    storage = _make_storage()
    storage.db.query.return_value = {"has_data": False}
    assert await storage.is_empty() is True


@pytest.mark.asyncio
async def test_is_empty_true_on_backend_error():
    storage = _make_storage()
    storage.db.query.side_effect = RuntimeError("boom")
    assert await storage.is_empty() is True
