"""Unit tests for ADB storage pure-logic helper methods.

Covers helpers that do not require a live database connection:
  * _format_datetime, _to_mysql_datetime, _parse_json_field, _parse_row (DocStatusStorage)
  * _scheduling_record_from_row (DocStatusStorage)
  * _decode_cursor / _encode_cursor (DocStatusStorage)
  * _conflict_fingerprint / _decode_conflict_cursor (DocStatusStorage)
  * _escape_like, _json_loads, _node_props, _node_output (GraphStorage)
  * build_in_clause, missing_tables, sync_workspace, execute_transaction (AnalyticDB)
  * ADBClientManager (the process-wide pool shared by every ADB storage)
  * max_batch_size / in_clause_max_size (flush sizes on the shared client)
  * write-statement batchability (the aiomysql executemany fast path)
"""

from __future__ import annotations

import asyncio
import datetime
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiomysql.cursors import RE_INSERT_VALUES

from lightrag.base import (
    DocSchedulingRecord,
    DocStatus,
)
from lightrag.exceptions import StorageControlPlaneError
from lightrag.kg import adb_mysql_impl
from lightrag.kg.adb_mysql_impl import (
    ADBClientManager,
    ADBDocStatusStorage,
    ADBGraphStorage,
    AnalyticDB,
)
from lightrag.kg.shared_storage import initialize_share_data
from tests.kg.adb_impl.test_adb_doc_status import (
    _make_storage as _make_doc_status,
)
from tests.kg.adb_impl.test_adb_kv import _make_kv
from tests.kg.adb_impl.test_adb_vector import (
    _chunk_items,
)
from tests.kg.adb_impl.test_adb_vector import (
    _make_storage as _make_vector,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_graph() -> ADBGraphStorage:
    """Create an ADBGraphStorage bypassing __init__."""
    storage = ADBGraphStorage.__new__(ADBGraphStorage)
    storage.namespace = "chunk_entity_relation"
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = MagicMock()
    return storage


def _row(**overrides):
    """Build a mock DB row with sensible defaults."""
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
# AnalyticDB.build_in_clause
# ===================================================================


def test_placeholders_are_named_by_field():
    """Every value gets its OWN named bind; sharing one placeholder would drop
    all but the first item of the IN list."""
    cases = [
        ("id", [], ("", {})),
        ("id", ["alpha"], ("%(id_0)s", {"id_0": "alpha"})),
        (
            "status",
            ["a", "b", "c"],
            (
                "%(status_0)s,%(status_1)s,%(status_2)s",
                {"status_0": "a", "status_1": "b", "status_2": "c"},
            ),
        ),
    ]
    for field_name, values, expected in cases:
        assert AnalyticDB.build_in_clause(field_name, values) == expected, values


# ===================================================================
# AnalyticDB.execute_transaction
# ===================================================================


def _make_adb_with_pool() -> tuple[AnalyticDB, MagicMock, AsyncMock]:
    """Build an AnalyticDB with a fully mocked pool/connection/cursor chain."""
    db = AnalyticDB.__new__(AnalyticDB)
    cursor = AsyncMock()
    cursor_cm = MagicMock()
    cursor_cm.__aenter__ = AsyncMock(return_value=cursor)
    cursor_cm.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.begin = AsyncMock()
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    conn.cursor = MagicMock(return_value=cursor_cm)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    db.pool = MagicMock()
    db.pool.acquire = MagicMock(return_value=acquire_cm)
    return db, conn, cursor


class TestQueryAndExecute:
    """The routing every strict read and every batch write is built on."""

    @pytest.mark.asyncio
    async def test_query_single_row_shape(self):
        db, _conn, cursor = _make_adb_with_pool()
        cursor.fetchone = AsyncMock(return_value={"a": 1})
        row = await db.query("SELECT a FROM t")
        assert row == {"a": 1}
        # params None must not pass a second positional to the driver
        cursor.execute.assert_awaited_once_with("SELECT a FROM t")

    @pytest.mark.asyncio
    async def test_query_no_row_is_none_not_empty_dict(self):
        db, _conn, cursor = _make_adb_with_pool()
        cursor.fetchone = AsyncMock(return_value=None)
        assert await db.query("SELECT a FROM t WHERE 0") is None

    @pytest.mark.asyncio
    async def test_query_multirows_empty_is_list_not_none(self):
        """strict reads tell a miss from a failure on this value."""
        db, _conn, cursor = _make_adb_with_pool()
        cursor.fetchall = AsyncMock(return_value=None)
        assert await db.query("SELECT a FROM t WHERE 0", multirows=True) == []

    @pytest.mark.asyncio
    async def test_query_multirows_rows_are_dicts(self):
        db, _conn, cursor = _make_adb_with_pool()
        cursor.fetchall = AsyncMock(return_value=[{"a": 1}, {"a": 2}])
        assert await db.query("SELECT a FROM t", multirows=True) == [
            {"a": 1},
            {"a": 2},
        ]

    @pytest.mark.asyncio
    async def test_query_error_propagates(self):
        db, _conn, cursor = _make_adb_with_pool()
        cursor.execute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await db.query("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_dict_uses_executemany_for_list(self):
        db, _conn, cursor = _make_adb_with_pool()
        await db.execute("REPLACE INTO t VALUES (%(a)s)", {"a": 1})
        cursor.execute.assert_awaited_once_with(
            "REPLACE INTO t VALUES (%(a)s)", {"a": 1}
        )
        cursor.executemany.assert_not_awaited()

        await db.execute("REPLACE INTO t VALUES (%(a)s)", [{"a": 1}, {"a": 2}])
        cursor.executemany.assert_awaited_once_with(
            "REPLACE INTO t VALUES (%(a)s)", [{"a": 1}, {"a": 2}]
        )

    @pytest.mark.asyncio
    async def test_execute_error_propagates(self):
        db, _conn, cursor = _make_adb_with_pool()
        cursor.execute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await db.execute("DELETE FROM t")


class TestExecuteTransaction:
    """Regression: multi-statement atomicity must run on ONE connection.

    ``query``/``execute`` each acquire their own pooled connection, so bare
    START TRANSACTION/COMMIT issued through them cannot span statements.
    """

    @pytest.mark.asyncio
    async def test_commits_on_single_connection(self):
        db, conn, cursor = _make_adb_with_pool()
        await db.execute_transaction(
            [
                ("UPDATE t SET a=%(a)s", {"a": 1}),
                ("DELETE FROM t WHERE x=1", None),
                ("INSERT INTO t VALUES (%s)", [{"b": 2}, {"b": 3}]),
            ]
        )
        # Exactly one connection acquired for the whole batch
        assert db.pool.acquire.call_count == 1
        conn.begin.assert_awaited_once()
        conn.commit.assert_awaited_once()
        conn.rollback.assert_not_awaited()
        assert cursor.execute.await_count == 2
        cursor.executemany.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rolls_back_and_reraises_on_error(self):
        db, conn, cursor = _make_adb_with_pool()
        cursor.execute.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await db.execute_transaction([("UPDATE t SET a=%(a)s", {"a": 1})])
        conn.rollback.assert_awaited_once()
        conn.commit.assert_not_awaited()


# ===================================================================
# ADBDocStatusStorage decoding helpers (_format_datetime,
# _to_mysql_datetime, _parse_json_field, _parse_row,
# _scheduling_record_from_row)
# ===================================================================


def test_format_datetime_emission_is_utc_normalised():
    """Naive datetimes (as returned by aiomysql) are interpreted as UTC and
    emitted with the +00:00 suffix, matching PGDocStatusStorage. A tz-aware
    input keeps its own offset; only naive values are labelled UTC.
    """
    storage = _make_doc_status()
    utc = datetime.timezone.utc
    cases = [
        (datetime.datetime(2024, 6, 15, 10, 30, 0), "2024-06-15T10:30:00+00:00"),
        (
            datetime.datetime(2024, 6, 15, 10, 30, 0, tzinfo=utc),
            "2024-06-15T10:30:00+00:00",
        ),
        (
            datetime.datetime(
                2024,
                6,
                15,
                10,
                30,
                0,
                tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
            ),
            "2024-06-15T10:30:00+08:00",
        ),
        (None, ""),
        ("2024-01-01T00:00:00", "2024-01-01T00:00:00"),
    ]
    for value, expected in cases:
        assert storage._format_datetime(value) == expected, value


def test_to_mysql_datetime_normalises_payload_timestamps():
    """Callers pass tz-aware ISO-8601 strings (``...+00:00``); MySQL datetime
    literals do not accept a UTC offset suffix (strict mode error 1292), so
    every accepted shape is converted and re-verified from this table.
    """
    storage = _make_doc_status()
    utc = datetime.timezone.utc
    cases = [
        ("2024-01-15T12:00:00+00:00", "2024-01-15 12:00:00"),
        # Non-UTC offsets are converted to UTC, not merely stripped.
        ("2024-06-01T17:00:00+05:00", "2024-06-01 12:00:00"),
        ("2024-01-15T12:00:00", "2024-01-15 12:00:00"),
        (datetime.datetime(2024, 1, 15, 12, 0, 0, tzinfo=utc), "2024-01-15 12:00:00"),
        (datetime.datetime(2024, 1, 15, 12, 0, 0), "2024-01-15 12:00:00"),
        (None, None),
        # Not ISO-8601: bind as-is and let the server reject it.
        ("not-a-date", "not-a-date"),
    ]
    for value, expected in cases:
        assert storage._to_mysql_datetime(value) == expected, value


def test_parse_json_field_decodes_or_returns_default():
    storage = _make_doc_status()
    already = {"key": "value"}
    cases = [
        ('{"a": 1}', None, {"a": 1}),
        ("[1, 2, 3]", None, [1, 2, 3]),
        ("not-json", [], []),
        (None, {}, {}),
        # An already-decoded dict is passed through by identity, not copied.
        (already, None, already),
    ]
    for value, default, expected in cases:
        result = storage._parse_json_field(value, default)
        assert result == expected, value
        if isinstance(value, dict):
            assert result is value


def test_parse_row_normalises_every_column_shape():
    storage = _make_doc_status()
    dt = datetime.datetime(2024, 3, 1, 8, 0, 0)
    cases = [
        (
            "full row",
            _row(),
            {
                "content_length": 42,
                "status": "processed",
                "file_path": "report.pdf",
                "content_hash": "abc123",
                "chunks_list": [],
                "metadata": {},
            },
        ),
        (
            "naive datetimes are emitted as UTC",
            _row(created_at=dt, updated_at=dt),
            {
                "created_at": "2024-03-01T08:00:00+00:00",
                "updated_at": "2024-03-01T08:00:00+00:00",
            },
        ),
        (
            "null file_path defaults",
            _row(file_path=None),
            {"file_path": "no-file-path"},
        ),
        (
            "JSON columns parsed from strings",
            _row(chunks_list='["c1","c2"]', metadata='{"key":"val"}'),
            {"chunks_list": ["c1", "c2"], "metadata": {"key": "val"}},
        ),
        (
            "invalid JSON metadata defaults to {}",
            _row(metadata="not-json"),
            {"metadata": {}},
        ),
    ]
    for label, row, expected in cases:
        parsed = storage._parse_row(row)
        for key, value in expected.items():
            assert parsed[key] == value, f"{label}: {key}"


def test_scheduling_record_from_row_strict_and_relaxed():
    """strict=True fails loud on a broken row; strict=False drops it.

    A malformed row must never be silently coerced into a default status --
    the scheduler would then re-enqueue a document it cannot describe.
    """
    storage = _make_doc_status()
    record = storage._scheduling_record_from_row(_row(), strict=True)
    assert isinstance(record, DocSchedulingRecord)
    assert (record.id, record.status, record.file_path) == (
        "doc-1",
        DocStatus.PROCESSED,
        "report.pdf",
    )

    cases = [
        ("id", _row(id=None), KeyError),
        # An unknown status string is as fatal as a missing one.
        ("status", _row(status=None), (KeyError, ValueError)),
    ]
    for label, row, raises in cases:
        with pytest.raises(raises):
            storage._scheduling_record_from_row(row, strict=True)
        assert storage._scheduling_record_from_row(row, strict=False) is None, label


# ===================================================================
# ADBDocStatusStorage._decode_cursor / _encode_cursor
# ===================================================================


def test_cursor_round_trips_the_keyset_it_was_built_from():
    """A page cursor returns exactly its (created_at, id) key, including the
    NULL created_at that opens the first page."""
    storage = _make_doc_status()
    dt = datetime.datetime(2024, 6, 1, 12, 0, 0)
    for row, expected_dt in (
        ({"id": "doc-42", "created_at": dt}, dt),
        ({"id": "doc-99", "created_at": None}, None),
    ):
        decoded_dt, decoded_id = ADBDocStatusStorage._decode_cursor(
            storage._encode_cursor(row)
        )
        assert (decoded_id, decoded_dt) == (row["id"], expected_dt), row


def test_decode_cursor_rejects_anything_that_is_not_a_keyset():
    """A malformed cursor is a control-plane error, never a silent restart from
    the first page -- that would re-emit rows the caller already has."""
    for cursor in ("not-json", json.dumps([123, "doc"])):
        with pytest.raises(StorageControlPlaneError):
            ADBDocStatusStorage._decode_cursor(cursor)

    # A tz-aware keyset decodes to naive UTC, the convention every read uses.
    tz = datetime.timezone(datetime.timedelta(hours=5))
    dt_aware = datetime.datetime(2024, 6, 1, 17, 0, 0, tzinfo=tz)
    decoded_dt, decoded_id = ADBDocStatusStorage._decode_cursor(
        json.dumps([dt_aware.isoformat(), "doc-tz"])
    )
    assert (decoded_id, decoded_dt.tzinfo, decoded_dt) == (
        "doc-tz",
        None,
        datetime.datetime(2024, 6, 1, 12, 0, 0),
    )


# ===================================================================
# ADBDocStatusStorage._conflict_fingerprint / _decode_conflict_cursor
# ===================================================================


def test_conflict_fingerprint_is_order_sensitive():
    fp1 = ADBDocStatusStorage._conflict_fingerprint(["a", "b"])
    fp2 = ADBDocStatusStorage._conflict_fingerprint(["b", "a"])
    assert fp1 != fp2


def test_only_a_json_string_conflict_cursor_is_accepted():
    """Any other JSON type is rejected: a list or number would silently shift
    the conflict page instead of failing the caller's cursor."""
    assert (
        ADBDocStatusStorage._decode_conflict_cursor(json.dumps("some-key"))
        == "some-key"
    )
    for cursor in ("not-json", "null", "123", "[1, 2]", '{"k": 1}'):
        with pytest.raises(StorageControlPlaneError):
            ADBDocStatusStorage._decode_conflict_cursor(cursor)


# ===================================================================
# ADBGraphStorage static helpers (_escape_like, _json_loads,
# _node_props, _node_output, _gp)
# ===================================================================


def test_escape_like_quotes_every_like_metacharacter():
    for value, expected in (
        ("100%", "100\\%"),
        ("a_b", "a\\_b"),
        ("a\\b", "a\\\\b"),
        ("a%b_c\\d", "a\\%b\\_c\\\\d"),
        ("hello", "hello"),
    ):
        assert ADBGraphStorage._escape_like(value) == expected, value


def test_json_loads_normalises_non_dict_json_and_none():
    """Malformed strings RAISE; non-dict JSON and None normalize to {}."""
    for value, expected in (
        ('{"a": 1}', {"a": 1}),
        ({"x": 1}, {"x": 1}),
        (None, {}),
        ("[1, 2]", {}),
    ):
        assert ADBGraphStorage._json_loads(value) == expected, value
    with pytest.raises(json.JSONDecodeError):
        ADBGraphStorage._json_loads("not-json")


def test_node_props_and_node_output_force_the_identity_keys():
    """entity_id (and id) come from the row key, never from the stored payload,
    so a stale payload cannot rename a node. Row presence, not payload
    truthiness, decides membership."""
    props = ADBGraphStorage._node_props("n1", '{"entity_id": "wrong", "desc": "x"}')
    assert props == {"entity_id": "n1", "desc": "x"}
    assert ADBGraphStorage._node_props("n2", None) == {"entity_id": "n2"}
    assert ADBGraphStorage._node_output("n1", '{"desc": "x"}') == {
        "id": "n1",
        "entity_id": "n1",
        "desc": "x",
    }


def test_gp_carries_workspace_namespace_and_extras():
    storage = _make_graph()
    base = {"workspace": "test_ws", "namespace": "chunk_entity_relation"}
    assert storage._gp() == base
    assert storage._gp(node_id="n1", extra="val") == {
        **base,
        "node_id": "n1",
        "extra": "val",
    }


# ===================================================================
# AnalyticDB.close_pool
# ===================================================================


def _make_adb() -> AnalyticDB:
    """Create an AnalyticDB bypassing __init__ (no env required)."""
    db = AnalyticDB.__new__(AnalyticDB)
    db._lock = asyncio.Lock()
    db.pool = None
    return db


class TestClosePool:
    """Regression: aiomysql ``Pool.closed`` is a property, not a method."""

    @pytest.mark.asyncio
    async def test_closes_open_pool(self):
        db = _make_adb()
        db.pool = MagicMock()
        db.pool.closed = False
        db.pool.terminate = MagicMock()
        db.pool.wait_closed = AsyncMock()
        await db.close_pool()
        db.pool.terminate.assert_called_once()
        db.pool.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_pool_already_closed(self):
        db = _make_adb()
        db.pool = MagicMock()
        db.pool.closed = True
        db.pool.terminate = MagicMock()
        db.pool.wait_closed = AsyncMock()
        await db.close_pool()
        db.pool.terminate.assert_not_called()
        db.pool.wait_closed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_pool_is_none(self):
        db = _make_adb()
        await db.close_pool()  # must not raise


# ===================================================================
# AnalyticDB.initdb
# ===================================================================


class TestInitdb:
    @pytest.mark.asyncio
    async def test_creates_only_missing_tables(self, monkeypatch):
        async def fake_create_pool(**kwargs):
            return MagicMock()

        monkeypatch.setattr(adb_mysql_impl.aiomysql, "create_pool", fake_create_pool)

        db = _make_adb()
        db.db_config = {"db": "testdb"}
        db.workspace = "ws"

        table_names = list(adb_mysql_impl.TABLES)
        # kepler reports lowercased names; every table except the first exists.
        db.query = AsyncMock(
            return_value=[{"table_name": t.lower()} for t in table_names[1:]]
        )
        db.execute = AsyncMock()

        await db.initdb()

        # ONE probe for the whole TABLES dict, not one round trip per table.
        assert db.query.await_count == 1
        probe = db.query.await_args
        assert "kepler_meta_tables" in probe.args[0]
        assert probe.args[0].count("%(probe_") == len(table_names)
        assert probe.kwargs["multirows"] is True
        db.execute.assert_awaited_once()
        ddl = db.execute.call_args.args[0]
        assert "CREATE TABLE" in ddl
        assert table_names[0] in ddl

    @pytest.mark.asyncio
    async def test_skips_ddl_when_all_tables_exist(self, monkeypatch):
        async def fake_create_pool(**kwargs):
            return MagicMock()

        monkeypatch.setattr(adb_mysql_impl.aiomysql, "create_pool", fake_create_pool)

        db = _make_adb()
        db.db_config = {"db": "testdb"}
        db.workspace = "ws"
        db.query = AsyncMock(
            return_value=[{"table_name": t.lower()} for t in adb_mysql_impl.TABLES]
        )
        db.execute = AsyncMock()

        await db.initdb()

        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pool_is_pinned_to_a_utc_session(self, monkeypatch):
        """Every time read/write in this backend assumes a UTC session; the
        pin is what makes that true on a +08 server."""
        captured = {}

        async def fake_create_pool(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(adb_mysql_impl.aiomysql, "create_pool", fake_create_pool)

        db = _make_adb()
        db.db_config = {"db": "testdb", "autocommit": True, "pool_recycle": 3600}
        db.workspace = "ws"
        db.query = AsyncMock(
            return_value=[{"table_name": t.lower()} for t in adb_mysql_impl.TABLES]
        )
        db.execute = AsyncMock()

        await db.initdb()

        assert captured["init_command"] == "SET time_zone = '+00:00'"
        assert captured["cursorclass"] is adb_mysql_impl.aiomysql.DictCursor


# ===================================================================
# AnalyticDB env defaults — an unset ADB_WORKSPACE must not outrank
# the workspace each storage was built with
# ===================================================================


class TestAnalyticDBEnvDefaults:
    def _client(self, monkeypatch) -> AnalyticDB:
        for key in (
            "ADB_HOST",
            "ADB_PORT",
            "ADB_USER",
            "ADB_PASSWORD",
            "ADB_DATABASE",
            "ADB_WORKSPACE",
            "ADB_MAX_CONNECTIONS",
            "ADB_MAX_BATCH_SIZE",
            "ADB_IN_CLAUSE_MAX_SIZE",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ADB_USER", "u")
        monkeypatch.setenv("ADB_PASSWORD", "p")
        monkeypatch.setenv("ADB_DATABASE", "d")
        return AnalyticDB()

    def test_workspace_env_has_no_default(self, monkeypatch):
        assert self._client(monkeypatch).workspace is None

    def test_missing_credentials_are_rejected(self, monkeypatch):
        monkeypatch.setenv("ADB_USER", "u")
        monkeypatch.setenv("ADB_PASSWORD", "p")
        monkeypatch.delenv("ADB_DATABASE", raising=False)
        with pytest.raises(ValueError, match="Missing database"):
            AnalyticDB()

    def test_pool_size_and_recycle_defaults(self, monkeypatch):
        db = self._client(monkeypatch)
        assert db.db_config["maxsize"] == 25
        assert db.db_config["pool_recycle"] == 3600

    @pytest.mark.asyncio
    async def test_storage_keeps_its_workspace_with_the_default_client(
        self, monkeypatch
    ):
        initialize_share_data(1)
        storage = _make_doc_status()
        storage.db = self._client(monkeypatch)
        await storage.initialize()
        assert storage.workspace == "test_ws"

    def test_batch_sizes_live_on_the_shared_client(self, monkeypatch):
        db = self._client(monkeypatch)
        assert db.max_batch_size == 32
        assert db.in_clause_max_size == 1000


# ===================================================================
# ADBClientManager — one pool shared by every ADB storage
# ===================================================================


@pytest.fixture
def fresh_manager():
    """``_instances`` is class-level state; isolate it so tests and any later
    storage initialized in the same process cannot inherit a live slot."""
    saved = dict(ADBClientManager._instances)
    ADBClientManager._instances.update({"db": None, "ref_count": 0})
    try:
        yield ADBClientManager
    finally:
        ADBClientManager._instances.clear()
        ADBClientManager._instances.update(saved)


@pytest.fixture
def fake_adb(monkeypatch):
    """Patch ``AnalyticDB`` and return the list recording each pool built."""
    built: list = []

    class _FakeADB:
        def __init__(self):
            built.append(self)
            self.initdb = AsyncMock()
            self.close_pool = AsyncMock()

    monkeypatch.setattr(adb_mysql_impl, "AnalyticDB", _FakeADB)
    return built


class TestADBClientManager:
    @pytest.mark.asyncio
    async def test_twelve_storages_share_one_client(self, fresh_manager, fake_adb):
        """Regression: each storage used to build its own pool, so a graph write
        was capped at ADB_MAX_CONNECTIONS no matter how many were free elsewhere.
        """
        first = await fresh_manager.get_client()
        for _ in range(11):
            assert await fresh_manager.get_client() is first

        assert len(fake_adb) == 1
        first.initdb.assert_awaited_once()
        assert fresh_manager._instances["ref_count"] == 12

    @pytest.mark.asyncio
    async def test_concurrent_gets_build_one_client(self, fresh_manager, monkeypatch):
        """The manager lock guards the build-and-init window, so simultaneous
        initializations cannot each create a pool."""
        built: list = []

        class SlowADB:
            def __init__(self):
                built.append(self)
                self.close_pool = AsyncMock()

            async def initdb(self):
                await asyncio.sleep(0.01)

        monkeypatch.setattr(adb_mysql_impl, "AnalyticDB", SlowADB)

        clients = await asyncio.gather(*[fresh_manager.get_client() for _ in range(4)])
        assert len(built) == 1
        assert all(c is built[0] for c in clients)
        assert fresh_manager._instances["ref_count"] == 4

    @pytest.mark.asyncio
    async def test_pool_closes_only_on_last_reference(self, fresh_manager, fake_adb):
        db = await fresh_manager.get_client()
        await fresh_manager.get_client()

        await fresh_manager.release_client(db)
        assert fresh_manager._instances["db"] is db
        assert fresh_manager._instances["ref_count"] == 1
        db.close_pool.assert_not_awaited()

        await fresh_manager.release_client(db)
        db.close_pool.assert_awaited_once()
        assert fresh_manager._instances["db"] is None

    @pytest.mark.asyncio
    async def test_foreign_client_release_leaves_shared_slot_alone(
        self, fresh_manager, fake_adb
    ):
        db = await fresh_manager.get_client()
        stray = _make_adb()
        stray.close_pool = AsyncMock()

        await fresh_manager.release_client(stray)

        stray.close_pool.assert_awaited_once()
        db.close_pool.assert_not_awaited()
        assert fresh_manager._instances["db"] is db
        assert fresh_manager._instances["ref_count"] == 1

    @pytest.mark.asyncio
    async def test_storage_initialize_and_finalize_go_through_the_manager(
        self, fresh_manager, monkeypatch
    ):
        """Pins the wiring: a storage reverting to ``AnalyticDB()``/``close_pool()``
        would silently rebuild the per-storage pool this replaces."""
        initialize_share_data()
        shared = _make_adb()
        shared.workspace = "ws"
        shared.db_config = {"db": "testdb"}
        shared.query = AsyncMock(
            return_value=[
                {"table_name": t.lower()} for t in adb_mysql_impl.GRAPH_TABLES
            ]
        )  # graph tables exist
        shared.execute = AsyncMock()
        shared.close_pool = AsyncMock()
        get_client = AsyncMock(return_value=shared)
        release_client = AsyncMock()
        monkeypatch.setattr(ADBClientManager, "get_client", get_client)
        monkeypatch.setattr(ADBClientManager, "release_client", release_client)

        storage = _make_graph()
        storage.db = None
        await storage.initialize()
        assert storage.db is shared
        get_client.assert_awaited_once()

        await storage.finalize()
        release_client.assert_awaited_once_with(shared)
        shared.close_pool.assert_not_awaited()
        assert storage.db is None


# ===================================================================
# Batch-write contract
# ===================================================================


def test_every_write_statement_keeps_the_executemany_fast_path():
    """A VALUES tuple may hold placeholders ONLY, in every write statement.

    aiomysql batches only when RE_INSERT_VALUES matches, and that regex
    requires each element of the tuple to be ``%s`` / ``%(name)s``. One bare
    CURRENT_TIMESTAMP — or a ``VALUES(`` written without the space — silently
    turns every executemany into a per-row round trip. All 11 statements below
    carried that literal until the fix this test now pins.
    """
    statements = {
        name: sql
        for name, sql in adb_mysql_impl.SQL_TEMPLATES.items()
        if "VALUES" in sql.upper()
    }
    statements["doc_status._REPLACE_SQL"] = ADBDocStatusStorage._REPLACE_SQL
    assert len(statements) == 11

    for name, sql in statements.items():
        match = RE_INSERT_VALUES.match(sql)
        assert match, f"{name} cannot be batched"
        assert "CURRENT_TIMESTAMP" not in match.group(2), name
        head = match.group(1)
        assert "update_time" not in head and "updated_at" not in head, name


# ===================================================================
# Cross-storage contracts — the same rule holds in KV, vector and
# doc-status storage
# ===================================================================


@pytest.mark.asyncio
async def test_delete_contract_is_shared_by_the_id_storages():
    """One id-delete contract everywhere: the cap slices the IN list, a
    multi-chunk sweep goes in ONE transaction, sets and empty input behave, and
    each storage names its own table. Only the failure policy differs — a
    vector delete must reach the caller, the other two swallow it."""
    cases = [
        ("kv", _make_kv, "LIGHTRAG_DOC_CHUNKS", False),
        ("vector", _make_vector, "LIGHTRAG_VDB_CHUNKS", True),
        ("doc_status", _make_doc_status, "LIGHTRAG_DOC_STATUS", False),
    ]
    ids = [f"id-{i}" for i in range(5)]
    for label, make, table, reraises in cases:
        # within the cap: ONE statement, no transaction
        storage = make()
        storage.db.in_clause_max_size = 2
        await storage.delete(["id-0", "id-1"])
        storage.db.execute.assert_awaited_once()
        storage.db.execute_transaction.assert_not_called()
        sql, params = storage.db.execute.call_args.args
        assert f"DELETE FROM {table}" in sql, label
        assert "workspace=%(workspace)s" in sql, label
        assert params["workspace"] == "test_ws", label

        # over the cap: chunked, every id delivered, ONE transaction
        storage = make()
        storage.db.in_clause_max_size = 2
        await storage.delete(ids)
        storage.db.execute.assert_not_called()
        statements = storage.db.execute_transaction.call_args.args[0]
        assert [stmt.count("%(id_") for stmt, _ in statements] == [2, 2, 1], label
        for stmt, _ in statements:
            assert "workspace=%(workspace)s" in stmt, (label, "chunked")
        sent = sorted(
            value
            for _, bound in statements
            for key, value in bound.items()
            if key.startswith("id_")
        )
        assert sent == sorted(ids), label

        # cap <= 0 is the documented unbounded escape hatch
        storage = make()
        storage.db.in_clause_max_size = 0
        await storage.delete(ids)
        storage.db.execute.assert_awaited_once()
        assert storage.db.execute.call_args.args[0].count("%(id_") == 5, label

        # a set reaches the same chunked path as a list
        storage = make()
        storage.db.in_clause_max_size = 2
        await storage.delete(set(ids))
        statements = storage.db.execute_transaction.call_args.args[0]
        assert sum(stmt.count("%(id_") for stmt, _ in statements) == 5, label

        # empty input is a no-op
        storage = make()
        await storage.delete([])
        storage.db.execute.assert_not_called()
        storage.db.execute_transaction.assert_not_called()

        # failure policy, on the chunked AND the single-statement path
        for cap, method in ((2, "execute_transaction"), (0, "execute")):
            storage = make()
            storage.db.in_clause_max_size = cap
            getattr(storage.db, method).side_effect = RuntimeError("boom")
            if reraises:
                with pytest.raises(RuntimeError, match="boom"):
                    await storage.delete(ids)
            else:
                await storage.delete(ids)


@pytest.mark.asyncio
async def test_upsert_stamps_create_time_from_the_app_clock():
    """KV and vector read the clock once per batch and bind naive-UTC
    create_time; update_time is never sent, so the DDL's server default owns it
    and the write stays placeholder-only."""
    kv_payload = {
        f"chunk-{i}": {
            "tokens": 5,
            "chunk_order_index": i,
            "full_doc_id": f"doc-{i}",
            "content": "text",
            "file_path": "f.txt",
        }
        for i in range(8)
    }
    for label, make, payload in (
        ("kv", _make_kv, kv_payload),
        ("vector", _make_vector, _chunk_items(8)),
    ):
        storage = make()
        before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        await storage.upsert(payload)
        # the write is decided from the payload alone: no pre-fetch
        storage.db.query.assert_not_called()
        rows = storage.db.execute.call_args.args[1]
        assert len(rows) == 8, label
        stamps = {row["create_time"] for row in rows}
        assert len(stamps) == 1, label
        assert next(iter(stamps)) >= before, label
        assert all(row["create_time"].tzinfo is None for row in rows), label
        for row in rows:
            assert "update_time" not in row, label


def test_every_upsert_template_binds_create_time_only():
    """All 10 upsert templates carry ``%(create_time)s`` and never mention
    update_time or CURRENT_TIMESTAMP — a literal in the VALUES tuple would drop
    every batched write out of aiomysql's executemany path."""
    upserts = {
        name: sql
        for name, sql in adb_mysql_impl.SQL_TEMPLATES.items()
        if name.startswith("upsert_")
    }
    assert len(upserts) == 10
    for name, sql in upserts.items():
        assert "%(create_time)s" in sql, name
        assert "update_time" not in sql, name
        assert "CURRENT_TIMESTAMP" not in sql, name
