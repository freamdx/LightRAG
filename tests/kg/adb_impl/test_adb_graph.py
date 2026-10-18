"""Unit tests for ADBGraphStorage database-interacting methods (mock).

All tests mock ``db.query`` / ``db.execute`` to verify SQL generation and
result parsing without a live AnalyticDB connection.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiomysql.cursors import RE_INSERT_VALUES

from lightrag.kg.adb_mysql_impl import GRAPH_TABLES, ADBGraphStorage

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_storage() -> ADBGraphStorage:
    storage = ADBGraphStorage.__new__(ADBGraphStorage)
    storage.namespace = "chunk_entity_relation"
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = MagicMock()
    storage.db.max_batch_size = 10
    storage.db.query = AsyncMock()
    storage.db.execute = AsyncMock()
    storage.db.execute_transaction = AsyncMock()
    return storage


# ===================================================================
# has_node / has_edge
# ===================================================================


@pytest.mark.asyncio
async def test_has_node_exists():
    storage = _make_storage()
    storage.db.query.return_value = {"1": 1}
    assert await storage.has_node("n1") is True
    sql = storage.db.query.call_args.args[0]
    params = storage.db.query.call_args.args[1]
    assert "LIGHTRAG_GRAPH_NODES" in sql
    assert params["node_id"] == "n1"
    assert params["workspace"] == "test_ws"
    assert params["namespace"] == "chunk_entity_relation"


@pytest.mark.asyncio
async def test_has_node_not_exists():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.has_node("missing") is False


@pytest.mark.asyncio
async def test_has_edge_exists():
    storage = _make_storage()
    storage.db.query.return_value = {"1": 1}
    assert await storage.has_edge("b", "a") is True
    params = storage.db.query.call_args.args[1]
    # Canonical ordering: min first
    assert params["src"] == "a"
    assert params["tgt"] == "b"


@pytest.mark.asyncio
async def test_has_edge_not_exists():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.has_edge("x", "y") is False


# ===================================================================
# get_node / get_edge
# ===================================================================


@pytest.mark.asyncio
async def test_get_node_returns_props():
    storage = _make_storage()
    storage.db.query.return_value = {"properties": '{"desc": "hello"}'}
    result = await storage.get_node("n1")
    assert result is not None
    assert result["entity_id"] == "n1"
    assert result["desc"] == "hello"


@pytest.mark.asyncio
async def test_get_node_not_found():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.get_node("missing") is None


@pytest.mark.asyncio
async def test_get_edge_returns_props():
    storage = _make_storage()
    storage.db.query.return_value = {"properties": '{"weight": "5"}'}
    result = await storage.get_edge("b", "a")
    assert result is not None
    assert result["weight"] == "5"


@pytest.mark.asyncio
async def test_get_edge_not_found():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.get_edge("x", "y") is None


# ===================================================================
# upsert_node
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_node_requires_entity_id():
    storage = _make_storage()
    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("n1", {"desc": "no entity_id"})


@pytest.mark.asyncio
async def test_upsert_node_new():
    storage = _make_storage()
    storage.db.query.return_value = None  # get_node returns None → new node
    await storage.upsert_node("n1", {"entity_id": "n1", "desc": "new"})
    storage.db.execute.assert_called_once()
    sql, params = storage.db.execute.call_args.args
    assert "REPLACE INTO LIGHTRAG_GRAPH_NODES" in sql
    assert json.loads(params["properties"]) == {"entity_id": "n1", "desc": "new"}


@pytest.mark.asyncio
async def test_upsert_node_merge():
    storage = _make_storage()
    # get_node returns existing node
    storage.db.query.return_value = {"properties": '{"entity_id": "n1", "old": "val"}'}
    await storage.upsert_node("n1", {"entity_id": "n1", "new": "val2"})
    storage.db.execute.assert_called_once()
    sql, params = storage.db.execute.call_args.args
    assert "REPLACE INTO LIGHTRAG_GRAPH_NODES" in sql
    # merge keeps the stored property and overwrites only the incoming one
    assert json.loads(params["properties"]) == {
        "entity_id": "n1",
        "old": "val",
        "new": "val2",
    }


# ===================================================================
# upsert_edge
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_edge_canonical_order():
    storage = _make_storage()
    # Flow: has_nodes_batch([a, z]) → [] (both missing),
    #        get_node("a_node") inside upsert_node → None,
    #        get_node("z_node") inside upsert_node → None
    storage.db.query.side_effect = [
        [],  # has_nodes_batch
        None,  # get_node("a_node") inside upsert_node
        None,  # get_node("z_node") inside upsert_node
    ]
    await storage.upsert_edge("z_node", "a_node", {"weight": "1"})
    # 3 executes: insert src node, insert tgt node, insert edge
    assert storage.db.execute.call_count == 3
    # The edge insert should use canonical order
    edge_call = storage.db.execute.call_args_list[2]
    params = edge_call.args[1]
    assert params["src"] == "a_node"  # min
    assert params["tgt"] == "z_node"  # max


@pytest.mark.asyncio
async def test_upsert_edge_existing_endpoints_no_node_writes():
    storage = _make_storage()
    # Both endpoints exist: only the edge REPLACE may run.
    storage.db.query.return_value = [{"node_id": "a"}, {"node_id": "b"}]
    await storage.upsert_edge("b", "a", {"weight": "1"})
    storage.db.query.assert_awaited_once()  # single has_nodes_batch read
    storage.db.execute.assert_awaited_once()
    params = storage.db.execute.call_args.args[1]
    assert params["src"] == "a" and params["tgt"] == "b"


# ===================================================================
# get_nodes_batch / upsert_nodes_batch
# ===================================================================


@pytest.mark.asyncio
async def test_get_nodes_batch_includes_empty_properties():
    """Regression: row presence decides membership, not properties truthiness.

    A node holding NULL or '{}' properties still exists; the old truthiness
    filter silently dropped it, disagreeing with get_node/has_nodes_batch.
    """
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "n1", "properties": '{"description": "d"}'},
        {"node_id": "n2", "properties": None},
        {"node_id": "n3", "properties": "{}"},
    ]
    result = await storage.get_nodes_batch(["n1", "n2", "n3"])
    assert set(result) == {"n1", "n2", "n3"}
    assert result["n2"] == {"entity_id": "n2"}
    assert result["n3"] == {"entity_id": "n3"}


@pytest.mark.asyncio
async def test_upsert_nodes_batch_single_batch_read():
    """Regression: existing nodes are read in ONE batch query, not N serial
    get_node round trips, and the merge keeps omitted keys."""
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "a", "properties": '{"entity_id": "a", "old": "kept"}'}
    ]
    await storage.upsert_nodes_batch(
        [("a", {"entity_id": "a", "new": "x"}), ("b", {"entity_id": "b"})]
    )
    storage.db.query.assert_awaited_once()
    storage.db.execute.assert_awaited_once()
    datas = storage.db.execute.call_args.args[1]
    assert isinstance(datas, list) and len(datas) == 2
    merged = {d["node_id"]: json.loads(d["properties"]) for d in datas}
    assert merged["a"] == {"entity_id": "a", "old": "kept", "new": "x"}
    assert merged["b"] == {"entity_id": "b"}


@pytest.mark.asyncio
async def test_upsert_nodes_batch_flushes_in_max_batch_chunks():
    """Writes flush in db.max_batch_size chunks; partial tail kept in order."""
    storage = _make_storage()
    storage.db.max_batch_size = 2
    storage.db.query.return_value = []  # get_nodes_batch: nothing exists
    await storage.upsert_nodes_batch(
        [(f"n{i}", {"entity_id": f"n{i}"}) for i in range(5)]
    )
    assert storage.db.execute.await_count == 3  # ceil(5 / 2)
    chunk_sizes = [len(c.args[1]) for c in storage.db.execute.call_args_list]
    assert chunk_sizes == [2, 2, 1]
    # Deduped sorted write order preserved across chunk boundaries.
    written = [
        d["node_id"] for c in storage.db.execute.call_args_list for d in c.args[1]
    ]
    assert written == ["n0", "n1", "n2", "n3", "n4"]


# ===================================================================
# Write SQL stays executemany-batchable
# ===================================================================


@pytest.mark.asyncio
async def test_graph_write_sql_stays_executemany_batchable():
    """Regression: every graph REPLACE must keep matching aiomysql's
    RE_INSERT_VALUES, whose VALUES group accepts placeholders ONLY.

    ``CURRENT_TIMESTAMP`` used to sit in that tuple; the regex then missed and
    ``executemany`` silently fell back to one round trip per row, so the
    db.max_batch_size chunking bought nothing. update_time is left to the
    column DEFAULT and must never be bound from Python (the convention the KV
    and vector storages pin).
    """
    storage = _make_storage()
    storage.db.query.return_value = []  # nothing exists yet

    await storage.upsert_node("a", {"entity_id": "a"})
    await storage.upsert_nodes_batch(
        [("a", {"entity_id": "a"}), ("b", {"entity_id": "b"})]
    )
    await storage.upsert_edge("a", "b", {"weight": 1})
    await storage.upsert_edges_batch(
        [("a", "b", {"weight": 1}), ("c", "d", {"weight": 2})]
    )

    calls = storage.db.execute.call_args_list
    assert calls, "no write SQL captured"
    for call in calls:
        sql, payload = call.args[0], call.args[1]
        assert RE_INSERT_VALUES.match(sql), sql
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            assert "update_time" not in row


# ===================================================================
# delete_node / remove_nodes atomicity
# ===================================================================


@pytest.mark.asyncio
async def test_delete_node_edge_deletes_split_by_index():
    """Edge + node deletes share ONE transaction (no FK CASCADE on ADB).

    source_id and target_id get separate DELETEs, never a cross-index OR:
    a UNION ALL is not available to DELETE, and one OR can only be served by
    one of the two indexes, so the row set is taken in two single-index passes.
    """
    storage = _make_storage()
    await storage.delete_node("n1")
    storage.db.execute_transaction.assert_awaited_once()
    storage.db.execute.assert_not_called()
    statements = storage.db.execute_transaction.call_args.args[0]
    sqls = [sql for sql, _ in statements]
    assert len(sqls) == 3
    edge_sqls = [sql for sql, _ in statements if "GRAPH_EDGES" in sql]
    assert len(edge_sqls) == 2
    for sql, params in statements:
        assert " OR " not in sql, sql
        assert "DELETE" in sql, sql
        assert params["node_id"] == "n1"
    assert "LIGHTRAG_GRAPH_NODES" in sqls[2]
    assert any("source_id=%(node_id)s" in sql for sql in edge_sqls)
    assert any("target_id=%(node_id)s" in sql for sql in edge_sqls)


@pytest.mark.asyncio
async def test_remove_nodes_splits_the_edge_in_clause():
    """Atomic, and one IN list per statement: 2 items per id instead of 2 lists
    in one, which halves what ADB's MAX_IN_ITEMS_COUNT has to measure."""
    storage = _make_storage()
    await storage.remove_nodes(["n1", "n2", "n3"])
    storage.db.execute_transaction.assert_awaited_once()
    storage.db.execute.assert_not_called()
    statements = storage.db.execute_transaction.call_args.args[0]
    assert len(statements) == 3
    for sql, _ in statements:
        assert " OR " not in sql, sql
        assert "DELETE" in sql, sql
        assert sql.count("%(node_id_") == 3
    edge_sqls = [sql for sql, _ in statements if "GRAPH_EDGES" in sql]
    assert [("source_id IN" in sql, "target_id IN" in sql) for sql in edge_sqls] == [
        (True, False),
        (False, True),
    ]


# ===================================================================
# remove_edges
# ===================================================================


@pytest.mark.asyncio
async def test_remove_edges_empty():
    storage = _make_storage()
    await storage.remove_edges([])
    storage.db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_remove_edges_batch():
    storage = _make_storage()
    await storage.remove_edges([("a", "b"), ("c", "d")])
    storage.db.execute.assert_called_once()
    sql = storage.db.execute.call_args.args[0]
    params = storage.db.execute.call_args.args[1]
    assert "DELETE" in sql
    assert "LIGHTRAG_GRAPH_EDGES" in sql
    # Canonical ordering: min first
    assert params["src_0"] == "a"
    assert params["tgt_0"] == "b"
    assert params["src_1"] == "c"
    assert params["tgt_1"] == "d"


@pytest.mark.asyncio
async def test_remove_edges_canonical_ordering():
    storage = _make_storage()
    # Pass edges in reverse order
    await storage.remove_edges([("z", "a")])
    params = storage.db.execute.call_args.args[1]
    assert params["src_0"] == "a"  # min
    assert params["tgt_0"] == "z"  # max


# ===================================================================
# search_labels
# ===================================================================


@pytest.mark.asyncio
async def test_search_labels_empty_query():
    storage = _make_storage()
    result = await storage.search_labels("")
    assert result == []
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_search_labels_with_results():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "exact_match"},
        {"node_id": "prefix_match"},
    ]
    result = await storage.search_labels("match")
    assert result == ["exact_match", "prefix_match"]
    sql = storage.db.query.call_args.args[0]
    assert "CASE" in sql  # SQL-side scoring
    assert "LIKE" in sql


@pytest.mark.asyncio
async def test_search_labels_escapes_like_chars():
    storage = _make_storage()
    storage.db.query.return_value = []
    await storage.search_labels("100%_off")
    params = storage.db.query.call_args.args[1]
    # The contains parameter should have escaped LIKE chars
    assert "\\%" in params["contains"]
    assert "\\_" in params["contains"]


# ===================================================================
# get_popular_labels
# ===================================================================


@pytest.mark.asyncio
async def test_popular_labels_empty():
    storage = _make_storage()
    storage.db.query.return_value = []
    result = await storage.get_popular_labels(10)
    assert result == []


@pytest.mark.asyncio
async def test_popular_labels_with_results():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "popular", "degree": 10},
        {"node_id": "less_popular", "degree": 5},
    ]
    result = await storage.get_popular_labels(10)
    assert result == ["popular", "less_popular"]
    sql = storage.db.query.call_args.args[0]
    assert "LEFT JOIN" in sql  # includes isolated nodes
    assert "ORDER BY degree DESC" in sql


# ===================================================================
# get_all_labels
# ===================================================================


@pytest.mark.asyncio
async def test_get_all_labels():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "b"},
        {"node_id": "a"},
        {"node_id": "c"},
    ]
    result = await storage.get_all_labels()
    assert result == ["a", "b", "c"]  # sorted


# ===================================================================
# get_node_edges
# ===================================================================


@pytest.mark.asyncio
async def test_get_node_edges_not_found():
    storage = _make_storage()
    storage.db.query.return_value = None  # has_node returns None
    assert await storage.get_node_edges("missing") is None


@pytest.mark.asyncio
async def test_get_node_edges():
    storage = _make_storage()
    storage.db.query.side_effect = [
        {"1": 1},  # has_node → True
        [  # get_node_edges results
            {"source_id": "n1", "target_id": "n2"},
            {"source_id": "n3", "target_id": "n1"},
        ],
    ]
    result = await storage.get_node_edges("n1")
    assert result is not None
    assert len(result) == 2
    # Results sorted by target
    assert result[0][0] == "n1"


@pytest.mark.asyncio
async def test_get_node_edges_uses_two_index_prefixed_branches():
    storage = _make_storage()
    storage.db.query.side_effect = [{"1": 1}, []]
    await storage.get_node_edges("n1")
    sql = storage.db.query.call_args.args[0]
    assert "UNION ALL" in sql
    assert " OR " not in sql
    assert "source_id=%(node_id)s" in sql
    assert "target_id=%(node_id)s" in sql


@pytest.mark.asyncio
async def test_get_node_edges_collapses_the_self_loop():
    """UNION ALL delivers a self-loop from BOTH branches — one edge, not two."""
    storage = _make_storage()
    storage.db.query.side_effect = [
        {"1": 1},  # has_node → True
        [
            {"source_id": "n1", "target_id": "n1"},
            {"source_id": "n1", "target_id": "n1"},
            {"source_id": "n1", "target_id": "n2"},
        ],
    ]
    assert await storage.get_node_edges("n1") == [("n1", "n1"), ("n1", "n2")]


# ===================================================================
# drop
# ===================================================================


@pytest.mark.asyncio
async def test_drop_success():
    storage = _make_storage()
    result = await storage.drop()
    assert result["status"] == "success"
    # Atomicity goes through the single-connection transaction path; bare
    # START TRANSACTION/COMMIT via execute() would land on separate pooled
    # connections and silently lose the transaction boundary.
    storage.db.execute_transaction.assert_awaited_once()
    statements = storage.db.execute_transaction.call_args.args[0]
    sqls = [sql for sql, _ in statements]
    assert len(sqls) == 2
    assert any("LIGHTRAG_GRAPH_EDGES" in sql and "DELETE" in sql for sql in sqls)
    assert any("LIGHTRAG_GRAPH_NODES" in sql and "DELETE" in sql for sql in sqls)
    storage.db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_drop_error_rolls_back():
    storage = _make_storage()
    storage.db.execute_transaction = AsyncMock(side_effect=Exception("DB error"))
    result = await storage.drop()
    assert result["status"] == "error"
    assert "DB error" in result["message"]


# ===================================================================
# get_knowledge_graph — edges via ONE double-IN query, never O(N²) pairs
# ===================================================================


def _all_query_sqls(storage) -> list[str]:
    return [call.args[0] for call in storage.db.query.call_args_list]


@pytest.mark.asyncio
async def test_full_knowledge_graph_edges_single_in_query():
    storage = _make_storage()
    storage.db.query.side_effect = [
        # get_popular_labels
        [{"node_id": "a", "degree": 2}, {"node_id": "b", "degree": 1}],
        # get_nodes_batch
        [
            {"node_id": "a", "properties": '{"description": "A"}'},
            {"node_id": "b", "properties": '{"description": "B"}'},
        ],
        # _fetch_edges_among
        [
            {
                "source_id": "a",
                "target_id": "b",
                "properties": '{"weight": "1"}',
            }
        ],
    ]
    kg = await storage.get_knowledge_graph("*")
    assert [n.id for n in kg.nodes] == ["a", "b"]
    assert len(kg.edges) == 1
    assert kg.edges[0].source == "a" and kg.edges[0].target == "b"
    assert kg.edges[0].properties == {"weight": "1"}
    assert kg.is_truncated is False
    # No per-pair placeholders anywhere: the O(N^2) enumeration is gone.
    assert not any("src_0" in sql for sql in _all_query_sqls(storage))
    edges_sql = _all_query_sqls(storage)[-1]
    assert "source_id IN" in edges_sql and "target_id IN" in edges_sql


@pytest.mark.asyncio
async def test_full_knowledge_graph_tail_reads_are_issued_concurrently():
    """The `*` view costs one ranking scan, then the two tail reads overlap."""
    storage = _make_storage()
    rows = {
        "LEFT JOIN": [{"node_id": "a", "degree": 2}, {"node_id": "b", "degree": 1}],
        "SELECT node_id, properties": [
            {"node_id": "a", "properties": '{"description": "A"}'},
            {"node_id": "b", "properties": '{"description": "B"}'},
        ],
        "SELECT source_id, target_id, properties": [
            {"source_id": "a", "target_id": "b", "properties": "{}"}
        ],
    }
    in_flight = {"live": 0, "peak": 0}

    async def _query(sql, params=None, multirows=False):
        in_flight["live"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["live"])
        await asyncio.sleep(0)
        in_flight["live"] -= 1
        return next(r for key, r in rows.items() if key in sql)

    storage.db.query = AsyncMock(side_effect=_query)
    kg = await storage.get_knowledge_graph("*", max_nodes=100)
    # Ranking order survives: degree DESC, label ASC comes back as a, b.
    assert [n.id for n in kg.nodes] == ["a", "b"]
    assert len(kg.edges) == 1
    assert in_flight["peak"] == 2


@pytest.mark.asyncio
async def test_bfs_knowledge_graph_backfills_real_properties():
    storage = _make_storage()
    storage.db.query.side_effect = [
        # Level-1 candidates: neighbours ranked by whole-graph degree, in SQL.
        [{"nid": "n2", "degree": 3}, {"nid": "n3", "degree": 1}],
        # get_nodes_batch for the retained nodes
        [
            {"node_id": "seed", "properties": '{"description": "S"}'},
            {"node_id": "n2", "properties": '{"description": "N2"}'},
            {"node_id": "n3", "properties": '{"description": "N3"}'},
        ],
        # _fetch_edges_among
        [{"source_id": "n2", "target_id": "seed", "properties": "{}"}],
    ]
    kg = await storage.get_knowledge_graph("seed", max_depth=1, max_nodes=100)
    assert [n.id for n in kg.nodes] == ["seed", "n2", "n3"]
    # Regression: non-seed nodes carry their stored properties, not "{}".
    props = {n.id: n.properties for n in kg.nodes}
    assert props["n2"]["description"] == "N2"
    assert props["n3"]["description"] == "N3"
    assert len(kg.edges) == 1
    assert kg.is_truncated is False
    assert not any("src_0" in sql for sql in _all_query_sqls(storage))


@pytest.mark.asyncio
async def test_bfs_knowledge_graph_trims_overflow_node():
    """The level_cap overflow (+1 past max_nodes) never reaches the caller."""
    storage = _make_storage()
    storage.db.query.side_effect = [
        [{"nid": "n2", "degree": 3}, {"nid": "n3", "degree": 1}],
        # After trimming only the seed survives: get_nodes_batch(["seed"])
        [{"node_id": "seed", "properties": '{"description": "S"}'}],
        # _fetch_edges_among({seed}) short-circuits (< 2 ids) — no query.
    ]
    kg = await storage.get_knowledge_graph("seed", max_depth=1, max_nodes=1)
    assert kg.is_truncated is True
    assert [n.id for n in kg.nodes] == ["seed"]
    assert kg.edges == []


@pytest.mark.asyncio
async def test_bfs_multi_node_frontier_uses_one_batched_read_per_level():
    """Regression: a wide frontier must cost ONE query per level.

    The previous per-node neighbour loop issued |frontier| round trips per
    level (up to O(node_budget) queries on one KG request), and the shape
    after that still needed two. Level 2 here has a two-node frontier and
    must still produce a single ranked-candidate query.
    """
    storage = _make_storage()
    storage.db.query.side_effect = [
        # Level-1 frontier [seed]
        [{"nid": "n2", "degree": 2}, {"nid": "n3", "degree": 1}],
        # Level-2 frontier [n2, n3] — ONE query, both frontier nodes bound
        [{"nid": "n4", "degree": 1}],
        # get_nodes_batch backfill for the retained nodes
        [
            {"node_id": "seed", "properties": "{}"},
            {"node_id": "n2", "properties": "{}"},
            {"node_id": "n3", "properties": "{}"},
            {"node_id": "n4", "properties": "{}"},
        ],
        # _fetch_edges_among
        [],
    ]
    kg = await storage.get_knowledge_graph("seed", max_depth=2, max_nodes=100)
    assert [n.id for n in kg.nodes] == ["seed", "n2", "n3", "n4"]
    # 1 per level + the concurrent backfill/edge pair — a per-node frontier
    # loop would need 8, the two-query-per-level shape needed 7, and the
    # removed seed probe one more.
    assert storage.db.query.await_count == 4
    level2_sql, level2_params = storage.db.query.call_args_list[1].args[:2]
    assert "LIGHTRAG_GRAPH_EDGES" in level2_sql
    assert "node_id_0" in level2_params and "node_id_1" in level2_params
    assert set(level2_params.values()) >= {"n2", "n3"}


@pytest.mark.asyncio
async def test_bfs_never_binds_the_neighbour_set():
    """P0-3: a level's IN items are 2 x |frontier|, never 2 x |neighbours|.

    A hub one hop out can have thousands of neighbours. Handing that set back
    in as an IN list is what tripped ADB's MAX_IN_ITEMS_COUNT, so the
    neighbour set must stay inside the statement and only the ranked top-K
    rows may come back.
    """
    storage = _make_storage()
    storage.db.query.side_effect = [
        [{"nid": "n2", "degree": 2}],
        [
            {"node_id": "seed", "properties": "{}"},
            {"node_id": "n2", "properties": "{}"},
        ],  # get_nodes_batch backfill
        # _fetch_edges_among({seed, n2})
        [{"source_id": "seed", "target_id": "n2", "properties": "{}"}],
    ]
    await storage.get_knowledge_graph("seed", max_depth=1, max_nodes=100)
    bfs_sql, bfs_params = storage.db.query.call_args_list[0].args[:2]
    assert bfs_sql.strip().startswith("WITH nb AS")
    # One IN list per UNION arm, each keyed only by the frontier: the frontier
    # has one node, so the statement carries 2 IN items no matter how many
    # neighbours that node has.
    assert bfs_sql.count("IN (SELECT nid FROM nb)") == 2
    assert bfs_sql.count("%(node_id_") == 2
    assert [k for k in bfs_params if k.startswith("node_id_")] == ["node_id_0"]
    assert bfs_params["node_id_0"] == "seed"
    # The cap travels as a number, not as literals.
    assert bfs_params["fetch_limit"] == 101
    assert "nid" in bfs_sql and "degree" in bfs_sql


@pytest.mark.asyncio
async def test_bfs_visited_row_does_not_consume_an_admission_slot():
    """The over-fetch window must be filtered before the level_cap is spent.

    The seed is itself a neighbour of the frontier and comes back ranked first
    by degree. Counting it as an admission would drop a real candidate and
    hide the truncation behind a budget that was never actually filled.
    """
    storage = _make_storage()
    storage.db.query.side_effect = [
        [
            {"nid": "seed", "degree": 9},  # already visited, ranks first
            {"nid": "n2", "degree": 5},
            {"nid": "n3", "degree": 4},
            {"nid": "n4", "degree": 1},
        ],
        [
            {"node_id": "seed", "properties": "{}"},
            {"node_id": "n2", "properties": "{}"},
            {"node_id": "n3", "properties": "{}"},
        ],
        [],  # _fetch_edges_among
    ]
    kg = await storage.get_knowledge_graph("seed", max_depth=1, max_nodes=3)
    # level_cap = 3 - 1 + 1 = 3 real admissions on top of the seed, so the
    # budget is overfull and truncation must be reported.
    assert kg.is_truncated is True
    assert [n.id for n in kg.nodes] == ["seed", "n2", "n3"]
    assert storage.db.query.call_args_list[0].args[1]["fetch_limit"] == 4


@pytest.mark.asyncio
async def test_bfs_absent_seed_is_decided_by_the_backfill_read():
    """A dangling edge must not smuggle a graph out for a seed that is gone.

    The traversal reaches the neighbour through edge rows even when the node
    row is missing; only the batched node read can tell, and it is the read
    that replaces the seed probe.
    """
    storage = _make_storage()
    storage.db.query.side_effect = [
        [{"nid": "n2", "degree": 2}],  # level 1 reaches n2
        [{"node_id": "n2", "properties": "{}"}],  # no row for the seed
        [{"source_id": "seed", "target_id": "n2", "properties": "{}"}],
    ]
    kg = await storage.get_knowledge_graph("seed", max_depth=1, max_nodes=100)
    assert kg.nodes == [] and kg.edges == []
    assert kg.is_truncated is False
    sqls = _all_query_sqls(storage)
    assert len(sqls) == 3
    # The old shape spent a point read on every request to answer this.
    assert not any("SELECT 1 FROM LIGHTRAG_GRAPH_NODES" in sql for sql in sqls)


@pytest.mark.asyncio
async def test_bfs_tail_reads_are_issued_concurrently():
    """Backfill and edge fetch overlap; the traversal levels stay serial."""
    storage = _make_storage()
    rows = {
        "WITH nb AS": [{"nid": "n2", "degree": 2}],
        "LIGHTRAG_GRAPH_NODES": [
            {"node_id": "seed", "properties": "{}"},
            {"node_id": "n2", "properties": "{}"},
        ],
        "LIGHTRAG_GRAPH_EDGES": [
            {"source_id": "seed", "target_id": "n2", "properties": "{}"}
        ],
    }
    in_flight = {"live": 0, "peak": 0}

    async def _query(sql, params=None, multirows=False):
        in_flight["live"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["live"])
        await asyncio.sleep(0)  # yield so a concurrent sibling can start
        in_flight["live"] -= 1
        return next(r for key, r in rows.items() if key in sql)

    storage.db.query = AsyncMock(side_effect=_query)
    kg = await storage.get_knowledge_graph("seed", max_depth=1, max_nodes=100)
    assert [n.id for n in kg.nodes] == ["seed", "n2"]
    assert len(kg.edges) == 1
    # Serial reads never exceed one; the level query cannot overlap with
    # itself, so the only way to reach 2 is the concurrent tail.
    assert in_flight["peak"] == 2


# ===================================================================
# Graph DDL
# ===================================================================


def test_graph_ddl_time_columns_and_target_index():
    """Graph DDL: update_time (no created_at/updated_at) and a secondary
    index covering target_id for undirected lookups (PK prefix only serves
    source_id)."""
    nodes_ddl = GRAPH_TABLES["LIGHTRAG_GRAPH_NODES"]["ddl"]
    edges_ddl = GRAPH_TABLES["LIGHTRAG_GRAPH_EDGES"]["ddl"]
    for ddl in (nodes_ddl, edges_ddl):
        assert "update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in ddl
        assert "created_at" not in ddl
        assert "updated_at" not in ddl
    assert "KEY idx_edges_target (workspace, target_id)" in edges_ddl


# ===================================================================
# node_degree / edge_degree
# ===================================================================


@pytest.mark.asyncio
async def test_node_degree_returns_count():
    storage = _make_storage()
    storage.db.query.return_value = {"degree": 4}
    assert await storage.node_degree("n1") == 4
    sql = storage.db.query.call_args.args[0]
    assert "UNION ALL" in sql  # counts both edge directions


@pytest.mark.asyncio
async def test_node_degree_missing_returns_zero():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.node_degree("missing") == 0


@pytest.mark.asyncio
async def test_edge_degree_sums_endpoint_degrees():
    storage = _make_storage()
    storage.db.query.return_value = {"degree": 8}
    assert await storage.edge_degree("a", "b") == 8
    # One round trip for the pair, not node_degree(src) + node_degree(tgt).
    assert storage.db.query.await_count == 1
    sql, params = storage.db.query.call_args.args[:2]
    assert "UNION ALL" in sql
    # Each arm keeps a single-index prefix: source_id through the PK, target_id
    # through idx_edges_target. A cross-index OR in either arm serves neither.
    assert "source_id IN (%(src_id)s, %(tgt_id)s)" in sql
    assert "target_id IN (%(src_id)s, %(tgt_id)s)" in sql
    assert params["src_id"] == "a" and params["tgt_id"] == "b"


@pytest.mark.asyncio
async def test_edge_degree_missing_edge_returns_zero():
    storage = _make_storage()
    storage.db.query.return_value = None
    assert await storage.edge_degree("x", "y") == 0


# ===================================================================
# node_degrees_batch / edge_degrees_batch
# ===================================================================


@pytest.mark.asyncio
async def test_node_degrees_batch_fills_missing_with_zero():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "n1", "degree": 2},
        {"node_id": "n2", "degree": 1},
    ]
    result = await storage.node_degrees_batch(["n1", "n2", "n3"])
    assert result == {"n1": 2, "n2": 1, "n3": 0}


@pytest.mark.asyncio
async def test_node_degrees_batch_empty_input():
    storage = _make_storage()
    assert await storage.node_degrees_batch([]) == {}
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_edge_degrees_batch_sums_endpoint_degrees():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "a", "degree": 2},
        {"node_id": "b", "degree": 3},
        {"node_id": "c", "degree": 1},
    ]
    result = await storage.edge_degrees_batch([("a", "b"), ("b", "c")])
    assert result == {("a", "b"): 5, ("b", "c"): 4}


@pytest.mark.asyncio
async def test_edge_degrees_batch_empty_input():
    storage = _make_storage()
    assert await storage.edge_degrees_batch([]) == {}


# ===================================================================
# get_edges_batch
# ===================================================================


@pytest.mark.asyncio
async def test_get_edges_batch_canonical_lookup_and_reverse_key():
    """Reverse-ordered request pairs are keyed back to the caller's order."""
    storage = _make_storage()
    storage.db.query.return_value = [
        {"source_id": "a", "target_id": "b", "properties": '{"w": "1"}'},
        # Row outside the requested canonical pair set must be dropped.
        {"source_id": "a", "target_id": "c", "properties": '{"w": "x"}'},
    ]
    result = await storage.get_edges_batch([{"src": "b", "tgt": "a"}])
    assert set(result) == {("b", "a")}
    assert result[("b", "a")]["w"] == "1"
    sql = storage.db.query.call_args.args[0]
    assert "source_id IN" in sql and "target_id IN" in sql


@pytest.mark.asyncio
async def test_get_edges_batch_empty_input():
    storage = _make_storage()
    assert await storage.get_edges_batch([]) == {}
    storage.db.query.assert_not_called()


# ===================================================================
# get_nodes_edges_batch
# ===================================================================


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_attributes_both_directions():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"source_id": "n1", "target_id": "n2"},
        {"source_id": "n3", "target_id": "n1"},
    ]
    result = await storage.get_nodes_edges_batch(["n1", "n2"])
    # Edges touching n1 from either side, sorted by counterpart id.
    assert result["n1"] == [("n1", "n2"), ("n1", "n3")]
    assert result["n2"] == [("n2", "n1")]


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_no_rows_returns_empty_lists():
    storage = _make_storage()
    storage.db.query.return_value = []
    result = await storage.get_nodes_edges_batch(["n1", "n2"])
    assert result == {"n1": [], "n2": []}


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_empty_input():
    storage = _make_storage()
    assert await storage.get_nodes_edges_batch([]) == {}


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_uses_two_index_prefixed_branches():
    storage = _make_storage()
    storage.db.query.return_value = []
    await storage.get_nodes_edges_batch(["n1", "n2"])
    sql = storage.db.query.call_args.args[0]
    assert "UNION ALL" in sql
    assert " OR " not in sql
    # One IN list per branch, both keyed by the same node_id_* params.
    assert sql.count("source_id IN") == 1
    assert sql.count("target_id IN") == 1
    assert sql.count("%(node_id_0)s") == 2


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_collapses_the_self_loop():
    """The self-loop row comes back from both UNION ALL branches."""
    storage = _make_storage()
    storage.db.query.return_value = [
        {"source_id": "n1", "target_id": "n1"},
        {"source_id": "n1", "target_id": "n1"},
    ]
    assert await storage.get_nodes_edges_batch(["n1"]) == {"n1": [("n1", "n1")]}


# ===================================================================
# upsert_edges_batch
# ===================================================================


@pytest.mark.asyncio
async def test_upsert_edges_batch_dedups_and_creates_missing_endpoints():
    storage = _make_storage()
    storage.db.query.side_effect = [
        [{"node_id": "a"}],  # has_nodes_batch: only 'a' exists
        [],  # get_nodes_batch inside upsert_nodes_batch
    ]
    # Reverse duplicate pair: last write wins under canonical key.
    await storage.upsert_edges_batch([("b", "a", {"w": "1"}), ("a", "b", {"w": "2"})])
    # 2 executes: endpoint node REPLACE for 'b' + single edge REPLACE batch.
    assert storage.db.execute.await_count == 2
    edge_call = storage.db.execute.call_args_list[1]
    datas = edge_call.args[1]
    assert isinstance(datas, list) and len(datas) == 1
    assert datas[0]["src"] == "a" and datas[0]["tgt"] == "b"
    assert json.loads(datas[0]["properties"]) == {"w": "2"}


@pytest.mark.asyncio
async def test_upsert_edges_batch_all_endpoints_exist_skips_node_writes():
    storage = _make_storage()
    storage.db.query.return_value = [{"node_id": "a"}, {"node_id": "b"}]
    await storage.upsert_edges_batch([("a", "b", {"w": "1"})])
    storage.db.query.assert_awaited_once()  # single has_nodes_batch read
    storage.db.execute.assert_awaited_once()  # only the edge REPLACE


@pytest.mark.asyncio
async def test_upsert_edges_batch_flushes_in_max_batch_chunks():
    """Edge REPLACE flushes in db.max_batch_size chunks (no endpoint writes)."""
    storage = _make_storage()
    storage.db.max_batch_size = 2
    endpoints = [f"n{i}" for i in range(6)]
    storage.db.query.return_value = [
        {"node_id": nid} for nid in endpoints
    ]  # has_nodes_batch: every endpoint exists
    await storage.upsert_edges_batch(
        [(endpoints[i], endpoints[i + 1], {"w": str(i)}) for i in range(5)]
    )
    storage.db.query.assert_awaited_once()  # still ONE endpoint existence read
    assert storage.db.execute.await_count == 3  # ceil(5 / 2), no node writes
    chunk_sizes = [len(c.args[1]) for c in storage.db.execute.call_args_list]
    assert chunk_sizes == [2, 2, 1]


@pytest.mark.asyncio
async def test_upsert_edges_batch_empty_input():
    storage = _make_storage()
    await storage.upsert_edges_batch([])
    storage.db.query.assert_not_called()
    storage.db.execute.assert_not_called()


# ===================================================================
# get_all_nodes / get_all_edges
# ===================================================================


@pytest.mark.asyncio
async def test_get_all_nodes_sorted_with_id_and_entity_id():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"node_id": "b", "properties": '{"desc": "B"}'},
        {"node_id": "a", "properties": None},
    ]
    result = await storage.get_all_nodes()
    assert [p["id"] for p in result] == ["a", "b"]
    assert result[1] == {"desc": "B", "entity_id": "b", "id": "b"}
    assert result[0] == {"entity_id": "a", "id": "a"}


@pytest.mark.asyncio
async def test_get_all_nodes_empty():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_all_nodes() == []


@pytest.mark.asyncio
async def test_get_all_edges_merges_properties_with_endpoints():
    storage = _make_storage()
    storage.db.query.return_value = [
        {"source_id": "a", "target_id": "b", "properties": '{"w": "1"}'},
        {"source_id": "c", "target_id": "d", "properties": None},
    ]
    result = await storage.get_all_edges()
    assert result[0] == {"w": "1", "source": "a", "target": "b"}
    assert result[1] == {"source": "c", "target": "d"}


@pytest.mark.asyncio
async def test_get_all_edges_empty():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_all_edges() == []


# ===================================================================
# _fetch_edges_among short-circuit
# ===================================================================


@pytest.mark.asyncio
async def test_fetch_edges_among_single_node_short_circuits():
    storage = _make_storage()
    result = await storage._fetch_edges_among({"only"})
    assert result == []
    storage.db.query.assert_not_called()


# ===================================================================
# iter_labels / iter_edges — bounded keyset iteration
# ===================================================================


@pytest.mark.asyncio
async def test_iter_labels_batched_keyset():
    storage = _make_storage()
    storage.db.query.side_effect = [
        [{"node_id": f"n{i}"} for i in range(3)],
        [{"node_id": "n3"}],
        [],
    ]
    batches = [batch async for batch in storage.iter_labels(3)]
    assert batches == [["n0", "n1", "n2"], ["n3"]]
    # Second page resumes the keyset after the last label of the first page.
    second_sql, second_params = storage.db.query.call_args_list[1].args[:2]
    assert "LIGHTRAG_GRAPH_NODES" in second_sql
    assert "ORDER BY node_id ASC" in second_sql
    assert second_params["after"] == "n2"
    assert second_params["batch_size"] == 3
    assert second_params["workspace"] == "test_ws"


@pytest.mark.asyncio
async def test_iter_labels_empty_graph():
    storage = _make_storage()
    storage.db.query.return_value = []
    batches = [batch async for batch in storage.iter_labels(5)]
    assert batches == []


@pytest.mark.asyncio
async def test_iter_edges_batched_shape_and_keyset():
    storage = _make_storage()
    storage.db.query.side_effect = [
        [
            {"source_id": "a", "target_id": "b", "properties": '{"w": "1"}'},
            {"source_id": "a", "target_id": "c", "properties": None},
        ],
        [{"source_id": "b", "target_id": "d", "properties": '{"w": "2"}'}],
        [],
    ]
    batches = [batch async for batch in storage.iter_edges(2)]
    assert len(batches) == 2
    # Same dict shape as get_all_edges: properties merged + endpoint keys.
    assert batches[0][0] == {"w": "1", "source": "a", "target": "b"}
    assert batches[0][1] == {"source": "a", "target": "c"}
    assert batches[1][0] == {"w": "2", "source": "b", "target": "d"}
    # Second page resumes the composite keyset past (a, c), the last row
    # of the first page.
    second_sql, second_params = storage.db.query.call_args_list[1].args[:2]
    assert "LIGHTRAG_GRAPH_EDGES" in second_sql
    assert "ORDER BY source_id ASC, target_id ASC" in second_sql
    assert second_params["after_src"] == "a"
    assert second_params["after_tgt"] == "c"


@pytest.mark.asyncio
async def test_iter_edges_empty_graph():
    storage = _make_storage()
    storage.db.query.return_value = []
    batches = [batch async for batch in storage.iter_edges(5)]
    assert batches == []


@pytest.mark.asyncio
async def test_iter_labels_rejects_non_positive_batch_size():
    storage = _make_storage()
    with pytest.raises(ValueError, match="positive"):
        await anext(storage.iter_labels(0))


@pytest.mark.asyncio
async def test_iter_edges_rejects_non_positive_batch_size():
    storage = _make_storage()
    with pytest.raises(ValueError, match="positive"):
        await anext(storage.iter_edges(-1))
