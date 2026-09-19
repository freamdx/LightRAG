import asyncio
import datetime
import hashlib
import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, ClassVar, Union, final
import numpy as np

from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    CursorAfter,
    CURSOR_END,
    CURSOR_START,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceConflict,
    SourceConflictPage,
    SourceConflictRepairResult,
    SourceConflictSummary,
    SourceResolution,
    SourceUnique,
    BaseGraphStorage,
)
from ..exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from ..namespace import NameSpace, is_namespace
from ..constants import CUSTOM_CHUNK_PATCH_METADATA_KEY, DEFAULT_QUERY_PRIORITY
from ..utils import logger, validate_workspace
from ..kg.shared_storage import get_data_init_lock
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

import pipmaster

if not pipmaster.is_installed("aiomysql"):
    pipmaster.install("aiomysql")

import aiomysql

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)


class AnalyticDB:
    """AnalyticDB MySQL

    For more information, please visit
        [AnalyticDB MySQL official site](https://www.alibabacloud.com/en/product/analyticdb-for-mysql)

    For Example:
    .. code-block:: python
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func,
            ),
            #rerank_model_func=rerank_model_func,
            #tiktoken_model_name="gpt-4o-mini",
            #graph_storage="NetworkXStorage",
            kv_storage="ADBKVStorage",
            vector_storage="ADBVectorStorage",
            doc_status_storage="ADBDocStatusStorage",
        )
    """

    def __init__(self, **kwargs: Any):
        self._lock = asyncio.Lock()

        self.db_config = {
            "host": os.getenv("ADB_HOST", "localhost"),
            "port": int(os.getenv("ADB_PORT", "3306")),
            "user": os.getenv("ADB_USER"),
            "password": os.getenv("ADB_PASSWORD"),
            "db": os.getenv("ADB_DATABASE"),
            "maxsize": int(os.getenv("ADB_MAX_CONNECTIONS", "5")),
            "autocommit": True,
        }
        self.workspace = os.getenv("ADB_WORKSPACE", "graphrag")
        self.pool = None

        if not all(
            [self.db_config["user"], self.db_config["password"], self.db_config["db"]]
        ):
            raise ValueError("Missing database user, password, or database")

    async def initdb(self):
        # init pool
        try:
            self.pool = await aiomysql.create_pool(
                **self.db_config, cursorclass=aiomysql.DictCursor
            )
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to connect database, Got:{e}")
            raise e

        # check tables
        for k, v in TABLES.items():
            try:
                result = await self.query(
                    "SELECT 1 FROM information_schema.kepler_meta_tables "
                    "where table_schema=%(db)s and table_name=lower(%(table)s)",
                    {"db": self.db_config["db"], "table": k},
                )
                if result is None:
                    logger.info(f"AnalyticDB MySQL, Try Creating table {k} in database")
                    await self.execute(v["ddl"])
            except Exception as e:
                logger.error(
                    f"AnalyticDB MySQL, Failed to create table {k} in database, Got: {e}"
                )
                raise e

    async def close_pool(self):
        async with self._lock:
            # aiomysql ``Pool.closed`` is a property, not a method.
            if self.pool is not None and not self.pool.closed:
                self.pool.terminate()
                await self.pool.wait_closed()

    async def query(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        multirows: bool = False,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if params is None:
                        await cursor.execute(sql)
                    else:
                        await cursor.execute(sql, params)

                    if multirows:
                        rows = await cursor.fetchall()
                        return list(rows) if rows else []
                    else:
                        row = await cursor.fetchone()
                        return dict(row) if row else None
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, \nsql:{sql},\nparam:{params},\nerror:{e}")
            raise e

    async def execute(
        self,
        sql: str,
        datas: dict[str, Any] | list[dict[str, Any]] | None = None,
    ):
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if datas is None:
                        await cursor.execute(sql)
                    else:
                        if isinstance(datas, list):
                            await cursor.executemany(sql, datas)
                        else:
                            await cursor.execute(sql, datas)
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, \nsql:{sql},\ndata:{datas},\nerror:{e}")
            raise e

    async def execute_transaction(self, statements: list[tuple[str, Any]]) -> None:
        """Run multiple statements atomically on ONE pooled connection.

        ``query``/``execute`` each acquire their own pooled connection, so bare
        START TRANSACTION/COMMIT issued through them cannot span statements.
        Each entry is a ``(sql, params)`` tuple: params ``None`` runs a bare
        statement, a ``list`` uses executemany, otherwise the dict is bound to
        the named placeholders. Any failure rolls the whole batch back.
        """
        async with self.pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cursor:
                    for sql, params in statements:
                        if params is None:
                            await cursor.execute(sql)
                        elif isinstance(params, list):
                            await cursor.executemany(sql, params)
                        else:
                            await cursor.execute(sql, params)
                await conn.commit()
            except Exception:
                await conn.rollback()
                logger.error("AnalyticDB MySQL, transaction rolled back")
                raise

    @staticmethod
    def build_in_clause(
        field_name: str, values: list[str]
    ) -> tuple[str, dict[str, Any]]:
        if not values:
            return "", {}

        placeholder = [f"%({field_name}_{i})s" for i in range(len(values))]
        params = {f"{field_name}_{i}": val for i, val in enumerate(values)}
        return ",".join(placeholder), params


@final
@dataclass
class ADBKVStorage(BaseKVStorage):
    db: AnalyticDB | None = field(default=None)

    supports_strict_point_reads: ClassVar[bool] = True

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        self._max_delete_records_per_batch = int(
            os.getenv("ADB_DELETE_MAX_RECORDS_PER_BATCH", "1000")
        )

    @staticmethod
    def _parse_dict_field(value: Any) -> dict:
        """Parse a JSON dict column value; normalize None/missing/invalid to {}."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value if isinstance(value, dict) else {}

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.workspace, "id": id}

        response = await self.db.query(sql, params)

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list

            # Parse heading / sidecar JSON strings back to dicts; normalize
            # None/missing to {}
            response["heading"] = self._parse_dict_field(response.get("heading"))
            response["sidecar"] = self._parse_dict_field(response.get("sidecar"))

            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            # Parse chunk_options JSON string back to dict; normalize None/missing to {}
            response["chunk_options"] = self._parse_dict_field(
                response.get("chunk_options")
            )

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            # Parse queryparam JSON string back to dict
            queryparam = response.get("queryparam")
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type"),
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "queryparam": queryparam,
                "create_time": create_time,
                "update_time": create_time if update_time == 0 else update_time,
            }

        # Special handling for FULL_ENTITIES namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Parse entity_names JSON string back to list
            entity_names = response.get("entity_names", [])
            if isinstance(entity_names, str):
                try:
                    entity_names = json.loads(entity_names)
                except json.JSONDecodeError:
                    entity_names = []
            response["entity_names"] = entity_names
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Parse relation_pairs JSON string back to list
            relation_pairs = response.get("relation_pairs", [])
            if isinstance(relation_pairs, str):
                try:
                    relation_pairs = json.loads(relation_pairs)
                except json.JSONDecodeError:
                    relation_pairs = []
            response["relation_pairs"] = relation_pairs
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS
        ):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        return response if response else None

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        """Strict point read: complete-or-raise (base contract).

        ``db.query`` propagates every aiomysql error (nothing in this class
        swallows it), so a ``None`` from the legacy read is a positively
        confirmed absence — safe for callers that take destructive action on a
        miss.
        """
        return await self.get_by_id(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace].replace(
            "%(ids)s", placeholder
        )
        params = {"workspace": self.workspace, **id_params}

        results = await self.db.query(sql, params, multirows=True)

        def _order_results(
            rows: list[dict[str, Any]] | None,
        ) -> list[dict[str, Any] | None]:
            """Preserve the caller requested ordering for bulk id lookups."""
            if not rows:
                return [None for _ in ids]

            id_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row is None:
                    continue
                row_id = row.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = row

            ordered: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered.append(id_map.get(str(requested_id)))
            return ordered

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list for each result
            for result in results:
                llm_cache_list = result.get("llm_cache_list", [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result["llm_cache_list"] = llm_cache_list

                # Parse heading / sidecar JSON strings back to dicts; normalize
                # None/missing to {}
                result["heading"] = self._parse_dict_field(result.get("heading"))
                result["sidecar"] = self._parse_dict_field(result.get("sidecar"))

                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for result in results:
                # Parse chunk_options JSON string back to dict; normalize
                # None/missing to {}
                result["chunk_options"] = self._parse_dict_field(
                    result.get("chunk_options")
                )

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            processed_results = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                # Parse queryparam JSON string back to dict
                queryparam = row.get("queryparam")
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": queryparam,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)

            return _order_results(processed_results)

        # Special handling for FULL_ENTITIES namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            for result in results:
                # Parse entity_names JSON string back to list
                entity_names = result.get("entity_names", [])
                if isinstance(entity_names, str):
                    try:
                        entity_names = json.loads(entity_names)
                    except json.JSONDecodeError:
                        entity_names = []
                result["entity_names"] = entity_names
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            for result in results:
                # Parse relation_pairs JSON string back to list
                relation_pairs = result.get("relation_pairs", [])
                if isinstance(relation_pairs, str):
                    try:
                        relation_pairs = json.loads(relation_pairs)
                    except json.JSONDecodeError:
                        relation_pairs = []
                result["relation_pairs"] = relation_pairs
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result["chunk_ids"] = chunk_ids
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result["chunk_ids"] = chunk_ids
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        return _order_results(results)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)

        placeholder, id_params = AnalyticDB.build_in_clause("id", list(keys))
        sql = f"SELECT id FROM {table_name} WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        params = {"workspace": self.workspace, **id_params}

        res = await self.db.query(sql, params, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
        else:
            exist_keys = []
        new_keys = set([s for s in keys if s not in exist_keys])
        return new_keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # All fields are replaced on conflict (REPLACE INTO); create_time is
        # bound from the application clock, update_time is generated by the
        # server via CURRENT_TIMESTAMP.
        create_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_text_chunk"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "tokens": v["tokens"],
                    "chunk_order_index": v["chunk_order_index"],
                    "full_doc_id": v["full_doc_id"],
                    "content": v["content"],
                    "file_path": v["file_path"],
                    "llm_cache_list": json.dumps(v.get("llm_cache_list", [])),
                    "heading": json.dumps(v.get("heading") or {}),
                    "sidecar": json.dumps(v.get("sidecar") or {}),
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
            for k, v in data.items():
                _data = {
                    "id": k,
                    "content": v["content"],
                    "doc_name": v.get("file_path", ""),  # Map file_path to doc_name
                    "workspace": self.workspace,
                    "sidecar_location": v.get("sidecar_location"),
                    "parse_format": v.get("parse_format"),
                    "content_hash": v.get("content_hash"),
                    "process_options": v.get("process_options"),
                    "chunk_options": json.dumps(v.get("chunk_options") or {}),
                    "parse_engine": v.get("parse_engine"),
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,  # Use flattened key as id
                    "original_prompt": v["original_prompt"],
                    "return_value": v["return"],
                    "chunk_id": v.get("chunk_id"),
                    "cache_type": v.get(
                        "cache_type", "extract"
                    ),  # Get cache_type from data
                    "queryparam": json.dumps(v.get("queryparam"))
                    if v.get("queryparam")
                    else None,
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_full_entities"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "entity_names": json.dumps(v["entity_names"]),
                    "count": v["count"],
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_full_relations"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "relation_pairs": json.dumps(v["relation_pairs"]),
                    "count": v["count"],
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_entity_chunks"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "chunk_ids": json.dumps(v["chunk_ids"]),
                    "count": v["count"],
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_relation_chunks"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "chunk_ids": json.dumps(v["chunk_ids"]),
                    "count": v["count"],
                    "create_time": create_at,
                }
                datas.append(_data)
            for offset in range(0, len(datas), self._max_batch_size):
                await self.db.execute(
                    upsert_sql, datas[offset : offset + self._max_batch_size]
                )
        else:
            logger.error(f"Unknown namespace: {self.namespace}")
            raise ValueError(f"Unknown namespace: {self.namespace}")

    async def index_done_callback(self) -> None:
        pass

    async def is_empty(self) -> bool:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}"
            )
            return True

        sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace=%(workspace)s LIMIT 1) as has_data"

        try:
            result = await self.db.query(sql, {"workspace": self.workspace})

            return not result.get("has_data", False) if result else True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        # Chunk the id list so each IN clause stays bounded (a non-positive
        # cap disables chunking). Multiple chunks run in ONE transaction via
        # execute_transaction, preserving the single-statement all-or-nothing
        # behaviour.
        chunk = (
            self._max_delete_records_per_batch
            if self._max_delete_records_per_batch > 0
            else len(ids)
        )

        try:
            if len(ids) <= chunk:
                placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
                delete_sql = (
                    f"DELETE FROM {table_name} "
                    f"WHERE workspace=%(workspace)s AND id IN ({placeholder})"
                )
                await self.db.execute(
                    delete_sql, {"workspace": self.workspace, **id_params}
                )
            else:
                logger.info(
                    f"[{self.workspace}] {self.namespace} delete: {len(ids)} ids "
                    f"split into chunks (chunk={chunk})"
                )
                statements: list[tuple[str, dict[str, Any]]] = []
                for i in range(0, len(ids), chunk):
                    placeholder, id_params = AnalyticDB.build_in_clause(
                        "id", ids[i : i + chunk]
                    )
                    delete_sql = (
                        f"DELETE FROM {table_name} "
                        f"WHERE workspace=%(workspace)s AND id IN ({placeholder})"
                    )
                    statements.append(
                        (delete_sql, {"workspace": self.workspace, **id_params})
                    )
                await self.db.execute_transaction(statements)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specify_table_workspace"].format(
                table_name=table_name
            )

            await self.db.execute(drop_sql, {"workspace": self.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBVectorStorage(BaseVectorStorage):
    db: AnalyticDB | None = field(default=None)

    def __post_init__(self):
        if self.embedding_func is None:
            raise ValueError("embedding_func is required for vector storage")
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

            # check vector tables
            for k, v in VECTOR_TABLES.items():
                try:
                    result = await self.db.query(
                        "SELECT 1 FROM information_schema.kepler_meta_tables "
                        "where table_schema=%(db)s and table_name=lower(%(table)s)",
                        {"db": self.db.db_config["db"], "table": k},
                    )
                    if result is None:
                        logger.info(
                            f"AnalyticDB MySQL, Try Creating vector table {k} in database"
                        )
                        # Dimension comes from the embedding function metadata;
                        # no dummy embedding call is made at startup.
                        ddl = v["ddl"].replace(
                            "ARRAY<FLOAT>(EMBEDDING_DIM)",
                            f"ARRAY<FLOAT>({self.embedding_func.embedding_dim})",
                        )
                        await self.db.execute(ddl)
                        # XUANWU_V2 tables are provisioned asynchronously.
                        await asyncio.sleep(3)

                        # add ann index.
                        ann_ddl = v.get("ann_index_ddl")
                        if ann_ddl:
                            await self.db.execute(ann_ddl)
                            logger.info(f"AnalyticDB MySQL, Created ann index for {k}")
                except Exception as e:
                    logger.error(
                        f"AnalyticDB MySQL, Failed to create vector table {k} in database, Got: {e}"
                    )
                    raise e

    async def finalize(self):
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    def _upsert_chunks(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_chunk"]
        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "tokens": item["tokens"],
            "chunk_order_index": item["chunk_order_index"],
            "full_doc_id": item["full_doc_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "file_path": item["file_path"],
        }
        return upsert_sql, data

    def _upsert_entities(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_entity"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "entity_name": item["entity_name"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "chunk_ids": json.dumps(chunk_ids),
            "file_path": item.get("file_path", None),
        }
        return upsert_sql, data

    def _upsert_relationships(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_relationship"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "source_id": item["src_id"],
            "target_id": item["tgt_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "chunk_ids": json.dumps(chunk_ids),
            "file_path": item.get("file_path", None),
        }
        return upsert_sql, data

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [
            self.embedding_func(batch, context="document") for batch in batches
        ]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]

        # create_time is bound from the application clock (KV parity);
        # REPLACE INTO replaces the whole row, so relying on the DDL default
        # would re-stamp it on every write. update_time stays server-side.
        create_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        datas = []
        upsert_sql = ""
        for item in list_data:
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                upsert_sql, data = self._upsert_chunks(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                upsert_sql, data = self._upsert_entities(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                upsert_sql, data = self._upsert_relationships(item)
            else:
                raise ValueError(f"{self.namespace} is not supported")
            data["create_time"] = create_at
            datas.append(data)
        for offset in range(0, len(datas), self._max_batch_size):
            await self.db.execute(
                upsert_sql, datas[offset : offset + self._max_batch_size]
            )

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
            )  # higher priority for query
            embedding = embeddings[0]

        embedding_string = ",".join(map(str, embedding))

        sql = SQL_TEMPLATES[self.namespace].format(embedding_string=embedding_string)
        # The threshold is a cosine-similarity floor: passed through directly,
        # NOT inverted into an L2-style distance bound.
        params = {
            "workspace": self.workspace,
            "cosine_better_than_threshold": self.cosine_better_than_threshold,
            "top_k": top_k,
        }

        results = await self.db.query(sql, params=params, multirows=True)
        return results

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for vector deletion: {self.namespace}"
            )
            return

        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        delete_sql = f"DELETE FROM {table_name} WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        params = {"workspace": self.workspace, **id_params}

        try:
            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )
            raise

    async def delete_entity(self, entity_name: str) -> None:
        try:
            # Construct SQL to delete the entity
            delete_sql = "DELETE FROM LIGHTRAG_VDB_ENTITY WHERE workspace=%(workspace)s AND entity_name=%(entity_name)s"
            params = {"workspace": self.workspace, "entity_name": entity_name}

            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")
            raise

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_VDB_RELATION
                         WHERE workspace=%(workspace)s AND (source_id=%(entity_name)s OR target_id=%(entity_name)s)
                         """
            params = {"workspace": self.workspace, "entity_name": entity_name}

            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for entity {entity_name}: {e}"
            )
            raise

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for ID lookup: {self.namespace}"
            )
            return None

        query = (
            f"SELECT *, UNIX_TIMESTAMP(create_time) as created_at FROM {table_name} "
            f"WHERE workspace=%(workspace)s AND id=%(id)s"
        )
        params = {"workspace": self.workspace, "id": id}

        try:
            result = await self.db.query(query, params)
            if result:
                result_dict = dict(result)
                # Embedding vectors are never needed for point reads and can
                # be large; strip the column like the PG backend does.
                result_dict.pop("content_vector", None)
                return result_dict
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for IDs lookup: {self.namespace}"
            )
            return []

        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        query = (
            f"SELECT *, UNIX_TIMESTAMP(create_time) as created_at FROM {table_name} "
            f"WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        )
        params = {"workspace": self.workspace, **id_params}

        try:
            results = await self.db.query(query, params, multirows=True)
            if not results:
                return []

            # Preserve caller requested ordering while normalizing asyncpg rows to dicts.
            id_map: dict[str, dict[str, Any]] = {}
            for record in results:
                if record is None:
                    continue
                record_dict = dict(record)
                record_dict.pop("content_vector", None)
                row_id = record_dict.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = record_dict

            ordered_results: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered_results.append(id_map.get(str(requested_id)))
            return ordered_results
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for vector lookup: {self.namespace}"
            )
            return {}

        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        query = f"SELECT id, content_vector FROM {table_name} WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        params = {"workspace": self.workspace, **id_params}

        try:
            results = await self.db.query(query, params, multirows=True)
            vectors_dict = {}

            for result in results:
                if result and "content_vector" in result and "id" in result:
                    try:
                        # Parse JSON string to get vector as list of floats
                        vector_data = result["content_vector"]
                        if hasattr(vector_data, "tolist"):
                            vectors_dict[result["id"]] = vector_data.tolist()
                        else:
                            vectors_dict[result["id"]] = json.loads(vector_data)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(
                            f"[{self.workspace}] Failed to parse vector data for ID {result['id']}: {e}"
                        )

            return vectors_dict
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specify_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.workspace})

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBDocStatusStorage(DocStatusStorage):
    db: AnalyticDB | None = field(default=None)

    supports_strict_point_reads: ClassVar[bool] = True

    # Bounded upper limit on the sample of conflicting doc IDs surfaced by the
    # source-conflict listing/repair APIs — never materialize the whole set.
    _CONFLICT_SAMPLE_CAP: ClassVar[int] = 32

    # Whitelist for update_doc_status_fields: column names are interpolated
    # into SQL and must never come from caller input. created_at is absent:
    # it is the immutable keyset sort key.
    _UPDATABLE_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {
            "content_summary",
            "content_length",
            "chunks_count",
            "status",
            "file_path",
            "chunks_list",
            "track_id",
            "metadata",
            "error_msg",
            "content_hash",
            "updated_at",
        }
    )
    # JSON columns serialized with json.dumps exactly like the batch upsert does.
    _JSON_COLUMNS: ClassVar[frozenset[str]] = frozenset({"chunks_list", "metadata"})
    # TIMESTAMP columns normalized to naive-UTC strings (MySQL TIMESTAMP
    # columns reject the ``+00:00`` offset suffix of tz-aware ISO input).
    _DATETIME_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"updated_at", "created_at"}
    )

    # SQL predicate isolating PRIMARY (non-duplicate) rows. metadata is a
    # JSON column, so JSON_EXTRACT yields the JSON literals true/false that
    # compare cleanly; a NULL/absent key coalesces to false (primary).
    _PRIMARY_PREDICATE = (
        "COALESCE(JSON_EXTRACT(metadata, '$.is_duplicate'), false) = false"
    )

    # Shared full-column REPLACE INTO statement.
    _REPLACE_SQL = (
        "REPLACE INTO LIGHTRAG_DOC_STATUS(workspace, id, content_summary, "
        "content_length, chunks_count, status, file_path, chunks_list, "
        "track_id, metadata, error_msg, content_hash, created_at, updated_at) "
        "VALUES(%(workspace)s, %(id)s, %(content_summary)s, %(content_length)s, "
        "%(chunks_count)s, %(status)s, %(file_path)s, %(chunks_list)s, "
        "%(track_id)s, %(metadata)s, %(error_msg)s, "
        "%(content_hash)s, %(created_at)s, CURRENT_TIMESTAMP)"
    )

    def __post_init__(self):
        validate_workspace(self.workspace)
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        self._max_delete_records_per_batch = int(
            os.getenv("ADB_DELETE_MAX_RECORDS_PER_BATCH", "1000")
        )

    def _format_datetime(self, value: Any) -> Any:
        """Emit a timezone-aware ISO string for read paths (PG parity).

        aiomysql returns naive datetimes for TIMESTAMP columns; the values
        stored are UTC, so attach UTC before formatting — consumers (e.g. the
        JS frontend) parse a naive ISO string as LOCAL time. Non-datetime
        values pass through unchanged; None becomes "".
        """
        if value is None:
            return ""
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        return value

    def _to_mysql_datetime(self, value: Any, context: str = "") -> Any:
        """Normalize a datetime value to the naive-UTC ``YYYY-MM-DD HH:MM:SS``
        form the MySQL TIMESTAMP columns accept.

        Accepts datetime/date objects and ISO-format strings (tz-aware input
        is converted to UTC first — MySQL datetime literals reject the offset
        suffix, strict mode error 1292). None and unparseable values pass
        through unchanged (the latter is bound as-is, letting the server
        reject it); the optional context hint is logged on parse failure.
        """
        if value is None:
            return None
        if isinstance(value, datetime.datetime):
            if value.tzinfo is not None:
                value = value.astimezone(timezone.utc)
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, datetime.date):
            return value.strftime("%Y-%m-%d 00:00:00")
        try:
            dt = datetime.datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            logger.error(
                f"Unable to parse doc status datetime string"
                f"{f' ({context})' if context else ''}: {value!r}"
            )
            return value
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    def _parse_json_field(self, value: Any, default: Any = None) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value if value is not None else default

    def _parse_row(self, row: dict[str, Any]) -> dict[str, Any]:
        chunks_list = self._parse_json_field(row.get("chunks_list"), [])
        metadata = self._parse_json_field(row.get("metadata"), {})
        if not isinstance(metadata, dict):
            metadata = {}

        return {
            "content_length": row["content_length"],
            "content_summary": row["content_summary"],
            "status": row["status"],
            "chunks_count": row["chunks_count"],
            "created_at": self._format_datetime(row["created_at"]),
            "updated_at": self._format_datetime(row["updated_at"]),
            "file_path": row.get("file_path") or "no-file-path",
            "chunks_list": chunks_list,
            "metadata": metadata,
            "error_msg": row.get("error_msg"),
            "track_id": row.get("track_id"),
            "content_hash": row.get("content_hash"),
        }

    def _row_to_doc_status(self, row: dict[str, Any]) -> DocProcessingStatus:
        return DocProcessingStatus(
            content_summary=row.get("content_summary"),
            content_length=row.get("content_length"),
            status=row.get("status"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            chunks_count=row.get("chunks_count"),
            file_path=row.get("file_path"),
            chunks_list=row.get("chunks_list"),
            metadata=row.get("metadata"),
            error_msg=row.get("error_msg"),
            track_id=row.get("track_id"),
            content_hash=row.get("content_hash"),
        )

    def _doc_status_from_row(self, row: dict[str, Any]) -> DocProcessingStatus:
        """Raw DB row -> DocProcessingStatus (parse + hydrate in one step)."""
        return self._row_to_doc_status(self._parse_row(row))

    def _log_unusable_doc_row(self, element: dict[str, Any] | None, e: Exception):
        """Shared skip-and-log for a row failing required-field parsing."""
        doc_id_hint = element.get("id", "<unknown>") if element else "<unknown>"
        logger.error(
            f"[{self.workspace}] Skipping document '{doc_id_hint}' — "
            f"required field missing or wrong type while parsing DB row: {e!r}"
        )

    def _replace_record_from_row(
        self, row: dict[str, Any], doc_id: str, **overrides: Any
    ) -> dict[str, Any]:
        """Full-column REPLACE INTO params rebuilt from a raw DB row.

        REPLACE INTO replaces the WHOLE row, so every column is re-bound:
        untouched columns are carried over from the read row — created_at
        most of all: it is the immutable keyset sort key and must never fall
        back to the DDL default. JSON columns are re-serialized; ``**overrides``
        wins last (values already prepared for binding).
        """
        record = {
            "workspace": self.workspace,
            "id": doc_id,
            "content_summary": row.get("content_summary"),
            "content_length": row.get("content_length"),
            "chunks_count": row.get("chunks_count"),
            "status": row.get("status"),
            "file_path": row.get("file_path"),
            "chunks_list": json.dumps(
                self._parse_json_field(row.get("chunks_list"), [])
            ),
            "track_id": row.get("track_id"),
            "metadata": json.dumps(self._parse_json_field(row.get("metadata"), {})),
            "error_msg": row.get("error_msg"),
            "content_hash": row.get("content_hash"),
            "created_at": self._to_mysql_datetime(
                row.get("created_at"),
                f"[{self.workspace}] doc {doc_id} created_at",
            ),
        }
        record.update(overrides)
        return record

    def _scheduling_record_from_row(
        self, row: dict[str, Any], *, strict: bool
    ) -> DocSchedulingRecord | None:
        """Project one DB row into the lightweight scheduling record.

        strict raises on unusable rows; relaxed returns None (the row was
        still returned by the scan and stays consumed).
        """
        doc_id = str(row.get("id") or "")
        try:
            if not doc_id:
                raise KeyError("id")
            status = DocStatus(str(row["status"]))
            created_raw = row["created_at"]
            if not isinstance(created_raw, datetime.datetime):
                raise TypeError("created_at must be a timestamp")
            updated_raw = row.get("updated_at") or created_raw
            if not isinstance(updated_raw, datetime.datetime):
                raise TypeError("updated_at must be a timestamp")
            metadata = self._parse_json_field(row.get("metadata"), {})
            if not isinstance(metadata, dict):
                metadata = {}
            return DocSchedulingRecord(
                id=doc_id,
                status=status,
                created_at=self._format_datetime(created_raw),
                updated_at=self._format_datetime(updated_raw),
                file_path=row.get("file_path") or "no-file-path",
                track_id=row.get("track_id"),
                has_custom_chunk_journal=isinstance(
                    metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict
                ),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"[{self.workspace}] Unusable scheduling row "
                f"{doc_id or '<unknown>'}: {e}"
            )
            if strict:
                raise
            return None

    # ------------------------------------------------------------------
    # Keyset cursor codec for the bounded status sweep
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_cursor(opaque: str) -> tuple[datetime.datetime | None, str]:
        """Decode an opaque page cursor into (created_at, id).

        The opaque form is ``json.dumps([created_at_iso | None, id])``. A
        ``None`` first element marks the NULL-created_at bucket (sorted
        FIRST); a malformed cursor raises StorageControlPlaneError.
        """
        try:
            decoded = json.loads(opaque)
            created_iso, doc_id = decoded
            if not isinstance(doc_id, str):
                raise TypeError("cursor id must be a string")
            if created_iso is None:
                return None, doc_id
            if not isinstance(created_iso, str):
                raise TypeError("cursor created_at must be a string or null")
            created = datetime.datetime.fromisoformat(created_iso)
        except (ValueError, TypeError) as e:
            raise StorageControlPlaneError(
                f"Malformed scheduling cursor for ADBDocStatusStorage: {e}"
            ) from e
        if created.tzinfo is not None:
            created = created.astimezone(timezone.utc).replace(tzinfo=None)
        return created, doc_id

    def _encode_cursor(self, row: dict[str, Any]) -> str:
        """Encode the keyset key of a returned DB row as an opaque cursor.

        Rows without a usable created_at sort FIRST and encode as
        ``[null, id]`` so the sweep traverses past them instead of losing
        them behind a comparison NULL can never satisfy.
        """
        created = row.get("created_at")
        if not isinstance(created, datetime.datetime):
            return json.dumps([None, str(row["id"])])
        if created.tzinfo is not None:
            created = created.astimezone(timezone.utc).replace(tzinfo=None)
        return json.dumps([created.isoformat(), str(row["id"])])

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)

        placeholder, id_params = AnalyticDB.build_in_clause("id", list(keys))
        sql = f"SELECT id FROM {table_name} WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        params = {"workspace": self.workspace, **id_params}

        res = await self.db.query(sql, params, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
        else:
            exist_keys = []
        new_keys = set([s for s in keys if s not in exist_keys])

        return new_keys

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and id=%(id)s"
        params = {"workspace": self.workspace, "id": id}

        result = await self.db.query(sql, params, True)
        if result is None or result == []:
            return None

        return self._parse_row(result[0])

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        sql = f"SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        params = {"workspace": self.workspace, **id_params}

        results = await self.db.query(sql, params, True)
        if not results:
            return []

        processed_map: dict[str, dict[str, Any]] = {
            str(row.get("id")): self._parse_row(row) for row in results
        }

        return [processed_map.get(str(requested_id)) for requested_id in ids]

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and file_path=%(file_path)s"
        params = {"workspace": self.workspace, "file_path": file_path}

        result = await self.db.query(sql, params, True)
        if result is None or result == []:
            return None

        return self._parse_row(result[0])

    async def get_by_id_strict(self, id: str) -> Union[dict[str, Any], None]:
        """Strict point read: complete-or-raise (base contract).

        ``db.query`` propagates every transport/server error, so a ``None``
        from the aligned legacy read is a confirmed absence.
        """
        return await self.get_by_id(id)

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Basename-based document lookup on the canonical ``file_path`` column.

        ``file_path`` is one-to-many (duplicate-attempt rows keep the same
        canonical basename); this returns the single PRIMARY
        (``metadata.is_duplicate != true``) row. When only duplicate markers
        remain (primary deleted) the basename is free again and this returns
        ``None``.
        """
        if not basename:
            return None
        if basename == "unknown_source":
            return None

        sql = (
            "SELECT * FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace=%(workspace)s AND file_path=%(file_path)s "
            f"AND {self._PRIMARY_PREDICATE} "
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        params = {"workspace": self.workspace, "file_path": basename}
        result = await self.db.query(sql, params, multirows=True)
        if not result:
            return None
        row = result[0]
        return str(row["id"]), self._parse_row(row)

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """Content-hash document lookup (fail-closed, deterministic, see base).

        ``exclude_doc_id`` adds ``AND id <> %(exclude_id)s`` plus a predicate
        dropping any row that merely POINTS at that id (``is_duplicate``
        naming it as ``original_doc_id``), so the duplicate check gets the
        earliest holder that is neither the row being processed nor a record
        of it, in one bounded query. Both are WHERE predicates on the same
        scan, so skipping a pointer row cannot truncate the search —
        ``LIMIT 1`` still returns the earliest row that survives them. The
        pointer half is ALSO honoured in Python over the ordered window
        (shared ``_row_points_at_as_duplicate`` reading), so the exclusion
        holds regardless of the engine's JSON-literal comparison semantics.
        """
        if not content_hash:
            return None

        params: dict[str, Any] = {
            "workspace": self.workspace,
            "content_hash": content_hash,
        }
        exclude_clause = ""
        if exclude_doc_id is not None:
            params["exclude_id"] = exclude_doc_id
            exclude_clause = (
                " AND id <> %(exclude_id)s AND NOT ("
                "COALESCE(JSON_EXTRACT(metadata, '$.is_duplicate'), false) "
                "AND COALESCE(JSON_EXTRACT(metadata, '$.original_doc_id'), '') "
                "= %(exclude_id)s)"
            )
        sql = (
            "SELECT * FROM LIGHTRAG_DOC_STATUS "
            f"WHERE workspace=%(workspace)s AND content_hash=%(content_hash)s{exclude_clause} "
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        result = await self.db.query(sql, params, multirows=True)
        for row in result or []:
            if self._row_points_at_as_duplicate(
                {"metadata": self._parse_json_field(row.get("metadata"), {})},
                exclude_doc_id,
            ):
                continue  # pointer row: not an independent holder, keep searching
            return str(row["id"]), self._parse_row(row)
        return None

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:
        """Typed, conflict-aware source resolution (see base contract).

        Fetches up to two PRIMARY rows for the canonical basename and maps
        0/1/≥2 → Absent/Unique/Conflict. When two are found the exact count
        is a ``COUNT(*)`` on the same predicate. Every aiomysql transport/
        server error propagates out of ``db.query`` (nothing here swallows
        it), so a returned ``SourceAbsent`` IS a confirmed absence.
        """
        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()

        sql = (
            "SELECT id, status, created_at, updated_at, file_path, track_id, "
            "metadata FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace=%(workspace)s AND file_path=%(file_path)s "
            f"AND {self._PRIMARY_PREDICATE} "
            "ORDER BY created_at ASC, id ASC LIMIT 2"
        )
        params = {"workspace": self.workspace, "file_path": canonical_source_key}
        rows = await self.db.query(sql, params, multirows=True)
        if not rows:
            return SourceAbsent()
        if len(rows) == 1:
            row = rows[0]
            return SourceUnique(
                doc_id=str(row["id"]),
                doc=self._scheduling_record_from_row(row, strict=True),
            )
        # ≥2 primary candidates: exact count is cheap on the same predicate.
        count_row = await self.db.query(
            "SELECT COUNT(*) AS c FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace=%(workspace)s AND file_path=%(file_path)s "
            f"AND {self._PRIMARY_PREDICATE}",
            params,
        )
        candidate_count = int(count_row["c"]) if count_row else None
        return SourceConflict(
            candidate_count=candidate_count,
            sample_doc_ids=tuple(sorted(str(r["id"]) for r in rows)),
        )

    async def get_status_counts(self) -> dict[str, int]:
        sql = "SELECT status, count(1) as count FROM LIGHTRAG_DOC_STATUS where workspace=%(workspace)s GROUP BY status"
        params = {"workspace": self.workspace}

        result = await self.db.query(sql, params, True)

        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        """Fetch documents matching any of the given statuses in a single query.

        Query errors always propagate; ``strict=True`` additionally raises on
        any row that cannot be converted (complete-or-raise scheduling
        contract, see base class).
        """
        if not statuses:
            return {}

        status_values = [s.value for s in statuses]
        placeholder, status_params = AnalyticDB.build_in_clause("status", status_values)
        sql = f"SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=%(workspace)s AND status IN ({placeholder})"
        params = {"workspace": self.workspace, **status_params}

        result = await self.db.query(sql, params, multirows=True)

        docs: dict[str, DocProcessingStatus] = {}
        for element in result or []:
            try:
                docs[element["id"]] = self._doc_status_from_row(element)
            except (KeyError, TypeError) as e:
                self._log_unusable_doc_row(element, e)
                if strict:
                    raise
                continue

        return docs

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and track_id=%(track_id)s"
        params = {"workspace": self.workspace, "track_id": track_id}

        result = await self.db.query(sql, params, True)

        docs_by_track_id = {}
        for element in result or []:
            try:
                docs_by_track_id[element["id"]] = self._doc_status_from_row(element)
            except (KeyError, TypeError) as e:
                # Relaxed skip-and-log, matching get_docs_by_statuses: one row
                # with a missing or renamed column (schema drift) must not
                # abort the listing for every sibling sharing the track_id.
                self._log_unusable_doc_row(element, e)
                continue

        return docs_by_track_id

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        # Whitelist validation for sort_field to prevent SQL injection
        allowed_sort_fields = {"created_at", "updated_at", "id", "file_path"}
        if sort_field not in allowed_sort_fields:
            sort_field = "updated_at"

        # Whitelist validation for sort_direction to prevent SQL injection
        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"
        else:
            sort_direction = sort_direction.lower()

        # Calculate offset
        offset = (page - 1) * page_size

        status_filter_values = self.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )

        # Build parameterized query components
        params: dict[str, Any] = {"workspace": self.workspace}

        # Build WHERE clause with parameterized query
        if status_filter_values is not None and len(status_filter_values) == 1:
            where_clause = "WHERE workspace=%(workspace)s AND status=%(status)s"
            params["status"] = next(iter(status_filter_values))
        elif status_filter_values is not None:
            placeholder, status_params = AnalyticDB.build_in_clause(
                "status", sorted(status_filter_values)
            )
            where_clause = (
                f"WHERE workspace=%(workspace)s AND status IN ({placeholder})"
            )
            params.update(status_params)
        else:
            where_clause = "WHERE workspace=%(workspace)s"

        # Build ORDER BY clause using validated whitelist values
        order_clause = f"ORDER BY {sort_field} {sort_direction.upper()}"

        # Query for total count
        count_sql = f"SELECT COUNT(*) as total FROM LIGHTRAG_DOC_STATUS {where_clause}"
        count_result = await self.db.query(count_sql, params)
        total_count = count_result["total"] if count_result else 0

        # Query for paginated data with parameterized LIMIT and OFFSET.
        # chunks_list is intentionally excluded from the column list:
        # DocStatusResponse does not expose it, so transferring the full JSON
        # array would be pure overhead.
        data_sql = f"""
                    SELECT id, content_summary, content_length, chunks_count,
                           status, file_path, track_id, metadata, error_msg,
                           content_hash, created_at, updated_at
                    FROM LIGHTRAG_DOC_STATUS
                    {where_clause}
                    {order_clause}
                    LIMIT %(limit)s OFFSET %(offset)s
                    """
        params["limit"] = page_size
        params["offset"] = offset

        result = await self.db.query(data_sql, params, True)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for element in result or []:
            metadata = self._parse_json_field(element.get("metadata"), {})
            if not isinstance(metadata, dict):
                metadata = {}
            doc_status = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=self._format_datetime(element["created_at"]),
                updated_at=self._format_datetime(element["updated_at"]),
                chunks_count=element["chunks_count"],
                file_path=element["file_path"],
                chunks_list=[],  # not fetched: unused by pagination response
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
                content_hash=element.get("content_hash"),
            )
            documents.append((element["id"], doc_status))

        return documents, total_count

    # ------------------------------------------------------------------
    # Memory-bounding scheduling API (Phase 1)
    # ------------------------------------------------------------------

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        """Bounded keyset page over LIGHTRAG_DOC_STATUS.

        **One branch per status, UNION ALL'd**, each carrying the same keyset
        predicate, the same ``(created_at ASC NULLS FIRST, id ASC)`` order and
        the same ``LIMIT``; the wrapper re-sorts and re-limits (see the PG
        implementation for why the single ``IN (...)`` shape degenerates into
        a full scan plus sort under LIMIT on large tables).

        NULL created_at (corrupt writes) sorts FIRST and the keyset
        comparison is bucket-aware — a plain ``(created_at, id) > (c, i)``
        row-value comparison evaluates to NULL for them, which would silently
        starve them out of every page after the first. They stay reachable:
        raised under strict, skipped (but consumed) under relaxed.

        Consumed-position contract: every predicate is part of the DB scan,
        so ``next_position`` is the key of the LAST RETURNED row and
        ``returned < limit`` proves exhaustion.

        ``strict=True``: any DB error or row-conversion failure raises
        without returning partial docs or a cursor.
        """
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if not statuses or position is CURSOR_END:
            return DocStatusPage(docs={}, next_position=CURSOR_END)

        params: dict[str, Any] = {"workspace": self.workspace}

        # The keyset predicate is identical in every branch; build it once.
        cursor_sql = ""
        if isinstance(position, CursorAfter):
            cur_created, cur_id = self._decode_cursor(position.opaque)
            if cur_created is None:
                # Cursor inside the NULL bucket (sorted first): continue
                # through the remaining NULL rows by id, then everything
                # with a real timestamp.
                params["cursor_id"] = cur_id
                cursor_sql = (
                    " AND ((created_at IS NULL AND id > %(cursor_id)s) "
                    "OR created_at IS NOT NULL)"
                )
            else:
                # Past the NULL bucket: only real-timestamp rows can follow.
                params["cursor_created_at"] = cur_created
                params["cursor_id"] = cur_id
                cursor_sql = (
                    " AND created_at IS NOT NULL AND "
                    "(created_at > %(cursor_created_at)s OR "
                    "(created_at = %(cursor_created_at)s AND id > %(cursor_id)s))"
                )

        order_by = "ORDER BY created_at ASC, id ASC"
        select_cols = (
            "SELECT id, status, created_at, updated_at, file_path, track_id, metadata"
        )
        branches: list[str] = []
        for i, status in enumerate(statuses):
            params[f"status_{i}"] = status.value
            branches.append(
                f"({select_cols} FROM LIGHTRAG_DOC_STATUS "
                f"WHERE workspace=%(workspace)s AND status=%(status_{i})s{cursor_sql} "
                f"{order_by} LIMIT %(limit)s)"
            )
        params["limit"] = limit
        if len(branches) == 1:
            sql = branches[0]
        else:
            sql = (
                f"SELECT * FROM ({' UNION ALL '.join(branches)}) u "
                f"{order_by} LIMIT %(limit)s"
            )

        # Any aiomysql error propagates out of db.query — strict pages never
        # commit a new cursor on failure.
        rows = await self.db.query(sql, params, multirows=True) or []

        docs: dict[str, DocSchedulingRecord] = {}
        for row in rows:
            record = self._scheduling_record_from_row(row, strict=strict)
            if record is None:
                continue  # relaxed skip is still consumed (see docstring)
            docs[record.id] = record

        if len(rows) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            next_position = CursorAfter(self._encode_cursor(rows[-1]))
        return DocStatusPage(docs=docs, next_position=next_position)

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        """Fail-closed status count: an accurate number or an exception.

        Unlike ``get_status_counts`` implementations that swallow errors,
        every DB failure propagates — admission control treats an error as
        "refuse", never as "capacity available".
        """
        if not statuses:
            return 0
        status_values = [s.value for s in statuses]
        placeholder, status_params = AnalyticDB.build_in_clause("status", status_values)
        sql = (
            "SELECT COUNT(*) AS cnt FROM LIGHTRAG_DOC_STATUS "
            f"WHERE workspace=%(workspace)s AND status IN ({placeholder})"
        )
        row = await self.db.query(sql, {"workspace": self.workspace, **status_params})
        if row is None or row.get("cnt") is None:
            raise StorageControlPlaneError(
                f"[{self.workspace}] COUNT query returned no row for "
                "count_docs_by_statuses; refusing to report a count"
            )
        return int(row["cnt"])

    def _prepare_doc_status_field_value(self, column: str, value: Any) -> Any:
        """Serialize one field for a row rewrite, matching the batch
        upsert's handling of JSON and TIMESTAMP columns."""
        if column in self._JSON_COLUMNS:
            return value if isinstance(value, str) else json.dumps(value)
        if column in self._DATETIME_COLUMNS:
            return self._to_mysql_datetime(
                value, f"[{self.workspace}] doc status {column}"
            )
        return value

    async def update_doc_status_fields(
        self,
        doc_id: str,
        fields: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        """Targeted field update implemented as read-modify-write REPLACE INTO.

        ``created_at`` is refused (immutable keyset sort key); unknown field
        names are refused too — column names must come from the whitelist.
        An unknown ``doc_id`` raises
        :class:`~lightrag.exceptions.StorageRecordNotFoundError` unless
        ``missing_ok=True`` (checked up front, off the SAME full-row read the
        merge needs).
        """
        if "created_at" in fields:
            raise ValueError(
                "created_at is an immutable scheduling sort key and cannot "
                "be changed via update_doc_status_fields"
            )
        unknown = set(fields) - self._UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"update_doc_status_fields received unknown doc_status "
                f"column(s): {sorted(unknown)}"
            )

        # Existence contract (honoured for empty and non-empty updates alike)
        # doubles as the merge source for the full-row REPLACE below.
        row = await self.db.query(
            "SELECT * FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace=%(workspace)s AND id=%(id)s",
            {"workspace": self.workspace, "id": doc_id},
            multirows=True,
        )
        row = (row or [None])[0]
        if row is None:
            if missing_ok:
                return
            raise StorageRecordNotFoundError(doc_id)
        if not fields:
            return

        overrides: dict[str, Any] = {}
        for column, value in fields.items():
            overrides[column] = self._prepare_doc_status_field_value(column, value)
        record = self._replace_record_from_row(row, doc_id, **overrides)
        await self.db.execute(self._REPLACE_SQL, record)

    # ------------------------------------------------------------------
    # Strict batch reads
    # ------------------------------------------------------------------

    async def get_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocSchedulingRecord]:
        """Batch strict read of scheduling records (see base contract).

        One indexed round-trip: a missing id is positively confirmed absent
        (simply not in the result set) and omitted. ``strict=True`` fails the
        WHOLE call rather than returning a partial mapping the feeder would
        mistake for stale ids. Results use the lightweight projection.
        """
        ids = [str(d) for d in doc_ids]
        if not ids:
            return {}
        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        sql = (
            "SELECT id, status, created_at, updated_at, file_path, track_id, "
            f"metadata FROM LIGHTRAG_DOC_STATUS WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        )
        rows = (
            await self.db.query(
                sql, {"workspace": self.workspace, **id_params}, multirows=True
            )
            or []
        )
        result: dict[str, DocSchedulingRecord] = {}
        for row in rows:
            record = self._scheduling_record_from_row(row, strict=strict)
            if record is None:
                continue  # relaxed skip of an unusable row (still consumed)
            result[record.id] = record
        return result

    async def get_full_docs_by_ids(
        self,
        doc_ids: Sequence[str],
        *,
        strict: bool = False,
    ) -> dict[str, DocProcessingStatus]:
        """Batch hydration of FULL DocProcessingStatus records (see base).

        Mirrors :meth:`get_docs_by_ids` but reuses the SAME raw ->
        :class:`DocProcessingStatus` normalisation as
        :meth:`get_docs_by_statuses` so every full field (content_summary /
        content_length / chunks_list / metadata / ...) is populated.
        """
        ids = [str(d) for d in doc_ids]
        if not ids:
            return {}
        placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
        sql = (
            "SELECT * FROM LIGHTRAG_DOC_STATUS "
            f"WHERE workspace=%(workspace)s AND id IN ({placeholder})"
        )
        rows = (
            await self.db.query(
                sql, {"workspace": self.workspace, **id_params}, multirows=True
            )
            or []
        )
        result: dict[str, DocProcessingStatus] = {}
        for element in rows:
            try:
                result[element["id"]] = self._doc_status_from_row(element)
            except (KeyError, TypeError) as e:
                self._log_unusable_doc_row(element, e)
                if strict:
                    raise
                continue
        return result

    # ------------------------------------------------------------------
    # Source-conflict listing and explicit CAS repair
    # ------------------------------------------------------------------

    @staticmethod
    def _conflict_fingerprint(sorted_doc_ids: list[str]) -> str:
        """Deterministic digest over candidate doc IDs in stable sort order."""
        digest = hashlib.sha256()
        for doc_id in sorted_doc_ids:
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    @staticmethod
    def _decode_conflict_cursor(opaque: str) -> str:
        try:
            key = json.loads(opaque)
            if not isinstance(key, str):
                raise TypeError("conflict cursor must be a string")
        except (ValueError, TypeError) as e:
            raise StorageControlPlaneError(
                f"Malformed source-conflict cursor for ADBDocStatusStorage: {e}"
            ) from e
        return key

    async def list_source_conflicts_page(
        self,
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
    ) -> SourceConflictPage:
        """Page canonical source keys with >1 primary candidate (see base).

        ``GROUP BY file_path HAVING COUNT(*) >= 2`` over PRIMARY rows only,
        keyset-ordered by the canonical key so pages are stable and bounded.
        Each key's bounded sample is fetched with its own ``ORDER BY id LIMIT
        _CONFLICT_SAMPLE_CAP`` query, so no group ever materializes its whole
        candidate set.
        """
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if position is CURSOR_END:
            return SourceConflictPage(conflicts=(), next_position=CURSOR_END)

        params: dict[str, Any] = {"workspace": self.workspace}
        sql = (
            "SELECT file_path, COUNT(*) AS c FROM LIGHTRAG_DOC_STATUS "
            f"WHERE workspace=%(workspace)s AND {self._PRIMARY_PREDICATE} "
            "AND file_path IS NOT NULL "
            "AND file_path NOT IN ('', 'unknown_source', 'no-file-path')"
        )
        if isinstance(position, CursorAfter):
            params["cursor_key"] = self._decode_conflict_cursor(position.opaque)
            sql += " AND file_path > %(cursor_key)s"
        params["limit"] = limit
        sql += (
            " GROUP BY file_path HAVING COUNT(*) >= 2 "
            "ORDER BY file_path ASC LIMIT %(limit)s"
        )
        rows = await self.db.query(sql, params, multirows=True) or []

        conflicts: list[SourceConflictSummary] = []
        for row in rows:
            key = row["file_path"]
            sample = (
                await self.db.query(
                    "SELECT id FROM LIGHTRAG_DOC_STATUS "
                    "WHERE workspace=%(workspace)s AND file_path=%(file_path)s "
                    f"AND {self._PRIMARY_PREDICATE} "
                    "ORDER BY id ASC LIMIT %(limit)s",
                    {
                        "workspace": self.workspace,
                        "file_path": key,
                        "limit": self._CONFLICT_SAMPLE_CAP,
                    },
                    multirows=True,
                )
                or []
            )
            conflicts.append(
                SourceConflictSummary(
                    canonical_source_key=key,
                    candidate_count=int(row["c"]),
                    sample_doc_ids=tuple(str(r["id"]) for r in sample),
                )
            )

        if len(rows) < limit:
            next_position: CursorPosition = CURSOR_END
        else:
            next_position = CursorAfter(
                json.dumps(rows[-1]["file_path"], ensure_ascii=False)
            )
        return SourceConflictPage(
            conflicts=tuple(conflicts), next_position=next_position
        )

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> SourceConflictRepairResult:
        """Demote all-but-one primary to duplicate, CAS-guarded (see base).

        The candidate set is re-read, the count/fingerprint recomputed and
        compared against the operator-echoed expectation; on commit the
        demotions land atomically via ``execute_transaction`` (one REPLACE
        INTO per demoted doc). Losing candidates get
        ``metadata.is_duplicate=true`` + ``original_doc_id=primary_doc_id``;
        content is never deleted. ``primary_doc_id`` not in the current
        candidate set raises ValueError.
        """
        sql = (
            "SELECT id FROM LIGHTRAG_DOC_STATUS "
            "WHERE workspace=%(workspace)s AND file_path=%(file_path)s "
            f"AND {self._PRIMARY_PREDICATE} "
            "ORDER BY id ASC"
        )
        rows = (
            await self.db.query(
                sql,
                {
                    "workspace": self.workspace,
                    "file_path": canonical_source_key,
                },
                multirows=True,
            )
            or []
        )
        candidates = sorted(str(r["id"]) for r in rows)
        count = len(candidates)
        fingerprint = self._conflict_fingerprint(candidates)
        if primary_doc_id not in candidates:
            raise ValueError(
                f"primary_doc_id {primary_doc_id!r} is not a current "
                f"primary candidate for {canonical_source_key!r}"
            )
        demoted = [d for d in candidates if d != primary_doc_id]
        result_kwargs = {
            "canonical_source_key": canonical_source_key,
            "primary_doc_id": primary_doc_id,
            "candidate_count": count,
            "fingerprint": fingerprint,
            "demoted_sample_doc_ids": tuple(demoted[: self._CONFLICT_SAMPLE_CAP]),
        }
        if dry_run:
            return SourceConflictRepairResult(committed=False, **result_kwargs)
        if (
            count != expected_candidate_count
            or fingerprint != expected_candidate_fingerprint
        ):
            raise SourceConflictRepairCASError(
                f"[{self.workspace}] source-conflict repair CAS failed for "
                f"{canonical_source_key!r}: candidate set changed "
                f"(count {count} vs {expected_candidate_count})"
            )
        if demoted:
            # REPLACE INTO rewrites the whole row, so one bounded batch read
            # fetches the demoted rows in full; is_duplicate/original_doc_id
            # are merged into each row's existing metadata and every other
            # column (created_at included) is carried over as read.
            placeholder, id_params = AnalyticDB.build_in_clause("id", demoted)
            rows = (
                await self.db.query(
                    "SELECT * FROM LIGHTRAG_DOC_STATUS "
                    f"WHERE workspace=%(workspace)s AND id IN ({placeholder})",
                    {"workspace": self.workspace, **id_params},
                    multirows=True,
                )
                or []
            )
            rows_by_id = {str(r.get("id")): r for r in rows}
            statements: list[tuple[str, dict[str, Any]]] = []
            for doc_id in demoted:
                row = rows_by_id.get(doc_id) or {}
                metadata = self._parse_json_field(row.get("metadata"), {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["is_duplicate"] = True
                metadata["original_doc_id"] = primary_doc_id
                record = self._replace_record_from_row(
                    row, doc_id, metadata=json.dumps(metadata)
                )
                statements.append((self._REPLACE_SQL, record))
            await self.db.execute_transaction(statements)
        return SourceConflictRepairResult(committed=True, **result_kwargs)

    async def get_all_status_counts(self) -> dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def index_done_callback(self) -> None:
        pass

    async def is_empty(self) -> bool:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}"
            )
            return True

        sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace=%(workspace)s LIMIT 1) as has_data"
        try:
            result = await self.db.query(sql, {"workspace": self.workspace})

            return not result.get("has_data", False) if result else True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        # Chunk the id list so each IN clause stays bounded (a non-positive
        # cap disables chunking). Multiple chunks run in ONE transaction via
        # execute_transaction, preserving the single-statement all-or-nothing
        # behaviour.
        chunk = (
            self._max_delete_records_per_batch
            if self._max_delete_records_per_batch > 0
            else len(ids)
        )

        try:
            if len(ids) <= chunk:
                placeholder, id_params = AnalyticDB.build_in_clause("id", ids)
                delete_sql = (
                    f"DELETE FROM {table_name} "
                    f"WHERE workspace=%(workspace)s AND id IN ({placeholder})"
                )
                await self.db.execute(
                    delete_sql, {"workspace": self.workspace, **id_params}
                )
            else:
                logger.info(
                    f"[{self.workspace}] {self.namespace} delete: {len(ids)} ids "
                    f"split into chunks (chunk={chunk})"
                )
                statements: list[tuple[str, dict[str, Any]]] = []
                for i in range(0, len(ids), chunk):
                    placeholder, id_params = AnalyticDB.build_in_clause(
                        "id", ids[i : i + chunk]
                    )
                    delete_sql = (
                        f"DELETE FROM {table_name} "
                        f"WHERE workspace=%(workspace)s AND id IN ({placeholder})"
                    )
                    statements.append(
                        (delete_sql, {"workspace": self.workspace, **id_params})
                    )
                await self.db.execute_transaction(statements)
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status.

        created_at is bound from the payload — it is the immutable scheduling
        sort key behind ORDER BY created_at keyset pages, and a REPLACE INTO
        omitting it would silently reset it to the DDL default on every
        re-enqueue, corrupting FIFO ordering. updated_at is server-stamped
        via CURRENT_TIMESTAMP. Both are normalized to the naive-UTC form the
        MySQL TIMESTAMP columns accept (tz-aware ISO input rejected
        otherwise).

        NOTE: unlike PGDocStatusStorage's COALESCE write-once guard, REPLACE
        INTO cannot reference the prior row, so callers must re-supply a
        persisted content_hash whenever they re-upsert an existing doc.
        """
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        datas: list[dict[str, Any]] = []
        for k, v in data.items():
            # chunks_count, chunks_list, track_id, metadata, error_msg,
            # content_hash and created_at are optional
            record = {
                "workspace": self.workspace,
                "id": k,
                "content_summary": v["content_summary"],
                "content_length": v["content_length"],
                "chunks_count": v.get("chunks_count", -1),
                "status": v["status"],
                "file_path": v["file_path"],
                "chunks_list": json.dumps(v.get("chunks_list", [])),
                "track_id": v.get("track_id"),
                "metadata": json.dumps(v.get("metadata", {})),
                "error_msg": v.get("error_msg"),
                "content_hash": v.get("content_hash"),
                "created_at": self._to_mysql_datetime(
                    v.get("created_at"), f"[{self.workspace}] doc {k} created_at"
                ),
            }
            datas.append(record)
        for offset in range(0, len(datas), self._max_batch_size):
            await self.db.execute(
                self._REPLACE_SQL, datas[offset : offset + self._max_batch_size]
            )

    async def drop(self) -> dict[str, str]:
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specify_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.workspace})

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBGraphStorage(BaseGraphStorage):
    """AnalyticDB MySQL Graph Storage implementation.

    Stores graph data (nodes and edges) in AnalyticDB MySQL using relational tables.
    Nodes are stored in LIGHTRAG_GRAPH_NODES table, edges in LIGHTRAG_GRAPH_EDGES table.

    Edge storage / undirected semantics:
        Edges are stored in canonical order source_id = min(a, b),
        target_id = max(a, b) via Python min/max. All write paths normalise
        before REPLACE, so upsert_edge(A, B) and upsert_edge(B, A) map to one
        row.

    Contract behaviour (parity with PGTableGraphStorage):
      - Edge upsert CREATES missing endpoint nodes with a minimal
        {"entity_id": id} payload; node upsert REQUIRES entity_id
        (ValueError if absent) and MERGES properties so omitted keys survive.
      - get_knowledge_graph uses a level-capped BFS whose retained set is
        bounded by max_nodes (+1 overflow probe), ranked degree DESC with
        label-ascending tie-break, seed pinned first.
      - search_labels scores and truncates in SQL (CASE scoring, LIKE with
        escaped wildcards); get_popular_labels ranks the WHOLE node set via
        LEFT JOIN so isolated (degree-0) entities still rank.
    """

    db: AnalyticDB | None = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _gp(self, **extras: Any) -> dict[str, Any]:
        """Param helper: workspace + graph namespace plus named-bind extras.

        The graph tables are workspace-partitioned (the DDL carries no
        namespace column); the namespace rides along for contract parity
        with the PG backends — an unreferenced key is ignored by the
        driver's named-bind formatting.
        """
        return {
            "workspace": self.workspace,
            "namespace": self.namespace,
            **extras,
        }

    @staticmethod
    def _escape_like(value: str) -> str:
        """Escape LIKE wildcards so user input matches literally."""
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _json_loads(value: Any) -> dict[str, Any]:
        """Parse a JSON properties cell; malformed strings RAISE, non-dict
        JSON and None normalize to {} (row presence, not payload truthiness,
        decides membership everywhere this is used)."""
        if isinstance(value, str):
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        return dict(value or {})

    @staticmethod
    def _node_props(node_id: str, properties: Any) -> dict[str, Any]:
        props = ADBGraphStorage._json_loads(properties)
        props["entity_id"] = node_id
        return props

    @staticmethod
    def _node_output(node_id: str, properties: Any) -> dict[str, Any]:
        props = ADBGraphStorage._node_props(node_id, properties)
        props["id"] = node_id
        return props

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Initialize database connection and create graph tables if not exist."""
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                self.workspace = self.db.workspace
            elif not hasattr(self, "workspace") or not self.workspace:
                self.workspace = "default"

            # Create graph tables if not exist
            for k, v in GRAPH_TABLES.items():
                try:
                    result = await self.db.query(
                        "SELECT 1 FROM information_schema.kepler_meta_tables "
                        "where table_schema=%(db)s and table_name=lower(%(table)s)",
                        {"db": self.db.db_config["db"], "table": k},
                    )
                    if result is None:
                        logger.info(
                            f"AnalyticDB MySQL, Try Creating graph table {k} in database"
                        )
                        await self.db.execute(v["ddl"])
                except Exception as e:
                    logger.error(
                        f"AnalyticDB MySQL, Failed to create graph table {k} in database, Got: {e}"
                    )
                    raise e

    async def finalize(self):
        """Close database connection pool."""
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    async def index_done_callback(self) -> None:
        """No-op callback for index completion."""
        pass

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        sql = "SELECT 1 FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s AND node_id=%(node_id)s LIMIT 1"
        result = await self.db.query(sql, self._gp(node_id=node_id))
        return result is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes (canonical order)."""
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        sql = (
            "SELECT 1 FROM LIGHTRAG_GRAPH_EDGES "
            "WHERE workspace=%(workspace)s AND source_id=%(src)s AND target_id=%(tgt)s "
            "LIMIT 1"
        )
        result = await self.db.query(sql, self._gp(src=src, tgt=tgt))
        return result is not None

    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node."""
        sql = """
            SELECT COUNT(*) as degree FROM (
                SELECT target_id FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s AND source_id=%(node_id)s
                UNION ALL
                SELECT source_id FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s AND target_id=%(node_id)s
            ) as edges
        """
        result = await self.db.query(sql, self._gp(node_id=node_id))
        return result["degree"] if result else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes)."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning node properties with entity_id forced."""
        sql = "SELECT properties FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s AND node_id=%(node_id)s"
        result = await self.db.query(sql, self._gp(node_id=node_id))
        return self._node_props(node_id, result["properties"]) if result else None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes (canonical order)."""
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)
        sql = (
            "SELECT properties FROM LIGHTRAG_GRAPH_EDGES "
            "WHERE workspace=%(workspace)s AND source_id=%(src)s AND target_id=%(tgt)s"
        )
        result = await self.db.query(sql, self._gp(src=src, tgt=tgt))
        return self._json_loads(result["properties"]) if result else None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node, sorted by counterpart id."""
        if not await self.has_node(source_node_id):
            return None

        sql = """
            SELECT source_id, target_id FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace=%(workspace)s AND (source_id=%(node_id)s OR target_id=%(node_id)s)
        """
        results = await self.db.query(
            sql, self._gp(node_id=source_node_id), multirows=True
        )
        if not results:
            return []

        # Normalize: emit (source_node_id, counterpart) pairs
        edges = [
            (
                source_node_id,
                row["target_id"]
                if row["source_id"] == source_node_id
                else row["source_id"],
            )
            for row in results
        ]
        edges.sort(key=lambda e: e[1])
        return edges

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Batch-fetch node properties. Row presence decides membership: a
        node holding NULL or '{}' properties still exists."""
        if not node_ids:
            return {}

        placeholder, id_params = AnalyticDB.build_in_clause("node_id", node_ids)
        sql = f"""
            SELECT node_id, properties FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace=%(workspace)s AND node_id IN ({placeholder})
        """
        results = await self.db.query(sql, self._gp(**id_params), multirows=True)
        if not results:
            return {}

        return {
            row["node_id"]: self._node_props(row["node_id"], row.get("properties"))
            for row in results
        }

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Batch-fetch edge properties for (src, tgt) request pairs.

        One double-IN query over canonical endpoints (never per-pair OR
        expansion); rows outside the requested canonical pair set are
        dropped, and results are keyed back to the CALLER's pair order.
        """
        if not pairs:
            return {}

        canonical = [(min(p["src"], p["tgt"]), max(p["src"], p["tgt"])) for p in pairs]
        canonical_set = set(canonical)
        srcs = sorted({c[0] for c in canonical})
        tgts = sorted({c[1] for c in canonical})

        src_ph, src_params = AnalyticDB.build_in_clause("source_id", srcs)
        tgt_ph, tgt_params = AnalyticDB.build_in_clause("target_id", tgts)
        sql = f"""
            SELECT source_id, target_id, properties FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace=%(workspace)s AND source_id IN ({src_ph})
            AND target_id IN ({tgt_ph})
        """
        results = await self.db.query(
            sql, self._gp(**src_params, **tgt_params), multirows=True
        )
        if not results:
            return {}

        canonical_props: dict[tuple[str, str], dict] = {}
        for row in results:
            key = (row["source_id"], row["target_id"])
            if key in canonical_set:
                canonical_props[key] = self._json_loads(row.get("properties"))

        return {
            (p["src"], p["tgt"]): canonical_props[c]
            for p, c in zip(pairs, canonical)
            if c in canonical_props
        }

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Check existence of multiple nodes in a single batch call."""
        if not node_ids:
            return set()

        placeholder, id_params = AnalyticDB.build_in_clause("node_id", node_ids)
        sql = f"""
            SELECT node_id FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s
            AND node_id IN ({placeholder})
        """
        result = await self.db.query(sql, self._gp(**id_params), multirows=True)
        return {row["node_id"] for row in result} if result else set()

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Calculate the degree (number of connected edges) for multiple nodes in batch."""
        if not node_ids:
            return {}

        placeholder, id_params = AnalyticDB.build_in_clause("node_id", node_ids)
        sql = f"""
            SELECT node_id, COUNT(*) as degree FROM (
                SELECT source_id as node_id FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s AND source_id IN ({placeholder})
                UNION ALL
                SELECT target_id as node_id FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s AND target_id IN ({placeholder})
            ) as all_nodes
            GROUP BY node_id
        """
        results = await self.db.query(sql, self._gp(**id_params), multirows=True)

        degrees = {row["node_id"]: row["degree"] for row in results} if results else {}
        return {node_id: degrees.get(node_id, 0) for node_id in node_ids}

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.
        """
        if not edge_pairs:
            return {}

        # Collect all unique nodes
        unique_nodes: set[str] = set()
        for src, tgt in edge_pairs:
            unique_nodes.add(src)
            unique_nodes.add(tgt)

        # Get all node degrees in one batch
        degrees = await self.node_degrees_batch(list(unique_nodes))

        # Calculate edge degrees
        edge_degrees: dict[tuple[str, str], int] = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)

        return edge_degrees

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes in one query."""
        if not node_ids:
            return {}

        placeholder, id_params = AnalyticDB.build_in_clause(
            "node_id", list(dict.fromkeys(node_ids))
        )
        sql = f"""
            SELECT source_id, target_id FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace=%(workspace)s AND (source_id IN ({placeholder}) OR target_id IN ({placeholder}))
        """
        results = await self.db.query(sql, self._gp(**id_params), multirows=True)

        result = {node_id: [] for node_id in node_ids}
        if not results:
            return result

        for row in results:
            src = row["source_id"]
            tgt = row["target_id"]

            if src in result:
                result[src].append((src, tgt))
            if tgt in result and tgt != src:
                # self-loop (src == tgt) is one edge, not two — match
                # get_node_edges() and NetworkX.
                result[tgt].append((tgt, src))

        for edges in result.values():
            edges.sort(key=lambda edge: edge[1])
        return result

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Requires ``entity_id`` (PGGraphStorage parity), forces it to
        node_id, and MERGES with the stored properties so omitted keys
        survive a partial update.
        """
        if "entity_id" not in node_data:
            raise ValueError(
                "AnalyticDB: node properties must contain an 'entity_id' field"
            )
        existing = await self.get_node(node_id)
        merged = {**(existing or {}), **node_data, "entity_id": node_id}

        sql = """
            REPLACE INTO LIGHTRAG_GRAPH_NODES (workspace, node_id, properties, update_time)
            VALUES (%(workspace)s, %(node_id)s, %(properties)s, CURRENT_TIMESTAMP)
        """
        await self.db.execute(
            sql,
            self._gp(node_id=node_id, properties=json.dumps(merged)),
        )

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Insert or update multiple nodes in a single batch call.

        Existing nodes are read in ONE batch query (never N serial
        get_node round trips); duplicate node_ids dedupe last-write-wins.
        """
        if not nodes:
            return

        deduped: dict[str, dict[str, str]] = {}
        for node_id, node_data in nodes:
            if "entity_id" not in node_data:
                raise ValueError(
                    "AnalyticDB: node properties must contain an 'entity_id' field"
                )
            deduped[node_id] = node_data

        existing = await self.get_nodes_batch(list(deduped))

        sql = """
            REPLACE INTO LIGHTRAG_GRAPH_NODES (workspace, node_id, properties, update_time)
            VALUES (%(workspace)s, %(node_id)s, %(properties)s, CURRENT_TIMESTAMP)
        """
        datas = []
        for node_id in sorted(deduped):
            # Merge (not replace), same as upsert_node — omitted keys survive.
            merged = {
                **(existing.get(node_id) or {}),
                **deduped[node_id],
                "entity_id": node_id,
            }
            datas.append(
                {
                    "workspace": self.workspace,
                    "node_id": node_id,
                    "properties": json.dumps(merged),
                }
            )
        for offset in range(0, len(datas), self._max_batch_size):
            await self.db.execute(sql, datas[offset : offset + self._max_batch_size])

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Missing endpoints are created with a minimal {"entity_id": id}
        payload (NetworkX add_edge semantics) before the edge write.
        """
        src = min(source_node_id, target_node_id)
        tgt = max(source_node_id, target_node_id)

        existing = await self.has_nodes_batch([src, tgt])
        missing = [nid for nid in (src, tgt) if nid not in existing]
        for nid in missing:
            await self.upsert_node(nid, {"entity_id": nid})

        sql = """
            REPLACE INTO LIGHTRAG_GRAPH_EDGES (workspace, source_id, target_id, properties, update_time)
            VALUES (%(workspace)s, %(src)s, %(tgt)s, %(properties)s, CURRENT_TIMESTAMP)
        """
        await self.db.execute(
            sql,
            self._gp(src=src, tgt=tgt, properties=json.dumps(edge_data)),
        )

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Insert or update multiple edges in a single batch call.

        Canonical-order dedup (last write wins); missing endpoints are
        created in ONE batched node write before the edge REPLACE batch.
        """
        if not edges:
            return

        deduped: dict[tuple[str, str], dict[str, str]] = {}
        for src, tgt, edge_data in edges:
            key = (min(src, tgt), max(src, tgt))
            deduped[key] = edge_data

        endpoints = sorted({nid for key in deduped for nid in key})
        existing = await self.has_nodes_batch(endpoints)
        missing = [nid for nid in endpoints if nid not in existing]
        if missing:
            await self.upsert_nodes_batch(
                [(nid, {"entity_id": nid}) for nid in missing]
            )

        sql = """
            REPLACE INTO LIGHTRAG_GRAPH_EDGES (workspace, source_id, target_id, properties, update_time)
            VALUES (%(workspace)s, %(src)s, %(tgt)s, %(properties)s, CURRENT_TIMESTAMP)
        """
        datas = [
            {
                "workspace": self.workspace,
                "src": src,
                "tgt": tgt,
                "properties": json.dumps(edge_data),
            }
            for (src, tgt), edge_data in sorted(deduped.items())
        ]
        for offset in range(0, len(datas), self._max_batch_size):
            await self.db.execute(sql, datas[offset : offset + self._max_batch_size])

    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph (including all its edges).

        ADB has no FK CASCADE, so the edge and node deletes share ONE
        transaction via execute_transaction (a bare START TRANSACTION over
        pooled connections would lose the transaction boundary).
        """
        statements = [
            (
                "DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s "
                + "AND (source_id=%(node_id)s OR target_id=%(node_id)s)",
                self._gp(node_id=node_id),
            ),
            (
                "DELETE FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s "
                + "AND node_id=%(node_id)s",
                self._gp(node_id=node_id),
            ),
        ]
        await self.db.execute_transaction(statements)

    async def remove_nodes(self, nodes: list[str]) -> None:
        """Delete multiple nodes atomically (edges first, ONE transaction)."""
        if not nodes:
            return

        placeholder, id_params = AnalyticDB.build_in_clause("node_id", nodes)
        statements = [
            (
                "DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s "
                + f"AND (source_id IN ({placeholder}) OR target_id IN ({placeholder}))",
                self._gp(**id_params),
            ),
            (
                "DELETE FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s "
                + f"AND node_id IN ({placeholder})",
                self._gp(**id_params),
            ),
        ]
        await self.db.execute_transaction(statements)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Delete multiple edges with one parameterised statement."""
        if not edges:
            return

        conditions = []
        params = self._gp()
        for i, (s, t) in enumerate(edges):
            src = min(s, t)
            tgt = max(s, t)
            conditions.append(f"(source_id=%(src_{i})s AND target_id=%(tgt_{i})s)")
            params[f"src_{i}"] = src
            params[f"tgt_{i}"] = tgt

        delete_sql = (
            "DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s AND ("
            + " OR ".join(conditions)
            + ")"
        )
        await self.db.execute(delete_sql, params)

    async def get_all_labels(self) -> list[str]:
        """Get all labels(entity names) in the graph, sorted alphabetically."""
        sql = "SELECT DISTINCT node_id FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s"
        results = await self.db.query(sql, self._gp(), multirows=True)
        return sorted(row["node_id"] for row in results) if results else []

    async def iter_labels(self, batch_size: int) -> AsyncIterator[list[str]]:
        """Yield all graph labels in bounded keyset batches.

        Whole-graph tools use this instead of ``get_all_labels`` so their
        client-side memory does not grow with the graph; batches stream in
        the same node_id order as ``get_all_labels``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        after = ""
        while True:
            sql = (
                "SELECT node_id FROM LIGHTRAG_GRAPH_NODES "
                "WHERE workspace=%(workspace)s AND node_id > %(after)s "
                "ORDER BY node_id ASC LIMIT %(batch_size)s"
            )
            rows = await self.db.query(
                sql, self._gp(after=after, batch_size=batch_size), multirows=True
            )
            if not rows:
                return
            batch = [row["node_id"] for row in rows]
            yield batch
            after = batch[-1]

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph (BFS) or the whole graph (``*``).

        Retained-set selection ranks degree DESC with label-ascending
        tie-break (base contract), seed pinned first; one overflow node
        past max_nodes is fetched to detect truncation but never reaches
        the caller.
        """
        cap = self.global_config.get("max_graph_nodes", 1000)
        node_budget: int = cap if max_nodes is None else min(max_nodes, cap)

        if node_label == "*":
            return await self._get_full_knowledge_graph(node_budget)
        return await self._bfs_knowledge_graph(node_label, max_depth, node_budget)

    async def _fetch_edges_among(self, node_ids: set[str]) -> list[dict[str, Any]]:
        """Fetch edges with BOTH endpoints in ``node_ids``.

        One double-IN query — never the O(N^2) per-pair enumeration.
        Fewer than two ids cannot hold an edge, so short-circuit.
        """
        if len(node_ids) < 2:
            return []

        ids = sorted(node_ids)
        placeholder, id_params = AnalyticDB.build_in_clause("node_id", ids)
        sql = f"""
            SELECT source_id, target_id, properties FROM LIGHTRAG_GRAPH_EDGES
            WHERE workspace=%(workspace)s AND source_id IN ({placeholder})
            AND target_id IN ({placeholder})
        """
        results = await self.db.query(sql, self._gp(**id_params), multirows=True)
        return results or []

    def _kg_edges_from_rows(self, edge_rows: list[dict[str, Any]]) -> list:
        edges = []
        for row in sorted(edge_rows, key=lambda r: (r["source_id"], r["target_id"])):
            edges.append(
                KnowledgeGraphEdge(
                    id=f"{row['source_id']}-{row['target_id']}",
                    type="DIRECTED",
                    source=row["source_id"],
                    target=row["target_id"],
                    properties=self._json_loads(row.get("properties")),
                )
            )
        return edges

    async def _get_full_knowledge_graph(self, node_budget: int) -> KnowledgeGraph:
        """Whole-graph view: popular labels, then ONE batched node read and
        ONE double-IN edge fetch (no per-pair enumeration)."""
        labels = await self.get_popular_labels(node_budget + 1)
        if not labels:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        is_truncated = len(labels) > node_budget
        labels = labels[:node_budget]

        nodes_data = await self.get_nodes_batch(labels)
        nodes = [
            KnowledgeGraphNode(
                id=node_id,
                labels=[node_id],
                properties=nodes_data.get(node_id, {"entity_id": node_id}),
            )
            for node_id in labels
        ]

        edge_rows = await self._fetch_edges_among(set(labels))
        edges = self._kg_edges_from_rows(edge_rows)

        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)

    async def _bfs_knowledge_graph(
        self, node_label: str, max_depth: int, node_budget: int
    ) -> KnowledgeGraph:
        """Level-capped BFS from the seed.

        Each level admits at most ``remaining + 1`` neighbours (the +1
        overflow probe distinguishes "exactly full" from "truncated")
        ranked degree DESC, label ASC. Every level costs exactly TWO
        round trips (one batched neighbour read for the whole frontier,
        one node_degrees_batch) no matter how wide the frontier grows.
        Node properties are backfilled in ONE get_nodes_batch read AFTER
        the traversal — never placeholder "{}" payloads.
        """
        if not await self.has_node(node_label):
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        # (node_id, depth, degree); the seed is pinned at depth 0.
        collected: dict[str, tuple[int, int]] = {node_label: (0, 0)}
        frontier: list[str] = [node_label]
        depth = 0
        while frontier and depth < max_depth and len(collected) <= node_budget:
            depth += 1
            # ONE batched neighbour read for the whole frontier — never a
            # per-node round trip. Frontier nodes were admitted from edge
            # rows (or are the entry-verified seed), so no existence guard
            # is needed. get_nodes_edges_batch keys each pair as
            # (frontier_node, counterpart).
            neighbor_set: set[str] = set()
            edges_map = await self.get_nodes_edges_batch(frontier)
            for pairs in edges_map.values():
                for _, counterpart in pairs:
                    if counterpart not in collected:
                        neighbor_set.add(counterpart)
            if not neighbor_set:
                break

            degrees = await self.node_degrees_batch(sorted(neighbor_set))
            # degree DESC, label ASC tie-break (base contract), then cap:
            # one past the budget so truncation stays detectable.
            level_cap = node_budget - len(collected) + 1
            ranked = sorted(neighbor_set, key=lambda nid: (-degrees.get(nid, 0), nid))
            frontier = []
            for nid in ranked[:level_cap]:
                collected[nid] = (depth, degrees.get(nid, 0))
                frontier.append(nid)

        # Sort before truncation: seed pinned first, shallower depth first,
        # higher degree within a level, label as the final tie-break.
        ordered = sorted(
            collected,
            key=lambda nid: (
                nid != node_label,
                collected[nid][0],
                -collected[nid][1],
                nid,
            ),
        )
        is_truncated = len(ordered) > node_budget
        ordered = ordered[:node_budget]

        # One batched read backfills the REAL stored properties for every
        # retained node (regression: non-seed nodes used to carry "{}").
        nodes_data = await self.get_nodes_batch(ordered)
        nodes = [
            KnowledgeGraphNode(
                id=node_id,
                labels=[node_id],
                properties=nodes_data.get(node_id, {"entity_id": node_id}),
            )
            for node_id in ordered
        ]

        edge_rows = await self._fetch_edges_among(set(ordered))
        edges = self._kg_edges_from_rows(edge_rows)

        return KnowledgeGraph(nodes=nodes, edges=edges, is_truncated=is_truncated)

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes; row existence decides, properties may be empty."""
        sql = "SELECT node_id, properties FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s"
        results = await self.db.query(sql, self._gp(), multirows=True)
        if not results:
            return []
        nodes = [
            self._node_output(row["node_id"], row.get("properties")) for row in results
        ]
        return sorted(nodes, key=lambda props: props["id"])

    async def get_all_edges(self) -> list[dict]:
        """Get all edges; properties merged with endpoint keys."""
        sql = (
            "SELECT source_id, target_id, properties FROM LIGHTRAG_GRAPH_EDGES "
            "WHERE workspace=%(workspace)s"
        )
        results = await self.db.query(sql, self._gp(), multirows=True)
        if not results:
            return []
        edges = [
            {
                **self._json_loads(row.get("properties")),
                "source": row["source_id"],
                "target": row["target_id"],
            }
            for row in results
        ]
        return sorted(edges, key=lambda e: (e["source"], e["target"]))

    async def iter_edges(self, batch_size: int) -> AsyncIterator[list[dict]]:
        """Yield all graph edges in bounded keyset batches.

        Each edge has the same dict shape as ``get_all_edges`` — properties
        merged with the ``source``/``target`` endpoint keys — batched on the
        composite (source_id, target_id) keyset.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        after_src = ""
        after_tgt = ""
        while True:
            sql = (
                "SELECT source_id, target_id, properties FROM LIGHTRAG_GRAPH_EDGES "
                "WHERE workspace=%(workspace)s AND (source_id > %(after_src)s "
                "OR (source_id = %(after_src)s AND target_id > %(after_tgt)s)) "
                "ORDER BY source_id ASC, target_id ASC LIMIT %(batch_size)s"
            )
            rows = await self.db.query(
                sql,
                self._gp(
                    after_src=after_src, after_tgt=after_tgt, batch_size=batch_size
                ),
                multirows=True,
            )
            if not rows:
                return
            batch = [
                {
                    **self._json_loads(row.get("properties")),
                    "source": row["source_id"],
                    "target": row["target_id"],
                }
                for row in rows
            ]
            yield batch
            after_src = batch[-1]["source"]
            after_tgt = batch[-1]["target"]

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels(entity names) by node degree (most connected entities).

        Ranks ALL nodes by degree, including isolated (degree 0) nodes, via a
        LEFT JOIN over the node table: aggregating the edge table alone would
        silently drop isolated entities. Self-loops count twice (no
        source_id <> target_id guard), consistent with node_degree.
        """
        sql = """
            SELECT n.node_id AS node_id, COALESCE(d.degree, 0) AS degree
            FROM LIGHTRAG_GRAPH_NODES n
            LEFT JOIN (
                SELECT node_id, COUNT(*) AS degree FROM (
                    SELECT source_id AS node_id FROM LIGHTRAG_GRAPH_EDGES
                    WHERE workspace=%(workspace)s
                    UNION ALL
                    SELECT target_id AS node_id FROM LIGHTRAG_GRAPH_EDGES
                    WHERE workspace=%(workspace)s
                ) sub
                GROUP BY node_id
            ) d ON d.node_id = n.node_id
            WHERE n.workspace=%(workspace)s
            ORDER BY degree DESC, node_id ASC
            LIMIT %(limit)s
        """
        results = await self.db.query(sql, self._gp(limit=limit), multirows=True)
        return [row["node_id"] for row in results] if results else []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels(entity names) with SQL-side scoring and fuzzy matching.

        The CASE mirrors _search_score / NetworkXStorage.search_labels' scoring
        rules: exact 1000, prefix 500, else 100-length plus a +50 word-boundary
        bonus nested INSIDE the ELSE branch so it never applies to exact or
        prefix matches. LIKE metacharacters in the query are escaped; MySQL
        treats backslash as the default LIKE escape char, so no explicit
        ESCAPE clause is needed.
        """
        q = query.strip().lower()
        if not q:
            return []
        escaped = self._escape_like(q)
        sql = """
            SELECT node_id FROM (
                SELECT node_id,
                       CASE
                           WHEN LOWER(node_id)=%(exact)s THEN 1000
                           WHEN LOWER(node_id) LIKE %(prefix)s THEN 500
                           ELSE 100 - LENGTH(node_id)
                                + CASE
                                      WHEN LOWER(node_id) LIKE %(space_q)s
                                        OR LOWER(node_id) LIKE %(underscore_q)s
                                      THEN 50
                                      ELSE 0
                                  END
                       END AS score
                FROM LIGHTRAG_GRAPH_NODES
                WHERE workspace=%(workspace)s
                  AND LOWER(node_id) LIKE %(contains)s
            ) scored
            ORDER BY score DESC, node_id ASC
            LIMIT %(limit)s
        """
        params = self._gp(
            exact=q,
            prefix=f"{escaped}%",
            contains=f"%{escaped}%",
            space_q=f"% {escaped}%",
            # Literal underscore: '_' is a LIKE wildcard, so the word-boundary
            # probe for "_query" must escape it or it would match any character.
            underscore_q=rf"%\_{escaped}%",
            limit=limit,
        )
        results = await self.db.query(sql, params, multirows=True)
        return [row["node_id"] for row in results] if results else []

    async def drop(self) -> dict[str, str]:
        """Drop all graph data for the current workspace atomically.

        Edges first, then nodes, in ONE execute_transaction: ADB has no FK
        CASCADE, and bare START TRANSACTION via execute() would land on
        separate pooled connections and lose the transaction boundary.
        """
        try:
            statements = [
                (
                    "DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=%(workspace)s",
                    self._gp(),
                ),
                (
                    "DELETE FROM LIGHTRAG_GRAPH_NODES WHERE workspace=%(workspace)s",
                    self._gp(),
                ),
            ]
            await self.db.execute_transaction(statements)
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping graph data: {e}")
            return {"status": "error", "message": str(e)}


NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "LIGHTRAG_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.KV_STORE_FULL_ENTITIES: "LIGHTRAG_FULL_ENTITIES",
    NameSpace.KV_STORE_FULL_RELATIONS: "LIGHTRAG_FULL_RELATIONS",
    NameSpace.KV_STORE_ENTITY_CHUNKS: "LIGHTRAG_ENTITY_CHUNKS",
    NameSpace.KV_STORE_RELATION_CHUNKS: "LIGHTRAG_RELATION_CHUNKS",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "LIGHTRAG_LLM_CACHE",
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_VDB_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_VDB_RELATION",
    NameSpace.DOC_STATUS: "LIGHTRAG_DOC_STATUS",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSON,
                    sidecar_location TEXT NULL,
                    parse_format VARCHAR(32) NULL DEFAULT 'raw',
                    content_hash TEXT NULL,
                    process_options TEXT NULL,
                    chunk_options JSON NULL DEFAULT CAST('{}' as JSON),
                    parse_engine TEXT NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    file_path TEXT NULL,
                    llm_cache_list JSON NULL DEFAULT CAST('[]' as JSON),
                    heading JSON NULL DEFAULT CAST('{}' as JSON),
                    sidecar JSON NULL DEFAULT CAST('{}' as JSON),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    chunk_id VARCHAR(255) NULL,
                    cache_type VARCHAR(32),
                    queryparam JSON NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length INTEGER NULL,
	               chunks_count INTEGER NULL,
	               status varchar(64) NULL,
	               file_path TEXT NULL,
	               chunks_list JSON NULL DEFAULT CAST('[]' as JSON),
	               track_id varchar(255) NULL,
	               metadata JSON NULL DEFAULT CAST('{}' as JSON),
	               error_msg TEXT NULL,
	               content_hash varchar(255) NULL,
	               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               PRIMARY KEY (workspace, id)
	              )"""
    },
    "LIGHTRAG_FULL_ENTITIES": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_ENTITIES (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_names JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_FULL_RELATIONS": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_RELATIONS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    relation_pairs JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_ENTITY_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_ENTITY_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_RELATION_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_RELATION_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
}

VECTOR_TABLES = {
    "LIGHTRAG_VDB_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector ARRAY<FLOAT>(EMBEDDING_DIM),
                    file_path TEXT NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    ) ENGINE='XUANWU_V2' """,
        "ann_index_ddl": """ALTER TABLE LIGHTRAG_VDB_CHUNKS
                    ADD ANN INDEX idx_content_vector(content_vector) distancemeasure=CosineSimilarity""",
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector ARRAY<FLOAT>(EMBEDDING_DIM),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids ARRAY<VARCHAR(255)> NULL,
                    file_path TEXT NULL,
                    PRIMARY KEY (workspace, id)
                    ) ENGINE='XUANWU_V2' """,
        "ann_index_ddl": """ALTER TABLE LIGHTRAG_VDB_ENTITY
                    ADD ANN INDEX idx_content_vector(content_vector) distancemeasure=CosineSimilarity""",
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector ARRAY<FLOAT>(EMBEDDING_DIM),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids ARRAY<VARCHAR(255)> NULL,
                    file_path TEXT NULL,
                    PRIMARY KEY (workspace, id)
                    ) ENGINE='XUANWU_V2' """,
        "ann_index_ddl": """ALTER TABLE LIGHTRAG_VDB_RELATION
                    ADD ANN INDEX idx_content_vector(content_vector) distancemeasure=CosineSimilarity""",
    },
}

GRAPH_TABLES = {
    "LIGHTRAG_GRAPH_NODES": {
        "ddl": """CREATE TABLE LIGHTRAG_GRAPH_NODES (
                    workspace VARCHAR(255) NOT NULL,
                    node_id VARCHAR(512) NOT NULL,
                    properties JSON NULL DEFAULT CAST('{}' as JSON),
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, node_id)
                    )"""
    },
    "LIGHTRAG_GRAPH_EDGES": {
        "ddl": """CREATE TABLE LIGHTRAG_GRAPH_EDGES (
                    workspace VARCHAR(255) NOT NULL,
                    source_id VARCHAR(512) NOT NULL,
                    target_id VARCHAR(512) NOT NULL,
                    properties JSON NULL DEFAULT CAST('{}' as JSON),
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, source_id, target_id),
                    KEY idx_edges_target (workspace, target_id)
                    )"""
    },
}

SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content,
                             COALESCE(doc_name, '') as file_path,
                             sidecar_location,
                             parse_format,
                             content_hash,
                             process_options,
                             COALESCE(chunk_options, cast('{}' as json)) as chunk_options,
                             parse_engine
                             FROM LIGHTRAG_DOC_FULL WHERE workspace=%(workspace)s AND id=%(id)s
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, cast('[]' as json)) as llm_cache_list,
                                COALESCE(heading, cast('{}' as json)) as heading,
                                COALESCE(sidecar, cast('{}' as json)) as sidecar,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=%(workspace)s AND id=%(id)s
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content,
                                 COALESCE(doc_name, '') as file_path,
                                 sidecar_location,
                                 parse_format,
                                 content_hash,
                                 process_options,
                                 COALESCE(chunk_options, cast('{}' as json)) as chunk_options,
                                 parse_engine
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, cast('[]' as json)) as llm_cache_list,
                                  COALESCE(heading, cast('{}' as json)) as heading,
                                  COALESCE(sidecar, cast('{}' as json)) as sidecar,
                                  UNIX_TIMESTAMP(create_time) as create_time,
                                  UNIX_TIMESTAMP(update_time) as update_time
                                  FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_id_full_entities": """SELECT id, entity_names, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_id_full_relations": """SELECT id, relation_pairs, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_ids_full_entities": """SELECT id, entity_names, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_ids_full_relations": """SELECT id, relation_pairs, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_id_entity_chunks": """SELECT id, chunk_ids, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_ENTITY_CHUNKS WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_id_relation_chunks": """SELECT id, chunk_ids, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_RELATION_CHUNKS WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_ids_entity_chunks": """SELECT id, chunk_ids, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_ENTITY_CHUNKS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_ids_relation_chunks": """SELECT id, chunk_ids, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_RELATION_CHUNKS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "upsert_doc_full": """REPLACE INTO LIGHTRAG_DOC_FULL (id, content, doc_name, workspace,
                        sidecar_location, parse_format, content_hash,
                        process_options, chunk_options, parse_engine,
                        create_time, update_time)
                        VALUES (%(id)s, %(content)s, %(doc_name)s, %(workspace)s,
                        %(sidecar_location)s, %(parse_format)s, %(content_hash)s,
                        %(process_options)s, %(chunk_options)s, %(parse_engine)s,
                        %(create_time)s, CURRENT_TIMESTAMP)
                       """,
    "upsert_llm_response_cache": """REPLACE INTO LIGHTRAG_LLM_CACHE(workspace, id, original_prompt, return_value,
                                  chunk_id, cache_type, queryparam, create_time, update_time)
                                  VALUES (%(workspace)s, %(id)s, %(original_prompt)s, %(return_value)s,
                                  %(chunk_id)s, %(cache_type)s, %(queryparam)s,
                                  %(create_time)s, CURRENT_TIMESTAMP)
                                 """,
    "upsert_text_chunk": """REPLACE INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path,
                      llm_cache_list, heading, sidecar, create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(tokens)s, %(chunk_order_index)s, %(full_doc_id)s,
                      %(content)s, %(file_path)s, %(llm_cache_list)s, %(heading)s, %(sidecar)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_full_entities": """REPLACE INTO LIGHTRAG_FULL_ENTITIES (workspace, id, entity_names, count,
                      create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(entity_names)s, %(count)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_full_relations": """REPLACE INTO LIGHTRAG_FULL_RELATIONS (workspace, id, relation_pairs, count,
                      create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(relation_pairs)s, %(count)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_entity_chunks": """REPLACE INTO LIGHTRAG_ENTITY_CHUNKS (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(chunk_ids)s, %(count)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_relation_chunks": """REPLACE INTO LIGHTRAG_RELATION_CHUNKS (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(chunk_ids)s, %(count)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    # SQL for VectorStorage
    "upsert_chunk": """REPLACE INTO LIGHTRAG_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path,
                      create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(tokens)s, %(chunk_order_index)s, %(full_doc_id)s,
                      %(content)s, %(content_vector)s, %(file_path)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_entity": """REPLACE INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(entity_name)s, %(content)s,
                      %(content_vector)s, %(chunk_ids)s, %(file_path)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_relationship": """REPLACE INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path,
                      create_time, update_time)
                      VALUES (%(workspace)s, %(id)s, %(source_id)s, %(target_id)s,
                      %(content)s, %(content_vector)s, %(chunk_ids)s, %(file_path)s,
                      %(create_time)s, CURRENT_TIMESTAMP)
                     """,
    "relationships": """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            UNIX_TIMESTAMP(r.create_time) AS created_at,
                            cosine_similarity(r.content_vector, '[{embedding_string}]') AS similarity
                     FROM LIGHTRAG_VDB_RELATION r
                     WHERE r.workspace = %(workspace)s
                       AND cosine_similarity(r.content_vector, '[{embedding_string}]') > %(cosine_better_than_threshold)s
                     ORDER BY similarity DESC
                     LIMIT %(top_k)s;
                     """,
    "entities": """
                SELECT e.entity_name,
                       UNIX_TIMESTAMP(e.create_time) AS created_at,
                       cosine_similarity(e.content_vector, '[{embedding_string}]') AS similarity
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = %(workspace)s
                  AND cosine_similarity(e.content_vector, '[{embedding_string}]') > %(cosine_better_than_threshold)s
                ORDER BY similarity DESC
                LIMIT %(top_k)s;
                """,
    "chunks": """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     UNIX_TIMESTAMP(c.create_time) AS created_at,
                     cosine_similarity(c.content_vector, '[{embedding_string}]') AS similarity
              FROM LIGHTRAG_VDB_CHUNKS c
              WHERE c.workspace = %(workspace)s
                AND cosine_similarity(c.content_vector, '[{embedding_string}]') > %(cosine_better_than_threshold)s
              ORDER BY similarity DESC
              LIMIT %(top_k)s;
              """,
    # DROP tables
    "drop_specify_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=%(workspace)s
       """,
}
