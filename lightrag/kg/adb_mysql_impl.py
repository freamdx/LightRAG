import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np

from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger
from ..kg.shared_storage import get_data_init_lock, get_storage_lock

import pipmaster

if not pipmaster.is_installed("aiomysql"):
    pipmaster.install("aiomysql")

import aiomysql

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env")


class AnalyticDB:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.config = config
        self.workspace = config["workspace"]
        self.pool = None

        if not all([self.config["user"], self.config["password"], self.config["db"]]):
            raise ValueError("Missing database user, password or database.")

    async def initdb(self):
        # init pool
        try:
            self.pool = await aiomysql.create_pool(**self.config)
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to connect database, Got:{e}")
            raise

        # create table
        for k, v in TABLES.items():
            try:
                await self.query(f"SELECT 1 FROM {k} LIMIT 1")
            except Exception:
                try:
                    logger.info(f"AnalyticDB MySQL, Try Creating table {k} in database")
                    await self.execute(v["ddl"])
                except Exception as e:
                    logger.error(
                        f"AnalyticDB MySQL, Failed to create table {k} in database, "
                        f"Please verify the connection with database, Got: {e}"
                    )
                    raise e

    async def query(
            self,
            sql: str,
            params: list[Any] | None = None,
            multirows: bool = False,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        async with self.pool.acquire() as connection:  # type: ignore
            try:
                if params:
                    rows = await connection.fetch(sql, *params)
                else:
                    rows = await connection.fetch(sql)

                if multirows:
                    if rows:
                        columns = [col for col in rows[0].keys()]
                        data = [dict(zip(columns, row)) for row in rows]
                    else:
                        data = []
                else:
                    if rows:
                        columns = rows[0].keys()
                        data = dict(zip(columns, rows[0]))
                    else:
                        data = None

                return data
            except Exception as e:
                logger.error(f"AnalyticDB MySQL, query error:{e}")
                raise

    async def execute(
            self,
            sql: str,
            data: dict[str, Any] | None = None,
    ):
        try:
            async with self.pool.acquire() as connection:  # type: ignore
                if data is None:
                    await connection.execute(sql)
                else:
                    await connection.execute(sql, *data.values())
        except Exception as e:
            logger.error(f"AnalyticDB MySQL,\nsql:{sql},\ndata:{data},\nerror:{e}")
            raise


class ClientManager:
    _instances: dict[str, Any] = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @staticmethod
    def get_config() -> dict[str, Any]:
        return {
            "host": os.getenv("ADB_HOST", "localhost"),
            "port": os.getenv("ADB_PORT", "3306"),
            "user": os.getenv("ADB_USER"),
            "password": os.getenv("ADB_PASSWORD"),
            "db": os.getenv("ADB_DATABASE", "graphrag"),
            "workspace": os.getenv("ADB_WORKSPACE", "graphrag"),
            "maxsize": os.getenv("ADB_MAX_CONNECTIONS", "20"),
            "autocommit": "True",
        }

    @classmethod
    async def get_client(cls) -> AnalyticDB:
        async with cls._lock:
            if cls._instances["db"] is None:
                config = ClientManager.get_config()
                db = AnalyticDB(config)
                await db.initdb()
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: AnalyticDB):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        await db.pool.close()
                        logger.info("Closed database connection pool")
                        cls._instances["db"] = None
                else:
                    await db.pool.close()


@final
@dataclass
class ADBKVStorage(BaseKVStorage):
    db: AnalyticDB | None = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: AnalyticDB.workspace > self.workspace > "graphrag"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "graphrag" for compatibility (lowest priority)
                self.workspace = "graphrag"

    async def finalize(self):
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    async def get_all(self) -> dict[str, Any]:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for get_all: {self.namespace}")
            return {}

        sql = f"SELECT * FROM {table_name} WHERE workspace=$1"
        params = {"workspace": self.workspace}

        try:
            results = await self.db.query(sql, list(params.values()), multirows=True)

            # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
            if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
                processed_results = {}
                for row in results:
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    # Map field names and add cache_type for compatibility
                    processed_row = {
                        **row,
                        "return": row.get("return_value", ""),
                        "cache_type": row.get("original_prompt", "unknow"),
                        "original_prompt": row.get("original_prompt", ""),
                        "chunk_id": row.get("chunk_id"),
                        "mode": row.get("mode", "default"),
                        "create_time": create_time,
                        "update_time": create_time if update_time == 0 else update_time,
                    }
                    processed_results[row["id"]] = processed_row
                return processed_results

            # For text_chunks namespace, parse llm_cache_list JSON string back to list
            if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                processed_results = {}
                for row in results:
                    llm_cache_list = row.get("llm_cache_list", [])
                    if isinstance(llm_cache_list, str):
                        try:
                            llm_cache_list = json.loads(llm_cache_list)
                        except json.JSONDecodeError:
                            llm_cache_list = []
                    row["llm_cache_list"] = llm_cache_list
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    row["create_time"] = create_time
                    row["update_time"] = (
                        create_time if update_time == 0 else update_time
                    )
                    processed_results[row["id"]] = row
                return processed_results

            # For FULL_ENTITIES namespace, parse entity_names JSON string back to list
            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
                processed_results = {}
                for row in results:
                    entity_names = row.get("entity_names", [])
                    if isinstance(entity_names, str):
                        try:
                            entity_names = json.loads(entity_names)
                        except json.JSONDecodeError:
                            entity_names = []
                    row["entity_names"] = entity_names
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    row["create_time"] = create_time
                    row["update_time"] = (
                        create_time if update_time == 0 else update_time
                    )
                    processed_results[row["id"]] = row
                return processed_results

            # For FULL_RELATIONS namespace, parse relation_pairs JSON string back to list
            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
                processed_results = {}
                for row in results:
                    relation_pairs = row.get("relation_pairs", [])
                    if isinstance(relation_pairs, str):
                        try:
                            relation_pairs = json.loads(relation_pairs)
                        except json.JSONDecodeError:
                            relation_pairs = []
                    row["relation_pairs"] = relation_pairs
                    create_time = row.get("create_time", 0)
                    update_time = row.get("update_time", 0)
                    row["create_time"] = create_time
                    row["update_time"] = (
                        create_time if update_time == 0 else update_time
                    )
                    processed_results[row["id"]] = row
                return processed_results

            # For other namespaces, return as-is
            return {row["id"]: row for row in results}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving all data from {self.namespace}: {e}")
            return {}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.workspace, "id": id}

        response = await self.db.query(sql, list(params.values()))

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            # Parse query_param JSON string back to dict
            query_param = response.get("query_param")
            if isinstance(query_param, str):
                try:
                    query_param = json.loads(query_param)
                except json.JSONDecodeError:
                    query_param = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type"),
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "query_param": query_param,
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

        return response if response else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        params = {"workspace": self.workspace}

        results = await self.db.query(sql, list(params.values()), multirows=True)

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
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            processed_results = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                # Parse query_param JSON string back to dict
                query_param = row.get("query_param")
                if isinstance(query_param, str):
                    try:
                        query_param = json.loads(query_param)
                    except json.JSONDecodeError:
                        query_param = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "query_param": query_param,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)

            return processed_results

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

        return results if results else []

    async def filter_keys(self, keys: set[str]) -> set[str]:
        sql = SQL_TEMPLATES["filter_keys"].format(
            table_name=namespace_to_table_name(self.namespace),
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.workspace}

        try:
            res = await self.db.query(sql, list(params.values()), multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            return new_keys
        except Exception as e:
            logger.error(f"[{self.workspace}] AnalyticDB MySQL,\nsql:{sql},\nparams:{params},\nerror:{e}")
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_text_chunk"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "tokens": v["tokens"],
                    "chunk_order_index": v["chunk_order_index"],
                    "full_doc_id": v["full_doc_id"],
                    "content": v["content"],
                    "file_path": v["file_path"],
                    "llm_cache_list": json.dumps(v.get("llm_cache_list", [])),
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
                _data = {
                    "id": k,
                    "content": v["content"],
                    "workspace": self.workspace,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,  # Use flattened key as id
                    "original_prompt": v["original_prompt"],
                    "return_value": v["return"],
                    "chunk_id": v.get("chunk_id"),
                    "cache_type": v.get(
                        "cache_type", "extract"
                    ),  # Get cache_type from data
                    "query_param": json.dumps(v.get("query_param"))
                    if v.get("query_param")
                    else None,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_full_entities"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "entity_names": json.dumps(v["entity_names"]),
                    "count": v["count"],
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_full_relations"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "relation_pairs": json.dumps(v["relation_pairs"]),
                    "count": v["count"],
                }
                await self.db.execute(upsert_sql, _data)

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}")
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids})
        except Exception as e:
            logger.error(f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}")

    async def drop(self) -> dict[str, str]:
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_TEMPLATES["drop_specific_table_workspace"].format(table_name=table_name)

                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBVectorStorage(BaseVectorStorage):
    db: AnalyticDB | None = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError("cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs")
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: AnalyticDB.workspace > self.workspace > "graphrag"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "graphrag" for compatibility (lowest priority)
                self.workspace = "graphrag"

    async def finalize(self):
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    def _upsert_chunks(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        try:
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
        except Exception as e:
            logger.error(f"[{self.workspace}] Error to prepare upsert,\nsql: {e}\nitem: {item}")
            raise

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
            "chunk_ids": chunk_ids,
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
            "chunk_ids": chunk_ids,
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
            contents[i: i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        for item in list_data:
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                upsert_sql, data = self._upsert_chunks(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                upsert_sql, data = self._upsert_entities(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                upsert_sql, data = self._upsert_relationships(item)
            else:
                raise ValueError(f"{self.namespace} is not supported")

            await self.db.execute(upsert_sql, data)

    async def query(
            self,
            query: str,
            top_k: int,
            query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func([query], _priority=5)
            embedding = embeddings[0]

        embedding_string = ",".join(map(str, embedding))

        sql = SQL_TEMPLATES[self.namespace].format(embedding_string=embedding_string)
        params = {
            "workspace": self.workspace,
            "closer_than_threshold": 1 - self.cosine_better_than_threshold,
            "top_k": top_k,
        }

        results = await self.db.query(sql, params=list(params.values()), multirows=True)
        return results

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for vector deletion: {self.namespace}")
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids})
        except Exception as e:
            logger.error(f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        try:
            delete_sql = "DELETE FROM LIGHTRAG_VDB_ENTITY WHERE workspace=$1 AND entity_name=$2"

            await self.db.execute(delete_sql, {"workspace": self.workspace, "entity_name": entity_name})
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            delete_sql = "DELETE FROM LIGHTRAG_VDB_RELATION WHERE workspace=$1 AND (source_id=$2 OR target_id=$2)"

            await self.db.execute(delete_sql, {"workspace": self.workspace, "entity_name": entity_name})
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting relations for entity {entity_name}: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for ID lookup: {self.namespace}")
            return None

        query = (f"SELECT *, UNIX_TIMESTAMP(create_time) as created_at "
                 f"FROM {table_name} WHERE workspace=$1 AND id=$2")
        params = {"workspace": self.workspace, "id": id}

        try:
            result = await self.db.query(query, list(params.values()))
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for IDs lookup: {self.namespace}")
            return []

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = (f"SELECT *, UNIX_TIMESTAMP(create_time) as created_at "
                 f"FROM {table_name} WHERE workspace=$1 AND id IN ({ids_str})")
        params = {"workspace": self.workspace}

        try:
            results = await self.db.query(query, list(params.values()), multirows=True)
            return [dict(record) for record in results]
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}")
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for vector lookup: {self.namespace}")
            return {}

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = f"SELECT id, content_vector FROM {table_name} WHERE workspace=$1 AND id IN ({ids_str})"
        params = {"workspace": self.workspace}

        try:
            results = await self.db.query(query, list(params.values()), multirows=True)
            vectors_dict = {}

            for result in results:
                if result and "content_vector" in result and "id" in result:
                    try:
                        # Parse JSON string to get vector as list of floats
                        vector_data = json.loads(result["content_vector"])
                        if isinstance(vector_data, list):
                            vectors_dict[result["id"]] = vector_data
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"[{self.workspace}] Failed to parse vector data for ID {result['id']}: {e}")

            return vectors_dict
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}")
            return {}

    async def drop(self) -> dict[str, str]:
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_TEMPLATES["drop_specific_table_workspace"].format(table_name=table_name)

                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBDocStatusStorage(DocStatusStorage):
    db: AnalyticDB | None = field(default=None)

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: AnalyticDB.workspace > self.workspace > "graphrag"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "graphrag" for compatibility (lowest priority)
                self.workspace = "graphrag"

    async def finalize(self):
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        sql = SQL_TEMPLATES["filter_keys"].format(
            table_name=namespace_to_table_name(self.namespace),
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.workspace}

        try:
            res = await self.db.query(sql, list(params.values()), multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            return new_keys
        except Exception as e:
            logger.error(f"[{self.workspace}] AnalyticDB MySQL,\nsql:{sql},\nparams:{params},\nerror:{e}")
            raise

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and id=$2"
        params = {"workspace": self.workspace, "id": id}

        result = await self.db.query(sql, list(params.values()), True)
        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            return dict(
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=result[0]["created_at"],
                updated_at=result[0]["updated_at"],
                file_path=result[0]["file_path"],
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=result[0].get("error_msg"),
                track_id=result[0].get("track_id"),
            )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        sql = "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND id = ANY($2)"
        params = {"workspace": self.workspace, "ids": ids}

        results = await self.db.query(sql, list(params.values()), True)
        if not results:
            return []

        processed_results = []
        for row in results:
            # Parse chunks_list JSON string back to list
            chunks_list = row.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            processed_results.append(
                {
                    "content_length": row["content_length"],
                    "content_summary": row["content_summary"],
                    "status": row["status"],
                    "chunks_count": row["chunks_count"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "file_path": row["file_path"],
                    "chunks_list": chunks_list,
                    "metadata": metadata,
                    "error_msg": row.get("error_msg"),
                    "track_id": row.get("track_id"),
                }
            )

        return processed_results

    async def get_status_counts(self) -> dict[str, int]:
        sql = "SELECT status, COUNT(1) as count FROM LIGHTRAG_DOC_STATUS where workspace=$1 GROUP BY status"
        params = {"workspace": self.workspace}

        result = await self.db.query(sql, list(params.values()), True)
        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and status=$2"
        params = {"workspace": self.workspace, "status": status.value}

        result = await self.db.query(sql, list(params.values()), True)

        docs_by_status = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get("file_path")
            if file_path is None:
                file_path = "no-file-path"

            docs_by_status[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=element.get("error_msg"),
                track_id=element.get("track_id"),
            )

        return docs_by_status

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and track_id=$2"
        params = {"workspace": self.workspace, "track_id": track_id}

        result = await self.db.query(sql, list(params.values()), True)

        docs_by_track_id = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get("file_path")
            if file_path is None:
                file_path = "no-file-path"

            docs_by_track_id[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
            )

        return docs_by_track_id

    async def get_docs_paginated(
            self,
            status_filter: DocStatus | None = None,
            page: int = 1,
            page_size: int = 50,
            sort_field: str = "updated_at",
            sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """

        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"

        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        # Calculate offset
        offset = (page - 1) * page_size

        # Build WHERE clause
        where_clause = "WHERE workspace=$1"
        params = {"workspace": self.workspace}
        param_count = 1

        if status_filter is not None:
            param_count += 1
            where_clause += f" AND status=${param_count}"
            params["status"] = status_filter.value

        # Build ORDER BY clause
        order_clause = f"ORDER BY {sort_field} {sort_direction.upper()}"

        # Query for total count
        count_sql = f"SELECT COUNT(*) as total FROM LIGHTRAG_DOC_STATUS {where_clause}"
        count_result = await self.db.query(count_sql, list(params.values()))
        total_count = count_result["total"] if count_result else 0

        # Query for paginated data
        data_sql = f"""
                    SELECT * FROM LIGHTRAG_DOC_STATUS
                    {where_clause} 
                    {order_clause} 
                    LIMIT ${param_count + 1} OFFSET ${param_count + 2}
                    """
        params["limit"] = page_size
        params["offset"] = offset

        result = await self.db.query(data_sql, list(params.values()), True)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for element in result:
            doc_id = element["id"]

            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            doc_status = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=element["file_path"],
                chunks_list=chunks_list,
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
            )
            documents.append((doc_id, doc_status))

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) as count FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 GROUP BY status"
        params = {"workspace": self.workspace}

        result = await self.db.query(sql, list(params.values()), True)

        counts = {}
        total_count = 0
        for row in result:
            counts[row["status"]] = row["count"]
            total_count += row["count"]

        # Add 'all' field with total count
        counts["all"] = total_count

        return counts

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}")
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids})
        except Exception as e:
            logger.error(f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        # All fields are updated from the input data in both INSERT and UPDATE cases
        sql = """replace into LIGHTRAG_DOC_STATUS(workspace, id, 
                 content_summary, content_length, chunks_count, status, file_path, 
                 chunks_list, track_id, metadata, error_msg, created_at, updated_at)
                 values($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)"""
        for k, v in data.items():
            # chunks_count, chunks_list, track_id, metadata, and error_msg are optional
            await self.db.execute(
                sql,
                {
                    "workspace": self.workspace,
                    "id": k,
                    "content_summary": v["content_summary"],
                    "content_length": v["content_length"],
                    "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                    "status": v["status"],
                    "file_path": v["file_path"],
                    "chunks_list": json.dumps(v.get("chunks_list", [])),
                    "track_id": v.get("track_id"),  # Add track_id support
                    "metadata": json.dumps(
                        v.get("metadata", {})
                    ),  # Add metadata support
                    "error_msg": v.get("error_msg"),  # Add error_msg support
                    "created_at": v.get("created_at"),
                    "updated_at": v.get("updated_at"),
                },
            )

    async def drop(self) -> dict[str, str]:
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_TEMPLATES["drop_specific_table_workspace"].format(table_name=table_name)

                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "LIGHTRAG_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.KV_STORE_FULL_ENTITIES: "LIGHTRAG_FULL_ENTITIES",
    NameSpace.KV_STORE_FULL_RELATIONS: "LIGHTRAG_FULL_RELATIONS",
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
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_CHUNKS": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector ARRAY<FLOAT>({os.environ.get("EMBEDDING_DIM", 1024)}),
                    file_path TEXT NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ANN INDEX idx_content_vector(content_vector),
	                PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector ARRAY<FLOAT>({os.environ.get("EMBEDDING_DIM", 1024)}),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids ARRAY<VARCHAR(255)> NULL,
                    file_path TEXT NULL,
                    ANN INDEX idx_content_vector(content_vector),
	                PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector ARRAY<FLOAT>({os.environ.get("EMBEDDING_DIM", 1024)}),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids ARRAY<VARCHAR(255)> NULL,
                    file_path TEXT NULL,
                    ANN INDEX idx_content_vector(content_vector),
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
                    query_param JSON NULL,
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
}

SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, cast('[]' as json)) as llm_cache_list,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, query_param,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id IN ({ids})
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, cast('[]' as json)) as llm_cache_list,
                                  UNIX_TIMESTAMP(create_time) as create_time,
                                  UNIX_TIMESTAMP(update_time) as update_time
                                  FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, query_param,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_id_full_entities": """SELECT id, entity_names, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=$1 AND id=$2
                               """,
    "get_by_id_full_relations": """SELECT id, relation_pairs, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_entities": """SELECT id, entity_names, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_ids_full_relations": """SELECT id, relation_pairs, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=$1 AND id IN ({ids})
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})",
    "upsert_doc_full": """REPLACE INTO LIGHTRAG_DOC_FULL (id, content, workspace, update_time)
                        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                       """,
    "upsert_llm_response_cache": """REPLACE INTO LIGHTRAG_LLM_CACHE(workspace, id, 
                                      original_prompt, return_value, chunk_id, cache_type, query_param, update_time)
                                      VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
                                     """,
    "upsert_text_chunk": """REPLACE INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path, llm_cache_list, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                     """,
    "upsert_full_entities": """REPLACE INTO LIGHTRAG_FULL_ENTITIES (workspace, id, entity_names, count, update_time)
                      VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                     """,
    "upsert_full_relations": """REPLACE INTO LIGHTRAG_FULL_RELATIONS (workspace, id, relation_pairs, count, update_time)
                      VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                     """,
    # SQL for VectorStorage
    "upsert_chunk": """REPLACE INTO LIGHTRAG_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                     """,
    "upsert_entity": """REPLACE INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, update_time)
                      VALUES ($1, $2, $3, $4, $5, cast($6 as array<varchar>), $7, CURRENT_TIMESTAMP)
                     """,
    "upsert_relationship": """REPLACE INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, cast($7 as array<varchar>), $8, CURRENT_TIMESTAMP)
                     """,
    "relationships": """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            UNIX_TIMESTAMP(r.create_time) AS created_at
                     FROM LIGHTRAG_VDB_RELATION r
                     WHERE r.workspace = $1
                       AND l2_distance(r.content_vector, '[{embedding_string}]') < $2
                     ORDER BY l2_distance(r.content_vector, '[{embedding_string}]')
                     LIMIT $3;
                     """,
    "entities": """
                SELECT e.entity_name,
                       UNIX_TIMESTAMP(e.create_time) AS created_at
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = $1
                  AND l2_distance(e.content_vector, '[{embedding_string}]') < $2
                ORDER BY l2_distance(e.content_vector, '[{embedding_string}]')
                LIMIT $3;
                """,
    "chunks": """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     UNIX_TIMESTAMP(c.create_time) AS created_at
              FROM LIGHTRAG_VDB_CHUNKS c
              WHERE c.workspace = $1
                AND l2_distance(c.content_vector, '[{embedding_string}]') < $2
              ORDER BY l2_distance(c.content_vector, '[{embedding_string}]')
              LIMIT $3;
              """,
    # DROP tables
    "drop_specific_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=$1
       """,
}
