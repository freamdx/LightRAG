import asyncio
import json
import os
import re
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np
import configparser
import itertools

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger
from ..constants import GRAPH_FIELD_SEP
from ..kg.shared_storage import get_data_init_lock, get_graph_db_lock, get_storage_lock

import pipmaster

if not pipmaster.is_installed("aiomysql"):
    pipmaster.install("aiomysql")

import aiomysql

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env")


class ADBMYSQL:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.config = config
        self.pool = None

        if not all([self.config["user"], self.config["password"]]):
            raise ValueError("Missing database user, password")

    async def initdb(self):
        try:
            self.pool = await aiomysql.create_pool(**self.config)
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to connect database, Got:{e}")
            raise

    async def _migrate_doc_chunks_to_vdb_chunks(self):
        """
        Migrate data from LIGHTRAG_DOC_CHUNKS to LIGHTRAG_VDB_CHUNKS if specific conditions are met.
        This migration is intended for users who are upgrading and have an older table structure
        where LIGHTRAG_DOC_CHUNKS contained a `content_vector` column.

        """
        try:
            # 1. Check if the new table LIGHTRAG_VDB_CHUNKS is empty
            vdb_chunks_count_sql = "SELECT COUNT(1) as count FROM LIGHTRAG_VDB_CHUNKS"
            vdb_chunks_count_result = await self.query(vdb_chunks_count_sql)
            if vdb_chunks_count_result and vdb_chunks_count_result["count"] > 0:
                logger.info(
                    "Skipping migration: LIGHTRAG_VDB_CHUNKS already contains data."
                )
                return

            # 2. Check if `content_vector` column exists in the old table
            check_column_sql = """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_chunks' AND column_name = 'content_vector'
            """
            column_exists = await self.query(check_column_sql)
            if not column_exists:
                logger.info(
                    "Skipping migration: `content_vector` not found in LIGHTRAG_DOC_CHUNKS"
                )
                return

            # 3. Check if the old table LIGHTRAG_DOC_CHUNKS has data
            doc_chunks_count_sql = "SELECT COUNT(1) as count FROM LIGHTRAG_DOC_CHUNKS"
            doc_chunks_count_result = await self.query(doc_chunks_count_sql)
            if not doc_chunks_count_result or doc_chunks_count_result["count"] == 0:
                logger.info("Skipping migration: LIGHTRAG_DOC_CHUNKS is empty.")
                return

            # 4. Perform the migration
            logger.info(
                "Starting data migration from LIGHTRAG_DOC_CHUNKS to LIGHTRAG_VDB_CHUNKS..."
            )
            migration_sql = """
            INSERT INTO LIGHTRAG_VDB_CHUNKS (
                id, workspace, full_doc_id, chunk_order_index, tokens, content,
                content_vector, file_path, create_time, update_time
            )
            SELECT
                id, workspace, full_doc_id, chunk_order_index, tokens, content,
                content_vector, file_path, create_time, update_time
            FROM LIGHTRAG_DOC_CHUNKS
            ON CONFLICT (workspace, id) DO NOTHING;
            """
            await self.execute(migration_sql)
            logger.info("Data migration to LIGHTRAG_VDB_CHUNKS completed successfully.")

        except Exception as e:
            logger.error(f"Failed during data migration to LIGHTRAG_VDB_CHUNKS: {e}")
            # Do not re-raise, to allow the application to start

    async def _check_llm_cache_needs_migration(self):
        """Check if LLM cache data needs migration by examining any record with old format"""
        try:
            # Optimized query: directly check for old format records without sorting
            check_sql = """
            SELECT 1 FROM LIGHTRAG_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            LIMIT 1
            """
            result = await self.query(check_sql)

            # If any old format record exists, migration is needed
            return result is not None

        except Exception as e:
            logger.warning(f"Failed to check LLM cache migration status: {e}")
            return False

    async def _migrate_llm_cache_to_flattened_keys(self):
        """Optimized version: directly execute single UPDATE migration to migrate old format cache keys to flattened format"""
        try:
            # Check if migration is needed
            check_sql = """
            SELECT COUNT(*) as count FROM LIGHTRAG_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            """
            result = await self.query(check_sql)

            if not result or result["count"] == 0:
                logger.info("No old format LLM cache data found, skipping migration")
                return

            old_count = result["count"]
            logger.info(f"Found {old_count} old format cache records")

            # Check potential primary key conflicts (optional but recommended)
            conflict_check_sql = """
            WITH new_ids AS (
                SELECT
                    workspace,
                    mode,
                    id as old_id,
                    mode || ':' ||
                    CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END || ':' ||
                    md5(original_prompt) as new_id
                FROM LIGHTRAG_LLM_CACHE
                WHERE id NOT LIKE '%:%'
            )
            SELECT COUNT(*) as conflicts
            FROM new_ids n1
            JOIN LIGHTRAG_LLM_CACHE existing
            ON existing.workspace = n1.workspace
            AND existing.mode = n1.mode
            AND existing.id = n1.new_id
            WHERE existing.id LIKE '%:%'  -- Only check conflicts with existing new format records
            """

            conflict_result = await self.query(conflict_check_sql)
            if conflict_result and conflict_result["conflicts"] > 0:
                logger.warning(
                    f"Found {conflict_result['conflicts']} potential ID conflicts with existing records"
                )
                # Can choose to continue or abort, here we choose to continue and log warning

            # Execute single UPDATE migration
            logger.info("Starting optimized LLM cache migration...")
            migration_sql = """
            UPDATE LIGHTRAG_LLM_CACHE
            SET
                id = mode || ':' ||
                     CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END || ':' ||
                     md5(original_prompt),
                cache_type = CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END,
                update_time = CURRENT_TIMESTAMP
            WHERE id NOT LIKE '%:%'
            """

            # Execute migration
            await self.execute(migration_sql)

            # Verify migration results
            verify_sql = """
            SELECT COUNT(*) as remaining_old FROM LIGHTRAG_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            """
            verify_result = await self.query(verify_sql)
            remaining = verify_result["remaining_old"] if verify_result else -1

            if remaining == 0:
                logger.info(
                    f"✅ Successfully migrated {old_count} LLM cache records to flattened format"
                )
            else:
                logger.warning(
                    f"⚠️ Migration completed but {remaining} old format records remain"
                )

        except Exception as e:
            logger.error(f"Optimized LLM cache migration failed: {e}")
            raise

    async def _migrate_doc_status_add_chunks_list(self):
        """Add chunks_list column to LIGHTRAG_DOC_STATUS table if it doesn't exist"""
        try:
            # Check if chunks_list column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'chunks_list'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info("Adding chunks_list column to LIGHTRAG_DOC_STATUS table")
                add_column_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN chunks_list JSONB NULL DEFAULT '[]'::jsonb
                """
                await self.execute(add_column_sql)
                logger.info(
                    "Successfully added chunks_list column to LIGHTRAG_DOC_STATUS table"
                )
            else:
                logger.info(
                    "chunks_list column already exists in LIGHTRAG_DOC_STATUS table"
                )
        except Exception as e:
            logger.warning(
                f"Failed to add chunks_list column to LIGHTRAG_DOC_STATUS: {e}"
            )

    async def _migrate_text_chunks_add_llm_cache_list(self):
        """Add llm_cache_list column to LIGHTRAG_DOC_CHUNKS table if it doesn't exist"""
        try:
            # Check if llm_cache_list column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_chunks'
            AND column_name = 'llm_cache_list'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info("Adding llm_cache_list column to LIGHTRAG_DOC_CHUNKS table")
                add_column_sql = """
                ALTER TABLE LIGHTRAG_DOC_CHUNKS
                ADD COLUMN llm_cache_list JSONB NULL DEFAULT '[]'::jsonb
                """
                await self.execute(add_column_sql)
                logger.info(
                    "Successfully added llm_cache_list column to LIGHTRAG_DOC_CHUNKS table"
                )
            else:
                logger.info(
                    "llm_cache_list column already exists in LIGHTRAG_DOC_CHUNKS table"
                )
        except Exception as e:
            logger.warning(
                f"Failed to add llm_cache_list column to LIGHTRAG_DOC_CHUNKS: {e}"
            )

    async def _migrate_doc_status_add_track_id(self):
        """Add track_id column to LIGHTRAG_DOC_STATUS table if it doesn't exist and create index"""
        try:
            # Check if track_id column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'track_id'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info("Adding track_id column to LIGHTRAG_DOC_STATUS table")
                add_column_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN track_id VARCHAR(255) NULL
                """
                await self.execute(add_column_sql)
                logger.info(
                    "Successfully added track_id column to LIGHTRAG_DOC_STATUS table"
                )
            else:
                logger.info(
                    "track_id column already exists in LIGHTRAG_DOC_STATUS table"
                )

            # Check if track_id index exists
            check_index_sql = """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'lightrag_doc_status'
            AND indexname = 'idx_lightrag_doc_status_track_id'
            """

            index_info = await self.query(check_index_sql)
            if not index_info:
                logger.info(
                    "Creating index on track_id column for LIGHTRAG_DOC_STATUS table"
                )
                create_index_sql = """
                CREATE INDEX idx_lightrag_doc_status_track_id ON LIGHTRAG_DOC_STATUS (track_id)
                """
                await self.execute(create_index_sql)
                logger.info(
                    "Successfully created index on track_id column for LIGHTRAG_DOC_STATUS table"
                )
            else:
                logger.info(
                    "Index on track_id column already exists for LIGHTRAG_DOC_STATUS table"
                )

        except Exception as e:
            logger.warning(
                f"Failed to add track_id column or index to LIGHTRAG_DOC_STATUS: {e}"
            )

    async def _migrate_doc_status_add_metadata_error_msg(self):
        """Add metadata and error_msg columns to LIGHTRAG_DOC_STATUS table if they don't exist"""
        try:
            # Check if metadata column exists
            check_metadata_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'metadata'
            """

            metadata_info = await self.query(check_metadata_sql)
            if not metadata_info:
                logger.info("Adding metadata column to LIGHTRAG_DOC_STATUS table")
                add_metadata_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN metadata JSONB NULL DEFAULT '{}'::jsonb
                """
                await self.execute(add_metadata_sql)
                logger.info(
                    "Successfully added metadata column to LIGHTRAG_DOC_STATUS table"
                )
            else:
                logger.info(
                    "metadata column already exists in LIGHTRAG_DOC_STATUS table"
                )

            # Check if error_msg column exists
            check_error_msg_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'error_msg'
            """

            error_msg_info = await self.query(check_error_msg_sql)
            if not error_msg_info:
                logger.info("Adding error_msg column to LIGHTRAG_DOC_STATUS table")
                add_error_msg_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN error_msg TEXT NULL
                """
                await self.execute(add_error_msg_sql)
                logger.info(
                    "Successfully added error_msg column to LIGHTRAG_DOC_STATUS table"
                )
            else:
                logger.info(
                    "error_msg column already exists in LIGHTRAG_DOC_STATUS table"
                )

        except Exception as e:
            logger.warning(
                f"Failed to add metadata/error_msg columns to LIGHTRAG_DOC_STATUS: {e}"
            )

    async def _migrate_field_lengths(self):
        """Migrate database field lengths: entity_name, source_id, target_id, and file_path"""
        # Define the field changes needed
        field_migrations = [
            {
                "table": "LIGHTRAG_VDB_ENTITY",
                "column": "entity_name",
                "old_type": "character varying(255)",
                "new_type": "VARCHAR(512)",
                "description": "entity_name from 255 to 512",
            },
            {
                "table": "LIGHTRAG_VDB_RELATION",
                "column": "source_id",
                "old_type": "character varying(256)",
                "new_type": "VARCHAR(512)",
                "description": "source_id from 256 to 512",
            },
            {
                "table": "LIGHTRAG_VDB_RELATION",
                "column": "target_id",
                "old_type": "character varying(256)",
                "new_type": "VARCHAR(512)",
                "description": "target_id from 256 to 512",
            },
            {
                "table": "LIGHTRAG_DOC_CHUNKS",
                "column": "file_path",
                "old_type": "character varying(256)",
                "new_type": "TEXT",
                "description": "file_path to TEXT NULL",
            },
            {
                "table": "LIGHTRAG_VDB_CHUNKS",
                "column": "file_path",
                "old_type": "character varying(256)",
                "new_type": "TEXT",
                "description": "file_path to TEXT NULL",
            },
        ]

        for migration in field_migrations:
            try:
                # Check current column definition
                check_column_sql = """
                SELECT column_name, data_type, character_maximum_length, is_nullable
                FROM information_schema.columns
                WHERE table_name = $1 AND column_name = $2
                """
                params = {
                    "table_name": migration["table"].lower(),
                    "column_name": migration["column"],
                }
                column_info = await self.query(
                    check_column_sql,
                    list(params.values()),
                )

                if not column_info:
                    logger.warning(
                        f"Column {migration['table']}.{migration['column']} does not exist, skipping migration"
                    )
                    continue

                current_type = column_info.get("data_type", "").lower()
                current_length = column_info.get("character_maximum_length")

                # Check if migration is needed
                needs_migration = False

                if migration["column"] == "entity_name" and current_length == 255:
                    needs_migration = True
                elif (
                        migration["column"] in ["source_id", "target_id"]
                        and current_length == 256
                ):
                    needs_migration = True
                elif (
                        migration["column"] == "file_path"
                        and current_type == "character varying"
                ):
                    needs_migration = True

                if needs_migration:
                    logger.info(
                        f"Migrating {migration['table']}.{migration['column']}: {migration['description']}"
                    )

                    # Execute the migration
                    alter_sql = f"""
                    ALTER TABLE {migration['table']}
                    ALTER COLUMN {migration['column']} TYPE {migration['new_type']}
                    """

                    await self.execute(alter_sql)
                    logger.info(
                        f"Successfully migrated {migration['table']}.{migration['column']}"
                    )
                else:
                    logger.debug(
                        f"Column {migration['table']}.{migration['column']} already has correct type, no migration needed"
                    )

            except Exception as e:
                # Log error but don't interrupt the process
                logger.warning(
                    f"Failed to migrate {migration['table']}.{migration['column']}: {e}"
                )

    async def check_tables(self):
        # First create all tables
        for k, v in TABLES.items():
            try:
                await self.query(f"SELECT 1 FROM {k} LIMIT 1")
            except Exception:
                try:
                    logger.info(f"AnalyticDB MySQL, Try Creating table {k} in database")
                    await self.execute(v["ddl"])
                    logger.info(
                        f"AnalyticDB MySQL, Creation success table {k} in database"
                    )
                except Exception as e:
                    logger.error(
                        f"AnalyticDB MySQL, Failed to create table {k} in database, Please verify the connection with database, Got: {e}"
                    )
                    raise e

        # Finally, attempt to migrate old doc chunks data if needed
        try:
            await self._migrate_doc_chunks_to_vdb_chunks()
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to migrate doc_chunks to vdb_chunks: {e}")

        # Check and migrate LLM cache to flattened keys if needed
        try:
            if await self._check_llm_cache_needs_migration():
                await self._migrate_llm_cache_to_flattened_keys()
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, LLM cache migration failed: {e}")

        # Migrate doc status to add chunks_list field if needed
        try:
            await self._migrate_doc_status_add_chunks_list()
        except Exception as e:
            logger.error(
                f"AnalyticDB MySQL, Failed to migrate doc status chunks_list field: {e}"
            )

        # Migrate text chunks to add llm_cache_list field if needed
        try:
            await self._migrate_text_chunks_add_llm_cache_list()
        except Exception as e:
            logger.error(
                f"AnalyticDB MySQL, Failed to migrate text chunks llm_cache_list field: {e}"
            )

        # Migrate field lengths for entity_name, source_id, target_id, and file_path
        try:
            await self._migrate_field_lengths()
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to migrate field lengths: {e}")

        # Migrate doc status to add track_id field if needed
        try:
            await self._migrate_doc_status_add_track_id()
        except Exception as e:
            logger.error(
                f"AnalyticDB MySQL, Failed to migrate doc status track_id field: {e}"
            )

        # Migrate doc status to add metadata and error_msg fields if needed
        try:
            await self._migrate_doc_status_add_metadata_error_msg()
        except Exception as e:
            logger.error(
                f"AnalyticDB MySQL, Failed to migrate doc status metadata/error_msg fields: {e}"
            )

        # Create pagination optimization indexes for LIGHTRAG_DOC_STATUS
        try:
            await self._create_pagination_indexes()
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to create pagination indexes: {e}")

        # Migrate to ensure new tables LIGHTRAG_FULL_ENTITIES and LIGHTRAG_FULL_RELATIONS exist
        try:
            await self._migrate_create_full_entities_relations_tables()
        except Exception as e:
            logger.error(
                f"AnalyticDB MySQL, Failed to create full entities/relations tables: {e}"
            )

    async def _migrate_create_full_entities_relations_tables(self):
        """Create LIGHTRAG_FULL_ENTITIES and LIGHTRAG_FULL_RELATIONS tables if they don't exist"""
        tables_to_check = [
            {
                "name": "LIGHTRAG_FULL_ENTITIES",
                "ddl": TABLES["LIGHTRAG_FULL_ENTITIES"]["ddl"],
                "description": "Full entities storage table",
            },
            {
                "name": "LIGHTRAG_FULL_RELATIONS",
                "ddl": TABLES["LIGHTRAG_FULL_RELATIONS"]["ddl"],
                "description": "Full relations storage table",
            },
        ]

        for table_info in tables_to_check:
            table_name = table_info["name"]
            try:
                # Check if table exists
                check_table_sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = $1
                AND table_schema = 'public'
                """
                params = {"table_name": table_name.lower()}
                table_exists = await self.query(check_table_sql, list(params.values()))

                if not table_exists:
                    logger.info(f"Creating table {table_name}")
                    await self.execute(table_info["ddl"])
                    logger.info(
                        f"Successfully created {table_info['description']}: {table_name}"
                    )

                    # Create basic indexes for the new table
                    try:
                        # Create index for id column
                        index_name = f"idx_{table_name.lower()}_id"
                        create_index_sql = (
                            f"CREATE INDEX {index_name} ON {table_name}(id)"
                        )
                        await self.execute(create_index_sql)
                        logger.info(f"Created index {index_name} on table {table_name}")

                        # Create composite index for (workspace, id) columns
                        composite_index_name = f"idx_{table_name.lower()}_workspace_id"
                        create_composite_index_sql = f"CREATE INDEX {composite_index_name} ON {table_name}(workspace, id)"
                        await self.execute(create_composite_index_sql)
                        logger.info(
                            f"Created composite index {composite_index_name} on table {table_name}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to create indexes for table {table_name}: {e}"
                        )

                else:
                    logger.debug(f"Table {table_name} already exists")

            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")

    async def _create_pagination_indexes(self):
        """Create indexes to optimize pagination queries for LIGHTRAG_DOC_STATUS"""
        indexes = [
            {
                "name": "idx_lightrag_doc_status_workspace_status_updated_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_status_updated_at ON LIGHTRAG_DOC_STATUS (workspace, status, updated_at DESC)",
                "description": "Composite index for workspace + status + updated_at pagination",
            },
            {
                "name": "idx_lightrag_doc_status_workspace_status_created_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_status_created_at ON LIGHTRAG_DOC_STATUS (workspace, status, created_at DESC)",
                "description": "Composite index for workspace + status + created_at pagination",
            },
            {
                "name": "idx_lightrag_doc_status_workspace_updated_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_updated_at ON LIGHTRAG_DOC_STATUS (workspace, updated_at DESC)",
                "description": "Index for workspace + updated_at pagination (all statuses)",
            },
            {
                "name": "idx_lightrag_doc_status_workspace_created_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_created_at ON LIGHTRAG_DOC_STATUS (workspace, created_at DESC)",
                "description": "Index for workspace + created_at pagination (all statuses)",
            },
            {
                "name": "idx_lightrag_doc_status_workspace_id",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_id ON LIGHTRAG_DOC_STATUS (workspace, id)",
                "description": "Index for workspace + id sorting",
            },
            {
                "name": "idx_lightrag_doc_status_workspace_file_path",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_file_path ON LIGHTRAG_DOC_STATUS (workspace, file_path)",
                "description": "Index for workspace + file_path sorting",
            },
        ]

        for index in indexes:
            try:
                # Check if index already exists
                check_sql = """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'lightrag_doc_status'
                AND indexname = $1
                """

                params = {"indexname": index["name"]}
                existing = await self.query(check_sql, list(params.values()))

                if not existing:
                    logger.info(f"Creating pagination index: {index['description']}")
                    await self.execute(index["sql"])
                    logger.info(f"Successfully created index: {index['name']}")
                else:
                    logger.debug(f"Index already exists: {index['name']}")

            except Exception as e:
                logger.warning(f"Failed to create index {index['name']}: {e}")

    async def query(
            self,
            sql: str,
            params: list[Any] | None = None,
            multirows: bool = False,
            with_age: bool = False,
            graph_name: str | None = None,
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
                logger.error(f"AnalyticDB MySQL, error:{e}")
                raise

    async def execute(
            self,
            sql: str,
            data: dict[str, Any] | None = None,
            upsert: bool = False,
            ignore_if_exists: bool = False,
            with_age: bool = False,
            graph_name: str | None = None,
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
            "workspace": os.environ.get("ADB_WORKSPACE", "graphrag"),
            "maxsize": os.environ.get("ADB_MAX_CONNECTIONS", "20"),
            "autocommit": "True",
        }

    @classmethod
    async def get_client(cls) -> ADBMYSQL:
        async with cls._lock:
            if cls._instances["db"] is None:
                config = ClientManager.get_config()
                db = ADBMYSQL(config)
                await db.initdb()
                await db.check_tables()
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: ADBMYSQL):
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
class PGKVStorage(BaseKVStorage):
    db: ADBMYSQL = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

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
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    ################ QUERY METHODS ################
    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for get_all: {self.namespace}"
            )
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
            logger.error(
                f"[{self.workspace}] Error retrieving all data from {self.namespace}: {e}"
            )
            return {}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get data by id."""
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

        return response if response else None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get data by ids"""
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
        """Filter out duplicated content"""
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
            logger.error(
                f"[{self.workspace}] AnalyticDB MySQL,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
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
                    "create_time": current_time,
                    "update_time": current_time,
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
                    "queryparam": json.dumps(v.get("queryparam"))
                    if v.get("queryparam")
                    else None,
                }

                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_full_entities"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "entity_names": json.dumps(v["entity_names"]),
                    "count": v["count"],
                    "create_time": current_time,
                    "update_time": current_time,
                }
                await self.db.execute(upsert_sql, _data)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_full_relations"]
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "relation_pairs": json.dumps(v["relation_pairs"]),
                    "count": v["count"],
                    "create_time": current_time,
                    "update_time": current_time,
                }
                await self.db.execute(upsert_sql, _data)

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                    table_name=table_name
                )
                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


@final
@dataclass
class PGVectorStorage(BaseVectorStorage):
    db: ADBMYSQL | None = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
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
                self.db = await ClientManager.get_client()

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
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    def _upsert_chunks(
            self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, dict[str, Any]]:
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
                "create_time": current_time,
                "update_time": current_time,
            }
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error to prepare upsert,\nsql: {e}\nitem: {item}"
            )
            raise

        return upsert_sql, data

    def _upsert_entities(
            self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, dict[str, Any]]:
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
            "create_time": current_time,
            "update_time": current_time,
        }
        return upsert_sql, data

    def _upsert_relationships(
            self, item: dict[str, Any], current_time: datetime.datetime
    ) -> tuple[str, dict[str, Any]]:
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
            "create_time": current_time,
            "update_time": current_time,
        }
        return upsert_sql, data

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # Get current UTC time and convert to naive datetime for database storage
        current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
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
                upsert_sql, data = self._upsert_chunks(item, current_time)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                upsert_sql, data = self._upsert_entities(item, current_time)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                upsert_sql, data = self._upsert_relationships(item, current_time)
            else:
                raise ValueError(f"{self.namespace} is not supported")

            await self.db.execute(upsert_sql, data)

    #################### query method ###############
    async def query(
            self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query
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
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs from the storage.

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for vector deletion: {self.namespace}"
            )
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name from the vector storage.

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Construct SQL to delete the entity
            delete_sql = """DELETE FROM LIGHTRAG_VDB_ENTITY
                            WHERE workspace=$1 AND entity_name=$2"""

            await self.db.execute(
                delete_sql, {"workspace": self.workspace, "entity_name": entity_name}
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted entity {entity_name}"
            )
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity.

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_VDB_RELATION
                            WHERE workspace=$1 AND (source_id=$2 OR target_id=$2)"""

            await self.db.execute(
                delete_sql, {"workspace": self.workspace, "entity_name": entity_name}
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted relations for entity {entity_name}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for entity {entity_name}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for ID lookup: {self.namespace}"
            )
            return None

        query = f"SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at FROM {table_name} WHERE workspace=$1 AND id=$2"
        params = {"workspace": self.workspace, "id": id}

        try:
            result = await self.db.query(query, list(params.values()))
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for IDs lookup: {self.namespace}"
            )
            return []

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = f"SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at FROM {table_name} WHERE workspace=$1 AND id IN ({ids_str})"
        params = {"workspace": self.workspace}

        try:
            results = await self.db.query(query, list(params.values()), multirows=True)
            return [dict(record) for record in results]
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for vector lookup: {self.namespace}"
            )
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
        """Drop the storage"""
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                    table_name=table_name
                )
                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


@final
@dataclass
class PGDocStatusStorage(DocStatusStorage):
    db: ADBMYSQL = field(default=None)

    def _format_datetime_with_timezone(self, dt):
        """Convert datetime to ISO format string with timezone info"""
        if dt is None:
            return None
        # If no timezone info, assume it's UTC time (as stored in database)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # If datetime already has timezone info, keep it as is
        return dt.isoformat()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

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
        async with get_storage_lock():
            if self.db is not None:
                await ClientManager.release_client(self.db)
                self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
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
            # print(f"keys: {keys}")
            # print(f"new_keys: {new_keys}")
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] AnalyticDB MySQL,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
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

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(result[0]["created_at"])
            updated_at = self._format_datetime_with_timezone(result[0]["updated_at"])

            return dict(
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=result[0]["file_path"],
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=result[0].get("error_msg"),
                track_id=result[0].get("track_id"),
            )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by multiple IDs."""
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

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(row["created_at"])
            updated_at = self._format_datetime_with_timezone(row["updated_at"])

            processed_results.append(
                {
                    "content_length": row["content_length"],
                    "content_summary": row["content_summary"],
                    "status": row["status"],
                    "chunks_count": row["chunks_count"],
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "file_path": row["file_path"],
                    "chunks_list": chunks_list,
                    "metadata": metadata,
                    "error_msg": row.get("error_msg"),
                    "track_id": row.get("track_id"),
                }
            )

        return processed_results

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  where workspace=$1 GROUP BY STATUS
                 """
        params = {"workspace": self.workspace}
        result = await self.db.query(sql, list(params.values()), True)
        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(
            self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """all documents with a specific status"""
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

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["created_at"])
            updated_at = self._format_datetime_with_timezone(element["updated_at"])

            docs_by_status[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=element.get("error_msg"),
                track_id=element.get("track_id"),
            )

        return docs_by_status

    async def get_docs_by_track_id(
            self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
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

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["created_at"])
            updated_at = self._format_datetime_with_timezone(element["updated_at"])

            docs_by_track_id[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
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

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["created_at"])
            updated_at = self._format_datetime_with_timezone(element["updated_at"])

            doc_status = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
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
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        sql = """
            SELECT status, COUNT(*) as count
            FROM LIGHTRAG_DOC_STATUS
            WHERE workspace=$1
            GROUP BY status
        """
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
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
            )
            return

        delete_sql = f"DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids})
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}"
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        def parse_datetime(dt_str):
            """Parse datetime and ensure it's stored as UTC time in database"""
            if dt_str is None:
                return None
            if isinstance(dt_str, (datetime.date, datetime.datetime)):
                # If it's a datetime object
                if isinstance(dt_str, datetime.datetime):
                    # If no timezone info, assume it's UTC
                    if dt_str.tzinfo is None:
                        dt_str = dt_str.replace(tzinfo=timezone.utc)
                    # Convert to UTC and remove timezone info for storage
                    return dt_str.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_str
            try:
                # Process ISO format string with timezone
                dt = datetime.datetime.fromisoformat(dt_str)
                # If no timezone info, assume it's UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # Convert to UTC and remove timezone info for storage
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            except (ValueError, TypeError):
                logger.warning(
                    f"[{self.workspace}] Unable to parse datetime string: {dt_str}"
                )
                return None

        # Modified SQL to include created_at, updated_at, chunks_list, track_id, metadata, and error_msg in both INSERT and UPDATE operations
        # All fields are updated from the input data in both INSERT and UPDATE cases
        sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content_summary,content_length,chunks_count,status,file_path,chunks_list,track_id,metadata,error_msg,created_at,updated_at)
                 values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                  on conflict(id,workspace) do update set
                  content_summary = EXCLUDED.content_summary,
                  content_length = EXCLUDED.content_length,
                  chunks_count = EXCLUDED.chunks_count,
                  status = EXCLUDED.status,
                  file_path = EXCLUDED.file_path,
                  chunks_list = EXCLUDED.chunks_list,
                  track_id = EXCLUDED.track_id,
                  metadata = EXCLUDED.metadata,
                  error_msg = EXCLUDED.error_msg,
                  created_at = EXCLUDED.created_at,
                  updated_at = EXCLUDED.updated_at"""
        for k, v in data.items():
            # Remove timezone information, store utc time in db
            created_at = parse_datetime(v.get("created_at"))
            updated_at = parse_datetime(v.get("updated_at"))

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
                    "created_at": created_at,  # Use the converted datetime object
                    "updated_at": updated_at,  # Use the converted datetime object
                },
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        async with get_storage_lock():
            try:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    return {
                        "status": "error",
                        "message": f"Unknown namespace: {self.namespace}",
                    }

                drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                    table_name=table_name
                )
                await self.db.execute(drop_sql, {"workspace": self.workspace})
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                return {"status": "error", "message": str(e)}


# Note: Order matters! More specific namespaces (e.g., "full_entities") must come before
# more general ones (e.g., "entities") because is_namespace() uses endswith() matching
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
                    meta JSONB,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
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
                    llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
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
                    content_vector VECTOR({os.environ.get("EMBEDDING_DIM", 1024)}),
                    file_path TEXT NULL,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({os.environ.get("EMBEDDING_DIM", 1024)}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": f"""CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({os.environ.get("EMBEDDING_DIM", 1024)}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
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
                    queryparam JSONB NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length int4 NULL,
	               chunks_count int4 NULL,
	               status varchar(64) NULL,
	               file_path TEXT NULL,
	               chunks_list JSONB NULL DEFAULT '[]'::jsonb,
	               track_id varchar(255) NULL,
	               metadata JSONB NULL DEFAULT '{}'::jsonb,
	               error_msg TEXT NULL,
	               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
	              )"""
    },
    "LIGHTRAG_FULL_ENTITIES": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_ENTITIES (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_names JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_FULL_ENTITIES_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_FULL_RELATIONS": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_RELATIONS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    relation_pairs JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_FULL_RELATIONS_PK PRIMARY KEY (workspace, id)
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
                                COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id IN ({ids})
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                  EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                  EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                   FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_id_full_entities": """SELECT id, entity_names, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=$1 AND id=$2
                               """,
    "get_by_id_full_relations": """SELECT id, relation_pairs, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_entities": """SELECT id, entity_names, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_ids_full_relations": """SELECT id, relation_pairs, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=$1 AND id IN ({ids})
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})",
    "upsert_doc_full": """INSERT INTO LIGHTRAG_DOC_FULL (id, content, workspace)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = $2, update_time = CURRENT_TIMESTAMP
                       """,
    "upsert_llm_response_cache": """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,chunk_id,cache_type,queryparam)
                                      VALUES ($1, $2, $3, $4, $5, $6, $7)
                                      ON CONFLICT (workspace,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      chunk_id=EXCLUDED.chunk_id,
                                      cache_type=EXCLUDED.cache_type,
                                      queryparam=EXCLUDED.queryparam,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    "upsert_text_chunk": """INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path, llm_cache_list,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      file_path=EXCLUDED.file_path,
                      llm_cache_list=EXCLUDED.llm_cache_list,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_full_entities": """INSERT INTO LIGHTRAG_FULL_ENTITIES (workspace, id, entity_names, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_names=EXCLUDED.entity_names,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_full_relations": """INSERT INTO LIGHTRAG_FULL_RELATIONS (workspace, id, relation_pairs, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET relation_pairs=EXCLUDED.relation_pairs,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    # SQL for VectorStorage
    "upsert_chunk": """INSERT INTO LIGHTRAG_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_entity": """INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7, $8, $9)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time=EXCLUDED.update_time
                     """,
    "upsert_relationship": """INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    "relationships": """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            EXTRACT(EPOCH FROM r.create_time)::BIGINT AS created_at
                     FROM LIGHTRAG_VDB_RELATION r
                     WHERE r.workspace = $1
                       AND r.content_vector <=> '[{embedding_string}]'::vector < $2
                     ORDER BY r.content_vector <=> '[{embedding_string}]'::vector
                     LIMIT $3;
                     """,
    "entities": """
                SELECT e.entity_name,
                       EXTRACT(EPOCH FROM e.create_time)::BIGINT AS created_at
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = $1
                  AND e.content_vector <=> '[{embedding_string}]'::vector < $2
                ORDER BY e.content_vector <=> '[{embedding_string}]'::vector
                LIMIT $3;
                """,
    "chunks": """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     EXTRACT(EPOCH FROM c.create_time)::BIGINT AS created_at
              FROM LIGHTRAG_VDB_CHUNKS c
              WHERE c.workspace = $1
                AND c.content_vector <=> '[{embedding_string}]'::vector < $2
              ORDER BY c.content_vector <=> '[{embedding_string}]'::vector
              LIMIT $3;
              """,
    # DROP tables
    "drop_specifiy_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=$1
       """,
}
