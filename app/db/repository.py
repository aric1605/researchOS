"""Data access repository for ResearchOS runs, events, and evaluation reports.
Automatically falls back to thread-safe in-memory storage if DATABASE_URL is not configured.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.db.connection import get_db_pool, get_connection_string

logger = logging.getLogger("researchos.repository")

# In-memory storage for offline testing / fallback
_in_memory_runs: Dict[str, Dict[str, Any]] = {}
_in_memory_events: List[Dict[str, Any]] = []
_in_memory_evals: List[Dict[str, Any]] = []


class Repository:
    @staticmethod
    def _is_db_enabled() -> bool:
        return bool(get_connection_string())

    @classmethod
    async def create_run(
        cls,
        run_id: str,
        question: str,
        depth: str = "standard",
        max_papers: int = 6,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        record = {
            "run_id": run_id,
            "question": question,
            "depth": depth,
            "max_papers": max_papers,
            "status": "pending",
            "summary_md": None,
            "reference_list": [],
            "claims": [],
            "comparison": {},
            "eval_scores": {},
            "trace": [],
            "timings": {},
            "call_counts": {},
            "error_message": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            INSERT INTO runs (run_id, question, depth, max_papers, status, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (run_id, question, depth, max_papers, "pending", now, now),
                        )
                return record
            except Exception as e:
                logger.error(f"Failed to create run in Postgres: {e}")

        _in_memory_runs[run_id] = record
        return record

    @classmethod
    async def update_run(
        cls,
        run_id: str,
        status: Optional[str] = None,
        summary_md: Optional[str] = None,
        reference_list: Optional[List[Dict[str, Any]]] = None,
        claims: Optional[List[Dict[str, Any]]] = None,
        comparison: Optional[Dict[str, Any]]] = None,
        eval_scores: Optional[Dict[str, Any]]] = None,
        trace: Optional[List[Dict[str, Any]]] = None,
        timings: Optional[Dict[str, Any]]] = None,
        call_counts: Optional[Dict[str, Any]]] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        updates = {"updated_at": now.isoformat()}
        if status is not None:
            updates["status"] = status
        if summary_md is not None:
            updates["summary_md"] = summary_md
        if reference_list is not None:
            updates["reference_list"] = reference_list
        if claims is not None:
            updates["claims"] = claims
        if comparison is not None:
            updates["comparison"] = comparison
        if eval_scores is not None:
            updates["eval_scores"] = eval_scores
        if trace is not None:
            updates["trace"] = trace
        if timings is not None:
            updates["timings"] = timings
        if call_counts is not None:
            updates["call_counts"] = call_counts
        if error_message is not None:
            updates["error_message"] = error_message

        pool = await get_db_pool()
        if pool:
            try:
                set_clauses = ["updated_at = %s"]
                params: List[Any] = [now]
                for k, v in updates.items():
                    if k == "updated_at":
                        continue
                    set_clauses.append(f"{k} = %s")
                    if isinstance(v, (dict, list)):
                        params.append(json.dumps(v))
                    else:
                        params.append(v)
                params.append(run_id)

                sql = f"UPDATE runs SET {', '.join(set_clauses)} WHERE run_id = %s RETURNING *;"
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(sql, tuple(params))
                        row = await cur.fetchone()
                        if row:
                            return dict(row)
            except Exception as e:
                logger.error(f"Failed to update run in Postgres: {e}")

        # In-memory fallback
        if run_id in _in_memory_runs:
            _in_memory_runs[run_id].update(updates)
            return _in_memory_runs[run_id]
        return None

    @classmethod
    async def get_run(cls, run_id: str) -> Optional[Dict[str, Any]]:
        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("SELECT * FROM runs WHERE run_id = %s;", (run_id,))
                        row = await cur.fetchone()
                        if row:
                            return dict(row)
            except Exception as e:
                logger.error(f"Failed to get run from Postgres: {e}")

        return _in_memory_runs.get(run_id)

    @classmethod
    async def list_runs(cls, limit: int = 10, cursor: Optional[str] = None) -> List[Dict[str, Any]]:
        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        if cursor:
                            await cur.execute(
                                """
                                SELECT * FROM runs
                                WHERE created_at < %s
                                ORDER BY created_at DESC
                                LIMIT %s;
                                """,
                                (cursor, limit),
                            )
                        else:
                            await cur.execute(
                                "SELECT * FROM runs ORDER BY created_at DESC LIMIT %s;",
                                (limit,),
                            )
                        rows = await cur.fetchall()
                        return [dict(r) for r in rows]
            except Exception as e:
                logger.error(f"Failed to list runs from Postgres: {e}")

        # In-memory fallback
        runs = sorted(_in_memory_runs.values(), key=lambda r: r.get("created_at", ""), reverse=True)
        if cursor:
            runs = [r for r in runs if r.get("created_at", "") < cursor]
        return runs[:limit]

    @classmethod
    async def add_event(cls, run_id: str, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        record = {
            "run_id": run_id,
            "event_type": event_type,
            "data": data,
            "created_at": now.isoformat(),
        }

        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            INSERT INTO events (run_id, event_type, data, created_at)
                            VALUES (%s, %s, %s, %s)
                            RETURNING id, created_at;
                            """,
                            (run_id, event_type, json.dumps(data), now),
                        )
                        row = await cur.fetchone()
                        if row:
                            record["id"] = row["id"]
                            record["created_at"] = str(row["created_at"])
                            return record
            except Exception as e:
                logger.error(f"Failed to add event to Postgres: {e}")

        record["id"] = len(_in_memory_events) + 1
        _in_memory_events.append(record)
        return record

    @classmethod
    async def get_events(cls, run_id: str, after_id: int = 0) -> List[Dict[str, Any]]:
        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT id, run_id, event_type, data, created_at
                            FROM events
                            WHERE run_id = %s AND id > %s
                            ORDER BY id ASC;
                            """,
                            (run_id, after_id),
                        )
                        rows = await cur.fetchall()
                        return [dict(r) for r in rows]
            except Exception as e:
                logger.error(f"Failed to get events from Postgres: {e}")

        return [e for e in _in_memory_events if e["run_id"] == run_id and e.get("id", 0) > after_id]

    @classmethod
    async def save_eval_report(
        cls,
        retrieval_ablation: Dict[str, Any],
        factual_consistency: Dict[str, Any],
        detailed_results: List[Dict[str, Any]],
        dataset_name: str = "default_8_qa",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        record = {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "retrieval_ablation": retrieval_ablation,
            "factual_consistency": factual_consistency,
            "detailed_results": detailed_results,
            "created_at": now.isoformat(),
        }

        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            INSERT INTO eval_reports (run_id, dataset_name, retrieval_ablation, factual_consistency, detailed_results, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id, created_at;
                            """,
                            (
                                run_id,
                                dataset_name,
                                json.dumps(retrieval_ablation),
                                json.dumps(factual_consistency),
                                json.dumps(detailed_results),
                                now,
                            ),
                        )
                        row = await cur.fetchone()
                        if row:
                            record["id"] = row["id"]
                            return record
            except Exception as e:
                logger.error(f"Failed to save eval report to Postgres: {e}")

        record["id"] = len(_in_memory_evals) + 1
        _in_memory_evals.append(record)
        return record

    @classmethod
    async def get_latest_eval_report(cls) -> Optional[Dict[str, Any]]:
        pool = await get_db_pool()
        if pool:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT * FROM eval_reports ORDER BY created_at DESC LIMIT 1;"
                        )
                        row = await cur.fetchone()
                        if row:
                            return dict(row)
            except Exception as e:
                logger.error(f"Failed to get eval report from Postgres: {e}")

        if _in_memory_evals:
            return _in_memory_evals[-1]
        return None
