"""FastAPI API routes, SSE streaming, rate limiting, and background job queue."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.db.repository import Repository
from app.graph.state import ResearchState
from app.graph.build import build_research_graph
from app.clients.llm import LLMClient

logger = logging.getLogger("researchos.api")

router = APIRouter(prefix="/api", tags=["Research"])

# Job queue and concurrency structures
_job_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.MAX_QUEUE_SIZE)
_worker_task: Optional[asyncio.Task] = None
_rate_limits: Dict[str, List[float]] = {}  # IP -> list of request timestamps (last hour)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500, description="Scientific research question")
    max_papers: Optional[int] = Field(default=None, ge=1, le=10, description="Max papers to retrieve")
    depth: Optional[str] = Field(default="standard", pattern="^(quick|standard)$")


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    message: str


def check_rate_limit(client_ip: str) -> None:
    """Enforces 5 runs per hour per IP."""
    now = time.time()
    one_hour_ago = now - 3600.0

    timestamps = _rate_limits.get(client_ip, [])
    # Filter to timestamps in last hour
    timestamps = [t for t in timestamps if t > one_hour_ago]
    if len(timestamps) >= settings.RATE_LIMIT_RUNS_PER_HOUR:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: maximum {settings.RATE_LIMIT_RUNS_PER_HOUR} research runs per hour allowed.",
        )
    timestamps.append(now)
    _rate_limits[client_ip] = timestamps


def get_remaining_runs(client_ip: str) -> int:
    """Returns number of remaining runs allowed for this IP in current hour."""
    now = time.time()
    one_hour_ago = now - 3600.0
    timestamps = [t for t in _rate_limits.get(client_ip, []) if t > one_hour_ago]
    return max(0, settings.RATE_LIMIT_RUNS_PER_HOUR - len(timestamps))


async def _run_job(job_data: Dict[str, Any]) -> None:
    """Executes a single research job through the LangGraph pipeline."""
    run_id = job_data["run_id"]
    question = job_data["question"]
    depth = job_data["depth"]
    max_papers = job_data["max_papers"]

    logger.info(f"Starting research run {run_id}: '{question[:50]}...'")
    await Repository.update_run(run_id, status="running")

    # Construct shared LLM and graph
    llm_client = LLMClient()
    graph = build_research_graph(llm_client=llm_client)

    initial_state = ResearchState(
        run_id=run_id,
        question=question,
        depth=depth,
        max_papers=max_papers,
        collection_name=f"papers_{run_id.replace('-', '_')}",
    )

    t0 = time.time()
    try:
        final_state_dict = await graph.ainvoke(initial_state)

        # Update run in repository
        summary = final_state_dict.get("synthesis", {}).get("markdown", "")
        references = final_state_dict.get("synthesis", {}).get("reference_list", [])
        claims = final_state_dict.get("claims", [])
        comparison = final_state_dict.get("comparison", {})
        eval_scores = final_state_dict.get("eval_scores", {})
        trace = final_state_dict.get("trace", [])
        timings = final_state_dict.get("timings", {})
        call_counts = {
            "llm_calls": llm_client.budget_tracker.call_count,
            "cached_calls": llm_client.budget_tracker.cached_count,
        }

        await Repository.update_run(
            run_id=run_id,
            status="completed",
            summary_md=summary,
            reference_list=references,
            claims=claims,
            comparison=comparison,
            eval_scores=eval_scores,
            trace=trace,
            timings=timings,
            call_counts=call_counts,
        )
        await Repository.add_event(run_id, "run_completed", {"run_id": run_id, "duration": round(time.time() - t0, 2)})
        logger.info(f"Completed research run {run_id} in {time.time() - t0:.2f}s.")
    except Exception as e:
        logger.error(f"Fatal error during research run {run_id}: {e}", exc_info=True)
        await Repository.update_run(
            run_id=run_id,
            status="failed",
            error_message=str(e),
        )
        await Repository.add_event(run_id, "run_failed", {"run_id": run_id, "error": str(e)})


async def _worker_loop() -> None:
    """Single-worker background loop: processes one research job at a time."""
    while True:
        job = await _job_queue.get()
        try:
            await _run_job(job)
        except Exception as e:
            logger.error(f"Worker exception: {e}")
        finally:
            _job_queue.task_done()


def ensure_worker_started() -> None:
    """Starts background worker if not already running."""
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())


@router.post("/research", response_model=ResearchResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research(
    req: ResearchRequest,
    request: Request,
    x_access_code: Optional[str] = Header(None, alias="X-Access-Code"),
):
    """Initiates an autonomous multi-agent research run in background queue."""
    # 1. Access code security check (if configured)
    if settings.ACCESS_CODE and x_access_code != settings.ACCESS_CODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Access-Code header.",
        )

    # 2. Rate limit check (5 runs per hour per IP)
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)

    # 3. Concurrency / queue limit check (max 3 pending)
    if _job_queue.full():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Research queue is full (max 3 concurrent queued jobs). Please try again shortly.",
        )

    run_id = str(uuid.uuid4())
    max_papers = req.max_papers or settings.MAX_PAPERS_DEFAULT
    depth = req.depth or "standard"

    # Persist initial record
    await Repository.create_run(
        run_id=run_id,
        question=req.question,
        depth=depth,
        max_papers=max_papers,
    )

    # Enqueue job
    ensure_worker_started()
    await _job_queue.put({
        "run_id": run_id,
        "question": req.question,
        "depth": depth,
        "max_papers": max_papers,
    })

    return ResearchResponse(
        run_id=run_id,
        status="pending",
        message="Research job queued successfully.",
    )


@router.get("/research/{run_id}/events")
async def stream_events(run_id: str):
    """Server-Sent Events (SSE) endpoint streaming real-time agent progression."""
    run = await Repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    async def event_generator():
        last_id = 0
        poll_interval = 0.5
        idle_cycles = 0

        while True:
            events = await Repository.get_events(run_id, after_id=last_id)
            for ev in events:
                last_id = ev["id"]
                event_name = ev["event_type"]
                event_payload = json.dumps(ev["data"])
                yield f"event: {event_name}\ndata: {event_payload}\n\n"

            # Check if run reached final state
            current_run = await Repository.get_run(run_id)
            run_status = current_run.get("status") if current_run else "unknown"

            if run_status in ("completed", "failed"):
                # Deliver any final remaining events
                final_events = await Repository.get_events(run_id, after_id=last_id)
                for ev in final_events:
                    last_id = ev["id"]
                    yield f"event: {ev['event_type']}\ndata: {json.dumps(ev['data'])}\n\n"
                yield f"event: close\ndata: {json.dumps({'status': run_status})}\n\n"
                break

            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/research/{run_id}")
async def get_research_run(run_id: str):
    """Retrieves full run state, synthesized markdown report, references, claims, trace, and evaluations."""
    run = await Repository.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@router.get("/research")
async def list_research_runs(limit: int = 10, cursor: Optional[str] = None):
    """Lists recent research runs with pagination."""
    return await Repository.list_runs(limit=limit, cursor=cursor)


@router.get("/limits")
async def get_user_limits(request: Request):
    """Returns remaining research runs allowed under per-IP rate limit."""
    client_ip = request.client.host if request.client else "unknown"
    remaining = get_remaining_runs(client_ip)
    return {
        "remaining_runs": remaining,
        "max_per_hour": settings.RATE_LIMIT_RUNS_PER_HOUR,
        "requires_access_code": bool(settings.ACCESS_CODE),
    }


@router.get("/eval/latest")
async def get_latest_eval():
    """Retrieves the latest retrieval ablation and factual consistency evaluation report."""
    report = await Repository.get_latest_eval_report()
    if not report:
        return {
            "status": "no_report",
            "message": "No evaluation has been recorded yet.",
        }
    return report
