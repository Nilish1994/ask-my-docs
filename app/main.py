"""HTTP API: /ask (SSE), /search, /collections, /health.

Run locally:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import anthropic
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from answer.claude import AnswerConfig, stream_answer
from retrieval.embeddings import Embedder, FastEmbedEmbedder
from retrieval.index import Retriever

from .config import Settings
from .guardrails import DailyCap, SlidingWindowLimiter, client_ip
from .uploads import CollectionStore, UploadError

log = logging.getLogger("ask_my_docs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Mode = Literal["bm25", "vector", "hybrid"]
PUBLIC_CHUNK_FIELDS = (
    "id",
    "chunk_id",
    "source",
    "page",
    "page_title",
    "heading_path",
    "url",
    "text",
    "rank",
    "scores",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Mode = "hybrid"
    collection_id: str | None = None


def public_hit(hit: dict) -> dict:
    return {key: hit.get(key) for key in PUBLIC_CHUNK_FIELDS}


def sse(event: str, data: dict) -> str:
    """One Server-Sent Event; the payload repeats the event name as ``type``."""
    payload = {"type": event, **data}
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    embedder: Embedder | None = None,
    anthropic_client: anthropic.AsyncAnthropic | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal retriever, embedder, anthropic_client
        embedder = embedder or FastEmbedEmbedder()
        if retriever is None:
            if not Retriever.exists(settings.index_dir):
                raise RuntimeError(f"no index at {settings.index_dir}; run `python -m ingest.build_index`")
            retriever = Retriever.load(settings.index_dir, embedder, stemming=settings.bm25_stemming)
        if anthropic_client is None and settings.answering_enabled:
            anthropic_client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value(), max_retries=2, timeout=120.0
            )
        app.state.retriever = retriever
        app.state.anthropic = anthropic_client
        app.state.collections = CollectionStore(
            embedder,
            ttl=settings.upload_ttl_seconds,
            max_collections=settings.max_collections,
            max_files=settings.upload_max_files,
            max_file_bytes=settings.upload_max_file_bytes,
            max_chunks=settings.upload_max_chunks,
            stemming=settings.bm25_stemming,
        )
        sweeper = asyncio.create_task(_sweep_forever(app))
        log.info(
            "ready: %d chunks, answering %s",
            len(retriever.chunks),
            "enabled" if anthropic_client else "disabled (no ANTHROPIC_API_KEY)",
        )
        try:
            yield
        finally:
            sweeper.cancel()
            if anthropic_client is not None:
                await anthropic_client.close()

    async def _sweep_forever(app: FastAPI) -> None:
        while True:
            await asyncio.sleep(60)
            removed = app.state.collections.sweep()
            for limiter in limiters.values():
                limiter.sweep()
            if removed:
                log.info("expired %d uploaded collection(s)", removed)

    app = FastAPI(title="ask-my-docs", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    limiters = {
        "ask": SlidingWindowLimiter(settings.ask_rate_limit_per_hour, 3600),
        "search": SlidingWindowLimiter(settings.search_rate_limit_per_minute, 60),
        "upload": SlidingWindowLimiter(settings.upload_rate_limit_per_hour, 3600),
    }
    daily_cap = DailyCap(settings.daily_question_cap)
    upload_lock = asyncio.Semaphore(1)  # one embedding job at a time on a small VPS
    app.state.limiters = limiters
    app.state.daily_cap = daily_cap

    def ip_of(request: Request) -> str:
        return client_ip(request, settings.forwarded_hops)

    def enforce(name: str, request: Request) -> None:
        wait = limiters[name].retry_after(ip_of(request))
        if wait:
            raise HTTPException(
                429, f"rate limit reached; try again in {wait} seconds", headers={"Retry-After": str(wait)}
            )
        limiters[name].consume(ip_of(request))

    def target_retriever(request: Request, collection_id: str | None) -> Retriever:
        if not collection_id:
            return request.app.state.retriever
        collection = request.app.state.collections.get(collection_id)
        if collection is None:
            raise HTTPException(404, "collection not found or expired")
        return collection.retriever

    @app.get("/health")
    async def health(request: Request) -> dict:
        return {
            "status": "ok",
            "chunks": len(request.app.state.retriever.chunks),
            "answering_enabled": request.app.state.anthropic is not None,
            "model": settings.anthropic_model,
            "limits": {
                "questions_per_hour": settings.ask_rate_limit_per_hour,
                "questions_remaining_this_hour": limiters["ask"].remaining(ip_of(request)),
                "daily_question_cap": settings.daily_question_cap,
                "daily_questions_used": daily_cap.used,
                "max_question_chars": settings.max_question_chars,
                "upload_max_files": settings.upload_max_files,
                "upload_max_file_bytes": settings.upload_max_file_bytes,
                "upload_ttl_seconds": settings.upload_ttl_seconds,
            },
        }

    @app.get("/search")
    async def search(
        request: Request,
        q: Annotated[str, Query(min_length=1)],
        mode: Mode = "hybrid",
        top_k: Annotated[int, Query(ge=1, le=20)] = 5,
        collection_id: str | None = None,
    ) -> dict:
        """Retrieval only: no LLM call, no API cost."""
        if len(q) > settings.max_question_chars:
            raise HTTPException(400, f"query longer than {settings.max_question_chars} characters")
        enforce("search", request)
        index = target_retriever(request, collection_id)
        hits = await run_in_threadpool(index.search, q, mode, top_k)
        return {"query": q, "mode": mode, "results": [public_hit(h) for h in hits]}

    @app.post("/ask")
    async def ask(request: Request, body: AskRequest):
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "question is empty")
        if len(question) > settings.max_question_chars:
            raise HTTPException(400, f"question longer than {settings.max_question_chars} characters")
        index = target_retriever(request, body.collection_id)
        client = request.app.state.anthropic

        if client is not None:
            # Check both budgets before consuming either, so a rejected request costs nothing.
            ip = ip_of(request)
            wait = limiters["ask"].retry_after(ip)
            if wait:
                raise HTTPException(
                    429,
                    f"you have used your {settings.ask_rate_limit_per_hour} questions "
                    f"for this hour; try again in {wait // 60 + 1} minutes",
                    headers={"Retry-After": str(wait)},
                )
            if daily_cap.exhausted():
                raise HTTPException(
                    503,
                    "the demo's daily question budget is used up; it resets at 00:00 UTC. "
                    "Retrieval-only search still works.",
                )
            limiters["ask"].consume(ip)
            daily_cap.consume()

        hits = await run_in_threadpool(index.search, question, body.mode, settings.answer_top_k)
        config = AnswerConfig(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            effort=settings.anthropic_effort,
            corpus="the uploaded documents" if body.collection_id else "the FastAPI documentation",
        )

        async def events() -> AsyncIterator[str]:
            yield sse("retrieval", {"mode": body.mode, "results": [public_hit(h) for h in hits]})
            if client is None:
                yield sse(
                    "error",
                    {
                        "code": "answering_disabled",
                        "retryable": False,
                        "message": "Answering is disabled on this server (no API key configured). "
                        "The retrieved sections are shown on the right.",
                    },
                )
                return
            if not hits:
                yield sse(
                    "error",
                    {
                        "code": "no_results",
                        "retryable": False,
                        "message": "Nothing relevant was found for that question.",
                    },
                )
                return
            async for event in stream_answer(client, question, hits, config):
                yield sse(event["type"], event)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Bound the whole multipart body so nothing large is buffered, and keep
    # parts in memory: Starlette spools parts above spool_max_size to a temp
    # file, so raise that above the request cap.
    max_upload_body = settings.upload_max_files * settings.upload_max_file_bytes + 64 * 1024
    MultiPartParser.spool_max_size = max_upload_body

    @app.post("/collections", status_code=201)
    async def create_collection(request: Request) -> dict:
        """Upload up to N .md/.txt/.pdf files as multipart field ``files``."""
        length = request.headers.get("content-length")
        if length is None:
            raise HTTPException(411, "Content-Length required")
        if int(length) > max_upload_body:
            raise HTTPException(
                413,
                f"upload too large; at most {settings.upload_max_files} files of "
                f"{settings.upload_max_file_bytes // (1024 * 1024)} MB each",
            )
        enforce("upload", request)
        try:
            async with request.form(max_files=settings.upload_max_files, max_fields=10) as form:
                uploads = [item for item in form.getlist("files") if isinstance(item, StarletteUploadFile)]
                payload = [(item.filename or "upload", await item.read()) for item in uploads]
        except MultiPartException as error:
            raise HTTPException(400, str(error)) from error
        try:
            async with upload_lock:
                collection = await run_in_threadpool(request.app.state.collections.create, payload)
        except UploadError as error:
            raise HTTPException(error.status_code, str(error)) from error
        log.info("created collection with %d chunks from %d file(s)", collection.chunks, len(payload))
        return collection.info()

    @app.get("/collections/{collection_id}")
    async def get_collection(request: Request, collection_id: str) -> dict:
        collection = request.app.state.collections.get(collection_id)
        if collection is None:
            raise HTTPException(404, "collection not found or expired")
        return collection.info()

    @app.delete("/collections/{collection_id}", status_code=204)
    async def delete_collection(request: Request, collection_id: str) -> None:
        if not request.app.state.collections.delete(collection_id):
            raise HTTPException(404, "collection not found or expired")

    return app


app = create_app()
