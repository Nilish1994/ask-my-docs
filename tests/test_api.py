"""API tests. Claude is replaced by FakeAnthropic: no network, no cost."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import (
    FakeAnthropic,
    citation_delta,
    ev,
    final_message,
    text_block_start,
    text_delta,
)


def make_client(retriever, embedder, fake=None, **overrides) -> TestClient:
    overrides.setdefault("anthropic_api_key", None)
    settings = Settings(_env_file=None, **overrides)
    app = create_app(settings, retriever=retriever, embedder=embedder, anthropic_client=fake)
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def ask(client: TestClient, question: str = "How do I enable CORS for my frontend?", **body):
    return client.post("/ask", json={"question": question, **body})


def test_health_reports_limits(retriever, embedder):
    with make_client(retriever, embedder, ask_rate_limit_per_hour=7) as client:
        data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["answering_enabled"] is False
    assert data["limits"]["questions_per_hour"] == 7


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_search_returns_scored_results_without_llm(retriever, embedder, mode):
    fake = FakeAnthropic()
    with make_client(retriever, embedder, fake) as client:
        response = client.get("/search", params={"q": "CORS middleware origins", "mode": mode, "top_k": 2})
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["page"] == "tutorial/cors"
    assert results[0]["url"].startswith("https://fastapi.tiangolo.com/tutorial/cors/")
    assert "fused" in results[0]["scores"]
    assert fake.calls == []


def test_search_rejects_bad_mode(retriever, embedder):
    with make_client(retriever, embedder) as client:
        assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


def test_ask_streams_retrieval_text_citations_and_done(retriever, embedder):
    fake = FakeAnthropic(
        events=[
            ev("message_start"),
            text_block_start(),
            text_delta("Add "),
            citation_delta(0, start=0, cited_text="CORS"),
            text_delta("CORSMiddleware."),
            ev("content_block_stop"),
        ]
    )
    with make_client(retriever, embedder, fake) as client:
        response = ask(client)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds == ["retrieval", "segment", "text", "citation", "text", "done"]
    # Clients switch on the payload's own "type", so every event must carry it.
    assert all(data["type"] == kind for kind, data in events)

    retrieval = events[0][1]["results"]
    citation = events[3][1]
    assert citation["chunk_id"] == retrieval[0]["chunk_id"]
    assert citation["url"].startswith("https://fastapi.tiangolo.com/")
    assert events[-1][1]["usage"] == {"input_tokens": 1200, "output_tokens": 150}


def test_ask_sends_the_configured_claude_request(retriever, embedder):
    fake = FakeAnthropic(events=[text_block_start(), text_delta("ok")])
    with make_client(retriever, embedder, fake, answer_top_k=2) as client:
        ask(client)
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 4096
    assert call["output_config"] == {"effort": "low"}
    assert call["thinking"] == {"type": "adaptive"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    content = call["messages"][0]["content"]
    documents = [block for block in content if block["type"] == "document"]
    assert len(documents) == 2
    assert all(doc["citations"] == {"enabled": True} for doc in documents)
    assert content[-1] == {"type": "text", "text": "How do I enable CORS for my frontend?"}


def test_ask_maps_citation_to_most_specific_heading(retriever, embedder):
    cors_intro = next(c for c in retriever.chunks if c["page"] == "tutorial/cors")
    sub = next(s for s in cors_intro["sections"] if s["anchor"] == "use-corsmiddleware")
    fake = FakeAnthropic(events=[text_block_start(), citation_delta(0, start=sub["start"] + 5)])
    with make_client(retriever, embedder, fake) as client:
        events = parse_sse(ask(client, mode="bm25").text)
    citation = next(data for kind, data in events if kind == "citation")
    assert citation["url"] == "https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware"
    assert citation["heading"] == "Use CORSMiddleware"


def test_ask_reports_server_side_fallback(retriever, embedder):
    fallback = SimpleNamespace(
        type="fallback", from_=SimpleNamespace(model="claude-opus-5-5"), to=SimpleNamespace(model="claude-opus-5")
    )
    fake = FakeAnthropic(
        events=[ev("content_block_start", content_block=fallback), text_block_start(), text_delta("Answer.")],
        message=final_message(model="claude-opus-5"),
    )
    with make_client(retriever, embedder, fake) as client:
        events = parse_sse(ask(client).text)
    assert ("fallback", {"type": "fallback", "from": "claude-opus-5-5", "to": "claude-opus-5"}) in events
    assert events[-1][1]["model"] == "claude-opus-5"


def test_ask_handles_refusal(retriever, embedder):
    fake = FakeAnthropic(
        events=[text_block_start(), text_delta("partial")],
        message=final_message(stop_reason="refusal", category="cyber"),
    )
    with make_client(retriever, embedder, fake) as client:
        events = parse_sse(ask(client).text)
    kind, data = events[-1]
    assert kind == "refusal"
    assert data["category"] == "cyber"
    assert "discarded" in data["message"]


def test_ask_flags_truncated_answers(retriever, embedder):
    fake = FakeAnthropic(events=[text_block_start(), text_delta("long")], message=final_message("max_tokens"))
    with make_client(retriever, embedder, fake) as client:
        events = parse_sse(ask(client).text)
    assert events[-1][1]["truncated"] is True


def _status_error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("upstream", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (_status_error(anthropic.RateLimitError, 429), "upstream_rate_limited", True),
        (_status_error(anthropic.InternalServerError, 500), "upstream_error", True),
        (_status_error(anthropic.BadRequestError, 400), "upstream_error", False),
        (
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
            "upstream_unreachable",
            True,
        ),
    ],
)
def test_ask_turns_sdk_errors_into_error_events(retriever, embedder, error, code, retryable):
    fake = FakeAnthropic(error=error)
    with make_client(retriever, embedder, fake) as client:
        events = parse_sse(ask(client).text)
    kind, data = events[-1]
    assert kind == "error"
    assert data["code"] == code
    assert data["retryable"] is retryable
    assert "upstream" not in data["message"], "raw upstream messages are not shown to users"


def test_ask_without_api_key_still_returns_retrieval(retriever, embedder):
    with make_client(retriever, embedder, fake=None) as client:
        events = parse_sse(ask(client).text)
    assert [kind for kind, _ in events] == ["retrieval", "error"]
    assert all(data["type"] == kind for kind, data in events)
    assert events[1][1]["code"] == "answering_disabled"


def test_ask_rejects_long_questions_before_calling_claude(retriever, embedder):
    fake = FakeAnthropic()
    with make_client(retriever, embedder, fake, max_question_chars=20) as client:
        response = ask(client, "x" * 21)
    assert response.status_code == 400
    assert fake.calls == []


def test_ask_per_ip_rate_limit(retriever, embedder):
    fake = FakeAnthropic(events=[text_block_start(), text_delta("ok")])
    with make_client(retriever, embedder, fake, ask_rate_limit_per_hour=2) as client:
        assert ask(client).status_code == 200
        assert ask(client).status_code == 200
        blocked = ask(client)
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0
    assert len(fake.calls) == 2


def test_ask_rate_limit_uses_forwarded_client_ip(retriever, embedder):
    fake = FakeAnthropic(events=[text_block_start(), text_delta("ok")])
    with make_client(retriever, embedder, fake, ask_rate_limit_per_hour=1, forwarded_hops=1) as client:
        first = client.post("/ask", json={"question": "cors?"}, headers={"x-forwarded-for": "203.0.113.1"})
        second = client.post("/ask", json={"question": "cors?"}, headers={"x-forwarded-for": "203.0.113.2"})
        repeat = client.post("/ask", json={"question": "cors?"}, headers={"x-forwarded-for": "203.0.113.1"})
    assert (first.status_code, second.status_code, repeat.status_code) == (200, 200, 429)


def test_ask_global_daily_cap(retriever, embedder):
    fake = FakeAnthropic(events=[text_block_start(), text_delta("ok")])
    with make_client(retriever, embedder, fake, daily_question_cap=1, forwarded_hops=1) as client:
        first = client.post("/ask", json={"question": "a"}, headers={"x-forwarded-for": "198.51.100.1"})
        assert first.status_code == 200
        capped = client.post("/ask", json={"question": "b"}, headers={"x-forwarded-for": "198.51.100.2"})
    assert capped.status_code == 503
    assert len(fake.calls) == 1


# --- uploads ---------------------------------------------------------------------------


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF with a single line of Helvetica text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_upload_markdown_txt_and_pdf_then_search_and_ask(retriever, embedder):
    files = [
        ("files", ("guide.md", b"# Deploy guide\n\n## Rollback { #rollback }\n\nRun make rollback.", "text/markdown")),
        ("files", ("notes.txt", b"The staging database is refreshed nightly.", "text/plain")),
        ("files", ("policy.pdf", minimal_pdf("Refunds are processed within ten days"), "application/pdf")),
    ]
    fake = FakeAnthropic(events=[text_block_start(), text_delta("Use make rollback."), citation_delta(0, 0)])
    with make_client(retriever, embedder, fake) as client:
        created = client.post("/collections", files=files)
        assert created.status_code == 201, created.text
        info = created.json()
        assert [f["name"] for f in info["files"]] == ["guide.md", "notes.txt", "policy.pdf"]
        assert info["expires_in_seconds"] > 3500

        collection_id = info["collection_id"]
        found = client.get("/search", params={"q": "refunds processed", "collection_id": collection_id}).json()
        assert found["results"][0]["page"] == "policy.pdf"
        assert found["results"][0]["url"] is None

        events = parse_sse(ask(client, "How do I roll back?", collection_id=collection_id).text)
        assert "uploaded documents" in fake.calls[0]["system"]
        citation = next(data for kind, data in events if kind == "citation")
        assert citation["url"] is None

        assert client.delete(f"/collections/{collection_id}").status_code == 204
        assert client.get(f"/collections/{collection_id}").status_code == 404
        assert client.get("/search", params={"q": "x", "collection_id": collection_id}).status_code == 404


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ([("files", ("a.exe", b"MZ", "application/octet-stream"))], 400),
        ([("files", (f"{i}.txt", b"hello", "text/plain")) for i in range(6)], 400),
        ([("files", ("big.txt", b"x" * (2 * 1024 * 1024 + 1), "text/plain"))], 413),
        ([("files", ("empty.md", b"", "text/markdown"))], 400),
    ],
)
def test_upload_limits(retriever, embedder, files, status):
    with make_client(retriever, embedder) as client:
        assert client.post("/collections", files=files).status_code == status


def test_upload_rejects_oversized_body_before_parsing(retriever, embedder):
    with make_client(retriever, embedder, upload_max_files=1, upload_max_file_bytes=1024) as client:
        files = [("files", ("a.txt", b"x" * 200_000, "text/plain"))]
        assert client.post("/collections", files=files).status_code == 413


def test_uploaded_collections_expire(retriever, embedder):
    with make_client(retriever, embedder) as client:
        created = client.post("/collections", files=[("files", ("a.txt", b"temporary text", "text/plain"))]).json()
        store = client.app.state.collections
        store.clock = lambda: created["expires_at"] + 1
        assert client.get(f"/collections/{created['collection_id']}").status_code == 404
        assert len(store) == 0
