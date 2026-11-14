"""Temporary collections built from user uploads (.md, .txt, .pdf).

Files are parsed from memory and never written to disk. Each collection gets
its own BM25 + vector index and expires after ``ttl`` seconds; a background
task sweeps expired collections, and lookups also check expiry.
"""

from __future__ import annotations

import io
import re
import time
import uuid
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ingest.chunker import estimate_tokens
from ingest.corpus import chunk_page
from ingest.markdown import Section, parse_sections
from retrieval.embeddings import Embedder
from retrieval.index import Retriever

ALLOWED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Collection:
    id: str
    files: list[dict]
    retriever: Retriever
    created_at: float
    expires_at: float
    chunks: int = field(default=0)

    def info(self) -> dict:
        return {
            "collection_id": self.id,
            "files": self.files,
            "chunks": self.chunks,
            "expires_at": self.expires_at,
            "expires_in_seconds": max(0, int(self.expires_at - time.time())),
        }


def _safe_name(name: str) -> str:
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"[^\w.\- ]", "_", name)[:120] or "upload"


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def sections_for_file(name: str, data: bytes) -> list[Section]:
    lower = name.lower()
    if lower.endswith((".md", ".markdown")):
        sections = parse_sections(_decode(data))
        # A file with no headings becomes one section titled by its filename.
        for section in sections:
            if not section.heading:
                section.heading, section.heading_path = name, [name]
        return sections
    if lower.endswith(".txt"):
        text = _decode(data)
        return [Section(1, name, "", [name], [f"# {name}", "", *text.splitlines()])]
    if lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise UploadError(f"{name}: encrypted PDFs are not supported")
            sections = []
            for number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    heading = f"{name}, page {number}"
                    sections.append(
                        Section(
                            2,
                            heading,
                            f"page-{number}",
                            [name, f"page {number}"],
                            [f"## {heading}", "", *text.splitlines()],
                        )
                    )
            return sections
        except PdfReadError as error:
            raise UploadError(f"{name}: could not read PDF ({error})") from error
    raise UploadError(f"{name}: unsupported file type; use {', '.join(ALLOWED_SUFFIXES)}")


def chunks_for_files(files: list[tuple[str, bytes]]) -> tuple[list[dict], list[dict]]:
    chunks: list[dict] = []
    summary = []
    for name, data in files:
        sections = sections_for_file(name, data)
        if sum(estimate_tokens(s.body) for s in sections) == 0:
            raise UploadError(f"{name}: no extractable text")
        file_chunks = chunk_page(sections, source="upload", page=name, url_for=None)
        summary.append({"name": name, "bytes": len(data), "chunks": len(file_chunks)})
        chunks.extend(file_chunks)
    for position, chunk in enumerate(chunks):
        chunk["id"] = position
    return chunks, summary


class CollectionStore:
    def __init__(
        self,
        embedder: Embedder,
        *,
        ttl: int,
        max_collections: int,
        max_files: int,
        max_file_bytes: int,
        max_chunks: int,
        stemming: bool = False,
        clock=time.time,
    ):
        self.embedder = embedder
        self.ttl = ttl
        self.max_collections = max_collections
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_chunks = max_chunks
        self.stemming = stemming
        self.clock = clock
        self._collections: dict[str, Collection] = {}

    def validate(self, files: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
        if not files:
            raise UploadError("attach at least one file")
        if len(files) > self.max_files:
            raise UploadError(f"at most {self.max_files} files per collection")
        cleaned = []
        for raw_name, data in files:
            name = _safe_name(raw_name)
            if not name.lower().endswith(ALLOWED_SUFFIXES):
                raise UploadError(f"{name}: unsupported file type; use {', '.join(ALLOWED_SUFFIXES)}")
            if len(data) > self.max_file_bytes:
                raise UploadError(f"{name}: larger than {self.max_file_bytes // (1024 * 1024)} MB", 413)
            cleaned.append((name, data))
        return cleaned

    def has_capacity(self) -> bool:
        self.sweep()
        return len(self._collections) < self.max_collections

    def create(self, files: list[tuple[str, bytes]]) -> Collection:
        """Parse, chunk and embed. CPU-bound: call from a worker thread."""
        files = self.validate(files)
        if not self.has_capacity():
            raise UploadError("the demo is holding too many uploads right now; try again later", 503)
        chunks, summary = chunks_for_files(files)
        if len(chunks) > self.max_chunks:
            raise UploadError(
                f"these files produce {len(chunks)} chunks; the limit is {self.max_chunks}. Upload less text.", 413
            )
        retriever = Retriever.build(chunks, self.embedder, stemming=self.stemming)
        now = self.clock()
        collection = Collection(uuid.uuid4().hex, summary, retriever, now, now + self.ttl, len(chunks))
        self._collections[collection.id] = collection
        return collection

    def get(self, collection_id: str) -> Collection | None:
        collection = self._collections.get(collection_id)
        if collection and collection.expires_at <= self.clock():
            self._collections.pop(collection_id, None)
            return None
        return collection

    def delete(self, collection_id: str) -> bool:
        return self._collections.pop(collection_id, None) is not None

    def sweep(self) -> int:
        now = self.clock()
        expired = [cid for cid, c in self._collections.items() if c.expires_at <= now]
        for cid in expired:
            del self._collections[cid]
        return len(expired)

    def __len__(self) -> int:
        return len(self._collections)
