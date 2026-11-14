"""Build the chunk list for the FastAPI docs (and for uploaded files).

Each chunk carries everything the API and the UI need to cite it:
``url`` (fastapi.tiangolo.com path + anchor), ``heading_path`` and the
character offsets of the sections inside it.
"""

from __future__ import annotations

from pathlib import Path

from .chunker import ChunkConfig, chunk_sections, estimate_tokens
from .markdown import Section, inline_includes, parse_sections

DOCS_BASE_URL = "https://fastapi.tiangolo.com/"

# Pages that are about the FastAPI project rather than about using FastAPI,
# or that are includes/banners rather than real pages. release-notes.md alone
# is ~8k lines of changelog and would swamp retrieval with near-duplicates.
EXCLUDED_PAGES = {
    "release-notes",
    "fastapi-people",
    "external-links",
    "newsletter",
    "management",
    "management-tasks",
    "contributing",
    "translations",
    "agent-banner",
    "translation-banner",
}
MIN_PAGE_TOKENS = 40


def page_url(page: str, anchor: str = "") -> str:
    """Map ``tutorial/first-steps`` to the published MkDocs URL."""
    if page == "index":
        path = ""
    elif page.endswith("/index"):
        path = page[: -len("index")]
    else:
        path = f"{page}/"
    url = DOCS_BASE_URL + path
    return f"{url}#{anchor}" if anchor else url


def _include_page(page: str) -> bool:
    name = page.rsplit("/", 1)[-1]
    return page not in EXCLUDED_PAGES and not name.startswith("_")


def chunk_page(
    sections: list[Section],
    *,
    source: str,
    page: str,
    url_for: callable | None,
    config: ChunkConfig | None = None,
) -> list[dict]:
    page_title = next((s.heading for s in sections if s.level == 1 and s.heading), page)
    chunks = []
    for position, raw in enumerate(chunk_sections(sections, config)):
        # The h1 anchor is the top of the page; link to the page itself.
        top_level = raw["heading_path"][:1] == [page_title] and len(raw["heading_path"]) == 1
        anchor = "" if top_level else raw["anchor"]
        for section in raw["sections"]:
            if section["heading"] == page_title:
                section["anchor"] = ""
        heading_path = raw["heading_path"] or [page_title]
        chunks.append(
            {
                "chunk_id": f"{page}#{position}",
                "source": source,
                "page": page,
                "page_title": page_title,
                "title": " > ".join(heading_path),
                "heading_path": heading_path,
                "anchor": anchor,
                "url": url_for(page, anchor) if url_for else None,
                "text": raw["text"],
                "sections": raw["sections"],
                "tokens": raw["tokens"],
            }
        )
    return chunks


def build_fastapi_chunks(raw_root: Path, config: ChunkConfig | None = None) -> list[dict]:
    docs_root = raw_root / "docs/en/docs"
    chunks: list[dict] = []
    for path in sorted(docs_root.rglob("*.md")):
        page = path.relative_to(docs_root).with_suffix("").as_posix()
        if not _include_page(page):
            continue
        text = inline_includes(path.read_text(encoding="utf-8"), raw_root)
        sections = parse_sections(text)
        if sum(estimate_tokens(s.body) for s in sections) < MIN_PAGE_TOKENS:
            continue
        chunks.extend(chunk_page(sections, source="fastapi", page=page, url_for=page_url, config=config))
    for position, chunk in enumerate(chunks):
        chunk["id"] = position
    return chunks
