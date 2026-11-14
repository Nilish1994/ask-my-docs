"""Turn one MkDocs Markdown page into clean, heading-delimited sections.

FastAPI's docs are written for MkDocs Material, so raw pages contain things
that are noise for retrieval and for the LLM:

- ``{* ../../docs_src/x.py hl[3] *}`` code includes (inlined here as fenced code)
- ``/// tip`` ... ``///`` admonitions (kept as a "Tip:" line)
- ``<div class="termy">`` terminal demos full of ``<font>``/``<span>`` tags
- ``::: fastapi.FastAPI`` mkdocstrings directives (dropped)

Headings carry explicit ids like ``## First Steps { #first-steps }``; those ids
are the anchors on fastapi.tiangolo.com, so we keep them for citation links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*(?:\{\s*#([\w-]+)\s*\})?\s*$")
INCLUDE_RE = re.compile(r"\{\*\s*(\S+)(.*?)\*\}")
LINE_RANGE_RE = re.compile(r"ln\[(\d+):(\d+)\]")
ADMONITION_OPEN_RE = re.compile(r"^///\s*(\w+)\s*(?:\|\s*(.*))?$")
HTML_TAG_RE = re.compile(r"</?(?:font|span|u|b|i|div|br|img|a|abbr|p|strong|em)\b[^>]*>", re.I)
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Section:
    level: int
    heading: str
    anchor: str
    heading_path: list[str]
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def slugify(text: str) -> str:
    """Fallback anchor when a heading has no explicit ``{ #id }``."""
    text = re.sub(r"[`*_]", "", text.lower())
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text).strip("-")


def clean_heading(text: str) -> str:
    return HTML_TAG_RE.sub("", text).replace("`", "").strip()


def _read_include(raw_root: Path, target: str, options: str) -> str | None:
    relative = target
    while relative.startswith("../"):
        relative = relative[3:]
    path = (raw_root / relative).resolve()
    if not path.is_relative_to(raw_root.resolve()) or not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    line_range = LINE_RANGE_RE.search(options)
    if line_range:
        start, end = int(line_range.group(1)), int(line_range.group(2))
        lines = lines[start - 1 : end]
    return "\n".join(lines).strip()


def inline_includes(text: str, raw_root: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        code = _read_include(raw_root, match.group(1), match.group(2))
        if code is None:
            return ""
        return f"```python\n{code}\n```"

    return INCLUDE_RE.sub(replace, text)


def clean_lines(lines: list[str]) -> list[str]:
    """Drop MkDocs-only syntax and terminal-demo HTML; keep prose and code."""
    out: list[str] = []
    in_fence = False
    skipping_directive = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            # Terminal demos put HTML colour tags inside ```console fences.
            out.append(HTML_TAG_RE.sub("", line))
            continue
        if skipping_directive:
            if line.startswith((" ", "\t")) or not line.strip():
                continue
            skipping_directive = False
        if line.startswith("::: "):
            skipping_directive = True
            continue
        stripped = line.strip()
        admonition = ADMONITION_OPEN_RE.match(stripped)
        if admonition:
            kind = admonition.group(1).capitalize()
            title = (admonition.group(2) or "").strip()
            out.append(f"{kind}: {title}" if title else f"{kind}:")
            continue
        if stripped == "///":
            continue
        line = HTML_TAG_RE.sub("", line)
        out.append(line)
    # Collapse runs of blank lines.
    collapsed: list[str] = []
    for line in out:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line.rstrip())
    return collapsed


def parse_sections(text: str) -> list[Section]:
    """Split a page into sections at h1-h4 headings, ignoring '#' inside code."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current: Section | None = None
    in_fence = False
    preamble: list[str] = []

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            title = clean_heading(heading.group(2))
            anchor = heading.group(3) or slugify(title)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = Section(level, title, anchor, [t for _, t in stack])
            # Keep the heading inside the text: it helps BM25, embeddings and Claude.
            current.lines.append(f"{'#' * level} {title}")
            sections.append(current)
            continue
        if current is None:
            preamble.append(line)
        else:
            current.lines.append(line)

    if preamble and any(p.strip() for p in preamble):
        intro = Section(1, "", "", [], preamble)
        sections.insert(0, intro)
    for section in sections:
        section.lines = clean_lines(section.lines)
    return sections
