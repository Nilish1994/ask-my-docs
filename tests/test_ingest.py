from pathlib import Path

from ingest.chunker import ChunkConfig, chunk_sections, estimate_tokens, split_long
from ingest.corpus import page_url
from ingest.markdown import clean_lines, inline_includes, parse_sections


def test_parse_sections_uses_explicit_anchors_and_heading_path():
    text = "# Body { #request-body }\n\nIntro.\n\n## Create a model { #create-your-data-model }\n\nText."
    sections = parse_sections(text)
    assert [s.anchor for s in sections] == ["request-body", "create-your-data-model"]
    assert sections[1].heading_path == ["Body", "Create a model"]
    assert sections[1].body.startswith("## Create a model")


def test_hash_inside_code_fence_is_not_a_heading():
    text = "# Page\n\n```bash\n# not a heading\nls\n```\n"
    sections = parse_sections(text)
    assert len(sections) == 1
    assert "# not a heading" in sections[0].body


def test_fallback_slug_when_heading_has_no_id():
    sections = parse_sections("# Hello `World` Again\n\ntext")
    assert sections[0].anchor == "hello-world-again"


def test_clean_lines_handles_admonitions_html_and_mkdocstrings():
    lines = [
        "/// tip | Pro tip",
        "Use <b>bold</b> carefully.",
        "///",
        "::: fastapi.FastAPI",
        "    options:",
        "        members: true",
        "After.",
    ]
    assert clean_lines(lines) == ["Tip: Pro tip", "Use bold carefully.", "After."]


def test_inline_includes_reads_docs_src_and_line_ranges(tmp_path: Path):
    (tmp_path / "docs_src/app").mkdir(parents=True)
    (tmp_path / "docs_src/app/main.py").write_text("line1\nline2\nline3\n")
    full = inline_includes("{* ../../docs_src/app/main.py hl[2] *}", tmp_path)
    assert full == "```python\nline1\nline2\nline3\n```"
    ranged = inline_includes("{* ../../docs_src/app/main.py ln[2:3] *}", tmp_path)
    assert ranged == "```python\nline2\nline3\n```"
    assert inline_includes("{* ../../../../etc/passwd *}", tmp_path) == ""


def test_small_sections_merge_and_record_offsets():
    text = "# Page { #page }\n\nOne.\n\n## A { #a }\n\nAlpha text.\n\n## B { #b }\n\nBeta text."
    chunks = chunk_sections(parse_sections(text), ChunkConfig(min_tokens=400, max_tokens=800))
    assert len(chunks) == 1
    chunk = chunks[0]
    for section in chunk["sections"]:
        heading = f"{'#' * (2 if section['anchor'] != 'page' else 1)} {section['heading']}"
        assert chunk["text"][section["start"] :].startswith(heading)


def test_long_section_splits_with_overlap_and_size_limit():
    paragraphs = [f"Paragraph {i} " + "word " * 60 for i in range(30)]
    text = "# Big { #big }\n\n" + "\n\n".join(paragraphs)
    config = ChunkConfig(min_tokens=100, target_tokens=300, max_tokens=400, overlap_tokens=80)
    chunks = chunk_sections(parse_sections(text), config)
    assert len(chunks) > 3
    assert all(c["tokens"] <= config.max_tokens + 20 for c in chunks)
    # The last paragraph of one window opens the next one.
    first_tail = chunks[0]["text"].split("\n\n")[-1]
    assert first_tail in chunks[1]["text"]
    assert chunks[1]["text"].startswith("# Big (continued)")


def test_split_long_keeps_code_fences_whole():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(20)) + "\n```"
    body = "intro " * 50 + "\n\n" + code + "\n\n" + "outro " * 50
    windows = split_long(body, ChunkConfig(target_tokens=120, max_tokens=200, overlap_tokens=0))
    assert any(w.count("```") == 2 for w in windows)
    assert all(w.count("```") % 2 == 0 for w in windows)


def test_page_url_mapping():
    assert page_url("index") == "https://fastapi.tiangolo.com/"
    assert page_url("tutorial/index") == "https://fastapi.tiangolo.com/tutorial/"
    assert page_url("tutorial/first-steps", "check-it") == "https://fastapi.tiangolo.com/tutorial/first-steps/#check-it"


def test_estimate_tokens_counts_words_and_punctuation():
    assert estimate_tokens("def f(x): return x") == 8  # def f ( x ) : return x
