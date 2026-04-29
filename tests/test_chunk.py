"""
Unit tests for pure-Python functions in src/chunk.py.
No Azure credentials or network calls required.
"""

from src.chunk import (
    _page_for_offset,
    _split_by_headers_with_offsets,
    _strip_page_markers,
    make_chunk_id,
    make_doc_id,
)


# ---------------------------------------------------------------------------
# make_doc_id
# ---------------------------------------------------------------------------

def test_make_doc_id_deterministic():
    assert make_doc_id("manuals/device_a.pdf") == make_doc_id("manuals/device_a.pdf")


def test_make_doc_id_different_paths_differ():
    assert make_doc_id("manuals/device_a.pdf") != make_doc_id("manuals/device_b.pdf")


def test_make_doc_id_length():
    result = make_doc_id("any/path.pdf")
    assert len(result) == 16


# ---------------------------------------------------------------------------
# make_chunk_id
# ---------------------------------------------------------------------------

def test_make_chunk_id_deterministic():
    cid = make_chunk_id("manuals/a.pdf", 0, "some content")
    assert cid == make_chunk_id("manuals/a.pdf", 0, "some content")


def test_make_chunk_id_source_path_matters():
    assert make_chunk_id("a.pdf", 0, "x") != make_chunk_id("b.pdf", 0, "x")


def test_make_chunk_id_index_matters():
    assert make_chunk_id("a.pdf", 0, "x") != make_chunk_id("a.pdf", 1, "x")


def test_make_chunk_id_content_matters():
    assert make_chunk_id("a.pdf", 0, "foo") != make_chunk_id("a.pdf", 0, "bar")


# ---------------------------------------------------------------------------
# _strip_page_markers
# ---------------------------------------------------------------------------

def test_strip_page_markers_basic():
    md = "Page one content<!-- PageNumber=\"2\" -->Page two content"
    clean, page_starts = _strip_page_markers(md)
    assert "PageNumber" not in clean
    assert "Page one content" in clean
    assert "Page two content" in clean
    # page_starts must contain an entry for page 2
    pages = [p for _, p in page_starts]
    assert 2 in pages


def test_strip_page_markers_pagebreak_bumps_counter():
    md = "first<!-- PageBreak -->second"
    clean, page_starts = _strip_page_markers(md)
    assert "PageBreak" not in clean
    pages = [p for _, p in page_starts]
    assert 2 in pages


def test_strip_page_markers_no_markers():
    md = "# Just a heading\n\nSome body text."
    clean, page_starts = _strip_page_markers(md)
    assert clean == md
    assert page_starts == [(0, 1)]


def test_strip_page_markers_removes_header_footer_chrome():
    md = "content<!-- PageHeader=\"Header text\" -->more content"
    clean, _ = _strip_page_markers(md)
    assert "PageHeader" not in clean
    assert "Header text" not in clean


# ---------------------------------------------------------------------------
# _page_for_offset
# ---------------------------------------------------------------------------

def test_page_for_offset_single_page():
    page_starts = [(0, 1)]
    assert _page_for_offset(page_starts, 0) == 1
    assert _page_for_offset(page_starts, 999) == 1


def test_page_for_offset_two_pages():
    page_starts = [(0, 1), (100, 2)]
    assert _page_for_offset(page_starts, 0) == 1
    assert _page_for_offset(page_starts, 99) == 1
    assert _page_for_offset(page_starts, 100) == 2
    assert _page_for_offset(page_starts, 500) == 2


def test_page_for_offset_three_pages():
    page_starts = [(0, 1), (50, 2), (150, 3)]
    assert _page_for_offset(page_starts, 49) == 1
    assert _page_for_offset(page_starts, 50) == 2
    assert _page_for_offset(page_starts, 149) == 2
    assert _page_for_offset(page_starts, 150) == 3


def test_page_for_offset_empty():
    assert _page_for_offset([], 0) is None


# ---------------------------------------------------------------------------
# _split_by_headers_with_offsets
# ---------------------------------------------------------------------------

def test_split_headers_no_headers():
    md = "Plain text with no headings."
    sections = _split_by_headers_with_offsets(md)
    assert len(sections) == 1
    assert sections[0]["heading_path"] is None
    assert sections[0]["start"] == 0
    assert sections[0]["end"] == len(md)


def test_split_headers_single_h1():
    md = "# Introduction\n\nSome body text."
    sections = _split_by_headers_with_offsets(md)
    assert len(sections) == 1
    assert sections[0]["heading_path"] == "Introduction"


def test_split_headers_two_h1_sections():
    md = "# First\n\nBody A.\n# Second\n\nBody B."
    sections = _split_by_headers_with_offsets(md)
    assert len(sections) == 2
    assert sections[0]["heading_path"] == "First"
    assert sections[1]["heading_path"] == "Second"
    # offsets must be contiguous and span the whole document
    assert sections[0]["start"] == 0
    assert sections[1]["end"] == len(md)


def test_split_headers_nested():
    md = "# H1\n\n## H2\n\nBody."
    sections = _split_by_headers_with_offsets(md)
    # First section is the H1 (with just the newlines before H2)
    # Second section is the H2
    paths = [s["heading_path"] for s in sections]
    assert any("H2" in (p or "") for p in paths)
    # H2 section should carry the H1 breadcrumb
    h2_section = next(s for s in sections if "H2" in (s["heading_path"] or ""))
    assert "H1" in h2_section["heading_path"]


def test_split_headers_offsets_cover_full_text():
    md = "# Section A\n\nContent A.\n\n# Section B\n\nContent B."
    sections = _split_by_headers_with_offsets(md)
    # start of first == 0, end of last == len(md)
    assert sections[0]["start"] == 0
    assert sections[-1]["end"] == len(md)
