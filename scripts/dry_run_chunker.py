"""
Dry-run validator for chunk.py.

First run: calls Document Intelligence on MIELE_DRYER.pdf (~$1.50) and caches
the ExtractedDocument to /tmp/miele_di_cache.pkl. Subsequent runs: load cache,
only re-run chunk_document. Use to iterate on the chunker for free after the
one-time DI call.

Reports: page coverage, figure attribution coverage, chunks-per-page histogram,
and any pages where DI emitted a figure but no chunk got attributed to that page.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

CACHE_PATH = Path("/tmp/miele_di_cache.pkl")
PDF_PATH = REPO_ROOT / "data" / "manuals" / "MIELE_DRYER.pdf"
BLOB_PATH = "manuals/MIELE_DRYER.pdf"


def load_or_extract():
    if CACHE_PATH.exists():
        print(f"  cache hit → {CACHE_PATH}")
        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    print(f"  cache miss — calling DI on {PDF_PATH.name} (one-time, ~$1.50)")
    from src.extract import extract
    ext = extract(PDF_PATH, BLOB_PATH)
    with CACHE_PATH.open("wb") as f:
        pickle.dump(ext, f)
    print(f"  cached to {CACHE_PATH}")
    return ext


def main() -> int:
    print(f"[1/3] Loading DI extract for {PDF_PATH.name}")
    ext = load_or_extract()
    print(f"  markdown: {len(ext.markdown):,} chars  |  figures: {len(ext.figures)}  |  tables: {len(ext.tables)}")

    # Mock figure blob_urls so chunk.py's figures_by_page lookup populates.
    # (figures_by_page only collects refs that have a blob_url — that's normally
    # set by render_and_upload_figures in ingest.py, which we're skipping.)
    for i, fig in enumerate(ext.figures):
        if not fig.blob_url:
            fig.blob_url = f"mock://figure_{i}"

    print("\n[2/3] Running chunk_document()")
    from src.chunk import chunk_document
    chunks = chunk_document(ext)
    print(f"  produced {len(chunks)} chunks")

    print("\n[3/3] Validating page + figure coverage")
    page_dist = Counter(c.page_number for c in chunks)
    chunk_pages = sorted(p for p in page_dist if p)
    di_pages = list(range(1, 89))    # MIELE has 88 pages

    print(f"\n  Unique chunk pages : {len(chunk_pages)} / 88")
    missing_chunk_pages = sorted(set(di_pages) - set(chunk_pages))
    if missing_chunk_pages:
        print(f"  Pages with NO chunk attributed: {missing_chunk_pages}")
    else:
        print("  ✓ Every page 1-88 has at least one chunk attributed")

    fig_pages_attached = sorted({c.page_number for c in chunks if c.figures and c.page_number})
    fig_pages_di = sorted({f.page for f in ext.figures})
    print(f"\n  Figure pages from DI ({len(fig_pages_di)}): {fig_pages_di}")
    print(f"  Pages where chunks got figures ({len(fig_pages_attached)}): {fig_pages_attached}")
    missing_fig_pages = sorted(set(fig_pages_di) - set(fig_pages_attached))
    if missing_fig_pages:
        print(f"  ✗ MISSING figure pages (DI has figures but no chunk attached): {missing_fig_pages}")
    else:
        print("  ✓ Every page with a DI figure has at least one chunk carrying it")

    # chunks-per-page histogram preview
    print("\n  Chunks per page (first 30 pages):")
    for p in sorted(page_dist.keys())[:30]:
        bar = "█" * page_dist[p]
        print(f"    p{p:>3}: {bar} ({page_dist[p]})")

    return 0 if not missing_fig_pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
