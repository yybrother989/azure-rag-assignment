"""
Document extraction via Azure Document Intelligence.

The Layout API returns each document as Markdown with `<!-- PageNumber=N -->`
HTML-comment markers between page boundaries — exactly the input the
markdown-header chunker downstream wants. For native Markdown / plain text
files we skip DI entirely (no point sending pre-formatted text through OCR).

DI gives us:
  - heading hierarchy (rendered as `#`, `##`, ...)
  - page number per content span
  - tables (rendered as Markdown tables)
  - reading-order for multi-column scanned PDFs

Path → metadata normalisation
──────────────────────────────
Supported blob path structures:

  devices/{device_family}/{model}/{doc_folder}/{filename}
      scope=device  device_family={device_family}  device={model}
  devices/{device}/{doc_folder}/{filename}
      scope=device  device_family=None  device={device}   (legacy 3-level)
  shared/{doc_folder}/{filename}
      scope=shared   is_shared=True
  {doc_folder}/{filename}
      scope=device  device_family=None  device=None  (legacy flat)

`doc_folder` is "manuals" | "troubleshooting" | "policies"; it maps to
`doc_type` "manual" | "troubleshooting" | "policy".

`topic` is derived from the filename stem after stripping the version suffix
(e.g. "error101.md" → "error101", "user_manual_v1.2.pdf" → "user_manual").

`version` is extracted from the filename if present ("_v1.2" → "1.2").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential

CATEGORY_MAP = {
    "manuals": "manual",
    "troubleshooting": "troubleshooting",
    "policies": "policy",
}

# Matches version suffixes like "_v1.2", "-v2.0", "_v10" in a filename stem.
_VERSION_RE = re.compile(r"[_\-]v(\d[\d.]*)(?=[_\-.]|$)", re.IGNORECASE)

_DOC_TYPE_MAP = {
    "manuals": "manual",
    "troubleshooting": "troubleshooting",
    "policies": "policy",
}

# Files we send through DI (need layout / OCR / table extraction).
DI_FORMATS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heif"}


@dataclass
class FigureRef:
    """A figure / diagram extracted by Document Intelligence.

    `polygon` is DI's 8-number `[x1,y1,x2,y2,x3,y3,x4,y4]` bounding quad in
    PAGE INCHES (DI's default unit) — used by the renderer in figures.py to
    crop the PNG out of the source PDF. `figure_id` is `p{page}_f{idx}` and
    becomes the blob path stem under `kb-figures/{doc_stem}/`. `blob_url`
    is filled in once the crop is uploaded; `caption` is filled in by the
    optional vision pre-pass.

    `markdown_offset/length` point into the raw DI markdown's `<figure>...
    </figure>` HTML span — used by the caption splicer so the description
    text lands inside the same chunk as the figure callout."""
    figure_id: str
    page: int
    polygon: list[float]
    markdown_offset: int = 0
    markdown_length: int = 0
    blob_url: str = ""
    caption: str = ""

    def to_dict(self) -> dict:
        return {
            "figure_id": self.figure_id,
            "page": self.page,
            "blob_url": self.blob_url,
            "caption": self.caption,
        }


@dataclass
class TableRef:
    """A structured table extracted by Document Intelligence.

    `markdown_offset` is the character position in the RAW DI markdown where the
    `<table>` HTML span begins; chunk.py uses this to associate the table with
    whichever chunk it ends up inside. `headers` is best-effort — DI marks
    cells as `columnHeader` when it can detect them, otherwise we fall back to
    treating the first row as headers if the table is wider than 2 cols, or
    use a generic 'Field | Value' for spec-style 2-column tables."""
    page: int
    headers: list[str]
    rows: list[list[str]]
    markdown_offset: int
    markdown_length: int

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "headers": self.headers,
            "rows": self.rows,
        }


@dataclass
class ExtractedDocument:
    source_path: str            # blob path, e.g. "devices/deviceA/manuals/user_manual_v1.2.pdf"
    file_name: str              # basename, e.g. "user_manual_v1.2.pdf"
    file_type: str              # "pdf" | "md" | "txt" | "html"
    category: str               # legacy: "manual" | "troubleshooting" | "policy" | "other"
    markdown: str               # document content as Markdown
                                # (PDFs include `<!-- PageNumber=N -->` markers)
    page_count: int             # 1 for non-paginated formats
    tables: list[TableRef]      # structured tables (from DI for PDFs; empty for MD/TXT)
    figures: list[FigureRef] = field(default_factory=list)  # figure refs (PDFs only)
    # Structured metadata derived deterministically from the blob path.
    scope: str = "device"       # "device" | "shared"
    device_family: str | None = None
    device: str | None = None   # e.g. "deviceA" — None for shared docs or legacy paths
    doc_type: str | None = None # "manual" | "troubleshooting" | "policy" | "other"
    topic: str | None = None    # filename-derived topic tag, e.g. "error101"
    version: str | None = None  # version from filename, e.g. "1.2"
    is_shared: bool = False     # True when scope == "shared"


def _category_from_path(blob_path: str) -> str:
    parts = blob_path.replace("\\", "/").split("/")
    if parts[0] == "devices" and len(parts) >= 5:
        return _DOC_TYPE_MAP.get(parts[3], "other")
    if parts[0] == "shared" and len(parts) >= 3:
        return _DOC_TYPE_MAP.get(parts[1], "other")
    return CATEGORY_MAP.get(parts[0], "other")


def _parse_path_metadata(blob_path: str) -> dict:
    """Derive structured metadata deterministically from the blob storage path.

    Supported structures
    --------------------
    devices/{device_family}/{model}/{doc_folder}/{filename}
        scope="device"  device_family={device_family}  device={model}
        doc_type from doc_folder

    devices/{device}/{doc_folder}/{filename}
        scope="device"  device_family=None  device={device}  doc_type from doc_folder

    shared/{doc_folder}/{filename}
        scope="shared"  device_family=None  is_shared=True  doc_type from doc_folder

    {doc_folder}/{filename}   (legacy flat layout)
        scope="device"  device_family=None  device=None  doc_type from doc_folder

    All other shapes fall back to scope="device", device=None, doc_type="other".
    """
    parts = blob_path.replace("\\", "/").split("/")
    file_stem = Path(parts[-1]).stem  # e.g. "user_manual_v1.2"

    # Version: "_v1.2", "-v2.0", etc.
    version_match = _VERSION_RE.search(file_stem)
    version: str | None = version_match.group(1) if version_match else None

    # Topic: stem after stripping version suffix, normalised to lowercase-hyphens.
    topic_stem = _VERSION_RE.sub("", file_stem).strip("_- ")
    topic = re.sub(r"[-\s_]+", "-", topic_stem.lower()).strip("-") if topic_stem else None

    if len(parts) >= 5 and parts[0].lower() == "devices":
        device_family = parts[1]
        device = parts[2]
        doc_folder = parts[3]
        return {
            "scope": "device",
            "device_family": device_family,
            "device": device,
            "doc_type": _DOC_TYPE_MAP.get(doc_folder, doc_folder or "other"),
            "topic": topic,
            "version": version,
            "is_shared": False,
        }

    if len(parts) >= 4 and parts[0].lower() == "devices":
        device = parts[1]
        doc_folder = parts[2]
        return {
            "scope": "device",
            "device_family": None,
            "device": device,
            "doc_type": _DOC_TYPE_MAP.get(doc_folder, doc_folder or "other"),
            "topic": topic,
            "version": version,
            "is_shared": False,
        }

    if len(parts) >= 2 and parts[0].lower() == "shared":
        doc_folder = parts[1] if len(parts) > 2 else ""
        return {
            "scope": "shared",
            "device_family": None,
            "device": None,
            "doc_type": _DOC_TYPE_MAP.get(doc_folder, doc_folder or "other"),
            "topic": topic,
            "version": version,
            "is_shared": True,
        }

    # Legacy flat layout: {manuals|troubleshooting|policies}/{filename}
    doc_folder = parts[0] if parts else ""
    return {
        "scope": "device",
        "device_family": None,
        "device": None,
        "doc_type": _DOC_TYPE_MAP.get(doc_folder, CATEGORY_MAP.get(doc_folder, "other")),
        "topic": topic,
        "version": version,
        "is_shared": False,
    }


def _file_type(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    if s == "htm":
        return "html"
    return s or "unknown"


_di_client: DocumentIntelligenceClient | None = None


def _get_di_client() -> DocumentIntelligenceClient:
    global _di_client
    if _di_client is None:
        endpoint = os.environ["AZURE_DOC_INTELLIGENCE_ENDPOINT"]
        key = os.environ["AZURE_DOC_INTELLIGENCE_KEY"]
        _di_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return _di_client


def _extract_via_di(local_path: Path) -> tuple[str, int, list[TableRef], list[FigureRef]]:
    """Run DI Layout API on a binary file.
    Returns (markdown, page_count, tables, figures)."""
    client = _get_di_client()
    with local_path.open("rb") as f:
        body = f.read()
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=body),
        output_content_format=DocumentContentFormat.MARKDOWN,
    )
    result = poller.result()
    markdown = result.content or ""
    page_count = len(result.pages) if result.pages else 1
    tables = _build_table_refs(result) if result.tables else []
    figures = _build_figure_refs(result) if getattr(result, "figures", None) else []
    return markdown, page_count, tables, figures


def _build_figure_refs(result) -> list[FigureRef]:
    """Convert DI's `result.figures[]` into our FigureRef structure.

    Each figure has one or more `bounding_regions` (page + polygon). We take
    the first region — multi-region figures are rare and usually duplicates.
    Polygon is DI's 8-number quad in inches at the source page's natural
    resolution; the renderer in ingest.py converts to pixel coords."""
    refs: list[FigureRef] = []
    per_page_idx: dict[int, int] = {}
    for fig in result.figures:
        if not fig.bounding_regions:
            continue
        region = fig.bounding_regions[0]
        page = region.page_number
        polygon = list(region.polygon) if region.polygon else []
        if len(polygon) < 8:
            continue
        idx = per_page_idx.get(page, 0) + 1
        per_page_idx[page] = idx
        offset = fig.spans[0].offset if fig.spans else 0
        length = fig.spans[0].length if fig.spans else 0
        refs.append(FigureRef(
            figure_id=f"p{page}_f{idx}",
            page=page,
            polygon=polygon,
            markdown_offset=offset,
            markdown_length=length,
        ))
    return refs


def _build_table_refs(result) -> list[TableRef]:
    """Convert DI's `result.tables[]` into our TableRef structure.

    Header detection: prefer cells with `kind == "columnHeader"`; if DI marked
    none (common for spec tables that look like {label: value}), fall back to:
      - 2-column tables  → headers = ["Field", "Value"], all rows treated as data
      - wider tables     → first row treated as headers
    """
    refs: list[TableRef] = []
    for t in result.tables:
        # Build a row-major grid first.
        grid: list[list[str]] = [["" for _ in range(t.column_count)] for _ in range(t.row_count)]
        explicit_header_cells = []
        for cell in t.cells:
            r, c = cell.row_index, cell.column_index
            if 0 <= r < t.row_count and 0 <= c < t.column_count:
                grid[r][c] = (cell.content or "").strip()
            if cell.kind == "columnHeader":
                explicit_header_cells.append(cell)

        if explicit_header_cells:
            # Use the explicit header cells, ordered by column.
            hdr_row = [None] * t.column_count
            for cell in explicit_header_cells:
                if cell.row_index == 0 and 0 <= cell.column_index < t.column_count:
                    hdr_row[cell.column_index] = (cell.content or "").strip()
            headers = [h or "" for h in hdr_row]
            rows = grid[1:]                          # skip the header row
        elif t.column_count == 2:
            headers = ["Field", "Value"]
            rows = grid
        elif t.row_count >= 2:
            headers = grid[0]
            rows = grid[1:]
        else:
            # Single-row table: treat as one row with positional headers.
            headers = [f"col{i+1}" for i in range(t.column_count)]
            rows = grid

        page = (t.bounding_regions[0].page_number
                if t.bounding_regions else 1)
        offset = t.spans[0].offset if t.spans else 0
        length = t.spans[0].length if t.spans else 0
        refs.append(TableRef(
            page=page, headers=headers, rows=rows,
            markdown_offset=offset, markdown_length=length,
        ))
    return refs


def _wrap_text_as_markdown(text: str, file_name: str) -> str:
    """For plain TXT files: emit a single H1 with the filename, then the text.
    Gives the markdown-header chunker something to anchor on."""
    title = Path(file_name).stem.replace("_", " ").replace("-", " ").title()
    return f"# {title}\n\n{text}"


def extract(local_path: Path, blob_path: str) -> ExtractedDocument:
    """
    Extract a single document into a uniform ExtractedDocument.

    PDFs and image formats go through Azure Document Intelligence (Layout API,
    Markdown output). MD files are read as-is. TXT is wrapped as a 1-heading
    Markdown doc so the chunker has structure to work with.
    """
    file_name = local_path.name
    file_type = _file_type(local_path.suffix)
    category = _category_from_path(blob_path)

    tables: list[TableRef] = []
    figures: list[FigureRef] = []
    if file_type in DI_FORMATS:
        markdown, page_count, tables, figures = _extract_via_di(local_path)
    elif file_type == "md":
        markdown = local_path.read_text(encoding="utf-8", errors="replace")
        page_count = 1
    elif file_type == "txt":
        text = local_path.read_text(encoding="utf-8", errors="replace")
        markdown = _wrap_text_as_markdown(text, file_name)
        page_count = 1
    elif file_type == "html":
        # Best-effort: keep raw HTML; markdown chunker will treat as one block.
        markdown = local_path.read_text(encoding="utf-8", errors="replace")
        page_count = 1
    else:
        # Unknown extension — fall back to text read.
        markdown = local_path.read_text(encoding="utf-8", errors="replace")
        page_count = 1

    path_meta = _parse_path_metadata(blob_path)
    return ExtractedDocument(
        source_path=blob_path,
        file_name=file_name,
        file_type=file_type,
        category=category,
        markdown=markdown,
        page_count=page_count,
        tables=tables,
        figures=figures,
        **path_meta,
    )
