"""
Document extraction.

Default backend: Docling — unified DocumentConverter for PDF (digital + scanned
via built-in OCR), Markdown, and HTML. Plain TXT is wrapped as Markdown so the
downstream chunker always sees a DoclingDocument.

Optional backend (env: OCR_BACKEND=azure_di): Azure Document Intelligence's
prebuilt-read model is used for PDFs, and its Markdown output is fed back
through Docling so the rest of the pipeline is uniform.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

CATEGORY_MAP = {
    "manuals": "manual",
    "troubleshooting": "troubleshooting",
    "policies": "policy",
}


@dataclass
class ExtractedDocument:
    source_path: str            # blob path, e.g. "manuals/deviceA.pdf"
    file_name: str              # basename, e.g. "deviceA.pdf"
    file_type: str              # "pdf" | "md" | "txt" | "html"
    category: str               # "manual" | "troubleshooting" | "policy" | "other"
    docling_document: Any       # DoclingDocument — input to HybridChunker


def _category_from_path(blob_path: str) -> str:
    top = blob_path.replace("\\", "/").split("/", 1)[0]
    return CATEGORY_MAP.get(top, "other")


def _file_type(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    if s in {"pdf", "md", "txt", "html", "htm"}:
        return "html" if s == "htm" else s
    return s or "unknown"


def _build_docling_converter() -> DocumentConverter:
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    pdf_opts.do_table_structure = True
    pdf_opts.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


_converter: DocumentConverter | None = None


def _converter_singleton() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = _build_docling_converter()
    return _converter


def _extract_pdf_via_azure_di(local_path: Path) -> str:
    """Run Azure Document Intelligence prebuilt-read; return Markdown text."""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        ContentFormat,
    )
    from azure.core.credentials import AzureKeyCredential

    endpoint = os.environ["AZURE_DOC_INTELLIGENCE_ENDPOINT"]
    key = os.environ["AZURE_DOC_INTELLIGENCE_KEY"]
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with local_path.open("rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=f.read()),
            output_content_format=ContentFormat.MARKDOWN,
        )
    return poller.result().content


def _markdown_to_docling(name: str, markdown_text: str):
    stream = DocumentStream(name=name, stream=io.BytesIO(markdown_text.encode("utf-8")))
    return _converter_singleton().convert(stream).document


def extract(local_path: Path, blob_path: str) -> ExtractedDocument:
    """
    Extract a single document into a uniform ExtractedDocument.

    Args:
        local_path: where the file currently lives on disk.
        blob_path:  its logical path within the blob container; used to derive
                    `source_path` (kept verbatim) and `category` (top folder).
    """
    file_name = local_path.name
    file_type = _file_type(local_path.suffix)
    category = _category_from_path(blob_path)
    ocr_backend = os.environ.get("OCR_BACKEND", "docling").lower()

    if file_type == "pdf" and ocr_backend == "azure_di":
        md_text = _extract_pdf_via_azure_di(local_path)
        docling_doc = _markdown_to_docling(file_name, md_text)
    elif file_type == "txt":
        text = local_path.read_text(encoding="utf-8", errors="replace")
        docling_doc = _markdown_to_docling(file_name, text)
    else:
        # PDF (digital or scanned via Docling OCR), MD, HTML — native Docling input.
        docling_doc = _converter_singleton().convert(str(local_path)).document

    return ExtractedDocument(
        source_path=blob_path,
        file_name=file_name,
        file_type=file_type,
        category=category,
        docling_document=docling_doc,
    )
