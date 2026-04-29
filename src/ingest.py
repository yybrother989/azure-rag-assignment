"""
Pipeline orchestrator: local /data -> Blob -> extract -> chunk -> embed -> Search.

Idempotent end-to-end:
- Blob uploads use overwrite=True
- Chunk IDs are content-hashed (see chunk.make_chunk_id), so re-running upserts
  identical chunks back into the same document slot.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from .chunk import Chunk, chunk_document
from .embed import embed_batch
from .extract import extract
from .figures import (
    ensure_figures_container,
    figures_container_name,
    render_and_upload_figures,
    splice_captions_into_markdown,
)
from .index import create_or_update_index, get_search_client

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".html", ".htm"}
# Recognised top-level folders in data/.
# "devices" and "shared" are the metadata-first layout;
# "manuals", "troubleshooting", "policies" are the legacy flat layout.
# Both are supported simultaneously so existing corpora still ingest correctly.
TOP_FOLDERS = {"devices", "shared", "manuals", "troubleshooting", "policies"}
SEARCH_UPLOAD_BATCH = 500


def _blob_service_client() -> BlobServiceClient:
    account = os.environ["AZURE_STORAGE_ACCOUNT"]
    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )


def _walk_data_dir(data_dir: Path):
    """Yield (local_path, blob_path) for every supported file under one of the
    three category folders, preserving the relative blob path."""
    for top in TOP_FOLDERS:
        root = data_dir / top
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                blob_path = path.relative_to(data_dir).as_posix()
                yield path, blob_path


def upload_local_data(data_dir: Path) -> list[str]:
    """Mirror /data into the blob container; return the list of blob paths."""
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
    container = _blob_service_client().get_container_client(container_name)
    uploaded: list[str] = []
    for local_path, blob_path in _walk_data_dir(data_dir):
        with local_path.open("rb") as f:
            container.upload_blob(name=blob_path, data=f, overwrite=True)
        uploaded.append(blob_path)
    return uploaded


def _delete_existing_chunks(source_path: str) -> int:
    """Delete all indexed chunks for source_path before re-upsert.

    Prevents stale chunks accumulating when a document is modified and
    re-ingested. Returns the count of deleted documents."""
    from .chunk import make_doc_id
    doc_id = make_doc_id(source_path)
    client = get_search_client()
    results = client.search(
        search_text="*",
        filter=f"doc_id eq '{doc_id}'",
        select=["chunk_id"],
        top=1000,
    )
    chunk_ids = [r["chunk_id"] for r in results]
    if chunk_ids:
        client.delete_documents(documents=[{"chunk_id": cid} for cid in chunk_ids])
    return len(chunk_ids)


def _delete_existing_figures(source_path: str) -> int:
    """Delete all figure PNGs under kb-figures/{stem}/ for source_path.

    Mirror of _delete_existing_chunks but for the figures container. Used by
    diff-sync when the source blob has been removed — the index chunks AND
    their attached figure PNGs both need to go.

    NOTE: figures are keyed by filename stem only (no folder context — see
    figures.py:135). Two PDFs with the same filename in different folders
    would write to the same kb-figures/{stem}/ folder; deleting one removes
    the other's figures too. Current data has no such collisions."""
    from .figures import figures_container_name

    stem = Path(source_path).stem
    if not stem:
        return 0
    container = _blob_service_client().get_container_client(figures_container_name())
    deleted = 0
    for blob in container.list_blobs(name_starts_with=f"{stem}/"):
        try:
            container.delete_blob(blob.name)
            deleted += 1
        except Exception:
            pass
    return deleted


def chunk_to_search_doc(c: Chunk, vector: list[float]) -> dict:
    d = asdict(c)
    # `tables` and `figures` are Python list[dict]; AI Search fields are
    # String columns. Serialize and drop the in-memory keys.
    tables = d.pop("tables", None) or []
    figures = d.pop("figures", None) or []
    d["tables_json"] = json.dumps(tables, ensure_ascii=False) if tables else ""
    d["figures_json"] = json.dumps(figures, ensure_ascii=False) if figures else ""
    d["content_vector"] = vector
    return d


def ingest_single(blob_path: str) -> dict:
    """Ingest one already-uploaded blob. Used by the Function App's blob trigger
    to handle live uploads without a full re-walk of /data.

    Pipeline: delete stale chunks → download blob → extract → chunk → embed → upsert.
    Idempotent: old chunks for the same source_path are deleted before upserting new ones.
    """
    import tempfile
    from .chunk import chunk_document
    from .embed import embed_batch
    from .extract import extract
    from .index import create_or_update_index, get_search_client

    create_or_update_index()
    _delete_existing_chunks(blob_path)
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
    blob_service = _blob_service_client()
    container = blob_service.get_container_client(container_name)
    ensure_figures_container(blob_service)
    figures_container = blob_service.get_container_client(figures_container_name())

    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(blob_path).name
        with local.open("wb") as f:
            f.write(container.download_blob(blob_path).readall())
        extracted = extract(local, blob_path)
        if extracted.figures:
            blob_stem = Path(blob_path).stem
            render_and_upload_figures(local, extracted.figures, blob_stem, figures_container)
            extracted.markdown = splice_captions_into_markdown(
                extracted.markdown, extracted.figures
            )
        chunks = chunk_document(extracted)

    if not chunks:
        return {"blob_path": blob_path, "chunks_indexed": 0, "skipped": True}

    vectors = embed_batch([c.content for c in chunks])
    docs = [chunk_to_search_doc(c, v) for c, v in zip(chunks, vectors)]
    get_search_client().upload_documents(docs)
    return {"blob_path": blob_path, "chunks_indexed": len(chunks)}


def ingest(data_dir: Path | str = "data") -> dict:
    """Run the full pipeline. Returns a summary dict."""
    data_dir = Path(data_dir)

    print("[1/5] Ensuring AI Search index exists")
    index_name = create_or_update_index()

    print(f"[2/5] Uploading documents from {data_dir} to blob storage")
    blob_paths = upload_local_data(data_dir)
    if not blob_paths:
        print(
            f"  no supported files found under {data_dir}/\n"
            f"  expected: devices/{{device}}/{{manuals|troubleshooting|policies}}/ "
            f"or shared/{{doc_type}}/ or legacy {{manuals|troubleshooting|policies}}/"
        )
        return {"index": index_name, "blobs_uploaded": 0, "chunks_indexed": 0}
    print(f"  uploaded {len(blob_paths)} blobs")

    print("[3/5] Extracting + chunking each document")
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
    blob_service = _blob_service_client()
    container = blob_service.get_container_client(container_name)
    ensure_figures_container(blob_service)
    figures_container = blob_service.get_container_client(figures_container_name())
    all_chunks: list[Chunk] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for blob_path in blob_paths:
            local = td_path / Path(blob_path).name
            with local.open("wb") as out:
                out.write(container.download_blob(blob_path).readall())
            try:
                _delete_existing_chunks(blob_path)
                extracted = extract(local, blob_path)
                # Crop + upload figures (and optionally caption via vision)
                # BEFORE chunking, then splice captions back into the markdown
                # so the chunker carries them into the right chunk.
                if extracted.figures:
                    blob_stem = Path(blob_path).stem
                    render_and_upload_figures(local, extracted.figures, blob_stem, figures_container)
                    extracted.markdown = splice_captions_into_markdown(
                        extracted.markdown, extracted.figures
                    )
                doc_chunks = chunk_document(extracted)
                all_chunks.extend(doc_chunks)
                print(f"  {blob_path}: {len(doc_chunks)} chunks, "
                      f"{len(extracted.figures)} figures, {len(extracted.tables)} tables")
            except Exception as e:  # extraction failures shouldn't kill the run
                print(f"  {blob_path}: SKIPPED ({type(e).__name__}: {e})")

    if not all_chunks:
        return {"index": index_name, "blobs_uploaded": len(blob_paths), "chunks_indexed": 0}

    print(f"[4/5] Embedding {len(all_chunks)} chunks")
    vectors = embed_batch([c.content for c in all_chunks])

    print(f"[5/5] Upserting into Azure AI Search index '{index_name}'")
    search = get_search_client()
    docs = [chunk_to_search_doc(c, v) for c, v in zip(all_chunks, vectors)]
    for i in range(0, len(docs), SEARCH_UPLOAD_BATCH):
        search.upload_documents(docs[i : i + SEARCH_UPLOAD_BATCH])

    return {
        "index": index_name,
        "blobs_uploaded": len(blob_paths),
        "chunks_indexed": len(all_chunks),
    }


if __name__ == "__main__":  # python -m src.ingest
    from dotenv import load_dotenv
    load_dotenv()
    summary = ingest()
    print("\nSummary:", summary)
