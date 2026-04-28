"""
Pipeline orchestrator: local /data -> Blob -> extract -> chunk -> embed -> Search.

Idempotent end-to-end:
- Blob uploads use overwrite=True
- Chunk IDs are content-hashed (see chunk.make_chunk_id), so re-running upserts
  identical chunks back into the same document slot.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from .chunk import Chunk, chunk_document
from .embed import embed_batch
from .extract import extract
from .index import create_or_update_index, get_search_client

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".html", ".htm"}
TOP_FOLDERS = {"manuals", "troubleshooting", "policies"}
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


def _chunk_to_search_doc(c: Chunk, vector: list[float]) -> dict:
    d = asdict(c)
    d["content_vector"] = vector
    return d


def ingest(data_dir: Path | str = "data") -> dict:
    """Run the full pipeline. Returns a summary dict."""
    data_dir = Path(data_dir)

    print("[1/5] Ensuring AI Search index exists")
    index_name = create_or_update_index()

    print(f"[2/5] Uploading documents from {data_dir} to blob storage")
    blob_paths = upload_local_data(data_dir)
    if not blob_paths:
        print(f"  no supported files found under {data_dir}/{{manuals,troubleshooting,policies}}")
        return {"index": index_name, "blobs_uploaded": 0, "chunks_indexed": 0}
    print(f"  uploaded {len(blob_paths)} blobs")

    print("[3/5] Extracting + chunking each document")
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
    container = _blob_service_client().get_container_client(container_name)
    all_chunks: list[Chunk] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for blob_path in blob_paths:
            local = td_path / Path(blob_path).name
            with local.open("wb") as out:
                out.write(container.download_blob(blob_path).readall())
            try:
                extracted = extract(local, blob_path)
                doc_chunks = chunk_document(extracted)
                all_chunks.extend(doc_chunks)
                print(f"  {blob_path}: {len(doc_chunks)} chunks")
            except Exception as e:  # extraction failures shouldn't kill the run
                print(f"  {blob_path}: SKIPPED ({type(e).__name__}: {e})")

    if not all_chunks:
        return {"index": index_name, "blobs_uploaded": len(blob_paths), "chunks_indexed": 0}

    print(f"[4/5] Embedding {len(all_chunks)} chunks")
    vectors = embed_batch([c.content for c in all_chunks])

    print(f"[5/5] Upserting into Azure AI Search index '{index_name}'")
    search = get_search_client()
    docs = [_chunk_to_search_doc(c, v) for c, v in zip(all_chunks, vectors)]
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
