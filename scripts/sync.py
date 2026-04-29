"""
Knowledge-base sync tool: keeps blob storage and AI Search index in sync with local data/.

Modes
-----
(default)        Compute diff between blob container and AI Search index.
                 Print additions (+) and deletions (-) explicitly, then
                 prompt Y/n before executing.
--full-rebuild   Wipe ALL chunks from the index, re-upload data/ to blob,
                 and re-ingest every document. Use after schema changes or
                 metadata bug fixes (e.g. category="other" for all chunks).
--show-diff      Print the diff and exit — no prompt, no changes.
--data-dir PATH  Override the data directory (default: data/ next to repo root).

Usage
-----
python scripts/sync.py                          # diff sync
python scripts/sync.py --full-rebuild           # wipe + rebuild
python scripts/sync.py --show-diff              # preview only
python scripts/sync.py --data-dir /path/to/data # custom data dir
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")


def _all_blob_paths() -> set[str]:
    from src.ingest import _blob_service_client

    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
    container = _blob_service_client().get_container_client(container_name)
    return {b.name for b in container.list_blobs()}


def _all_indexed_paths() -> set[str]:
    from src.index import get_search_client

    client = get_search_client()
    results = client.search("*", select=["source_path"], top=1000)
    return {r["source_path"] for r in results}


def _wipe_all_chunks() -> int:
    from src.index import get_search_client

    client = get_search_client()
    results = list(client.search("*", select=["chunk_id"], top=1000))
    if not results:
        return 0
    chunk_ids = [r["chunk_id"] for r in results]
    client.delete_documents(documents=[{"chunk_id": cid} for cid in chunk_ids])
    # Search index deletions are eventual; keep deleting in pages until empty.
    while True:
        remaining = list(client.search("*", select=["chunk_id"], top=1000))
        if not remaining:
            break
        client.delete_documents(documents=[{"chunk_id": r["chunk_id"]} for r in remaining])
    return len(chunk_ids)


def _confirm(prompt: str) -> bool:
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes", "")


@contextmanager
def _function_app_paused():
    """Stop the Function App for the duration of the block, then restart it.

    The blob trigger fires on every write to kb-docs (including overwrite=True
    re-uploads), so concurrent execution with sync.py would race. If the deployed
    code differs from the local repo, the two pipelines produce different chunk_ids
    for the same source and both sets land in the index. Stopping the Function App
    eliminates the race entirely.

    Skipped when AZURE_FUNCTION_APP / AZURE_RESOURCE_GROUP are unset, or when az
    CLI isn't on PATH — useful for environments without Function App control.
    """
    func_app = os.environ.get("AZURE_FUNCTION_APP")
    rg = os.environ.get("AZURE_RESOURCE_GROUP")
    az_path = shutil.which("az")

    if not func_app or not rg or not az_path:
        if not az_path:
            print("  ⚠ az CLI not found — skipping Function App stop/start. "
                  "Concurrent ingest may race.")
        elif not (func_app and rg):
            print("  ⚠ AZURE_FUNCTION_APP / AZURE_RESOURCE_GROUP not set — "
                  "skipping Function App stop/start.")
        yield
        return

    print(f"  Stopping Function App {func_app} …")
    subprocess.run([az_path, "functionapp", "stop", "--name", func_app,
                    "--resource-group", rg], check=True)
    try:
        yield
    finally:
        print(f"  Restarting Function App {func_app} …")
        subprocess.run([az_path, "functionapp", "start", "--name", func_app,
                        "--resource-group", rg], check=True)


def cmd_show_diff() -> int:
    print("Computing diff …")
    blob_paths = _all_blob_paths()
    indexed_paths = _all_indexed_paths()

    to_ingest = sorted(blob_paths - indexed_paths)
    to_delete = sorted(indexed_paths - blob_paths)

    if not to_ingest and not to_delete:
        print("  Already in sync — nothing to do.")
        return 0

    if to_ingest:
        print("\n  Blobs to ingest (in blob, not in index):")
        for p in to_ingest:
            print(f"    + {p}")

    if to_delete:
        print("\n  Chunks to delete (in index, blob removed):")
        for p in to_delete:
            print(f"    - {p}")

    print(f"\n  {len(to_ingest)} to ingest, {len(to_delete)} to delete")
    return 0


def cmd_diff_sync() -> int:
    print("Computing diff …")
    blob_paths = _all_blob_paths()
    indexed_paths = _all_indexed_paths()

    to_ingest = sorted(blob_paths - indexed_paths)
    to_delete = sorted(indexed_paths - blob_paths)

    if not to_ingest and not to_delete:
        print("  Already in sync — nothing to do.")
        return 0

    if to_ingest:
        print("\n  Blobs to ingest (in blob, not in index):")
        for p in to_ingest:
            print(f"    + {p}")

    if to_delete:
        print("\n  Chunks to delete (in index, blob removed):")
        for p in to_delete:
            print(f"    - {p}")

    print(f"\n  {len(to_ingest)} to ingest, {len(to_delete)} to delete")

    if not _confirm("\nProceed? [Y/n] "):
        print("Aborted.")
        return 0

    from src.ingest import (
        _delete_existing_chunks,
        _delete_existing_figures,
        ingest_single,
    )

    errors = 0
    for i, path in enumerate(to_ingest, 1):
        print(f"[{i}/{len(to_ingest)}] Ingesting {path} …")
        try:
            result = ingest_single(path)
            print(f"       → {result.get('chunks_indexed', 0)} chunks indexed")
        except Exception as e:
            print(f"       ✗ FAILED: {type(e).__name__}: {e}")
            errors += 1

    for i, path in enumerate(to_delete, 1):
        print(f"[{i}/{len(to_delete)}] Deleting chunks + figures for {path} …")
        try:
            n_chunks = _delete_existing_chunks(path)
            n_figs = _delete_existing_figures(path)
            print(f"       → {n_chunks} chunks, {n_figs} figure PNGs removed")
        except Exception as e:
            print(f"       ✗ FAILED: {type(e).__name__}: {e}")
            errors += 1

    print(f"\nDone. {len(to_ingest)} ingested, {len(to_delete)} deleted. {errors} error(s).")
    return 1 if errors else 0


def cmd_full_rebuild(data_dir: Path) -> int:
    from src.index import get_search_client

    blob_paths = _all_blob_paths()
    chunk_count = get_search_client().get_document_count()

    print("Full rebuild plan:")
    print(f"  1. Wipe all {chunk_count} chunks from AI Search index")
    print(f"  2. Re-upload {data_dir} to blob ({len(blob_paths)} blobs currently in container)")
    print(f"  3. Extract → chunk → embed → upsert every document")
    print("\n  WARNING: this will DELETE all existing index data before rebuilding.")

    if not _confirm("\nProceed with full rebuild? [Y/n] "):
        print("Aborted.")
        return 0

    with _function_app_paused():
        print("\n[1/3] Wiping index …")
        deleted = _wipe_all_chunks()
        print(f"       → {deleted} chunks removed")

        print(f"\n[2/3] Uploading {data_dir} to blob and re-ingesting …")
        from src.ingest import ingest

        summary = ingest(data_dir)
        print(f"       → {summary.get('blobs_uploaded', 0)} blobs uploaded, "
              f"{summary.get('chunks_indexed', 0)} chunks indexed")

    print("\nFull rebuild complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync local data/ with Azure Blob + AI Search.")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="Wipe all index chunks and re-ingest everything.")
    parser.add_argument("--show-diff", action="store_true",
                        help="Print diff only — no prompt, no changes.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="Local data directory (default: data/).")
    args = parser.parse_args()

    if args.show_diff:
        return cmd_show_diff()
    if args.full_rebuild:
        return cmd_full_rebuild(args.data_dir)
    return cmd_diff_sync()


if __name__ == "__main__":
    raise SystemExit(main())
