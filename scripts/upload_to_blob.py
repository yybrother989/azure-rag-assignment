"""
Upload specific local files to Azure Blob Storage (kb-docs container).
The Function App blob trigger picks up new/updated blobs and runs ingestion automatically.

Use this for targeted uploads only (a single new file or one device folder). For
bulk syncs across the whole data/ tree, use `scripts/sync.py --full-rebuild`
instead — it stops the Function App to avoid the upload→trigger race.

Usage
-----
python scripts/upload_to_blob.py data/devices/epson/manuals/tm.pdf  # single file
python scripts/upload_to_blob.py data/devices/epson/                # one device folder

The path argument is required. Files are uploaded with overwrite=True. A directory
is uploaded recursively (all supported formats inside).

Supported formats: .pdf .md .txt .html .htm
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".html", ".htm"}


def _upload_path(local: Path, data_dir: Path) -> int:
    from src.ingest import _blob_service_client

    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "kb-docs")
    container = _blob_service_client().get_container_client(container_name)

    files: list[Path] = []
    if local.is_file():
        if local.suffix.lower() in SUPPORTED_SUFFIXES:
            files = [local]
        else:
            print(f"Unsupported file type: {local.suffix}")
            return 1
    else:
        files = [p for p in local.rglob("*")
                 if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]

    if not files:
        print(f"No supported files found under {local}")
        return 1

    print(f"Uploading {len(files)} file(s) to container '{container_name}' …\n")
    errors = 0
    for f in sorted(files):
        blob_path = f.relative_to(data_dir).as_posix()
        try:
            with f.open("rb") as fh:
                container.upload_blob(name=blob_path, data=fh, overwrite=True)
            print(f"  ✓ {blob_path}")
        except Exception as e:
            print(f"  ✗ {blob_path}  ({type(e).__name__}: {e})")
            errors += 1

    print(f"\n{len(files) - errors} uploaded, {errors} failed.")
    if errors == 0 and len(files) > 0:
        print("Function App blob trigger will ingest new/updated files automatically.")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload local files to Azure Blob Storage for Function App ingestion."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory to upload (required). Must live inside --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root data directory used to compute blob paths (default: data/).",
    )
    args = parser.parse_args()

    target = args.path.resolve()
    data_dir = args.data_dir.resolve()

    if not target.exists():
        print(f"Path not found: {target}")
        return 1

    # Ensure target is inside data_dir so relative blob paths are well-formed.
    try:
        target.relative_to(data_dir)
    except ValueError:
        print(f"Path {target} is not inside data-dir {data_dir}.")
        print("Use --data-dir to set the root if your data lives elsewhere.")
        return 1

    return _upload_path(target, data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
