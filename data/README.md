# Sample Documents

Drop ~10 documents into the folders below before running ingestion. The folder name maps 1:1 to the chunk's `category` field in the AI Search index, which the LangGraph router uses to scope retrieval.

```
data/
  manuals/          # PDF — product manuals (digital or scanned)
  troubleshooting/  # Markdown — error / fix guides
  policies/         # Plain text — org policies
```

The ingestion pipeline (`python -m src.cli ingest`) will:
1. Upload everything under `data/` to the `kb-docs` blob container, preserving folder structure
2. Extract text via Docling (or Doc Intelligence, if `OCR_BACKEND=azure_di`)
3. Chunk with Docling's structure-aware `HybridChunker`
4. Embed each chunk with `text-embedding-3-small`
5. Upsert into the `kb-chunks` Azure AI Search index (idempotent — re-runs are safe)
