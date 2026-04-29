"""
Azure Function — blob-trigger auto-ingest.

When a blob is added/replaced under `kb-docs/{name}`, this function downloads
it, runs extract → chunk → embed → upsert into the AI Search index. The
operator no longer has to remember to `python -m src.ingest` after dropping
new files into the blob container.

Uses the SAME extract/chunk/embed/index modules as the local ingestor — to
make that import work, deploy_functions.sh copies the repo's `src/` directory
into this function package's site root.

Env vars (set on the Function App via Bicep + deploy.sh):
  AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_CONTAINER
  AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX, AZURE_SEARCH_API_KEY
  AZURE_AI_FOUNDRY_PROJECT_ENDPOINT, AZURE_AI_FOUNDRY_PROJECT_NAME
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_DEPLOYMENT, AZURE_OPENAI_CHAT_DEPLOYMENT
  AZURE_DOC_INTELLIGENCE_ENDPOINT, AZURE_DOC_INTELLIGENCE_KEY
  AZURE_FIGURES_CONTAINER, ENABLE_FIGURE_CAPTIONS
  APPLICATIONINSIGHTS_CONNECTION_STRING
"""

from __future__ import annotations

import logging

import azure.functions as func
from opentelemetry.trace import Status, StatusCode

app = func.FunctionApp()
logger = logging.getLogger("azure-observable-rag")
logger.setLevel(logging.INFO)


def _force_flush_telemetry() -> None:
    """Best-effort flush for short-lived executions."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
    except Exception:
        pass


@app.blob_trigger(
    arg_name="blob",
    path="kb-docs/{name}",
    connection="AzureWebJobsStorage",   # uses the function app's MSI / connection string
)
def auto_ingest(blob: func.InputStream) -> None:
    """Triggered for any add/update under kb-docs/. Triggers a single-file ingest."""
    from src.telemetry import get_tracer

    blob_path = blob.name.removeprefix("kb-docs/")
    size_kb = (blob.length or 0) // 1024
    tracer = get_tracer()

    with tracer.start_as_current_span("auto_ingest") as span:
        span.set_attribute("kb.blob_path", blob_path)
        span.set_attribute("kb.blob_size_kb", size_kb)
        logger.info("auto_ingest fired: blob=%s size=%dKB", blob_path, size_kb)

        # Late import — keeps cold-start fast for unrelated triggers.
        from src.ingest import ingest_single

        try:
            result = ingest_single(blob_path)
            chunks_indexed = int(result.get("chunks_indexed", 0))
            span.set_attribute("kb.chunks_indexed", chunks_indexed)
            span.set_attribute("kb.skipped", bool(result.get("skipped", False)))
            logger.info(
                "auto_ingest done: blob=%s chunks_indexed=%d skipped=%s",
                blob_path,
                chunks_indexed,
                bool(result.get("skipped", False)),
            )
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            logger.exception("auto_ingest FAILED for %s: %s", blob_path, e)
            raise   # surface to App Insights so we can alert
        finally:
            _force_flush_telemetry()
