"""
OpenTelemetry → Azure Application Insights wiring.

Set APPLICATIONINSIGHTS_CONNECTION_STRING in .env and call `configure()`
once at process start. After that, every `tracer.start_as_current_span()`
in agent.py / generate.py / etc. lights up as a request in App Insights,
queryable via KQL. The OpenAI auto-instrumentor adds chat-completion
spans with token-count + model attributes for free.
"""

from __future__ import annotations

import os
from functools import lru_cache

from opentelemetry import trace
from opentelemetry.trace import Tracer

_CONFIGURED = False


@lru_cache(maxsize=1)
def configure() -> None:
    """One-shot setup. Safe to call repeatedly (lru_cache makes it idempotent).
    Silent no-op if APPLICATIONINSIGHTS_CONNECTION_STRING isn't set, so local
    dev / unit tests don't need App Insights to run."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if not conn:
        # No connection string → spans become no-ops, code keeps working.
        _CONFIGURED = True
        return

    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(
        connection_string=conn,
        # Logger name maps to the cloud-role-name in App Insights;
        # set it so we can filter our app vs. other tenants in shared workspaces.
        logger_name="azure-observable-rag",
    )

    # Auto-instrument OpenAI calls — adds spans w/ model, prompt_tokens,
    # completion_tokens, latency. Saves ~20 LOC of manual span code.
    try:
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor  # type: ignore[import-not-found]

        OpenAIInstrumentor().instrument()
    except ImportError:
        pass  # auto-instrumentation is best-effort

    _CONFIGURED = True


def get_tracer() -> Tracer:
    configure()
    return trace.get_tracer("azure-observable-rag")
