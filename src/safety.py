"""
Azure AI Content Safety wrapper.

Two checks the agent uses:
  - check_input(user_query)  → run BEFORE intent_router. Blocks malicious /
    out-of-policy prompts up front; saves us LLM calls + cost.
  - check_output(answer)     → run AFTER generator_node. Blocks model output
    that drifts into unsafe territory even with a safe prompt.

Both return a SafetyResult: pass/fail + the worst category + raw severities.
The agent treats fail as a short-circuit equivalent to out_of_scope and
records the verdict in the FinalRagTrace.

If AZURE_CONTENT_SAFETY_ENDPOINT isn't set, both functions return PASS
silently — keeps local dev / unit tests working when CS isn't provisioned.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions,
    TextCategory,
)
from azure.core.credentials import AzureKeyCredential

# Severity threshold (Content Safety scores 0/2/4/6, low→high).
# 4 = "Medium" — block anything Medium or above.
SEVERITY_BLOCK_THRESHOLD = 4

CATEGORIES = (
    TextCategory.HATE,
    TextCategory.VIOLENCE,
    TextCategory.SELF_HARM,
    TextCategory.SEXUAL,
)


@dataclass
class SafetyResult:
    passed: bool
    blocked_category: str | None
    severities: dict[str, int]    # category → severity (0/2/4/6)
    skipped: bool = False         # true if Content Safety not configured
    error: str | None = None      # populated if the API call itself failed

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _get_client() -> ContentSafetyClient | None:
    endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_CONTENT_SAFETY_KEY", "").strip()
    if not endpoint or not key:
        return None
    return ContentSafetyClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def _analyze(text: str) -> SafetyResult:
    client = _get_client()
    if client is None:
        return SafetyResult(passed=True, blocked_category=None, severities={}, skipped=True)
    if not text or not text.strip():
        return SafetyResult(passed=True, blocked_category=None, severities={})

    try:
        resp = client.analyze_text(AnalyzeTextOptions(text=text, categories=list(CATEGORIES)))
    except Exception as e:
        # On API failure, fail OPEN (passed=True) so a CS outage doesn't break
        # the whole RAG pipeline. Record the error for the trace.
        return SafetyResult(passed=True, blocked_category=None, severities={}, error=f"{type(e).__name__}: {e}")

    severities = {item.category: int(item.severity) for item in (resp.categories_analysis or [])}  # type: ignore[arg-type]
    worst_cat, worst_sev = None, 0
    for cat, sev in severities.items():
        if sev > worst_sev:
            worst_sev = sev
            worst_cat = cat
    if worst_sev >= SEVERITY_BLOCK_THRESHOLD:
        return SafetyResult(passed=False, blocked_category=worst_cat, severities=severities)
    return SafetyResult(passed=True, blocked_category=None, severities=severities)


def check_input(user_query: str) -> SafetyResult:
    """Pre-router gate."""
    return _analyze(user_query)


def check_output(answer: str) -> SafetyResult:
    """Post-generator gate."""
    return _analyze(answer)
