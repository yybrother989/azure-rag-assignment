"""
Batched Azure OpenAI embeddings (`text-embedding-3-small`, 1536 dim).

Model access goes through an Azure AI Foundry Project by default —
`AIProjectClient.get_openai_client()` returns an `AzureOpenAI`-compatible
client that routes calls through the Foundry project (AAD-authed via
DefaultAzureCredential, deployments resolved by the project's connections).

If `AZURE_AI_FOUNDRY_PROJECT_ENDPOINT` is unset, falls back to a direct
AzureOpenAI client against `AZURE_OPENAI_ENDPOINT` with the API key.
This matters for running notebooks/CLI before the Foundry role
assignment has propagated.

Both code paths return objects with the same `.chat` / `.embeddings` API,
so callers (generate.py, agent.py, ingest.py) need no changes.
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import AzureOpenAI

EMBED_DIM = 1536
BATCH_SIZE = 16


@lru_cache(maxsize=1)
def get_azure_openai_client():
    """Return an OpenAI-compatible client. AAD-first, key-fallback.

    When AZURE_AI_FOUNDRY_PROJECT_ENDPOINT is set, we authenticate via
    DefaultAzureCredential against the underlying AI Services account
    (Foundry project still owns the deployments + RBAC; this routes
    inference calls through the proven `cognitiveservices.azure.com`
    OpenAI endpoint rather than the project's discovery URL, which
    in current SDKs constructs a path that 404s for inference).
    """
    foundry_endpoint = os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
    if foundry_endpoint:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def _embed_deployment() -> str:
    return os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]


def embed_query(text: str) -> list[float]:
    """Embed a single query string (used at search time)."""
    client = get_azure_openai_client()
    resp = client.embeddings.create(input=text, model=_embed_deployment())
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches, preserving input order."""
    if not texts:
        return []
    client = get_azure_openai_client()
    deployment = _embed_deployment()
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(input=batch, model=deployment)
        # API guarantees order matches input
        out.extend(item.embedding for item in resp.data)
    return out
