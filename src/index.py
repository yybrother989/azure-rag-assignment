"""
Azure AI Search index schema for `kb-chunks`.

One field per metadata dimension on a Chunk + a vector field for hybrid retrieval
+ a semantic configuration so the L2 reranker and extractive captions/answers
work out of the box.

`create_or_update_index()` is idempotent — re-running is safe.
"""

from __future__ import annotations

import os
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from .embed import EMBED_DIM

VECTOR_PROFILE = "kb-vector-profile"
HNSW_CONFIG = "kb-hnsw"


def _index_name() -> str:
    return os.environ.get("AZURE_SEARCH_INDEX", "kb-chunks")


def _semantic_config_name() -> str:
    return os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIG", "kb-semantic")


@lru_cache(maxsize=1)
def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=_index_name(),
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


def _build_index() -> SearchIndex:
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_path", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, retrievable=True),
        SearchableField(name="heading_path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=HNSW_CONFIG,
                parameters=HnswParameters(
                    m=4, ef_construction=400, ef_search=500, metric=VectorSearchAlgorithmMetric.COSINE
                ),
            )
        ],
        profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG)],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=_semantic_config_name(),
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="heading_path"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="category")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=_index_name(),
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def create_or_update_index() -> str:
    """Idempotent: creates the index if missing, updates the schema otherwise."""
    client = get_index_client()
    idx = _build_index()
    client.create_or_update_index(idx)
    return idx.name
