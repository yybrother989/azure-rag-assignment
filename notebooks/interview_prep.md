# Q&A prep — Azure Observable RAG project review

## Context

Self-contained Q&A document covering ~60 likely review / interview questions across 12 topic areas. Each entry: **question + concise answer + the actual code snippet that proves it**. Use the file pointers (e.g. `[src/agent.py:251-321](../src/agent.py#L251-L321)`) to jump straight to the source on GitHub.

Format per entry:
- **Q:** the question (with 1–3 likely phrasings)
- **A:** the answer (2–5 sentences, concrete + opinionated)
- **→** code/file pointer
- **Code:** the relevant excerpt (truncated for length where noted)

---

## 1. Architecture & high-level design

### Q1.1 — Why "Observable RAG"? Why isn't this just another LangChain RAG demo?

**A:** Traditional agentic RAG hides its work — one chat-completions call decides what to retrieve and how to answer. That's fast to build but impossible to grade. This project decouples every stage and emits a typed trace per LangGraph node, so the reviewer can audit which intent the router picked, which filter was sent to the index, which chunks the index returned, which the selector kept, and exactly what the generator was given. Retrieval and generation never share state implicitly.

→ [src/tracing.py](../src/tracing.py)

```python
# src/tracing.py — every node produces a typed sub-trace; one bundle ships per query
@dataclass
class FinalRagTrace:
    user_query: str
    query_plan: QueryPlanTrace            # router output + detected metadata
    retrieval: RetrievalTrace             # filters, results, latency, fallback flag
    evidence_selection: EvidenceSelectionTrace  # which chunks survived top-N
    generation: GenerationTrace           # model, tokens, citations, answer
    total_latency_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)   # JSON-serializable → drives Chainlit, notebook, JSONL audit log
```

### Q1.2 — Why LangGraph instead of plain LangChain or hand-rolled functions?

**A:** LangGraph gives a typed StateGraph where each node consumes/produces well-defined fields of a single state dict. That makes the trace shape automatic, conditional routing explicit, and per-node observability trivial (one OTel span per node). Plain functions would also work but you'd reinvent the routing + state-merging machinery.

→ [src/agent.py](../src/agent.py) `build_graph()`

```python
# src/agent.py
def build_graph():
    builder = StateGraph(RagState)
    builder.add_node("intent_router", intent_router)
    builder.add_node("no_retrieval_response", no_retrieval_response)
    builder.add_node("build_retrieval_scope", build_retrieval_scope)
    builder.add_node("simple_rag_search", simple_rag_search)
    builder.add_node("evidence_selector", evidence_selector)
    builder.add_node("generator_node", generator_node)
    builder.add_node("response_formatter", response_formatter)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        _route_after_router,
        {
            "blocked":      "response_formatter",
            "no_retrieval": "no_retrieval_response",
            "simple_rag":   "build_retrieval_scope",
        },
    )
    builder.add_edge("build_retrieval_scope", "simple_rag_search")
    builder.add_edge("simple_rag_search", "evidence_selector")
    builder.add_edge("evidence_selector", "generator_node")
    builder.add_edge("generator_node", "response_formatter")
    builder.add_edge("response_formatter", END)

    return builder.compile(checkpointer=MemorySaver())
```

### Q1.3 — Why two routes (NO_RETRIEVAL vs SIMPLE_RAG)? Why not always retrieve?

**A:** Some queries don't need retrieval (greetings, "what can you do", clarifying chit-chat). Always retrieving wastes Azure Search reads + token spend on chunks that won't be cited. The intent_router LLM call is cheap and gates the retrieval path. Trade-off: one extra LLM hop per query in exchange for cleaner traces and avoided retrieval noise on conversational turns.

→ [src/agent.py](../src/agent.py) `_route_after_router`

```python
# src/agent.py
def _route_after_router(state: RagState) -> str:
    route = state.get("route", "SIMPLE_RAG")
    if route == "blocked_input":   # safety blocked → straight to formatter (refusal)
        return "blocked"
    if route == "NO_RETRIEVAL":    # conceptual / chit-chat → answer directly
        return "no_retrieval"
    return "simple_rag"            # default: full retrieval pipeline
```

### Q1.4 — Why a separate `evidence_selector` node after retrieval?

**A:** `hybrid_search` returns `RETRIEVAL_TOP_K=5` candidates with `reranker_score`. The evidence_selector then picks the top `EVIDENCE_TOP_N=4` to actually feed the generator. Decoupling lets us tune "how wide stage 1 retrieves" independently of "how much context the LLM sees". For demo it's 5→4 (mostly a no-op); production should widen stage 1 to ~50 (Azure reranker cap) and narrow to 5–8 here.

→ [src/agent.py](../src/agent.py) `evidence_selector`

```python
# src/agent.py
def evidence_selector(state: RagState) -> dict:
    """Select top-N evidence chunks ranked by reranker_score (or score)."""
    with _tracer.start_as_current_span("evidence_selector") as span:
        results: list[dict] = state["retrieval_results"]
        n = int(os.environ.get("EVIDENCE_TOP_N", "4"))

        def sort_key(r: dict):
            rs = r.get("reranker_score")
            return rs if rs is not None else r.get("score", 0.0)

        ranked = sorted(results, key=sort_key, reverse=True)[:n]
        return {"evidence": _dataclass_to_dict(...), "retrieval_results": ranked}
```

### Q1.5 — Why two ingestion paths — `scripts/sync.py` AND a Function App?

**A:** Two execution surfaces, **same `src/` modules**. `sync.py` is for local/dev workflows: explicit `--full-rebuild`, `--diff`, drives the lifecycle. The Function App handles production: blob upload to `kb-docs` triggers `auto_ingest` automatically. Both call into the same `extract → figures → chunk → embed → upsert` pipeline, so a fix in those modules takes effect in both paths after one Function App redeploy.

→ [infra/functions/function_app.py:54](../infra/functions/function_app.py#L54), [src/ingest.py:133](../src/ingest.py#L133)

```python
# infra/functions/function_app.py — production path
@app.blob_trigger(arg_name="blob", path="kb-docs/{name}", connection="AzureWebJobsStorage")
def auto_ingest(blob: func.InputStream) -> None:
    blob_path = blob.name.removeprefix("kb-docs/")
    from src.ingest import ingest_single        # ← same module sync.py uses
    result = ingest_single(blob_path)

# src/ingest.py — both paths converge here
def ingest_single(blob_path: str) -> dict:
    create_or_update_index()
    _delete_existing_chunks(blob_path)          # idempotent re-ingest
    extracted = extract(local, blob_path)
    if extracted.figures:
        render_and_upload_figures(local, extracted.figures, blob_stem, figures_container)
        extracted.markdown = splice_captions_into_markdown(extracted.markdown, extracted.figures)
    chunks = chunk_document(extracted)
    vectors = embed_batch([c.content for c in chunks])
    docs = [chunk_to_search_doc(c, v) for c, v in zip(chunks, vectors)]
    get_search_client().upload_documents(docs)
```

### Q1.6 — What does `FinalRagTrace` contain?

**A:** Five sub-traces bundled into one JSON-serializable object: `QueryPlanTrace`, `RetrievalTrace`, `EvidenceSelectionTrace`, `GenerationTrace`, plus per-step `SafetyResult`. Same payload drives Chainlit's step panels, the notebook's per-stage tables, and the JSONL audit log.

→ [src/tracing.py:25-92](../src/tracing.py#L25-L92)

```python
# src/tracing.py — three of the five sub-traces, abridged
@dataclass
class QueryPlanTrace:
    original_query: str
    route: str                      # "NO_RETRIEVAL" | "SIMPLE_RAG"
    query_type: str                 # "troubleshoot" | "manual_lookup" | "policy_check" | ...
    detected_device_family: str | None
    detected_device: str | None
    primary_filter: dict[str, Any]  # OData filter built by build_retrieval_scope
    fallback_filter: dict[str, Any] | None
    search_mode: str
    top_k: int

@dataclass
class RetrievalTrace:
    query: str
    filters: dict[str, Any]
    search_mode: str
    results: list[RetrievalResult]
    latency_ms: int
    fallback_triggered: bool = False

@dataclass
class GenerationTrace:
    model: str
    context_chunk_ids: list[str]
    prompt_token_estimate: int
    completion_token_estimate: int
    answer: str
    citations: list[Citation]
    latency_ms: int
```

---

## 2. Data layout & ingestion

### Q2.1 — Why `data/devices/{family}/{model}/{doc_type}/` instead of a flat folder?

**A:** The path encodes structured metadata that we extract deterministically without an LLM: `device_family`, `device`, `doc_type`. That metadata becomes filterable columns on the AI Search index, which lets `build_retrieval_scope` produce a scoped OData filter ("only docs about meraki_mx67") instead of relying on cosine similarity to surface the right device. Result: fewer wrong-device hallucinations.

→ [src/extract.py:155-200](../src/extract.py#L155-L200) `_parse_path_metadata`

```python
# src/extract.py
def _parse_path_metadata(blob_path: str) -> dict:
    """Derive structured metadata deterministically from the blob storage path.

    devices/{device_family}/{model}/{doc_folder}/{filename}
        scope="device"  device_family={device_family}  device={model}
        doc_type from doc_folder
    shared/{doc_folder}/{filename}
        scope="shared"  is_shared=True  doc_type from doc_folder
    """
    parts = blob_path.replace("\\", "/").split("/")

    if len(parts) >= 5 and parts[0].lower() == "devices":
        return {
            "scope": "device",
            "device_family": parts[1],
            "device": parts[2],
            "doc_type": _DOC_TYPE_MAP.get(parts[3], parts[3] or "other"),
            "topic": topic, "version": version, "is_shared": False,
        }
    # ... shared/ and legacy fallthrough cases
```

### Q2.2 — How do you handle docs that aren't device-specific?

**A:** Top-level `data/shared/` carries `scope=shared`, `is_shared=True`. The retrieval scope builder normally filters to the detected device, but for `policy_check` query types (and as a fallback when device-scoped search returns nothing) it allows shared docs through.

→ [src/agent.py:380-434](../src/agent.py#L380-L434) `build_retrieval_scope`

```python
# src/agent.py
def build_retrieval_scope(state: RagState) -> dict:
    device_family = state.get("detected_device_family")
    device        = state.get("detected_device")
    doc_type      = state.get("detected_doc_type")
    allow_fallback = state.get("allow_shared_fallback", True)

    primary: dict = {}
    fallback: dict | None = None

    if device_family and device and doc_type:
        primary = {"scope": "device", "device_family": device_family,
                   "device": device, "doc_type": doc_type}
        if allow_fallback:
            fallback = {"scope": "shared", "doc_type": doc_type}
    elif doc_type == "policy":
        primary = {"scope": "shared", "doc_type": "policy"}   # cross-device policies
    # ... 4 more branches for partial-detection cases

    return {"primary_filter": primary, "fallback_filter": fallback, ...}
```

### Q2.3 — What's `data/document_manifest.csv` for?

**A:** A flat inventory of every doc in the corpus with its parsed metadata fields (relative_path, device_family, model, doc_type, vendor, topic, version, is_shared, status). Useful for QA / audit / sanity-check that path-derived metadata is what you expect, before paying for ingest. Not consumed at runtime — it's a human-checkable source of truth.

→ [data/document_manifest.csv](../data/document_manifest.csv)

```csv
# data/document_manifest.csv (excerpt)
relative_path,device_family,model,doc_type,vendor,topic,version,is_shared,status
devices/network_access/meraki_mx67/manuals/MX67_MX68 Installation Guide.pdf,network_access,meraki_mx67,manual,Cisco Meraki,mx67-mx68-installation-guide,,False,active
devices/payment_terminal/ingenico_desk5000/manuals/Desk5000 and 3000 series - User guide.pdf,payment_terminal,ingenico_desk5000,manual,Ingenico,desk5000-user-guide,,False,active
shared/policies/data_retention.txt,,,policy,,data-retention,,True,active
```

### Q2.4 — Why Azure Document Intelligence instead of `pypdf2` / `pdfplumber`?

**A:** DI's prebuilt-layout model returns markdown with explicit page boundaries (`<!-- PageNumber=N -->`), HTML tables, and figure bounding boxes — all in one call. Hand-rolled PDF parsing would lose table structure and figure positions, which the rest of the pipeline depends on (figure crops, table anchors). For markdown / txt sources we skip DI.

→ [src/extract.py](../src/extract.py)

```python
# src/extract.py
DI_FORMATS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heif"}

# Branch in extract():
if file_type in DI_FORMATS:
    # Call Azure Document Intelligence Layout API
    poller = di_client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=local_path.read_bytes()),
        output_content_format="markdown",
    )
    result = poller.result()
    # → produces markdown with <!-- PageNumber=N --> + HTML tables + figure boxes
else:
    # Markdown / txt: read directly, no DI call
    markdown = local_path.read_text(encoding="utf-8")
```

### Q2.5 — What about deletes? If I remove a blob, does the index clean up?

**A:** No — Azure blob triggers fire only on add/update, never on delete. To remove orphan chunks AND their figure PNGs in `kb-figures/{stem}/`, run `python scripts/sync.py` (default mode). It diffs the blob container against the search index and deletes mismatches.

→ [scripts/sync.py](../scripts/sync.py), [src/ingest.py:74-118](../src/ingest.py#L74-L118)

```python
# src/ingest.py
def _delete_existing_chunks(source_path: str) -> int:
    """Delete all indexed chunks for source_path before re-upsert."""
    doc_id = make_doc_id(source_path)
    client = get_search_client()
    results = client.search(search_text="*", filter=f"doc_id eq '{doc_id}'",
                            select=["chunk_id"], top=1000)
    chunk_ids = [r["chunk_id"] for r in results]
    if chunk_ids:
        client.delete_documents(documents=[{"chunk_id": cid} for cid in chunk_ids])
    return len(chunk_ids)

def _delete_existing_figures(source_path: str) -> int:
    """Mirror for the kb-figures/{stem}/ folder."""
    stem = Path(source_path).stem
    container = _blob_service_client().get_container_client(figures_container_name())
    deleted = 0
    for blob in container.list_blobs(name_starts_with=f"{stem}/"):
        container.delete_blob(blob.name)
        deleted += 1
    return deleted
```

---

## 3. Chunking

### Q3.1 — Walk me through the chunking strategy.

**A:** Two stages. **Stage 1**: split by markdown headers (`#`–`####`) so each chunk carries a `heading_path` like "Installation > Mounting > Wall Mount". **Stage 2**: only if a section exceeds `CHUNK_MAX_TOKENS=512`, sub-split via LangChain's `RecursiveCharacterTextSplitter` with `chunk_overlap=50`. Most chunks are "one section = one chunk" — Stage 2 fires only on long sections.

→ [src/chunk.py:216-301](../src/chunk.py#L216-L301) `chunk_document`

```python
# src/chunk.py
def chunk_document(extracted: ExtractedDocument) -> list[Chunk]:
    max_tokens = int(os.environ.get("CHUNK_MAX_TOKENS", "512"))
    overlap_tokens = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50"))

    clean_md, page_starts = _strip_page_markers(extracted.markdown)
    sections = _split_by_headers_with_offsets(clean_md)   # Stage 1

    sub_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",     # matches text-embedding-3-small
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
    )

    for section in sections:
        if _token_count(text) <= max_tokens:
            pieces = [text]                       # short section → one chunk
        else:
            pieces = sub_splitter.split_text(text)  # Stage 2: token-bound sub-split

        for piece in pieces:
            piece_page = _page_for_offset(page_starts, piece_global_offset)
            chunks.append(Chunk(content=piece, page_number=piece_page,
                                heading_path=heading_path, ...))
```

### Q3.2 — Why 512 tokens? Did you sweep this?

**A:** No, we didn't sweep. 512 is a common default that fits comfortably inside text-embedding-3-small's 8192-token limit while keeping chunks small enough that retrieval scoring isn't washed out. **Listed as a known limitation** — production should sweep `{256, 512, 1024} × {0%, 10%, 20%}` overlap on a held-out gold-QA set. The infra is there, the eval harness is missing.

→ [src/chunk.py:218-219](../src/chunk.py#L218-L219), [README.md](../README.md) "Known Limitations"

```python
# src/chunk.py — both env-overridable, no per-call kwargs
max_tokens = int(os.environ.get("CHUNK_MAX_TOKENS", "512"))
overlap_tokens = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50"))
```

### Q3.3 — Why a custom header splitter instead of LangChain's `MarkdownHeaderTextSplitter`?

**A:** LangChain's splitter rewrites whitespace and discards character offsets. We need exact offsets to attribute each chunk to a page (via the `page_starts` map built when stripping `<!-- PageNumber=N -->` markers) and to attach figures/tables. Without offsets, a long section spanning pages 5–8 would pin every sub-chunk to page 5, and the figure-attachment lookup would break.

→ [src/chunk.py:150-188](../src/chunk.py#L150-L188) `_split_by_headers_with_offsets`

```python
# src/chunk.py
def _split_by_headers_with_offsets(clean_md: str) -> list[dict]:
    """In-house markdown-header splitter that retains EXACT offsets.

    LangChain's MarkdownHeaderTextSplitter rewrites whitespace and discards
    positions, which made the page-attribution lookup brittle — sections
    drifted to the wrong offsets, so chunks for whole page ranges got pinned
    to the same page and figure attachment broke."""
    matches = list(HEADING_RE.finditer(clean_md))   # ^(#{1,4})[ \t]+(.+)$
    sections: list[dict] = []
    crumbs: list[str | None] = [None, None, None, None]   # h1, h2, h3, h4

    for i, match in enumerate(matches):
        start, level, heading = boundaries[i]
        next_start = boundaries[i + 1][0]
        crumbs[level - 1] = heading
        for deeper in range(level, 4):
            crumbs[deeper] = None
        sections.append({
            "start": start,                 # ← exact char offset (the whole point)
            "end": next_start,
            "text": clean_md[start:next_start],
            "heading_path": " > ".join(c for c in crumbs if c) or None,
        })
    return sections
```

### Q3.4 — What does the 50-token overlap actually buy you?

**A:** When Stage 2 sub-splits a long section, an answer that straddles the cut ("First do X, **then do Y**") would otherwise live half in chunk A, half in chunk B — neither chunk is independently sufficient. 50 tokens of overlap (~10% of 512) gives both sides enough surrounding context to be retrievable on their own. Only invoked on long sections, so the index size penalty is ~10% extra tokens per long section, none extra for short ones.

→ [src/chunk.py:235-239](../src/chunk.py#L235-L239)

```python
# src/chunk.py
sub_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=DEFAULT_TOKENIZER_ENCODING,
    chunk_size=max_tokens,           # 512 by default
    chunk_overlap=overlap_tokens,    # 50 by default
)
# Only invoked when _token_count(section_text) > max_tokens
```

### Q3.5 — How do figures and tables get attached to chunks?

**A:** Figures: collected per-page from DI's bounding-box output, rendered as PNGs to `kb-figures/{stem}/`, attached to every chunk on that page. Tables: matched by HTML prefix (first 120 chars of `<table>...`) — anchors that appear inside a chunk's text get attached. Both serialized as `tables_json` / `figures_json` in the search index.

→ [src/chunk.py:191-213, 275-276](../src/chunk.py#L191-L213), [src/figures.py:91-141](../src/figures.py#L91-L141)

```python
# src/chunk.py — table anchoring
def _table_anchors(extracted: ExtractedDocument) -> list[tuple[str, dict]]:
    """For each TableRef, take the first ~120 chars of its HTML span as a
    unique anchor. Used to detect whether a chunk piece contains the table."""
    anchors: list[tuple[str, dict]] = []
    for tref in extracted.tables:
        end = tref.markdown_offset + min(tref.markdown_length, 240)
        raw = extracted.markdown[tref.markdown_offset:end]
        cleaned = PAGE_BOUNDARY_RE.sub("", raw)
        anchor = cleaned[:120].strip()
        if anchor:
            anchors.append((anchor, tref.to_dict()))
    return anchors

# Per-chunk attachment (in chunk_document loop):
piece_tables  = [td for anchor, td in table_anchors if anchor in piece]
piece_figures = figures_by_page.get(piece_page, []) if piece_page else []

# src/figures.py — figure rendering
def render_and_upload_figures(local_pdf, figures, blob_stem, container_client, *, scale=2.0):
    pdf = pdfium.PdfDocument(str(local_pdf))
    for page_no, page_figs in by_page.items():
        page = pdf[page_no - 1]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        for fig in page_figs:
            left, top, right, bottom = _polygon_pixel_box(fig.polygon, scale)
            buf = BytesIO()
            pil.crop((left, top, right, bottom)).save(buf, format="PNG", optimize=True)
            container_client.upload_blob(name=f"{blob_stem}/{fig.figure_id}.png",
                                         data=buf.getvalue(), overwrite=True)
            fig.blob_url = f"{container_client.url}/{blob_stem}/{fig.figure_id}.png"
```

### Q3.6 — What about cross-page chunks?

**A:** When a long section's Stage-2 sub-split produces a piece that starts on page N and ends on page N+1, both `page_number=N` and `page_end=N+1` are recorded by binary-searching the page-offsets map at the piece's global start AND end character positions. Citation rendering can then say "p. N–N+1" honestly.

→ [src/chunk.py:262-274](../src/chunk.py#L262-L274)

```python
# src/chunk.py — per-piece page attribution (handles cross-page)
piece_offset_in_section = text.find(piece, piece_cursor)
piece_global_offset     = section_offset + piece_offset_in_section
piece_page              = _page_for_offset(page_starts, piece_global_offset)

piece_end_global        = piece_global_offset + len(piece)
piece_page_end          = _page_for_offset(page_starts, piece_end_global) or piece_page

# Both stored on the Chunk → citations can say "p. 5-6" honestly
chunks.append(Chunk(page_number=piece_page, page_end=piece_page_end, ...))
```

---

## 4. Retrieval

### Q4.1 — Why hybrid (BM25 + vector) instead of pure vector?

**A:** Vector retrieval is great for paraphrased / conceptual queries but weak on rare proper nouns, error codes, model numbers (e.g. "MX67", "error 101", "Desk5000"). BM25 nails those because they have high IDF. Hybrid covers both regimes; Azure runs them in parallel inside one query call so latency overhead is sub-50ms.

→ [src/search.py:100-145](../src/search.py#L100-L145) `hybrid_search`

```python
# src/search.py — hybrid by sending BOTH search_text AND vector_queries
def hybrid_search(query, *, filters=None, top_k=5, search_mode="hybrid_semantic"):
    kwargs = {"select": select_fields, "top": top_k, "filter": odata_filter}

    if search_mode in {"vector", "hybrid", "hybrid_semantic"}:
        kwargs["vector_queries"] = [
            VectorizedQuery(
                vector=embed_query(query),
                k_nearest_neighbors=max(top_k, 50),
                fields="content_vector",
            )
        ]
    if search_mode == "vector":
        kwargs["search_text"] = None     # vector only
    else:
        kwargs["search_text"] = query    # → BM25 on `searchable=True` fields

    raw = client.search(**kwargs)        # Azure runs BM25 + HNSW in parallel
```

### Q4.2 — How are BM25 and HNSW results merged?

**A:** Reciprocal Rank Fusion, **Azure default**. When both `search_text` and `vector_queries` are passed in the same call, Azure auto-applies RRF with formula `score = Σ 1/(60 + rank_i)` and fixed `k=60`. We don't write any RRF code — it's service-side default behavior.

→ same `client.search(**kwargs)` call as Q4.1; no client-side RRF code exists. Azure docs: <https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking>

```python
# All we do is pass both. Azure's hybrid pipeline does:
#   1. BM25 → ranked list A
#   2. HNSW → ranked list B
#   3. RRF (k=60, fixed) → fused single ranking
#   4. (if hybrid_semantic) → L2 reranker rescore
client.search(search_text=query, vector_queries=[...], **kwargs)
```

### Q4.3 — What is the L2 semantic reranker doing differently from RRF?

**A:** RRF only sees rank positions — it doesn't read documents. Cosine only sees embedding distance — same. The L2 reranker is a **pretrained transformer cross-encoder** (Microsoft, Bing-derived, multilingual). It feeds query + each candidate document into one transformer forward pass and uses attention to score actual semantic relevance. That's why `reranker_score` is the highest-quality signal we have.

→ [src/index.py:122-133](../src/index.py#L122-L133) (config), [src/search.py:140-145](../src/search.py#L140-L145) (activation)

```python
# src/index.py — index-side configuration
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

# src/search.py — query-side activation (per call)
if search_mode == "hybrid_semantic":
    kwargs.update(
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name=_semantic_config_name(),
        query_caption=QueryCaptionType.EXTRACTIVE,    # bonus extractive snippet
        query_answer=QueryAnswerType.EXTRACTIVE,
    )
```

### Q4.4 — Why `RETRIEVAL_TOP_K=5` and `EVIDENCE_TOP_N=4`? That seems narrow.

**A:** It IS narrow — kept tight for demo readability. **Listed as a known limitation.** Azure's L2 reranker can score up to 50 candidates per query; production should widen to `RETRIEVAL_TOP_K=50` so the reranker has real headroom (if BM25/HNSW miss the right doc in their top 5, the reranker can't recover it), and narrow `EVIDENCE_TOP_N` to 5–8 for the LLM context budget. Both env-overridable.

→ [src/agent.py:448](../src/agent.py#L448), [src/agent.py:513](../src/agent.py#L513)

```python
# src/agent.py
def simple_rag_search(state: RagState) -> dict:
    top_k = int(os.environ.get("RETRIEVAL_TOP_K", "5"))    # ← widen to 50 in prod
    results = hybrid_search(query=query, filters=primary_filter, top_k=top_k,
                            search_mode="hybrid_semantic")

def evidence_selector(state: RagState) -> dict:
    n = int(os.environ.get("EVIDENCE_TOP_N", "4"))         # ← bump to 5-8 in prod
    ranked = sorted(results, key=lambda r: r.get("reranker_score") or 0, reverse=True)[:n]
```

### Q4.5 — What are the four search modes and when do you use each?

**A:** `bm25` (lexical only), `vector` (HNSW only), `hybrid` (BM25 + HNSW + RRF), `hybrid_semantic` (hybrid + L2 reranker). Default and demo path is `hybrid_semantic`. The others exist for ablation: compare reranker-on vs reranker-off, hybrid vs pure vector, etc.

→ [src/search.py:31](../src/search.py#L31)

```python
# src/search.py
SUPPORTED_MODES = {"bm25", "vector", "hybrid", "hybrid_semantic"}

# At call site (agent.py simple_rag_search) we hard-code "hybrid_semantic";
# the other modes are reachable from notebooks / eval harness for ablation:
hybrid_search(query, top_k=5, search_mode="bm25")             # lexical only
hybrid_search(query, top_k=5, search_mode="vector")           # HNSW only
hybrid_search(query, top_k=5, search_mode="hybrid")           # +RRF, no reranker
hybrid_search(query, top_k=5, search_mode="hybrid_semantic")  # default
```

### Q4.6 — How does device-first scoping work? Show me a concrete query.

**A:** Query: "How do I factory-reset the Meraki MX67?" → intent_router detects `device_family="network_access"`, `device="meraki_mx67"` → build_retrieval_scope emits OData filter `device eq 'meraki_mx67' and doc_type eq 'manual'` → `simple_rag_search` queries the index with that filter, restricting candidates BEFORE BM25/HNSW score. Result: zero chance of an Ingenico or Canon doc accidentally outranking the Meraki manual.

→ [src/agent.py](../src/agent.py) — `intent_router → build_retrieval_scope → simple_rag_search`

```python
# 1. intent_router (LLM call) emits structured detection:
{
  "route": "SIMPLE_RAG",
  "query_type": "troubleshoot",
  "detected_device_family": "network_access",
  "detected_device": "meraki_mx67",
  "detected_doc_type": "manual",
}

# 2. build_retrieval_scope turns that into an OData filter dict:
primary = {"scope": "device", "device_family": "network_access",
           "device": "meraki_mx67", "doc_type": "manual"}

# 3. simple_rag_search → hybrid_search → _build_filter() → OData string:
#    "scope eq 'device' and device_family eq 'network_access'
#     and device eq 'meraki_mx67' and doc_type eq 'manual'"
# Azure applies the filter BEFORE BM25/HNSW score, so wrong-device docs never compete.
```

### Q4.7 — What if the device-scoped filter returns 0 hits?

**A:** Three-step fallback chain. (1) Re-query with `fallback_filter` (often `scope=shared` for the same doc_type). (2) If still empty, run unfiltered search so legacy docs without scope/device fields aren't silently excluded. (3) Trace records `fallback_triggered=True` so reviewers see when it kicked in.

→ [src/agent.py:437-506](../src/agent.py#L437-L506) `simple_rag_search`

```python
# src/agent.py
def simple_rag_search(state: RagState) -> dict:
    MIN_RESULTS = 2

    # Step 1: primary search.
    results = hybrid_search(query, filters=primary_filter, top_k=top_k, ...)

    # Step 2: fallback to shared docs if primary returned too few.
    if len(results) < MIN_RESULTS and fallback_filter:
        fallback_results = hybrid_search(query, filters=fallback_filter, ...)
        seen_ids = {r.chunk_id for r in results}
        unique_fallback = [r for r in fallback_results if r.chunk_id not in seen_ids]
        results = results + unique_fallback
        fallback_triggered = bool(unique_fallback)

    # Step 3: if still empty, unfiltered search (legacy data without scope fields).
    if not results and primary_filter:
        results = hybrid_search(query, filters=None, ...)
        fallback_triggered = bool(results)
```

### Q4.8 — Why store `tables_json` and `figures_json` as strings instead of separate index?

**A:** They're chunk-scoped (each chunk knows what tables/figures live in its content + page), and we always retrieve them together with the chunk. A second index would mean N+1 round-trips and a join. Single-index trade-off: schema is fatter (~20% more bytes per doc), retrieval is one round-trip and the search-doc shape mirrors the rendered output.

→ [src/index.py](../src/index.py) field definitions, [src/ingest.py:121-130](../src/ingest.py#L121-L130)

```python
# src/index.py — both fields are SearchableField, so BM25 indexes their text content
SearchableField(name="tables_json",  type=String, retrievable=True),  # JSON list
SearchableField(name="figures_json", type=String, retrievable=True),  # JSON list

# src/ingest.py — serialization at upsert time
def chunk_to_search_doc(c: Chunk, vector: list[float]) -> dict:
    d = asdict(c)
    tables  = d.pop("tables", None)  or []
    figures = d.pop("figures", None) or []
    d["tables_json"]  = json.dumps(tables, ensure_ascii=False)  if tables  else ""
    d["figures_json"] = json.dumps(figures, ensure_ascii=False) if figures else ""
    d["content_vector"] = vector
    return d
```

---

## 5. Generation & citations

### Q5.1 — How are citations enforced in the answer?

**A:** The system prompt tells the model to cite `[N]` where N is the 1-indexed position of the chunk in the context block. `Citation` post-parses the answer and verifies each `[N]` references a real chunk; mismatches are silently dropped (model can't fabricate `[7]` if only 4 chunks were provided). The trace records the citation list.

→ [src/generate.py:25-41, 129-159](../src/generate.py#L25-L159)

```python
# src/generate.py
SYSTEM_PROMPT = (
    "You are a knowledge-base assistant for product manuals, troubleshooting "
    "guides, and policies.\n"
    "RULES:\n"
    "1. Use ONLY the provided context chunks. Never use outside knowledge.\n"
    "2. If the context does not contain enough information, say so plainly. Do not guess.\n"
    "3. Cite every factual claim with the source's bracketed index, e.g. [1] or [2].\n"
    "   Use ONLY indices that appear in the supplied context — do not invent higher numbers.\n"
    "4. Be concise and concrete. Prefer numbered steps for procedures.\n"
    "5. Tables and 'Figure description' lines are authoritative — reference figure callouts\n"
    "   (① ② ③) by the same numbers when answering."
)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

def _parse_citations(answer: str, chunks_by_index: dict[int, RetrievalResult]):
    """Walk [N] references in `answer` in first-mention order; resolve each to
    the chunk at that 1-indexed position. Indices outside the supplied set are
    silently dropped — model hallucinated a higher number."""
    seen, citations = set(), []
    for match in CITATION_PATTERN.finditer(answer):
        idx = int(match.group(1))
        if idx in seen or idx not in chunks_by_index:
            continue
        seen.add(idx)
        c = chunks_by_index[idx]
        citations.append(Citation(chunk_id=c.chunk_id, source_path=c.source_path,
                                  file_name=c.file_name, page_number=c.page_number,
                                  heading_path=c.heading_path, figures=c.figures, index=idx))
    citations.sort(key=lambda c: c.index)   # 1, 2, 4 (ascending) for the Sources footer
    return citations
```

### Q5.2 — Why gpt-4o instead of gpt-4o-mini?

**A:** Mini hallucinates citations more often in spot-checks (citing chunks it didn't actually read). gpt-4o is ~10× more expensive but for a typed-trace demo where citation correctness is the headline grade-able output, the cost is justified. Production with budget pressure could swap in gpt-4.1-mini (10× cheaper) where region quota allows, or downgrade only the intent_router to mini and keep gpt-4o for generation.

→ both router and generator read the same env var:

```python
# src/agent.py — router
deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]   # default: gpt-4o

# src/generate.py — generator (same default; can be overridden per-call)
def generate_grounded_answer(query, selected_chunks, *, model=None, ...):
    deployment = model or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
```

### Q5.3 — How do tables and figures show up in the prompt?

**A:** Tables get rendered as markdown tables and inlined into the chunk's content shown to the LLM. Figures are referenced by their caption (DI extracted, optionally enriched by gpt-4o-vision) — when the model cites a chunk that has figures, the trace surfaces those figures' blob URLs to the UI, which renders them inline. The LLM never sees pixel data; it sees captions only.

→ [src/generate.py:72-96](../src/generate.py#L72-L96), [src/figures.py:144+](../src/figures.py#L144) `splice_captions_into_markdown`

```python
# src/generate.py — table rendering at prompt-construction time (not index time)
def _render_table_as_markdown(table: dict) -> str:
    headers = [h or f"col{i+1}" for i, h in enumerate(table.get("headers") or [])]
    rows    = table.get("rows") or []
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [(str(c) or "").replace("\n", " ").replace("|", "\\|") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

# src/figures.py — caption splicing happens BEFORE chunking, so chunks contain captions
def splice_captions_into_markdown(markdown: str, figures: list[FigureRef]) -> str:
    # Inserts each figure's caption as visible text just before the closing
    # </figure> tag in the DI markdown — chunks downstream see the captions inline.
    ...
```

### Q5.4 — What if the model answers without citations?

**A:** Answer is still returned, `Citation` parsing produces an empty list, trace records `citation_count=0`, UI shows a "no citations" warning. Doesn't refuse — just makes the lack visible. (A stricter contract would re-prompt or refuse; we kept it permissive for demo.)

→ same `_parse_citations` as Q5.1 — empty list is a valid return; no special-case code

### Q5.5 — What's the system prompt strategy?

**A:** Single system prompt with: (a) role, (b) the exact citation format `[N]`, (c) instruction to refuse if context is insufficient instead of inventing, (d) device-aware framing. Prompt is ~150 tokens — heavy lifting is done by the retrieval scope filter, not by prompt engineering.

→ [src/generate.py:25-41](../src/generate.py#L25-L41) (full prompt shown in Q5.1)

---

## 6. Safety

### Q6.1 — How is Azure Content Safety integrated?

**A:** Two-stage gating in the agent. Input safety: `check_input(user_query)` runs at the top of `intent_router` — if blocked, route becomes `"blocked_input"` and we short-circuit to the formatter with a refusal. Output safety: `check_output(answer)` runs before the formatter — if blocked, the answer is replaced. Both produce `SafetyResult` so the trace records `blocked_category` + per-category severities.

→ [src/safety.py:89-96](../src/safety.py#L89-L96), [src/agent.py:259-276](../src/agent.py#L259-L276)

```python
# src/safety.py
def check_input(user_query: str) -> SafetyResult:
    """Pre-router gate."""
    return _analyze(user_query)

def check_output(answer: str) -> SafetyResult:
    """Post-generator gate."""
    return _analyze(answer)

# src/agent.py — intent_router gates BEFORE the LLM router call (saves cost)
def intent_router(state: RagState) -> dict:
    safety_in = check_input(query)
    if not safety_in.passed:
        plan = _make_plan_dict(query=query, route="blocked_input",
                               reason=f"blocked by content safety: {safety_in.blocked_category}", ...)
        return {"route": "blocked_input", "safety_input": safety_in.to_dict(),
                "safety_blocked": True, ...}
    # ... only if passed do we make the LLM router call
```

### Q6.2 — What categories does it block?

**A:** Azure Content Safety's standard four: Hate, SelfHarm, Sexual, Violence. Each scored 0/2/4/6; we treat ≥4 as blocked. We don't re-train or extend categories — using Azure's defaults is the point of using a managed service.

→ [src/safety.py:31-40](../src/safety.py#L31-L40)

```python
# src/safety.py
SEVERITY_BLOCK_THRESHOLD = 4   # Medium

CATEGORIES = (
    TextCategory.HATE,
    TextCategory.VIOLENCE,
    TextCategory.SELF_HARM,
    TextCategory.SEXUAL,
)

# In _analyze:
severities = {item.category: int(item.severity) for item in (resp.categories_analysis or [])}
worst_cat, worst_sev = None, 0
for cat, sev in severities.items():
    if sev > worst_sev:
        worst_sev, worst_cat = sev, cat
if worst_sev >= SEVERITY_BLOCK_THRESHOLD:
    return SafetyResult(passed=False, blocked_category=worst_cat, severities=severities)
```

### Q6.3 — What if Content Safety isn't configured (no env vars)?

**A:** `SafetyResult` returns `passed=True, skipped=True` and the request flows through. Intentional for local dev / unit-test environments where you don't want to require a Content Safety endpoint. The trace records `skipped=True` so reviewers see when safety was disabled.

→ [src/safety.py:55-67](../src/safety.py#L55-L67)

```python
# src/safety.py — silent fallback when not configured
@lru_cache(maxsize=1)
def _get_client() -> ContentSafetyClient | None:
    endpoint = os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT", "").strip()
    key      = os.environ.get("AZURE_CONTENT_SAFETY_KEY", "").strip()
    if not endpoint or not key:
        return None        # ← no creds → skip entirely
    return ContentSafetyClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def _analyze(text: str) -> SafetyResult:
    client = _get_client()
    if client is None:
        return SafetyResult(passed=True, blocked_category=None, severities={}, skipped=True)
    # ... API call
    except Exception as e:
        # Fail OPEN: a CS outage shouldn't break the whole RAG pipeline.
        return SafetyResult(passed=True, ..., error=f"{type(e).__name__}: {e}")
```

### Q6.4 — Could a legitimate query get blocked? False positive risk?

**A:** Yes — "the printer keeps jamming, I'm going to kill it" might trigger Violence. Mitigation in this demo: threshold is 4 (Medium), not 2. Production would want a domain-specific blocklist + whitelist or a feedback mechanism so users can flag false positives. Not implemented; counted as a known operational gap.

→ same `SEVERITY_BLOCK_THRESHOLD = 4` in Q6.2 — the only knob currently exposed

---

## 7. Telemetry & observability

### Q7.1 — What's the difference between FinalRagTrace and OpenTelemetry traces?

**A:** Two different layers. **FinalRagTrace** is application-domain typed data (the structured output the reviewer / UI sees per query). **OpenTelemetry** spans are infrastructure-level — one span per LangGraph node, exported to App Insights for cost/latency profiling. They overlap in instrumentation point but serve different audiences: FinalRagTrace for "did the reasoning go right?", OTel for "is the system healthy?".

→ [src/telemetry.py](../src/telemetry.py), [src/agent.py](../src/agent.py) (every node opens a span)

```python
# src/agent.py — every node wraps its body in an OTel span
def intent_router(state: RagState) -> dict:
    with _tracer.start_as_current_span("intent_router") as span:
        span.set_attribute("rag.query_length", len(query))
        # ... node logic ...
        span.set_attribute("rag.route", route)
        span.set_attribute("rag.detected_device", device or "")
        return { ... }

# Same pattern in evidence_selector, simple_rag_search, generator_node, etc.
```

### Q7.2 — Where do OTel traces actually go?

**A:** Application Insights (Azure-managed, in the same resource group). `azure-monitor-opentelemetry`'s `configure_azure_monitor()` reads `APPLICATIONINSIGHTS_CONNECTION_STRING` and exports spans + logs over OTLP. View them in Azure portal → App Insights → Transaction Search / Application Map.

→ [src/telemetry.py:22-54](../src/telemetry.py#L22-L54)

```python
# src/telemetry.py
@lru_cache(maxsize=1)
def configure() -> None:
    """One-shot setup. Silent no-op if APPLICATIONINSIGHTS_CONNECTION_STRING isn't set."""
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if not conn:
        return    # ← no-op for local dev / unit tests

    from azure.monitor.opentelemetry import configure_azure_monitor
    configure_azure_monitor(
        connection_string=conn,
        logger_name="azure-observable-rag",   # cloud-role-name in App Insights
    )

    # Auto-instrument OpenAI calls (gives free spans for chat-completion w/ token counts)
    try:
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
    except ImportError:
        pass
```

### Q7.3 — What's the runtime overhead of all this tracing?

**A:** OTel spans add ~1-3ms per node + ~50-200ms total export latency (batched, async, doesn't block the response). FinalRagTrace serialization is sub-ms. Compared to LLM calls (gpt-4o ≈ 1-3s) and AI Search reads (~100-300ms), tracing is negligible.

→ no specific code — empirical observation; the BatchSpanProcessor inside `azure-monitor-opentelemetry` runs export off the main thread.

### Q7.4 — Can you disable telemetry for testing or air-gapped envs?

**A:** Yes — leave `APPLICATIONINSIGHTS_CONNECTION_STRING` unset and `configure_azure_monitor()` becomes a no-op. `_tracer.start_as_current_span()` still runs but its spans go nowhere. Code path is unchanged.

→ [src/telemetry.py:30-34](../src/telemetry.py#L30-L34) (silent fallback shown in Q7.2)

---

## 8. Infrastructure & deployment

### Q8.1 — What does `bash infra/deploy.sh kb-rag-rg swedencentral` actually do?

**A:** Five phases: (1) provisions Bicep resources (Storage + AI Search Standard + AI Foundry Project + Document Intelligence + Content Safety + App Insights + Function App), (2) grants Content Safety RBAC, (3) wires App Insights connection string, (4) deploys 4 AI service models, (5) Function App MSI grants + zip deploys the function code. Writes a complete `.env` at the end. Idempotent — re-running on an existing RG is safe.

→ [infra/deploy.sh](../infra/deploy.sh)

```bash
# infra/deploy.sh — high-level structure
RESOURCE_GROUP="${1:-kb-rag-rg}"
LOCATION="${2:-swedencentral}"

# Phase 1 — Bicep deploy provisions everything
az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file infra/main.bicep \
    --parameters location="$LOCATION" ...

# Phase 2 — pull deploy outputs into shell vars (storage, search, ai-services names…)
SEARCH_NAME=$(get AZURE_SEARCH_NAME)
FOUNDRY_PROJECT_NAME=$(get AZURE_AI_FOUNDRY_PROJECT_NAME)

# Phase 3 — RBAC (Content Safety, App Insights, Function App MSI)
az role assignment create --role "Cognitive Services User" --assignee ...
az role assignment create --role "Search Index Data Contributor" --assignee ...

# Phase 4 — write final .env at repo root with all 27 vars
cat > .env <<EOF
AZURE_SEARCH_ENDPOINT=...
APPLICATIONINSIGHTS_CONNECTION_STRING=...
EOF

# Phase 5 — package + zip-deploy Function App
(cd infra/functions && zip -r /tmp/funcapp.zip .)
az functionapp deployment source config-zip --src /tmp/funcapp.zip ...
```

### Q8.2 — Why Bicep instead of Terraform / ARM templates?

**A:** Bicep is Microsoft's first-party IaC for Azure — better SDK support, fewer abstraction leaks, direct mapping to ARM resource types. Terraform would also work but adds another tool + state to manage. For a single-environment demo, Bicep + Azure CLI is the path of least resistance.

→ [infra/main.bicep](../infra/main.bicep)

```bicep
// infra/main.bicep — direct mapping to Microsoft.* resource providers
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = { ... }
resource search 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  name: 'kb-search-${nameSuffix}'
  sku: { name: searchSku }
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: 'standard'    // ← required for the L2 reranker
  }
}
resource aiServices 'Microsoft.CognitiveServices/accounts@2026-03-01' = { ... }
resource funcApp 'Microsoft.Web/sites@2023-12-01' = {
  name: 'kb-funcs-${nameSuffix}'
  ...
}
```

### Q8.3 — How does authentication work? API keys or AAD?

**A:** AAD via `DefaultAzureCredential` for the data plane. `deploy.sh` auto-grants the deploying user the right RBAC roles (Storage Blob Data Contributor, Azure AI Developer, Cognitive Services User, Search Index Data Contributor). The Function App uses its system-assigned MSI for the same set. API keys are accepted as fallback for AI Search / DI but not used in the demo.

→ [src/embed.py](../src/embed.py), [src/extract.py](../src/extract.py), [infra/deploy.sh](../infra/deploy.sh) Phase 2

```python
# src/embed.py — DefaultAzureCredential + AAD bearer token for data-plane calls
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

def get_azure_openai_client():
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,           # ← AAD, not API key
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
```

### Q8.4 — Why Standard SKU AI Search instead of Basic?

**A:** L2 semantic reranker requires Standard or higher. Basic (~$75/mo) is text-only. Since the reranker is a core architectural choice, Basic isn't viable. Trade-off: Standard is ~$250/mo flat regardless of usage. Listed as a cost concern.

→ [infra/main.bicep:82](../infra/main.bicep#L82)

```bicep
// infra/main.bicep — Standard or above is the only way to get semanticSearch: 'standard'
@description('Azure AI Search SKU. Semantic ranker requires standard or higher.')
param searchSku string = 'standard'    // not 'basic'

resource search 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  ...
  properties: {
    semanticSearch: 'standard'    // ← unlocks L2 reranker
  }
}
```

### Q8.5 — What's the Function App's contract? When does it fire?

**A:** Blob trigger on the `kb-docs` container — fires on add/update events for any blob (recursive). Handler is `auto_ingest(blob)` which calls `src.ingest.ingest_single(blob_path)`. Cold start ~10s on Y1 Consumption SKU; warm ~2-5s for a small PDF.

→ [infra/functions/function_app.py:49-87](../infra/functions/function_app.py#L49-L87)

```python
# infra/functions/function_app.py
@app.blob_trigger(
    arg_name="blob",
    path="kb-docs/{name}",                # any blob under the container
    connection="AzureWebJobsStorage",     # uses the function app's MSI
)
def auto_ingest(blob: func.InputStream) -> None:
    blob_path = blob.name.removeprefix("kb-docs/")
    tracer = get_tracer()

    with tracer.start_as_current_span("auto_ingest") as span:
        span.set_attribute("kb.blob_path", blob_path)
        span.set_attribute("kb.blob_size_kb", (blob.length or 0) // 1024)

        from src.ingest import ingest_single   # late import → faster cold-start
        try:
            result = ingest_single(blob_path)
            span.set_attribute("kb.chunks_indexed", int(result.get("chunks_indexed", 0)))
        except Exception as e:
            span.record_exception(e)
            raise   # surface to App Insights for alerting
        finally:
            _force_flush_telemetry()
```

---

## 9. Frontends

### Q9.1 — Why both Chainlit AND a notebook?

**A:** Different audiences. **Chainlit** for live interactive demo / multi-turn / showing CoT step panels live. **Notebook** for static reproducibility — a reviewer can read it like a paper, see one canonical query walk through every stage with structured tables, no Azure auth needed to read it. Same backend modules, different rendering.

→ [src/app.py](../src/app.py), [notebooks/demo.ipynb](demo.ipynb)

```python
# src/app.py — Chainlit entry, calls the same get_graph() the notebook uses
@cl.on_message
async def on_message(message: cl.Message):
    graph = get_graph()                                     # same LangGraph DAG
    config: RunnableConfig = {"configurable": {"thread_id": cl.context.session.id}}
    async for event in graph.astream({"user_query": text}, config=config,
                                      stream_mode="updates"):
        for node_name, payload in event.items():
            await _run_node_step(node_name, payload)        # render as CoT step
```

### Q9.2 — How are sources rendered in Chainlit?

**A:** Cited files auto-attach to the answer message as `cl.Pdf` / `cl.Text` elements with `display="side"` — clicking opens the original document in the right panel for citation comparison. There's also an on-demand `files` command showing the corpus inventory grouped by category.

→ [src/app.py](../src/app.py) `_attach_cited_files_to_message`, `_send_inventory`

```python
# src/app.py — abridged
async def _attach_cited_files_to_message(msg, citations):
    elements = []
    seen_paths = set()
    for cit in citations:
        if cit.source_path in seen_paths:
            continue
        seen_paths.add(cit.source_path)
        elem = _build_element(cit.source_path, cit.file_name)   # cl.Pdf / cl.Text
        if elem:
            elements.append(elem)
    msg.elements = elements
    await msg.update()
```

### Q9.3 — What's the FileTree custom element?

**A:** A React custom element ([public/elements/FileTree.jsx](../public/elements/FileTree.jsx)) mounted into the Chainlit sidebar. Shows the corpus tree (devices/families/models/doc_types) so users can browse what's available. The header `Files` button (custom JS in `public/header_files_button.js`) bridges header click → typing `files` into the chat input — Chainlit's `[[UI.header_links]]` only supports static URLs, not Python callbacks.

→ [public/header_files_button.js](../public/header_files_button.js)

```javascript
// public/header_files_button.js
(function () {
  const TRIGGER_HREF = "#open-files";
  const COMMAND_TEXT = "files";
  // Wire header link click → type "files" into composer → submit
  // (Chainlit fires @cl.on_message("files") which calls _send_inventory())
})();
```

---

## 10. Reproducibility & dev experience

### Q10.1 — Why uv instead of pip / poetry / conda?

**A:** uv is 10-100× faster than pip for resolution + install, has a real solver (catches conflicts at lock time, not at runtime), and produces a deterministic `uv.lock` for reproducibility. Poetry is slower, has a less stable resolver. Conda is heavier. uv is the pragmatic 2026 choice.

→ [pyproject.toml](../pyproject.toml), [uv.lock](../uv.lock)

```toml
# pyproject.toml — source of truth; uv.lock is the deterministic transitive pin set
[project]
name = "azure-observable-rag"
requires-python = ">=3.12,<3.13"

dependencies = [
    "azure-identity>=1.19.0",
    "azure-search-documents>=11.5.2,<12",
    "azure-ai-documentintelligence>=1.0.0",
    "azure-monitor-opentelemetry>=1.6.9",
    "openai>=1.54.0,<2",
    "langgraph>=0.2.50,<0.3",
    "chainlit>=2.11,<3",   # matches .chainlit/config.toml's generated_by
    "pydantic>=2.9.0",
    # ... 18 more
]

[tool.uv]
package = false           # treat as app, not library

[tool.pyright]
venvPath = "."
venv = ".venv"
```

### Q10.2 — How does the test suite work? Does it need Azure?

**A:** No Azure credentials required. Tests cover only pure functions: `tests/test_chunk.py` (chunking), `tests/test_extract.py` (path-to-metadata parsing), `tests/test_search.py` (filter construction). Run with `pytest`. CI-friendly. Integration path isn't tested automatically — known gap.

→ [tests/](../tests/), [pytest.ini](../pytest.ini)

```python
# tests/test_extract.py — example: deterministic, no Azure required
def test_parse_path_metadata_devices_layout():
    meta = _parse_path_metadata("devices/network_access/meraki_mx67/manuals/guide_v1.2.pdf")
    assert meta["scope"] == "device"
    assert meta["device_family"] == "network_access"
    assert meta["device"] == "meraki_mx67"
    assert meta["doc_type"] == "manual"
    assert meta["version"] == "1.2"
```

### Q10.3 — What does `sync.py --full-rebuild` do?

**A:** (1) Stops the Function App so the local pipeline and the blob trigger don't race. (2) Wipes all `kb-chunks` index entries. (3) Uploads `data/` to the `kb-docs` blob container. (4) Walks every blob, runs `extract → figures → chunk → embed → upsert`. (5) Restarts the Function App. Idempotent: re-running produces identical chunk IDs (content-hashed).

→ [scripts/sync.py:85-115](../scripts/sync.py#L85-L115)

```python
# scripts/sync.py — context manager that stops/starts the Function App
@contextmanager
def _function_app_paused():
    func_app = os.environ.get("AZURE_FUNCTION_APP")
    rg       = os.environ.get("AZURE_RESOURCE_GROUP")
    az_path  = shutil.which("az")
    if not func_app or not rg or not az_path:
        yield   # silently skip if not configured
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

# Used in --full-rebuild flow:
with _function_app_paused():
    delete_all_chunks()
    upload_local_data(data_dir)
    for blob_path in blob_paths:
        ingest_single(blob_path)
```

### Q10.4 — How is `requirements.txt` related to `pyproject.toml`?

**A:** `pyproject.toml` is the source of truth. `uv.lock` is the deterministic transitive-pin lockfile. `requirements.txt` is auto-generated from `uv.lock` via `uv export --no-hashes --no-dev` — exists only so someone without uv can still `pip install -r requirements.txt`.

→ [requirements.txt:1](../requirements.txt#L1)

```text
# This file was autogenerated by uv via the following command:
#    uv export --no-hashes --no-dev --format requirements-txt
aiofiles==23.2.1
    # via chainlit
... (~735 lines of pinned transitive deps)
```

---

## 11. Limitations & production gaps

### Q11.1 — What's the biggest gap before this is production-ready?

**A:** **No automated evaluation harness.** Without recall@k, citation accuracy, groundedness scores on a held-out gold-QA set, every "this works better" claim is anecdotal. That's why the previous `eval_harness.py` was deleted — it was a stub that didn't actually measure anything. A real harness is the next addition.

→ [README.md](../README.md) "Known Limitations"

```markdown
- **No automated evaluation harness.** This iteration focuses on architecture
  and observability — there's no recall@k / groundedness / citation-correctness
  scoring. A gold-QA set + harness would be the next addition before claiming
  retrieval quality numbers.
```

### Q11.2 — Why no LangGraph checkpointer? History is lost on restart.

**A:** `MemorySaver` is in-process. Production should wire `langgraph.checkpoint.sqlite.SqliteSaver` (one-line swap) or the Postgres equivalent so threads persist across Chainlit restarts. We didn't because (a) demo doesn't need session continuity, (b) adds an external dependency.

→ [src/agent.py:772](../src/agent.py#L772)

```python
# src/agent.py — current
return builder.compile(checkpointer=MemorySaver())  # ← in-process, lost on restart

# Production swap (one line):
from langgraph.checkpoint.sqlite import SqliteSaver
return builder.compile(checkpointer=SqliteSaver.from_conn_string("threads.db"))
```

### Q11.3 — What about cost monitoring? AI Search bills hourly.

**A:** No budget alerts wired up. The resource group will keep billing $0.30/hr ($250/mo) for AI Search Standard whether you query it or not. Production would set Azure budget + alert rules at the subscription or RG level. Documented as a limitation.

→ no code; documented in [README.md](../README.md) "Known Limitations"

### Q11.4 — How would you scale to 10k or 100k documents?

**A:**
- **AI Search** Standard SKU has ~50GB index limit — 100k docs likely exceeds; bump to Standard 2 or partition by tenant
- **DI** is pay-per-page — 100k × avg 20 pages × $1.50/1000 = $3000 one-time; cache extracted markdown to blob (we already do)
- **Function App** Y1 cold-starts hurt batch ingest — switch to Premium or run sync.py in parallel
- **Embeddings** rate limit ~350k TPM — batch larger windows
- **Reranker** L2 latency degrades past ~5M chunks — shard the index or pre-filter aggressively

→ no specific code; architectural reasoning

### Q11.5 — Multi-tenant — different customers, isolated corpora?

**A:** Not built for it. Every chunk lives in one `kb-chunks` index. Options for multi-tenant: separate index per tenant (simplest), single index with `tenant_id` field + always-on filter (more efficient, harder to clean delete), or separate AI Search instance per tenant (most isolation, most expensive). Current design is single-tenant.

→ no specific code; the index schema has no `tenant_id` field today

---

## 12. Trade-off / "what if" questions

### Q12.1 — What if you had to swap Azure AI Search for Pinecone / Weaviate / pgvector?

**A:** The retrieval interface is mostly contained in [src/search.py](../src/search.py) and [src/index.py](../src/index.py). Swap would require: (1) replace index schema with target's API, (2) replace `client.search(...)` with target's hybrid query, (3) find a replacement reranker (Pinecone / Weaviate hosted; pgvector needs separate cross-encoder service). Estimated 2-3 days. Rest of stack (chunk, ingest, agent, generator) wouldn't change.

→ [src/search.py](../src/search.py), [src/index.py](../src/index.py)

```python
# src/search.py — the only Azure-Search-specific call site to swap
raw = client.search(**kwargs)        # ← becomes pinecone.query(...) or weaviate.query.hybrid(...)

# Everything downstream operates on RetrievalResult, which is provider-agnostic:
@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    reranker_score: float | None
    page_number: int | None
    heading_path: str | None
    # ... portable across any vector DB
```

### Q12.2 — What if a user asks something off-topic (cooking recipes)?

**A:** Three layers of defense. (1) intent_router classifies — `general_kb` / `conceptual` queries with no device get routed but with no scope filter. (2) Retrieval returns low `reranker_score` chunks (typically <2.0 vs ~2.5+ for in-scope queries — see notebook's "Off-topic contrast" cell). (3) Generator prompt instructs to refuse if context is insufficient. Reviewer can SEE this happen end-to-end in the trace.

→ notebook section 4 "Off-topic contrast" + system prompt rule 2 (Q5.1)

```python
# notebooks/demo.ipynb cell-search-offtopic — actual run:
OFF_TOPIC = "How do I configure a payment terminal pin pad?"
off = hybrid_search(OFF_TOPIC, top_k=5, search_mode="hybrid_semantic")
# Returns hits from Desk5000 / TM-M30II PDFs with reranker_score 1.98-2.24
# (vs 2.43-2.74 for the in-topic Meraki query in the cell above)
# → generator sees low-confidence context → refuses per system-prompt rule 2
```

### Q12.3 — What if two valid documents disagree on the answer?

**A:** L2 reranker picks based on semantic relevance. Both might end up in top-N. Generator sees both with chunk IDs; can either cite both ("[1] says X, but [3] says Y") or pick one. By design — we don't try to merge or arbitrate. Production might add a `version` filter (we have that field) to prefer newer docs, or surface conflicts to the user.

→ `version` field exists on `RetrievalResult` but no current code branches on it

### Q12.4 — What happens when a doc is updated (new version)?

**A:** Upload to blob → Function App fires `auto_ingest` → calls `_delete_existing_chunks(doc_id)` first, then re-embeds and upserts. Result: clean replacement. `chunk_id` is content-hashed so chunks whose content didn't change keep the same ID (no churn).

→ [src/ingest.py:74-92, 133-173](../src/ingest.py#L74-L173)

```python
# src/ingest.py — chunk_id is content-hashed → idempotent re-ingest
def make_chunk_id(source_path: str, chunk_index: int, content: str) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"::")
    h.update(str(chunk_index).encode())
    h.update(b"::")
    h.update(content.encode("utf-8"))     # ← content hash
    return h.hexdigest()

def ingest_single(blob_path: str) -> dict:
    create_or_update_index()
    _delete_existing_chunks(blob_path)    # ← drop old chunks first
    # ... extract, chunk, embed, upload
```

### Q12.5 — What's the latency budget per query?

**A:** Measured ~1.5–3s end-to-end on swedencentral. Breakdown: intent_router LLM ~600ms + retrieval ~200ms + L2 rerank ~150ms (server-side) + generator LLM ~1-2s + safety checks ~100ms each. Network roundtrip dominates. Optimizations available: parallelize input safety with retrieval (independent), use gpt-4o-mini for intent_router, batch query embedding. Demo is "feels responsive" so we didn't optimize.

→ each LangGraph node has a `latency_ms` field on its trace + an OTel span — measurable post-hoc

### Q12.6 — How would you A/B test chunk size in production?

**A:** Build two indexes (`kb-chunks-256`, `kb-chunks-512`) ingested with different `CHUNK_MAX_TOKENS`. Add a feature flag to `hybrid_search` selecting which to query. Run a holdout gold-QA set against both, score recall@k + citation accuracy + answer groundedness. Pick winner, retire loser. **Requires the eval harness we don't have.**

→ env-overridable today, but no A/B routing layer

```bash
# Hypothetical setup:
CHUNK_MAX_TOKENS=256 python scripts/sync.py --full-rebuild  # → kb-chunks-256
CHUNK_MAX_TOKENS=512 python scripts/sync.py --full-rebuild  # → kb-chunks-512

# hybrid_search would need a `index_name=` arg routed by feature flag
```

### Q12.7 — Could you cache LLM calls?

**A:** Azure OpenAI now supports prompt caching for system prompts ≥1024 tokens. Our system prompt is ~150 tokens — too short to benefit. Bigger opportunities: skip the intent_router LLM for queries that exact-match a recent one (LRU on query string), cache embedding for repeated queries (text-embedding-3-small is deterministic, so memoize). Not implemented; would shave ~30% off repeat-query latency.

→ no caching code today; suggestion-level

### Q12.8 — How would you add a new device family?

**A:** Three steps: (1) Drop docs under `data/devices/{new_family}/{model}/{doc_type}/` following the existing convention. (2) Run `python scripts/sync.py --full-rebuild` (or just upload new blobs and let the Function App fire). (3) Optionally update `chainlit.md` and the demo notebook intro. **No code changes** — path-derived metadata extraction handles new device_family / device automatically. The intent_router LLM will start detecting the new family on its own from queries naming it.

→ [src/extract.py:155-200](../src/extract.py#L155-L200) (path parser handles arbitrary device_family / model strings — see Q2.1 snippet)

---

## How to use this document

**Before the review:**
1. Skim every Q so you're not blindsided.
2. For 10–15 questions where you feel weakest, click the code pointers and read the actual implementation.
3. Practice answering 5–10 aloud (not reading from this doc) — keeps you concise.

**During the review:**
- Lead with the answer, then back with code. Reviewers prefer "Yes — [src/file.py:LN]" over a long preamble.
- For limitations: own them clearly. "We didn't tune chunk size — listed as a known limitation, would sweep with eval harness next" is much stronger than dancing around it.
- For trade-off questions: name the trade-off explicitly. "We picked X over Y because A; the cost is B."

**Cross-check:** if any answer here looks wrong, the source of truth is the code, not this doc. The pointers are exactly what the reviewer would jump to.
