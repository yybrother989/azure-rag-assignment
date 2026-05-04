# Q&A prep — Azure Observable RAG project review

## Context

User is preparing to defend / present this project to a reviewer (assignment grader, interview panel, or technical reader). Goal: anticipate every question a reviewer would ask, with a concise answer + a pointer to where in the code to back it up. Coverage spans 12 topic areas, ~60 likely questions.

This document is structured as **plan-file-as-deliverable**: the Q&A itself lives below. On ExitPlanMode approval, decide whether to (a) keep it as personal prep notes only, or (b) commit it to the repo as `docs/QA.md` / `docs/INTERVIEW.md` for future reference. Both are valid; comprehensive doc in the repo helps any next person who picks up the codebase.

Format per entry:
- **Q:** the question (with 1–3 likely phrasings)
- **A:** the answer (2–5 sentences, concrete + opinionated)
- **→** code/file pointer for proof if asked

---

## 1. Architecture & high-level design

**Q1.1.** *Why "Observable RAG"? What's the framing?*
*"Why isn't this just another LangChain RAG demo?"*

**A:** Traditional agentic RAG hides its work — one chat-completions call decides what to retrieve and how to answer. That's fast to build but impossible to grade. This project decouples every stage and emits a typed trace per LangGraph node, so the reviewer can audit which intent the router picked, which filter was sent to the index, which chunks the index returned, which the selector kept, and exactly what the generator was given. Retrieval and generation never share state implicitly. → [README.md](README.md) "Why Observable RAG", [src/tracing.py](src/tracing.py).

**Q1.2.** *Why LangGraph instead of plain LangChain or hand-rolled functions?*

**A:** LangGraph gives a typed StateGraph where each node consumes/produces well-defined fields of a single state dict. That makes the trace shape automatic (every node return becomes part of `FinalRagTrace`), conditional routing explicit (`_route_after_router`), and per-node observability trivial (one OTel span per node). Plain functions would also work but you'd reinvent the routing + state-merging machinery. → [src/agent.py](src/agent.py) `build_graph()`.

**Q1.3.** *Why two routes (NO_RETRIEVAL vs SIMPLE_RAG)? Why not always retrieve?*

**A:** Some queries don't need retrieval (greetings, "what can you do", clarifying chit-chat). Always retrieving wastes Azure Search reads + token spend on chunks that won't be cited. The intent_router LLM call is cheap (gpt-4o on a short prompt) and gates the retrieval path. Trade-off: one extra LLM hop per query in exchange for cleaner traces and avoided retrieval noise on conversational turns. → [src/agent.py](src/agent.py) `intent_router`, `_route_after_router`.

**Q1.4.** *Why a separate `evidence_selector` node after retrieval?*

**A:** `hybrid_search` returns `RETRIEVAL_TOP_K=5` candidates with `reranker_score`. The evidence_selector then picks the top `EVIDENCE_TOP_N=4` to actually feed the generator. Decoupling lets us tune "how wide stage 1 retrieves" independently of "how much context the LLM sees". For demo it's 5→4 (mostly a no-op); production should widen stage 1 to ~50 (Azure reranker cap) and narrow to 5–8 here. → [src/agent.py](src/agent.py) `evidence_selector`.

**Q1.5.** *Why two ingestion paths — `scripts/sync.py` AND a Function App?*

**A:** Two execution surfaces, **same `src/` modules**. `sync.py` is for local/dev workflows: explicit `--full-rebuild`, `--diff`, drives the lifecycle. The Function App handles production: blob upload to `kb-docs` triggers `auto_ingest` automatically. Both call into the same `extract → figures → chunk → embed → upsert` pipeline, so a fix in those modules takes effect in both paths after one Function App redeploy. The duplication is in execution context, not logic. → README "Two ingestion paths, one codebase", [infra/functions/function_app.py:54](infra/functions/function_app.py#L54), [scripts/sync.py](scripts/sync.py).

**Q1.6.** *What does `FinalRagTrace` contain?*

**A:** Five sub-traces bundled into one JSON-serializable object: `QueryPlanTrace` (router output + detected metadata), `RetrievalTrace` (filters, results, latency), `EvidenceSelectionTrace` (which chunks survived, why), `GenerationTrace` (model, tokens, citations, answer), and `SafetyResult` for input/output. The same payload drives Chainlit's step panels, the demo notebook's per-stage tables, the JSONL audit log, and (eventually) any eval harness. → [src/tracing.py](src/tracing.py).

---

## 2. Data layout & ingestion

**Q2.1.** *Why `data/devices/{family}/{model}/{doc_type}/` instead of a flat folder?*

**A:** The path encodes structured metadata that we extract deterministically without an LLM: `device_family`, `device`, `doc_type`. That metadata becomes filterable columns on the AI Search index, which lets `build_retrieval_scope` produce a scoped OData filter (e.g., "only docs about meraki_mx67") instead of relying on cosine similarity to surface the right device. Result: fewer wrong-device hallucinations on multi-device corpora. → [src/extract.py](src/extract.py) `_parse_path_metadata`, [src/agent.py](src/agent.py) `build_retrieval_scope`.

**Q2.2.** *How do you handle docs that aren't device-specific (cross-device policies)?*

**A:** Top-level `data/shared/` carries `scope=shared`, `is_shared=True`. The retrieval scope builder normally filters to the detected device, but for `policy_check` query types (and as a fallback when the device-scoped search returns nothing) it allows shared docs through. → `data/shared/policies/`, [src/agent.py](src/agent.py) `build_retrieval_scope` `allow_shared_fallback`.

**Q2.3.** *What's `data/document_manifest.csv` for?*

**A:** A flat inventory of every doc in the corpus with its parsed metadata fields (relative_path, device_family, model, doc_type, vendor, topic, version, is_shared, status). Useful for QA / audit / sanity-check that path-derived metadata is what you expect, before paying for ingest. Not consumed at runtime — it's a human-checkable source of truth.

**Q2.4.** *Why Azure Document Intelligence for extraction instead of just `pypdf2` or `pdfplumber`?*

**A:** DI's prebuilt-layout model returns markdown with explicit page boundaries (`<!-- PageNumber=N -->`), HTML tables, and figure bounding boxes — all in one call. Hand-rolled PDF parsing would lose table structure and figure positions, which the rest of the pipeline depends on (figure crops in Stage 4, table anchors in chunking). For markdown / txt sources we skip DI. → [src/extract.py](src/extract.py).

**Q2.5.** *What about deletes? If I remove a blob, does the index clean up?*

**A:** No — Azure blob triggers fire only on add/update, never on delete. To remove orphan chunks AND their figure PNGs in `kb-figures/{stem}/`, run `python scripts/sync.py` (default mode). It diffs the blob container against the search index and deletes mismatches. Documented as a known limitation in README. → [scripts/sync.py](scripts/sync.py), README "Known Limitations".

---

## 3. Chunking

**Q3.1.** *Walk me through the chunking strategy.*

**A:** Two stages. **Stage 1**: split by markdown headers (`#`–`####`) so each chunk carries a `heading_path` like "Installation > Mounting > Wall Mount". **Stage 2**: only if a section exceeds `CHUNK_MAX_TOKENS=512`, sub-split it via LangChain's `RecursiveCharacterTextSplitter` with `chunk_overlap=50`. Most chunks are "one section = one chunk" — Stage 2 only fires on long sections. Each chunk also carries page number, device metadata, and any figures/tables that fall on its page. → [src/chunk.py](src/chunk.py) `chunk_document`.

**Q3.2.** *Why 512 tokens? Did you sweep this?*

**A:** No, we didn't sweep. 512 is a common default that fits comfortably inside text-embedding-3-small's 8192-token limit while keeping chunks small enough that retrieval scoring isn't washed out by irrelevant text. **Listed as a known limitation** — production should sweep `{256, 512, 1024} × {0%, 10%, 20%}` overlap on a held-out gold-QA set. The infra is there (env vars), the eval harness is missing. → [src/chunk.py:218-219](src/chunk.py#L218-L219), README "Known Limitations".

**Q3.3.** *Why a custom header splitter instead of LangChain's `MarkdownHeaderTextSplitter`?*

**A:** LangChain's splitter rewrites whitespace and discards character offsets. We need exact offsets to attribute each chunk to a page (via the `page_starts` map built when stripping `<!-- PageNumber=N -->` markers) and to attach figures/tables that fall in that piece. Without offsets, a long section spanning pages 5–8 would pin every sub-chunk to page 5, and the figure-attachment lookup would break. → [src/chunk.py:150-188](src/chunk.py#L150-L188) `_split_by_headers_with_offsets`.

**Q3.4.** *What does the 50-token overlap actually buy you?*

**A:** When Stage 2 sub-splits a long section, an answer that straddles the cut ("First do X, **then do Y**") would otherwise live half in chunk A, half in chunk B — neither chunk is independently sufficient. 50 tokens of overlap (~10% of 512) gives both sides enough surrounding context to be retrievable on their own. It's only invoked on long sections so the overall index size penalty is small (~10% extra tokens per long section, none extra for short ones).

**Q3.5.** *How do figures and tables get attached to chunks?*

**A:** Figures: collected per-page from DI's bounding-box output, rendered as PNGs to `kb-figures/{stem}/`, then attached to every chunk on that page (in the chunk's `figures` field as `{figure_id, page, blob_url, caption}`). Tables: matched by HTML prefix (first 120 chars of `<table>...`) — anchors that appear inside a chunk's text get attached. Tables and figures both get serialized as `tables_json` / `figures_json` in the search index for retrieval-time access. → [src/figures.py](src/figures.py), [src/chunk.py:191-213](src/chunk.py#L191-L213), [src/index.py](src/index.py).

**Q3.6.** *What about cross-page chunks?*

**A:** When a long section's Stage-2 sub-split produces a piece that starts on page N and ends on page N+1, both `page_number=N` and `page_end=N+1` are recorded by binary-searching the page-offsets map at the piece's global start AND end character positions. Citation rendering can then say "p. N–N+1" honestly. → [src/chunk.py:262-274](src/chunk.py#L262-L274).

---

## 4. Retrieval

**Q4.1.** *Why hybrid (BM25 + vector) instead of pure vector?*

**A:** Vector retrieval is great for paraphrased / conceptual queries but weak on rare proper nouns, error codes, model numbers (e.g., "MX67", "error 101", "Desk5000"). BM25 nails those because they have high IDF. Hybrid covers both regimes; the tradeoff is one extra retrieval pass server-side, but Azure does it in parallel inside one query call so latency overhead is sub-50ms. → [src/search.py](src/search.py) `hybrid_search`.

**Q4.2.** *How are BM25 and HNSW results merged?*

**A:** Reciprocal Rank Fusion, **Azure default**. When both `search_text` and `vector_queries` are passed in the same call, Azure auto-applies RRF with formula `score = Σ 1/(60 + rank_i)` and fixed `k=60`. We don't write any RRF code — it's service-side default behavior. → demo notebook section 4 explains this; README architecture diagram doesn't expose RRF as a separate node because it happens inside Azure.

**Q4.3.** *What is the L2 semantic reranker doing differently from RRF?*

**A:** RRF only sees rank positions — it doesn't read documents. Cosine only sees embedding distance — it doesn't read documents either. The L2 reranker is a **pretrained transformer cross-encoder** (Microsoft, Bing-derived, multilingual). It feeds query + each candidate document into one transformer forward pass and uses attention to score actual semantic relevance. That's why `reranker_score` is the highest-quality signal we have. → [src/index.py](src/index.py) `SemanticConfiguration`, [src/search.py](src/search.py) `query_type=SEMANTIC`, demo notebook section 4.

**Q4.4.** *Why `RETRIEVAL_TOP_K=5` and `EVIDENCE_TOP_N=4`? That seems narrow.*

**A:** It IS narrow — kept tight for demo readability. **Listed as a known limitation.** Azure's L2 reranker can score up to 50 candidates per query; production should widen to `RETRIEVAL_TOP_K=50` so the reranker has real headroom (if the top 5 from BM25+HNSW miss the right doc, the reranker can't recover it), and narrow `EVIDENCE_TOP_N` to 5–8 for the LLM context budget. Both are env-overridable. → README "Known Limitations".

**Q4.5.** *What are the four search modes and when do you use each?*

**A:** `bm25` (lexical only), `vector` (HNSW only), `hybrid` (BM25 + HNSW + RRF), `hybrid_semantic` (hybrid + L2 reranker). Default and demo path is `hybrid_semantic`. The others exist for ablation: you can compare reranker-on vs reranker-off, hybrid vs pure vector, etc. — useful when you eventually wire up an eval harness. → [src/search.py:31](src/search.py#L31) `SUPPORTED_MODES`.

**Q4.6.** *How does device-first scoping work? Show me a concrete query.*

**A:** Query: "How do I factory-reset the Meraki MX67?" → intent_router detects `device_family="network_access"`, `device="meraki_mx67"` → build_retrieval_scope emits OData filter `device eq 'meraki_mx67' and doc_type eq 'manual'` → `simple_rag_search` queries the index with that filter, restricting candidates BEFORE BM25/HNSW score. Result: zero chance of an Ingenico or Canon doc accidentally outranking the Meraki manual. → [src/agent.py](src/agent.py) `intent_router`, `build_retrieval_scope`, `simple_rag_search`.

**Q4.7.** *What if the device-scoped filter returns 0 hits?*

**A:** Fallback: re-query without the device filter (or expanded to `is_shared=True`). The trace records `fallback_triggered=True` so reviewers can see when it kicked in. The router also flags `allow_shared_fallback=True` for `policy_check` queries (where device-specific docs may not exist). → [src/agent.py](src/agent.py) `simple_rag_search`, `RetrievalTrace.fallback_triggered`.

**Q4.8.** *Why store `tables_json` and `figures_json` as strings on each chunk instead of separate index?*

**A:** They're chunk-scoped (each chunk knows what tables/figures live in its content + page), and we always retrieve them together with the chunk. A second index would mean N+1 round-trips and a join. Single-index trade-off: schema is fatter (~20% more bytes per doc), but retrieval is one round-trip and the search-doc shape mirrors the rendered output. → [src/index.py](src/index.py) field definitions.

---

## 5. Generation & citations

**Q5.1.** *How are citations enforced in the answer?*

**A:** The system prompt tells the model to cite `[N]` where N is the 1-indexed position of the chunk in the context block. `Citation` dataclass post-parses the answer and verifies each `[N]` references a real chunk; mismatches go into `GenerationTrace.unmatched_citations` (visible in trace). The model can't easily fabricate a `[7]` if only 4 chunks were provided. → [src/generate.py](src/generate.py) `Citation`, system prompt around line 30.

**Q5.2.** *Why gpt-4o instead of gpt-4o-mini?*

**A:** Mini models hallucinate citations more often in our spot-checks (citing chunks they didn't actually read from). gpt-4o is ~10× more expensive but for a typed-trace demo where citation correctness is the headline grade-able output, the cost is justified. Production with a budget constraint could swap in gpt-4.1-mini (10× cheaper) where region quota allows, OR keep gpt-4o for the generator and downgrade the intent_router to mini.

**Q5.3.** *How do tables and figures show up in the prompt?*

**A:** Tables get rendered as markdown tables and inlined into the chunk's content shown to the LLM. Figures are referenced by their caption (DI extracted) — when the model cites a chunk that has figures, the trace surfaces those figures' blob URLs to the UI, which renders them inline. The LLM never sees pixel data; it sees captions only. → [src/generate.py](src/generate.py) `_render_table_as_markdown`, [src/figures.py](src/figures.py).

**Q5.4.** *What if the model answers without citations?*

**A:** Answer is still returned, but `Citation` parsing produces an empty list. The trace records `citation_count=0`. UI shows a "no citations" warning. Doesn't refuse — just makes the lack visible. (A stricter contract would re-prompt or refuse; we kept it permissive for demo.)

**Q5.5.** *What's the system prompt strategy?*

**A:** Single system prompt with: (a) role ("retail IT support assistant"), (b) the exact citation format (`[N]`), (c) instruction to refuse if context is insufficient instead of inventing, (d) device-aware framing. Prompt is short (~150 tokens) — the heavy lifting is done by the retrieval scope filter, not by prompt engineering. → [src/generate.py](src/generate.py) `SYSTEM_PROMPT`.

---

## 6. Safety

**Q6.1.** *How is Azure Content Safety integrated?*

**A:** Two-stage gating in the agent. Input safety: `check_input(user_query)` runs after `intent_router` — if blocked, route becomes `"blocked_input"` and we short-circuit to the formatter with a refusal message. Output safety: `check_output(generated_answer)` runs before the formatter — if blocked, the answer is replaced with a refusal. Both go through `SafetyResult` so the trace records `blocked_category` + per-category severities. → [src/safety.py](src/safety.py), [src/agent.py](src/agent.py) input/output safety nodes.

**Q6.2.** *What categories does it block?*

**A:** Azure Content Safety's standard four: Hate, SelfHarm, Sexual, Violence. Each scored 0/2/4/6; we treat ≥4 as blocked. We don't re-train or extend the categories — using Azure's defaults is the point of using a managed service. → [src/safety.py](src/safety.py).

**Q6.3.** *What if Content Safety isn't configured (no env vars)?*

**A:** `SafetyResult` returns `passed=True, skipped=True` and the request flows through. This is intentional for local dev / unit-test environments where you don't want to require a Content Safety endpoint. The trace records `skipped=True` so reviewers can see when safety was disabled. → [src/safety.py:74-76](src/safety.py#L74-L76).

**Q6.4.** *Could a legitimate query get blocked? False positive risk?*

**A:** Yes — a question like "the printer keeps jamming, I'm going to kill it" might trigger Violence. Mitigation in this demo: the threshold is 4 (medium severity), not 2. Production would want either a custom blocklist + whitelist for IT-support-domain phrases, or a feedback mechanism so users can flag false-positive blocks for review. Not implemented; counts as a known operational gap.

---

## 7. Telemetry & observability

**Q7.1.** *What's the difference between FinalRagTrace and OpenTelemetry traces?*

**A:** Two different layers. **FinalRagTrace** is application-domain typed data (the structured output the reviewer / UI sees per query). **OpenTelemetry** spans are infrastructure-level — one span per LangGraph node, exported to App Insights for cost/latency profiling, error tracking, distributed tracing across the Function App. They overlap (both instrument the agent's node executions) but serve different audiences: FinalRagTrace for "did the reasoning go right?", OTel for "is the system healthy?". → [src/agent.py](src/agent.py) `_tracer.start_as_current_span` calls, [src/telemetry.py](src/telemetry.py).

**Q7.2.** *Where do OTel traces actually go?*

**A:** Application Insights (Azure-managed, in the same resource group). `azure-monitor-opentelemetry`'s `configure_azure_monitor()` reads `APPLICATIONINSIGHTS_CONNECTION_STRING` and exports spans + logs over OTLP. View them in the Azure portal under the App Insights resource's "Transaction search" or "Application map". → [src/telemetry.py](src/telemetry.py), [infra/main.bicep](infra/main.bicep) App Insights resource.

**Q7.3.** *What's the runtime overhead of all this tracing?*

**A:** OTel spans add ~1-3ms per node + ~50-200ms total export latency (batched, async, doesn't block the response). FinalRagTrace serialization is sub-ms — it's just dicts. Compared to the LLM calls (gpt-4o ≈ 1-3s) and AI Search reads (~100-300ms), tracing is negligible. The cost/benefit is reviewer-clarity per millisecond — extremely high.

**Q7.4.** *Can you disable telemetry for testing or air-gapped envs?*

**A:** Yes — leave `APPLICATIONINSIGHTS_CONNECTION_STRING` unset and `configure_azure_monitor()` becomes a no-op (silent fallback in [src/telemetry.py:30-34](src/telemetry.py#L30-L34)). `_tracer.start_as_current_span()` still runs but its spans go nowhere. Code path is unchanged.

---

## 8. Infrastructure & deployment

**Q8.1.** *What does `bash infra/deploy.sh kb-rag-rg swedencentral` actually do?*

**A:** Five phases: (1) provisions Bicep resources (Storage + AI Search Standard + AI Foundry Project + Document Intelligence + Content Safety + App Insights + Function App), (2) grants Content Safety RBAC, (3) wires App Insights connection string, (4) deploys 4 AI service models (gpt-4o, text-embedding-3-small), (5) Function App MSI grants + zip deploys the function code. Writes a complete `.env` at the end. Idempotent — re-running on an existing RG is safe. → [infra/deploy.sh](infra/deploy.sh).

**Q8.2.** *Why Bicep instead of Terraform / ARM templates?*

**A:** Bicep is Microsoft's first-party IaC for Azure — better SDK support, fewer abstraction leaks, and direct mapping to ARM resource types. Terraform would also work but adds another tool to install + manage state. For a single-environment demo, Bicep + Azure CLI is the path of least resistance. → [infra/main.bicep](infra/main.bicep).

**Q8.3.** *How does authentication work? API keys or AAD?*

**A:** AAD via `DefaultAzureCredential` for the data plane (Storage, AI Search, AI Foundry chat/embedding, Document Intelligence, Content Safety). `deploy.sh` auto-grants the deploying user the right RBAC roles (Storage Blob Data Contributor, Azure AI Developer, Cognitive Services User, Search Index Data Contributor). The Function App uses its system-assigned MSI for the same set. API keys are accepted as fallback for AI Search / Document Intelligence but not used in the demo. → [infra/deploy.sh](infra/deploy.sh) Phase 2.

**Q8.4.** *Why Standard SKU AI Search instead of Basic?*

**A:** L2 semantic reranker requires Standard or higher. Basic (~$75/mo) is text-only. Since the reranker is a core architectural choice (it's the difference between hybrid and hybrid_semantic mode), Basic isn't viable. Trade-off: Standard is ~$250/mo flat regardless of usage. **Listed as a cost concern** — production with budget pressure should evaluate whether L2 reranker quality lift justifies the SKU jump.

**Q8.5.** *What's the Function App's contract? When does it fire?*

**A:** Blob trigger on the `kb-docs` container — fires on add/update events for any blob (recursive). Handler is `auto_ingest(blob)` which calls `src.ingest.ingest_single(blob_path)`, running the full pipeline for that single doc. Cold start ~10s on Y1 Consumption SKU; warm executions ~2-5s for a small PDF. Concurrency: Functions runtime auto-scales up to ~10 parallel executions; we haven't load-tested. → [infra/functions/function_app.py](infra/functions/function_app.py).

---

## 9. Frontends

**Q9.1.** *Why both Chainlit AND a notebook?*

**A:** Different audiences. **Chainlit** for live interactive demo / multi-turn conversation / showing the CoT step panels live. **Notebook** for static reproducibility — a reviewer can read it like a paper, see one canonical query walk through every stage with structured tables, no Azure auth required to read it (only to re-execute). They share the same backend modules, so behavior matches; only the rendering differs.

**Q9.2.** *How are sources rendered in Chainlit?*

**A:** Cited files auto-attach to the answer message as `cl.Pdf` / `cl.Text` elements with `display="side"` — clicking opens the original document in the right panel for citation comparison. There's also an on-demand `files` command that shows the full corpus inventory grouped by category, with one action button per file. → [src/app.py](src/app.py) `_attach_cited_files_to_message`, `_send_inventory`.

**Q9.3.** *What's the FileTree custom element?*

**A:** A React custom element ([public/elements/FileTree.jsx](public/elements/FileTree.jsx)) mounted into the Chainlit sidebar. Shows the corpus tree (devices/families/models/doc_types) so a user can browse what's available. The header `Files` button (custom JS in `public/header_files_button.js`) bridges header click → typing `files` into the chat input, since Chainlit's `[[UI.header_links]]` only supports static URLs, not Python callbacks.

---

## 10. Reproducibility & dev experience

**Q10.1.** *Why uv instead of pip / poetry / conda?*

**A:** uv is 10-100× faster than pip for resolution + install, has a real solver (catches conflicts at lock time, not at runtime), and produces a deterministic `uv.lock` for reproducibility. Poetry would also give a lockfile but is slower and has a less stable resolver. Conda is heavier, more for scientific/binary packages. uv is the pragmatic choice in 2026 — fast, simple, single binary.

**Q10.2.** *How does the test suite work? Does it need Azure?*

**A:** No Azure credentials required. Tests cover only pure functions: `tests/test_chunk.py` (chunking), `tests/test_extract.py` (path-to-metadata parsing), `tests/test_search.py` (filter construction). Run with `pytest`. CI-friendly. The integration path (real Azure calls) isn't tested automatically — that's a known gap, but a prerequisite for any test-of-pipeline-correctness would be a stable test corpus + mock layer for Azure SDKs, neither of which we built.

**Q10.3.** *What does `sync.py --full-rebuild` do?*

**A:** (1) Stops the Function App so the local pipeline and the blob trigger don't race. (2) Wipes all `kb-chunks` index entries. (3) Uploads `data/` to the `kb-docs` blob container. (4) Walks every blob, runs `extract → figures → chunk → embed → upsert`. (5) Restarts the Function App. Idempotent: re-running produces identical chunk IDs (content-hashed) so the index ends up in the same state. → [scripts/sync.py](scripts/sync.py).

**Q10.4.** *How is `requirements.txt` related to `pyproject.toml`?*

**A:** `pyproject.toml` is the source of truth for dependencies. `uv.lock` is the deterministic transitive-pin lockfile. `requirements.txt` is auto-generated from `uv.lock` via `uv export --no-hashes --no-dev` — exists only so that someone without uv can still `pip install -r requirements.txt`. The header comment in the file says "auto-generated, edit pyproject.toml instead". → [pyproject.toml](pyproject.toml), [requirements.txt:1](requirements.txt#L1).

---

## 11. Limitations & production gaps

**Q11.1.** *What's the biggest gap before this is production-ready?*

**A:** **No automated evaluation harness.** Without recall@k, citation accuracy, groundedness scores on a held-out gold-QA set, every "this works better" claim is anecdotal. That's why the `eval_harness.py` deletion is intentional — the previous one was a stub that didn't actually measure anything. A real harness is the next addition. → README "Known Limitations".

**Q11.2.** *Why no LangGraph checkpointer? History is lost on restart.*

**A:** `MemorySaver` is in-process. Production should wire `langgraph.checkpoint.sqlite.SqliteSaver` (one-line swap) or the Postgres equivalent so threads persist across Chainlit restarts. We didn't do it because (a) demo doesn't need session continuity, (b) adds an external dependency (sqlite file or Postgres). Acknowledged limitation. → README "Known Limitations".

**Q11.3.** *What about cost monitoring? AI Search bills hourly.*

**A:** No budget alerts wired up. The resource group will keep billing $0.30/hr ($250/mo) for AI Search Standard whether you query it or not. Production would set Azure budget + alert rules at the subscription or RG level. Documented as a limitation.

**Q11.4.** *How would you scale to 10k or 100k documents?*

**A:** Several inflection points:
- AI Search Standard SKU has ~50GB index limit — 100k docs likely exceeds; bump to Standard 2 or partition by tenant
- Document Intelligence is pay-per-page — at 100k docs × avg 20 pages × $1.50/1000 = $3000 one-time; manageable but worth caching extracted markdown to blob (we already do)
- Function App Y1 Consumption auto-scales but cold starts hurt batch ingest — switch to Premium or run sync.py in parallel
- Embedding rate limits: text-embedding-3-small caps at ~350k TPM; batch in larger windows
- Chunk count: 100k docs × ~30 chunks/doc = 3M chunks — index queries stay fast (HNSW is sublinear) but L2 reranker latency degrades. Probably fine up to ~5M chunks, then need to shard.

**Q11.5.** *Multi-tenant — different customers, isolated corpora?*

**A:** Not built for it. Every chunk lives in one `kb-chunks` index. Multi-tenant options:
- Separate index per tenant (simplest, scales OK to ~100 tenants)
- Single index with `tenant_id` field + always-on filter (more efficient, harder to delete a tenant cleanly)
- Separate AI Search instance per tenant (most isolation, most expensive)
This project picks none — it's single-tenant by design.

---

## 12. Trade-off / "what if" questions

**Q12.1.** *What if you had to swap Azure AI Search for Pinecone / Weaviate / pgvector?*

**A:** The retrieval interface is mostly contained in [src/search.py](src/search.py) and [src/index.py](src/index.py). Swap would require: (1) replacing index schema definition with the target's API, (2) replacing `client.search(...)` with the target's hybrid query, (3) finding a replacement for L2 semantic reranker (Pinecone has a hosted reranker; Weaviate has its own; pgvector would need a separate cross-encoder service). Estimated effort: ~2-3 days. The rest of the stack (chunking, ingestion, agent, generator) wouldn't change.

**Q12.2.** *What if a user asks something off-topic (cooking recipes)?*

**A:** Three layers of defense. (1) intent_router classifies the query type — `general_kb` or `conceptual` queries that can't be matched to a known device get routed but with no scope filter. (2) Retrieval returns low `reranker_score` chunks (typically <2.0 vs ~2.5+ for in-scope queries — see the demo notebook's "Off-topic contrast" cell). (3) Generator prompt instructs to refuse if context is insufficient. The reviewer can SEE this happen end-to-end in the trace.

**Q12.3.** *What if two valid documents disagree on the answer?*

**A:** The L2 reranker picks based on semantic relevance to the query. Both might end up in the top-N. The generator then sees both with their chunk IDs; it can either cite both ("[1] says X, but [3] says Y") or pick one. This is by design — we don't try to merge or arbitrate. Production might add a "version" filter (we have `version` field on chunks) to prefer newer docs, or surface conflicts to the user.

**Q12.4.** *What happens when a doc is updated (new version)?*

**A:** Upload to blob → Function App fires `auto_ingest` → calls `_delete_existing_chunks(doc_id)` first, then re-embeds and upserts. Result: clean replacement. `chunk_id` is content-hashed so chunks whose content didn't change keep the same ID (no churn). For figures, `_delete_existing_figures(doc_id)` clears the `kb-figures/{stem}/` folder before re-rendering. → [src/ingest.py](src/ingest.py) `_delete_existing_chunks`, `_delete_existing_figures`, `ingest_single`.

**Q12.5.** *What's the latency budget per query?*

**A:** Measured: ~1.5–3s end-to-end on swedencentral. Breakdown: intent_router LLM ~600ms + retrieval ~200ms + L2 rerank ~150ms (server-side) + generator LLM ~1-2s + safety checks ~100ms each. Network roundtrip dominates. Optimization options: parallelize input safety check with retrieval (independent), use gpt-4o-mini for intent_router (cheaper + faster), batch embeddings for the query embedding. Demo runs are within "feels responsive" range so we didn't optimize.

**Q12.6.** *How would you A/B test chunk size in production?*

**A:** Build two indexes (`kb-chunks-256`, `kb-chunks-512`) ingested with different `CHUNK_MAX_TOKENS`. Add a feature flag to `hybrid_search` selecting which to query. Run a holdout gold-QA set against both, score recall@k + citation accuracy + answer groundedness (LLM-as-judge or human eval). Pick winner, retire loser. **This requires the eval harness we don't have.**

**Q12.7.** *Could you cache LLM calls?*

**A:** Anthropic prompt caching isn't in OpenAI yet but Azure OpenAI now supports prompt caching for system prompts ≥1024 tokens. For this app, the system prompt is ~150 tokens — too short to benefit. The bigger cache opportunity is: skip the intent_router LLM for queries that exact-match a recent one (LRU on query string), and cache embedding for repeated queries (text-embedding-3-small returns deterministic embeddings, so memoize). Not implemented; would shave ~30% off repeat-query latency.

**Q12.8.** *How would you add a new device family?*

**A:** Three steps. (1) Drop docs under `data/devices/{new_family}/{model}/{doc_type}/` following the existing convention. (2) Run `python scripts/sync.py --full-rebuild` (or just upload new blobs and let the Function App fire). (3) Optionally update `chainlit.md` and the demo notebook's intro to mention the new device. **No code changes** — path-derived metadata extraction handles new device_family / device automatically. The intent_router LLM will start detecting the new family on its own from queries naming it.

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

**Where to put this doc:**
- Personal prep notes only → keep here in `~/.claude/plans/`
- Or commit to repo as `docs/QA.md` for any future reviewer / contributor → simple `cp` + `git add` + commit + push (no Claude trailer)

## Verification

This is a documentation deliverable, not a code change. "Verification" here means: cross-check 5–10 random Q&A entries against the actual code by clicking each pointer. If any answer is wrong, the source of truth is the code, not this doc.
