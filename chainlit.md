# Azure Observable RAG

Ask anything about your indexed knowledge base. Each query is processed through six explicit LangGraph steps — **Intent Detection → Query Planning → Retrieval → Evidence Selection → Generation → Final Answer** — and the trace from each step renders live in the Chain-of-Thought panel above the answer.

Every claim in the final answer is grounded in retrieved chunks and cited back to the source file + page.

## Try

- *How do I factory-reset Device A?*
- *What's the maximum operating temperature for Device B?*
- *What does error 101 mean and how do I fix it?*
- *What's our policy on storing customer payment data?*

Out-of-scope queries (e.g. *"tell me a joke"*) short-circuit through the conditional edge in the graph — no LLM cost, ~1 ms response.
