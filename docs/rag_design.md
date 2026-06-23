# RAG Design Note

## Stack Choices

The backend uses Python, Pandas, OpenPyXL, LangChain document/message primitives, ChromaDB, Ollama, and optional Claude inference. Streamlit provides the user-facing chat application while ingestion, retrieval, vector-store access, and answer generation remain separated in service modules.

## Why Local Ollama Mode

Ollama makes the default application local-first, private, and easy to demonstrate without paid API keys. On an Apple Silicon machine, `qwen3:8b` and `bge-m3` provide a strong balance between answer quality, retrieval quality, and local latency.

## Why Claude Is Optional Only For Inference

Claude is supported as an optional answer-generation provider for quality comparison. Embeddings remain Ollama-only so vector indexes are reproducible and do not change when the answer model changes.

## Why Ollama Embeddings Are Fixed

The embedding provider is fixed to Ollama to keep retrieval behavior predictable. The app stores the embedding signature in `vector_db/chroma_embedding_config.json` and protects retrieval quality by requiring the active embedding model to match the indexed vector database.

## Why Chroma

Chroma provides a local persistent vector database with straightforward metadata filtering. The implementation is isolated in `src/vectorstore/chroma_store.py`, which keeps the vector-store boundary clean if a managed vector database is introduced later.

## Chunking Strategy

The workbook already provides retrieval-ready rows in `RAG_Documents`. Each row is treated as one semantic document. This avoids arbitrary token-window chunking and preserves the relationship between a family office, the evidence text, metadata, and source URLs.

## Retrieval Architecture

Retrieval has four layers:

- Deterministic intent routing in `src/retrieval/intent_router.py`.
- Metadata filter construction in `src/retrieval/filters.py`.
- Chroma similarity search in `src/vectorstore/chroma_store.py`.
- Optional reranking in `src/retrieval/reranker.py`.

Source-coverage queries are routed internally toward source-profile records. Recent activity queries are routed toward recent-activity records. General queries search across available record types while the UI exposes only business-friendly filters such as region, country, and family office type.

## Grounding Strategy

The prompt instructs the LLM to answer only from the retrieved family-office records, never invent family offices or unsupported details, and state uncertainty plainly when evidence is partial. The LLM emits only compact source IDs; the app deterministically renders those IDs as source chips after generation so raw URLs and broken markdown do not leak into the answer.

## Failure Handling

The app and scripts include operational handling for dataset availability, vector database readiness, metadata validation, empty retrieval, model/provider health, Chroma errors, reranker issues, and embedding signature mismatches. Terminal logs include request IDs so retrieval, reranking, and answer streaming can be followed request by request.

## Scope And Next Improvements

The current design is intentionally compact and production-shaped for a 50-record Micro-RAG. Natural next improvements are claim-level evidence rows, exact entity lookup for named-family-office queries, richer golden-query evaluation, and a managed vector/chat store for multi-user hosted deployments.
