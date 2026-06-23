# Family Office Intelligence RAG

Family Office Intelligence RAG is a production-shaped Micro-RAG pipeline for querying a validated family office intelligence workbook. It ingests curated evidence records, stores vectors in ChromaDB, applies structured retrieval and reranking, and answers questions through Streamlit with grounded source chips.

## Architecture

```text
XLSX workbook
  -> src/data loader + validator
  -> LangChain Documents
  -> Ollama embeddings
  -> persistent Chroma collection
  -> retrieval router + metadata filters
  -> service layer
  -> Streamlit UI
  -> Ollama or optional Claude answer generation
```

The UI never talks directly to Chroma. Ingestion runs independently from the application, vector-store access is isolated in `src/vectorstore`, retrieval is isolated in `src/retrieval`, and user workflows go through `src/services`.

## Local-First Setup With Ollama

1. Install Ollama from https://ollama.com/download

2. Pull the default models:

```bash
ollama pull qwen3:8b
ollama pull bge-m3
```

3. Create env:

```bash
cp .env.example .env
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Ingest:

```bash
python scripts/ingest.py
```

6. Run app:

```bash
streamlit run app/streamlit_app.py
```

The same local workflow is available through `make`:

```bash
make setup
make ingest
make run
```

Default mode requires zero paid API keys:

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
OLLAMA_EMBEDDING_MODEL=bge-m3
```

## Claude Quality Testing Mode

Claude is optional and used only for answer generation. Embeddings always remain local through Ollama.

```env
LLM_PROVIDER=claude
EMBEDDING_PROVIDER=ollama
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-3-5-sonnet-latest
```

No index rebuild is needed when changing only the LLM provider. The same Chroma vectors are queried with the same Ollama embedding model.

## Provider Switching Rules

- Changing LLM provider: no rebuild required.
- Changing `OLLAMA_EMBEDDING_MODEL`: rebuild required.
- Changing dataset: rebuild required.
- Changing chunking strategy: rebuild required.
- Embedding provider must remain `ollama`.

The ingestion process writes `vector_db/chroma_embedding_config.json`. The app compares that stored embedding signature with the current config and refuses to query a mismatched index.

## Retrieval Tuning

Retrieval tuning is configured through environment variables instead of the UI:

```env
RETRIEVAL_TOP_K=6
RETRIEVAL_CANDIDATE_TOP_K=25
RETRIEVAL_SCORE_THRESHOLD=0.0
RERANKER_ENABLED=true
RERANKER_PROVIDER=flag_embedding
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=auto
RERANK_TOP_K=5
```

Without reranking, `RETRIEVAL_TOP_K` controls how many documents are sent to answer generation. With reranking enabled, Chroma retrieves `RETRIEVAL_CANDIDATE_TOP_K` candidates, the reranker scores those candidates, and only `RERANK_TOP_K` documents are sent to answer generation. `RETRIEVAL_SCORE_THRESHOLD` filters out low-relevance vector results before reranking and must be between `0.0` and `1.0`.

The default local reranker is `BAAI/bge-reranker-v2-m3` through `FlagEmbedding`. Install dependencies before enabling reranking:

```bash
python3 -m pip install -r requirements.txt
```

If the local reranker cannot load or score results because of a package/runtime issue, the app continues with vector retrieval and logs the reranker event for operators. Reinstalling from `requirements.txt` keeps `transformers` on the compatible 4.x line used by `FlagEmbedding`.

## Chat History

The app stores one retained chat timeline and exposes a clear-history button in the sidebar. There is no multiple-chat UI.

```env
CHAT_HISTORY_PATH=state/chat_history.sqlite3
```

SQLite keeps local and single-instance deployments simple while preserving chat continuity. The service boundary keeps the storage layer straightforward to replace for authenticated multi-user deployments.

## Production UI

The default UI hides internal retrieval controls, document types, confidence labels, model health, and index rebuild actions. To expose operational diagnostics locally, set:

```env
SHOW_ADMIN_PANEL=true
```

Keep this disabled for end-user deployments. Rebuild the index through `python scripts/ingest.py` or an operator workflow instead of exposing rebuild controls to users.

## Dataset

The default dataset path is:

```text
data/family_offices_final.xlsx
```

Required workbook sheets:

- `Family_Offices`
- `Recent_Activities`
- `Source_Log`
- `Record_Validation`
- `RAG_Documents`
- `Data_Dictionary`
- `QA_Checks`

Only `RAG_Documents` is embedded. `Family_Offices` is the clean entity table and supplies structured filter options. `Record_Validation`, `Data_Dictionary`, and `QA_Checks` preserve validation and auditability outside the main dataset table.

## Evaluation

After ingestion:

```bash
python scripts/run_eval.py
```

Golden query checks compare retrieved entities, source domains, and document types against `eval/golden_queries.json`:

```bash
python scripts/run_eval.py --golden eval/golden_queries.json
# or
make eval-golden
```

The evaluator prints the LLM provider/model, embedding provider/model, vector DB signature status, retrieved document IDs, generated answers, citations, sources, evidence status, and notes.

## Tests

```bash
pytest
```

The tests cover loader validation, duplicate handling, metadata parsing and normalization, deterministic intent routing, provider validation, Ollama-only embeddings, signature mismatch handling, health checks, and grounded refusal when no evidence is available.

## Deployment Notes

This project is local-first and container-ready. For a hosted deployment, run Streamlit alongside Ollama on a VPS/container host, or use the optional Claude inference path while keeping embeddings and index signatures reproducible.

Docker Compose is provided for local container execution:

```bash
make compose-up
```

By default, Compose loads `.env.docker.example` and points the app container at Ollama running on the host through `host.docker.internal`. Override the app's Ollama URL with `COMPOSE_OLLAMA_BASE_URL` when needed. To run the optional Ollama container profile instead:

```bash
make compose-up-ollama
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull bge-m3
```

For assessment review, the local demo path highlights the full pipeline: Ollama running, ingestion succeeding, retrieval/reranking logs in the terminal, and Streamlit returning grounded answers with source chips.
