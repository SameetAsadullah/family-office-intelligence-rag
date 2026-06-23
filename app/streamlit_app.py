from __future__ import annotations

import logging
import sys
from uuid import uuid4
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.settings import ProviderConfigError, get_settings
from src.data.loader import DatasetNotFoundError
from src.data.validator import DatasetValidationError
from src.health.providers import check_provider_health
from src.rag.citations import (
    answer_to_html_with_source_chips,
    citation_references_from_documents,
    references_for_answer,
    render_source_chip_row,
    strip_selected_sources_line,
    stream_preview_text,
)
from src.services.answer_service import AnswerService
from src.services.chat_history_service import ChatHistoryService
from src.services.ingestion_service import IngestionService
from src.services.search_service import SearchService
from src.utils.helpers import parse_source_urls
from src.utils.logging import configure_logging
from src.vectorstore.chroma_store import VectorStoreError, embedding_signature_matches, read_embedding_config

logger = logging.getLogger(__name__)


EXAMPLE_QUERIES = [
    "Which family offices invest in healthcare?",
    "Show European family offices with venture activity.",
    "Which family offices have recent investment activity?",
    "Find family offices with real estate exposure.",
    "Which family offices have publicly available source coverage?",
    "Give me five family offices relevant for fintech fundraising.",
]

SAFE_HTML_PREFIX = "__SAFE_HTML__\n"

SOURCE_CHIP_CSS = """
<style>
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.12rem 0.55rem;
    margin-left: 0.2rem;
    border-radius: 999px;
    background: rgba(120, 120, 120, 0.22);
    color: inherit !important;
    text-decoration: none !important;
    white-space: nowrap;
    font-size: 0.88em;
    line-height: 1.5;
}
.source-chip:hover {
    background: rgba(120, 120, 120, 0.34);
}
.source-chip-dot {
    width: 0.48rem;
    height: 0.48rem;
    border-radius: 999px;
    background: linear-gradient(135deg, #ff4f8b, #f6b73c);
    flex: 0 0 auto;
}
.source-chip-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.75rem;
}
.source-chip-heading {
    opacity: 0.72;
    font-size: 0.88em;
    margin-right: 0.15rem;
}
</style>
"""


def status_label(ok: bool, good: str, bad: str) -> str:
    return good if ok else bad


def render_admin_panel(settings) -> None:
    with st.expander("Admin", expanded=False):
        st.write(f"LLM provider: `{settings.llm_provider}`")
        st.write(f"LLM model: `{settings.llm_model}`")
        st.write(f"Embedding model: `{settings.ollama_embedding_model}`")
        st.write(status_label(settings.dataset_exists, "Dataset found", "Dataset missing"))
        st.write(status_label(settings.vector_db_exists, "Vector DB found", "Vector DB missing"))
        provider_health = check_provider_health(settings)
        if provider_health.ollama.ok:
            st.success("Ollama healthy")
        elif provider_health.ollama.server_running:
            st.warning(provider_health.ollama.message)
        else:
            st.error(provider_health.ollama.message)

        stored_config = None
        signature_matches = False
        if settings.vector_db_exists:
            try:
                stored_config = read_embedding_config(settings.chroma_path)
                signature_matches = embedding_signature_matches(settings.chroma_path, settings.embedding_signature)
            except VectorStoreError as exc:
                st.warning(str(exc))
        if stored_config:
            st.write(f"Indexed embedding: `{stored_config.embedding_signature}`")
        st.write(f"Current embedding: `{settings.embedding_signature}`")
        if settings.vector_db_exists and not signature_matches:
            st.warning("The current embedding model differs from the indexed vector DB. Please rebuild the index.")

        if settings.llm_provider == "claude":
            st.write(
                status_label(
                    settings.claude_api_key_configured,
                    "Claude API key configured",
                    "Claude API key missing",
                )
            )

        if st.button("Check Ollama status", use_container_width=True):
            if provider_health.ollama.ok:
                st.success(provider_health.ollama.message)
            else:
                st.error(provider_health.ollama.message)
            if provider_health.ollama.available_models:
                st.caption("Available models: " + ", ".join(provider_health.ollama.available_models))

        if st.button("Rebuild index", use_container_width=True):
            if not settings.dataset_exists:
                st.error(f"Dataset not found at {settings.data_path}")
            else:
                with st.spinner("Rebuilding Chroma index..."):
                    try:
                        result = IngestionService(settings).rebuild_index()
                        st.success(f"Indexed {result.documents_indexed} documents.")
                    except (DatasetNotFoundError, DatasetValidationError, ValueError) as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Index rebuild failed: {exc}")


def render_sidebar(settings, chat_history: ChatHistoryService):
    search_service = SearchService(settings)
    with st.sidebar:
        st.subheader("Chat")
        if st.button("Clear chat history", use_container_width=True):
            chat_history.clear_messages()
            st.session_state.selected_query = ""
            st.rerun()

        st.subheader("Filters")
        try:
            options = search_service.get_filter_options()
        except Exception:
            options = None

        region = st.selectbox("Region", ["All"] + (options.regions if options else []))
        country = st.selectbox("Country", ["All"] + (options.countries if options else []))
        family_office_type = st.selectbox(
            "Family Office Type", ["All"] + (options.family_office_types if options else [])
        )

        if settings.show_admin_panel:
            render_admin_panel(settings)

    filters = {
        "region": region,
        "country": country,
        "family_office_type": family_office_type,
    }
    return filters


def render_chat_history(chat_history: ChatHistoryService) -> None:
    for message in chat_history.list_messages():
        with st.chat_message(message.role):
            if message.role == "assistant" and message.content.startswith(SAFE_HTML_PREFIX):
                st.markdown(SOURCE_CHIP_CSS + message.content.removeprefix(SAFE_HTML_PREFIX), unsafe_allow_html=True)
            else:
                st.write(message.content)


def render_debug_evidence(retrieval):
    with st.expander("Debug evidence", expanded=False):
        for index, result in enumerate(retrieval.results, start=1):
            metadata = result.document.metadata
            confidence = metadata.get("confidence_level", "Unknown")
            family_office = metadata.get("family_office_name", "Unknown family office")
            st.markdown(f"**{index}. {family_office}**")
            st.caption(
                f"doc_id={metadata.get('doc_id', '')} | "
                f"doc_type={metadata.get('doc_type', '')} | "
                f"score={result.score:.3f}"
            )
            st.markdown(f"Confidence: `{confidence}`")
            st.write(result.document.page_content)
            source_urls = parse_source_urls(metadata.get("source_urls", ""))
            if source_urls:
                st.markdown("Sources: " + " · ".join(f"[{idx}]({url})" for idx, url in enumerate(source_urls, 1)))
            else:
                st.caption("No source URL listed.")


def stream_answer_to_placeholder(service: AnswerService, retrieval, request_id: str, answer_placeholder) -> tuple[str, int]:
    logger.info("[%s] answer_stream.start context_documents=%s", request_id, len(retrieval.results))
    stream = iter(service.stream_from_retrieval(retrieval))
    first_chunk = None
    with answer_placeholder.container():
        with st.spinner("Generating answer..."):
            try:
                first_chunk = next(stream)
            except StopIteration:
                first_chunk = None
    answer_placeholder.empty()

    answer_chunks: list[str] = []
    last_preview = ""
    if first_chunk:
        logger.info("[%s] answer_stream.first_chunk", request_id)
        answer_chunks.append(str(first_chunk))
        last_preview = stream_preview_text(answer_chunks)
        if last_preview:
            answer_placeholder.markdown(last_preview)
    for chunk in stream:
        answer_chunks.append(str(chunk))
        preview = stream_preview_text(answer_chunks)
        if preview != last_preview:
            last_preview = preview
            if preview:
                answer_placeholder.markdown(preview)
    return "".join(answer_chunks), len(answer_chunks)


def build_answer_html(answer_text: str, retrieval) -> tuple[str, int]:
    documents = [result.document for result in retrieval.results]
    references = citation_references_from_documents(documents)
    cleaned_answer = strip_selected_sources_line(answer_text)
    selected_references = references_for_answer(answer_text, references)
    answer_html = answer_to_html_with_source_chips(cleaned_answer, selected_references)
    source_html = render_source_chip_row(selected_references)
    return answer_html + source_html, len(selected_references)


def run_query(query: str, settings, filters, chat_history: ChatHistoryService) -> None:
    request_id = uuid4().hex[:8]
    logger.info(
        "[%s] request.start query_chars=%s llm_provider=%s llm_model=%s",
        request_id,
        len(query.strip()),
        settings.llm_provider,
        settings.llm_model,
    )
    if not query.strip():
        st.warning("Enter a query before searching.")
        logger.info("[%s] request.rejected reason=empty_query", request_id)
        return
    if not settings.dataset_exists:
        st.error(f"Dataset not found at {settings.data_path}")
        logger.info("[%s] request.rejected reason=dataset_missing", request_id)
        return
    if not settings.vector_db_exists:
        st.error("Search is not ready yet. Please ask an administrator to build the search index.")
        logger.info("[%s] request.rejected reason=vector_db_missing", request_id)
        return
    if not embedding_signature_matches(settings.chroma_path, settings.embedding_signature):
        st.error("The current embedding model differs from the indexed vector DB. Please rebuild the index.")
        logger.info("[%s] request.rejected reason=embedding_signature_mismatch", request_id)
        return

    chat_history.append_message("user", query)
    st.chat_message("user").write(query)

    service = AnswerService(settings)
    answer_text = ""
    final_html = ""
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        try:
            with answer_placeholder.container():
                with st.spinner("Searching knowledge base..."):
                    retrieval = service.retrieve(
                        query,
                        filters=filters,
                        top_k=settings.retrieval_candidate_top_k if settings.reranker_enabled else settings.retrieval_top_k,
                        score_threshold=settings.retrieval_score_threshold,
                        request_id=request_id,
                    )
        except VectorStoreError as exc:
            answer_placeholder.empty()
            st.error(f"Vector database error: {exc}")
            st.info("Try rebuilding the index. If the collection is corrupted, the rebuild will recreate it.")
            logger.exception("[%s] request.failed stage=vector_db", request_id)
            return
        except Exception as exc:
            answer_placeholder.empty()
            st.error(f"Retrieval failed: {exc}")
            logger.exception("[%s] request.failed stage=retrieval", request_id)
            return

        if len(retrieval.results) < 2:
            st.warning("I found limited matching information. Treat this answer as narrower than a full market scan.")

        answer_text, answer_chunk_count = stream_answer_to_placeholder(service, retrieval, request_id, answer_placeholder)
        final_html, selected_source_count = build_answer_html(answer_text, retrieval)
        if final_html:
            answer_placeholder.markdown(SOURCE_CHIP_CSS + final_html, unsafe_allow_html=True)
        logger.info(
            "[%s] answer_stream.done chunks=%s answer_chars=%s selected_sources=%s",
            request_id,
            answer_chunk_count,
            len(answer_text),
            selected_source_count,
        )
    if answer_text:
        chat_history.append_message("assistant", SAFE_HTML_PREFIX + final_html)
        logger.info("[%s] request.done stored_history=true", request_id)
    else:
        logger.info("[%s] request.done stored_history=false empty_answer=true", request_id)
    if settings.show_admin_panel:
        render_debug_evidence(retrieval)


def main() -> None:
    configure_logging()
    st.set_page_config(page_title="Family Office Intelligence RAG", layout="wide")
    try:
        settings = get_settings()
    except ProviderConfigError as exc:
        st.title("Family Office Intelligence RAG")
        st.error(str(exc))
        return

    st.title("Family Office Intelligence RAG")
    chat_history = ChatHistoryService(settings.chat_history_path)
    filters = render_sidebar(settings, chat_history)

    if "selected_query" not in st.session_state:
        st.session_state.selected_query = ""

    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        if cols[idx % 2].button(example, use_container_width=True):
            st.session_state.selected_query = example

    render_chat_history(chat_history)

    query = st.chat_input("Ask about family offices, investments, validation, or recent activity")
    active_query = query or st.session_state.pop("selected_query", "")
    if active_query:
        run_query(active_query, settings, filters, chat_history)


if __name__ == "__main__":
    main()
