from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings
from src.evaluation.evaluator import load_golden_queries, run_evaluation, summarize_golden_results
from src.vectorstore.chroma_store import embedding_signature_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval and answer quality checks.")
    parser.add_argument(
        "--golden",
        type=Path,
        help="Path to a golden query JSON file with expected retrieved entities and source domains.",
    )
    parser.add_argument("--top-k", type=int, help="Override retrieval candidate count for evaluation.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Override retrieval score threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if not settings.vector_db_exists:
        print("Vector database is missing. Run python scripts/ingest.py first.")
        return 1
    signature_matches = embedding_signature_matches(settings.chroma_path, settings.embedding_signature)
    payload = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.ollama_embedding_model,
        "vector_db_signature_matches": signature_matches,
        "results": [],
    }
    if not signature_matches:
        payload["error"] = "The current embedding model differs from the indexed vector DB. Please rebuild the index."
        print(json.dumps(payload, indent=2))
        return 1
    golden_queries = load_golden_queries(args.golden) if args.golden else None
    payload["evaluation_mode"] = "golden" if golden_queries else "representative"
    if args.golden:
        payload["golden_query_file"] = str(args.golden)
    payload["results"] = run_evaluation(
        settings,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        golden_queries=golden_queries,
    )
    if golden_queries:
        payload["golden_summary"] = summarize_golden_results(payload["results"])
    print(json.dumps(payload, indent=2))
    if golden_queries and payload["golden_summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
