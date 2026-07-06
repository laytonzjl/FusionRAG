from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_api_config, load_rag_config
from rag_core import build_engine


DEFAULT_QUESTIONS = [
    "汤姆索亚是谁？",
    "Who is Tom Sawyer?",
    "汤姆索亚最好的朋友是谁？",
    "API timeout 在 v1.2.0 中如何配置？",
]


def summarize_question(engine, question: str) -> dict:
    results = engine.retrieval.search(question)
    diagnostics = engine.retrieval_diagnostics()
    plan = diagnostics.get("query_plan", {})
    channels = diagnostics.get("candidates_by_channel", {})

    return {
        "question": question,
        "intent": plan.get("intent"),
        "query_language": plan.get("query_language"),
        "entities": [entity.get("surface") for entity in plan.get("entities", [])],
        "variants": [
            {
                "text": variant.get("text"),
                "language": variant.get("language"),
                "origin": variant.get("origin"),
            }
            for variant in plan.get("retrieval_queries", [])
        ],
        "entity_coverage_failed": diagnostics.get("entity_coverage_failed"),
        "evidence_judge_enabled": diagnostics.get("evidence_judge_enabled"),
        "relation_type": plan.get("relation_type"),
        "answer_type": plan.get("answer_type"),
        "channels": {name: len(items) for name, items in channels.items()},
        "top_results": [
            {
                "file": result.metadata.get("file_name")
                or result.metadata.get("document_title"),
                "page": result.metadata.get("page_start")
                or result.metadata.get("page"),
                "kind": result.chunk_kind or result.metadata.get("chunk_kind"),
                "language": result.metadata.get("chunk_language"),
                "rrf_score": round(float(result.rrf_score or 0.0), 4),
                "relation_evidence": result.diagnostics.get("relation_evidence"),
                "channels": [item.channel for item in result.contributions],
                "snippet": result.content[:120].replace("\n", " "),
            }
            for result in results[:3]
        ],
    }


def main(argv: list[str]) -> int:
    questions = argv or DEFAULT_QUESTIONS
    api_config = load_api_config()
    rag_config = load_rag_config().normalized()

    # 诊断默认只跑本地链路，避免命令行调试时误触发外部 LLM 请求。
    rag_config.enable_query_rewrite = False
    rag_config.enable_reranker = False
    rag_config.enable_evidence_judge = False
    rag_config.retrieval_candidate_k = min(rag_config.retrieval_candidate_k, 24)
    rag_config.top_k = min(rag_config.top_k, 5)

    engine = build_engine(api_config=api_config, rag_config=rag_config)
    payload = [summarize_question(engine, question) for question in questions]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
