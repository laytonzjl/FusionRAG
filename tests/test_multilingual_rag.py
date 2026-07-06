from __future__ import annotations

from pathlib import Path

import pytest

from config import RagConfig
from rag_core.hybrid_index import HybridIndex, HybridRecord
from rag_core.evaluation import recall_at_k, mrr_at_k, ndcg_at_k, entity_coverage_rate
from rag_core.engine import RAGEngine
from rag_core.models import SearchResult
from rag_core.multilingual import (
    detect_language,
    normalize_entity_name,
    normalize_for_lexical_search,
    query_focus_terms,
    tokenize_for_search,
)
from rag_core.query_planner import build_query_plan, build_rule_query_plan
from rag_core.reranker import CrossEncoderReranker
from rag_core.retrieval import RetrievalService
from rag_ui.views import _adjust_answer_meta_by_answer, _calibrated_confidence


def test_entity_normalization_variants_are_consistent() -> None:
    assert normalize_entity_name("汤姆索亚") == normalize_entity_name("汤姆·索亚")
    assert normalize_entity_name("汤姆 · 索亚") == normalize_entity_name("汤姆索亚")
    assert normalize_entity_name("TOM SAWYER") == normalize_entity_name("Tom Sawyer")
    assert normalize_entity_name("汤姆索亚？") == normalize_entity_name("汤姆索亚")


def test_latin_accents_and_casefold_for_lexical_search() -> None:
    assert normalize_for_lexical_search("Résumé") == normalize_for_lexical_search("resume")


def test_mixed_technical_tokens_are_preserved() -> None:
    tokens = tokenize_for_search(r"API timeout v1.2.0 C++ C# .NET C:\tmp\foo_bar.py")
    joined = " ".join(tokens)
    assert "api" in joined
    assert "timeout" in joined
    assert "v1.2.0" in joined
    assert "c++" in joined
    assert "c#" in joined
    assert ".net" in joined
    assert "foo_bar" in joined


def test_language_detection_handles_cjk_latin_mixed() -> None:
    profile = detect_language("Tom Sawyer 是汤姆·索亚。")
    assert profile.language in {"mixed", "zh", "en"}
    assert profile.script_distribution


def test_exact_entity_retrieval_prioritizes_alias(tmp_path: Path) -> None:
    index = HybridIndex(path=tmp_path / "hybrid.sqlite", collection_name="test")
    index.reset_collection()
    index.upsert_records(
        [
            HybridRecord(
                chunk_id="c1",
                document_id="d1",
                parent_chunk_id="p1",
                chunk_kind="entity_card",
                language="zh",
                title="汤姆·索亚历险记",
                section_path="Entity Card / 汤姆·索亚",
                content="实体：汤姆·索亚\n实体类型：人物\n所属文档：汤姆·索亚历险记",
                metadata={"file_name": "tom.pdf", "chunk_kind": "entity_card"},
                aliases=["汤姆·索亚", "汤姆索亚", "Tom Sawyer"],
            )
        ]
    )
    hits = index.exact_search(["汤姆索亚"], limit=5)
    assert hits
    assert hits[0].chunk_id == "c1"


def test_query_planner_rule_fallback_for_english_entity() -> None:
    plan = build_rule_query_plan("Who is Tom Sawyer?", RagConfig())
    assert plan.intent == "entity_definition"
    assert plan.query_language in {"en", "unknown"}
    assert plan.entities
    assert plan.entities[0].surface == "tom sawyer"
    assert plan.retrieval_queries[0].origin == "original"


def test_query_planner_relation_beats_definition_when_friend_is_present() -> None:
    plan = build_rule_query_plan("汤姆索亚最好的朋友是谁？", RagConfig())
    assert plan.intent == "entity_relation"


def test_semantic_query_planner_generalizes_without_keyword_pattern() -> None:
    class FakeSemanticClient:
        api_config = type("ApiConfig", (), {"embedding_provider": "fake", "embedding_model": "semantic-test"})()

        def embed_query(self, text: str):
            lowered = text.casefold()
            if "一起冒险" in lowered or "companions" in lowered or "relationship" in lowered:
                return [1.0, 0.0]
            return [0.0, 1.0]

        def embed_documents(self, texts, batch_size: int = 32):
            vectors = []
            for text in texts:
                lowered = text.casefold()
                if "朋友" in lowered or "companions" in lowered or "relationship" in lowered:
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return vectors

    plan = build_query_plan("汤姆通常和谁一起冒险？", RagConfig(), client=FakeSemanticClient())
    assert plan.intent == "entity_relation"
    assert plan.planner_source == "semantic_intent"


def test_query_focus_terms_strip_question_words() -> None:
    assert query_focus_terms("Who is Tom Sawyer?")[0] == "tom sawyer"
    assert query_focus_terms("汤姆索亚最好的朋友是谁？")[0] == "汤姆索亚"


def test_relation_query_splits_subject_from_predicate_in_chinese() -> None:
    plan = build_rule_query_plan("汤姆索亚认识哪些人？", RagConfig())

    assert plan.intent == "entity_relation"
    assert plan.answer_type == "person_list"
    assert plan.entities[0].surface == "汤姆索亚"
    assert "汤姆索亚认识人" not in [entity.surface for entity in plan.entities]
    assert any(variant.origin == "relation_expansion" for variant in plan.retrieval_queries)


def test_relation_query_supports_middle_dot_and_english_forms() -> None:
    chinese = build_rule_query_plan("汤姆·索亚认识谁？", RagConfig())
    english = build_rule_query_plan("Tom Sawyer interacts with whom?", RagConfig())

    assert chinese.intent == "entity_relation"
    assert chinese.entities[0].surface == "汤姆·索亚"
    assert english.intent == "entity_relation"
    assert english.entities[0].surface == "Tom Sawyer"


def test_relation_evidence_judge_prefers_body_interaction_over_supplemental_notes() -> None:
    service = RetrievalService.__new__(RetrievalService)
    service.rag_config = RagConfig(top_k=6).normalized()
    plan = build_rule_query_plan("某角色认识哪些人？", RagConfig())
    focus_terms = ["某角色"]
    supplemental = SearchResult(
        "附录：人物原型说明中提到某角色根据作者认识的一些居民改写而成。",
        {"chunk_id": "supplemental", "file_name": "novel.pdf", "chunk_kind": "child"},
        0,
        0,
        0.9,
        chunk_id="supplemental",
        chunk_kind="child",
    )
    body = SearchResult(
        "某角色和阿青一起去了学校，两人交谈后又帮助李老师处理争执。",
        {"chunk_id": "body", "file_name": "novel.pdf", "chunk_kind": "child"},
        0,
        0,
        0.6,
        chunk_id="body",
        chunk_kind="child",
    )

    ranked = service._rank_relation_evidence(plan, [supplemental, body], focus_terms)

    assert ranked[0].chunk_id == "body"
    relation = ranked[0].diagnostics["relation_evidence"]
    assert relation["judge_passed"] is True
    assert "阿青" in relation["candidate_people"]


def test_weighted_rrf_uses_rank_not_raw_scores(tmp_path: Path) -> None:
    service = RetrievalService.__new__(RetrievalService)
    service.rag_config = RagConfig(rrf_k=60).normalized()
    a = SearchResult("a", {"chunk_id": "a"}, vector_score=999, keyword_score=0, final_score=0, chunk_id="a")
    b = SearchResult("b", {"chunk_id": "b"}, vector_score=0.1, keyword_score=0, final_score=0, chunk_id="b")
    fused = service._weighted_rrf("fact_lookup", {"dense_original_query": [b, a]})
    assert fused[0].chunk_id == "b"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_reranker_failure_falls_back(monkeypatch) -> None:
    def boom(model_name: str, device: str):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("rag_core.reranker._load_cross_encoder", boom)
    reranker = CrossEncoderReranker(RagConfig(enable_reranker=True, reranker_model="missing-model"))
    result = SearchResult("hello", {"chunk_id": "x"}, 0, 0, 0, chunk_id="x")
    results, status = reranker.rerank("hello?", [result])
    assert results[0].chunk_id == "x"
    assert status.startswith("fallback_rrf")


def test_confidence_is_not_raw_vector_score() -> None:
    result = SearchResult(
        "证据",
        {"chunk_id": "x", "text_quality_score": 0.8},
        vector_score=0.99,
        keyword_score=0.0,
        final_score=0.99,
        chunk_id="x",
    )
    confidence = _calibrated_confidence([result])
    assert confidence != pytest.approx(0.99)
    assert 0.0 < confidence < 0.99


def test_focus_guard_prefers_entity_document_over_unrelated_document() -> None:
    service = RetrievalService.__new__(RetrievalService)
    service.rag_config = RagConfig(top_k=6).normalized()
    target = SearchResult(
        "吹牛大王的故事片段",
        {"chunk_id": "a", "file_name": "吹牛大王.pdf", "document_id": "book-a"},
        0,
        0,
        0,
        chunk_id="a",
    )
    unrelated = SearchResult(
        "汤姆索亚的故事片段",
        {"chunk_id": "b", "file_name": "汤姆索亚.pdf", "document_id": "book-b"},
        0,
        0,
        0,
        chunk_id="b",
    )

    anchors = service._anchor_documents([target], ["吹牛大王"])
    guarded = service._apply_focus_guard([unrelated, target], ["吹牛大王"], anchors)

    assert guarded[0].chunk_id == "a"


def test_evidence_requirements_lift_generic_numeric_evidence() -> None:
    service = RetrievalService.__new__(RetrievalService)
    plan = build_rule_query_plan("蓝星项目预算是多少？", RagConfig())
    focus_terms = ["蓝星项目"]
    weak = SearchResult(
        "蓝星项目是一项内部迁移计划，本文介绍背景和范围。",
        {"chunk_id": "weak", "file_name": "project.md"},
        0,
        0,
        0,
        chunk_id="weak",
    )
    strong = SearchResult(
        "蓝星项目预算为 120 万元，费用范围包括模型推理、存储和运维。",
        {"chunk_id": "strong", "file_name": "project.md"},
        0,
        0,
        0,
        chunk_id="strong",
    )

    ranked = service._apply_evidence_requirements([weak, strong], plan, focus_terms)

    assert ranked[0].chunk_id == "strong"
    assert "numeric_value" in ranked[0].diagnostics["covered_evidence_requirements"]


def test_evidence_requirements_lift_generic_location_evidence() -> None:
    service = RetrievalService.__new__(RetrievalService)
    plan = build_rule_query_plan("艾拉去过哪些地方？", RagConfig())
    focus_terms = ["艾拉"]
    weak = SearchResult(
        "艾拉在故事中多次谈起自己的计划。",
        {"chunk_id": "weak", "file_name": "novel.md"},
        0,
        0,
        0,
        chunk_id="weak",
    )
    strong = SearchResult(
        "艾拉先到达北海港口，又前往云杉山和红石城。",
        {"chunk_id": "strong", "file_name": "novel.md"},
        0,
        0,
        0,
        chunk_id="strong",
    )

    ranked = service._apply_evidence_requirements([weak, strong], plan, focus_terms)

    assert ranked[0].chunk_id == "strong"
    assert "location_name" in ranked[0].diagnostics["covered_evidence_requirements"]


def test_uncertain_answer_caps_confidence_to_low(monkeypatch) -> None:
    from rag_ui import views

    monkeypatch.setattr(views.st, "session_state", {"last_retrieval": []})
    meta = {
        "confidence_score": 0.68,
        "confidence_label": "中",
        "source_count": 6,
        "source_type_label": "部分证据",
    }
    answer = "根据当前知识库内容，无法确定汤姆·索亚大概多少岁。所有上下文片段均未提及年龄。"

    adjusted = views._adjust_answer_meta_by_answer(meta, answer)

    assert adjusted["confidence_score"] <= 0.28
    assert adjusted["source_type_label"] == "证据不足"


def test_partial_evidence_answer_is_not_downgraded_to_refusal() -> None:
    meta = {
        "confidence_score": 0.82,
        "confidence_label": "高",
        "source_count": 3,
        "source_type_label": "直接证据",
    }
    answer = (
        "根据当前知识库内容，无法直接给出完整名单。"
        "结论：现有证据可确认汤姆的朋友包括哈克。"
        "证据：来源：tom.pdf，第 10 页。"
    )
    adjusted = _adjust_answer_meta_by_answer(meta, answer)
    assert adjusted["source_type_label"] == "部分证据"
    assert adjusted["confidence_score"] > 0.32


def test_basic_evaluation_metrics() -> None:
    retrieved = ["a", "b", "c"]
    assert recall_at_k(["b", "x"], retrieved, 2) == pytest.approx(0.5)
    assert mrr_at_k(["b"], retrieved, 3) == pytest.approx(0.5)
    assert ndcg_at_k({"b": 2.0, "c": 1.0}, retrieved, 3) > 0
    assert entity_coverage_rate(["Tom Sawyer"], ["Tom Sawyer is mentioned."]) == 1.0


def test_chroma_metadata_sanitizer_keeps_only_supported_scalars() -> None:
    metadata = {
        "chunk_id": "c1",
        "page_start": 1,
        "language_confidence": 0.91,
        "is_scanned": False,
        "script_distribution": {"Han": 0.8, "Latin": 0.2},
        "aliases": ["汤姆·索亚", "Tom Sawyer"],
        "empty_page": None,
        "bad_float": float("nan"),
    }
    sanitized = RAGEngine._sanitize_chroma_metadata(metadata)

    assert sanitized["chunk_id"] == "c1"
    assert sanitized["page_start"] == 1
    assert sanitized["language_confidence"] == pytest.approx(0.91)
    assert sanitized["is_scanned"] is False
    assert isinstance(sanitized["script_distribution"], str)
    assert isinstance(sanitized["aliases"], str)
    assert "empty_page" not in sanitized
    assert "bad_float" not in sanitized
