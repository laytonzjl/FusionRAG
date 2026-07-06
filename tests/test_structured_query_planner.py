from __future__ import annotations

import json
from types import SimpleNamespace

from rag_core.query_planner import build_query_plan


class FakePlannerClient:
    """模拟模型只返回计划 JSON，不包含真实答案。"""

    def complete_chat(self, messages, max_tokens=0):
        return json.dumps(
            {
                "intent": "numeric_lookup",
                "entity_mentions": [{"text": "汤姆索亚", "entity_type": "person"}],
                "requested_property": "年龄范围",
                "operation": "estimate",
                "answer_mode": "inferred",
                "constraints": ["仅依据知识库", "不要输出精确年龄"],
                "answer_shape": "range_or_life_stage",
                "retrieval_queries": [
                    {"text": "汤姆索亚 身份 成长阶段 学校经历", "language": "zh", "origin": "llm_structured"}
                ],
                "planner_confidence": 0.91,
            },
            ensure_ascii=False,
        )


class BrokenPlannerClient:
    def complete_chat(self, messages, max_tokens=0):
        return "not-json"


def _rag_config():
    return SimpleNamespace(enable_query_planner=True, enable_query_rewrite=True, query_rewrite_count=2)


def test_llm_structured_planner_is_primary_path():
    plan = build_query_plan("能从文中推测汤姆索亚处于什么年龄阶段吗？", _rag_config(), FakePlannerClient())
    assert plan.planner_source == "llm_structured"
    assert plan.semantics.operation == "estimate"
    assert plan.semantics.answer_mode == "inferred"
    assert plan.entities[0].surface == "汤姆索亚"
    assert any(item.origin == "llm_structured" for item in plan.retrieval_queries)
    assert plan.required_evidence == ["target_entity", "answer_statement"]


def test_bad_llm_output_falls_back_without_crashing():
    plan = build_query_plan("请解释这个项目的部署流程", _rag_config(), BrokenPlannerClient())
    assert plan.planner_source in {"semantic_intent_fallback", "rules_fallback"}
    assert plan.retrieval_queries
