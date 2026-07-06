from __future__ import annotations

"""Execution policy for semantic, structure-aware retrieval.

This module deliberately makes no decision from surface keywords, document
names, language-specific phrase lists, or regular expressions. Expensive stages
are controlled only by the validated semantic plan and by source-coverage state.
"""

from typing import Sequence


def config_value(config: object, name: str, default: object) -> object:
    value = getattr(config, name, default)
    return default if value is None else value


def as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "enabled"}


def normalized_mode(config: object, name: str, default: str = "adaptive") -> str:
    value = str(config_value(config, name, default) or default).strip().casefold()
    return value if value in {"adaptive", "always", "never"} else default


def is_fast_path_enabled(config: object) -> bool:
    return as_bool(config_value(config, "enable_adaptive_fast_path", False), False)


def plan_requires_deep_read(plan: object) -> bool:
    semantics = getattr(plan, "semantics", None)
    regions = {str(item).strip().casefold() for item in (getattr(semantics, "regions", []) or [])}
    scope = str(getattr(semantics, "scope", "entity_or_document") or "entity_or_document")
    operation = str(getattr(semantics, "operation", "lookup") or "lookup")
    return bool(
        regions.intersection({"front", "middle", "terminal"})
        or bool(getattr(semantics, "need_timeline", False))
        or bool(getattr(semantics, "need_entity_neighborhood", False))
        or scope in {"collection", "document", "section", "entity_neighborhood"}
        or operation in {"enumerate", "compare", "summarize", "trace", "verify", "estimate"}
    )


def is_coverage_fallback(plan: object) -> bool:
    return str(getattr(plan, "planner_source", "") or "").casefold() == "coverage_fallback"


def should_use_llm_planner(query: str, config: object) -> bool:
    """Use semantic planning whenever it is configured.

    `adaptive` intentionally means semantic-first rather than phrase-triggered.
    A fallback plan is still available if the model, network, or JSON response
    fails, but no rules infer the question intent from wording.
    """

    if not (query or "").strip():
        return False
    if not as_bool(config_value(config, "enable_query_planner", False), False):
        return False
    return normalized_mode(config, "planner_mode", "adaptive") != "never"


def candidate_budget(config: object, plan: object, final_top_k: int, collection_count: int) -> int:
    top_k = max(1, int(final_top_k))
    configured = int(config_value(config, "retrieval_candidate_k", top_k * 6) or top_k)
    deep = plan_requires_deep_read(plan) or is_coverage_fallback(plan)
    floor = max(top_k * (8 if deep else 5), 36 if deep else 24)
    cap = int(config_value(config, "deep_candidate_cap", 72 if deep else 48) or (72 if deep else 48))
    return min(max(collection_count, 1), cap, max(floor, configured))


def rerank_candidate_budget(config: object, plan: object, final_top_k: int) -> int:
    top_k = max(1, int(final_top_k))
    deep = plan_requires_deep_read(plan) or is_coverage_fallback(plan)
    configured = int(config_value(config, "reranker_candidate_k", top_k * 4) or top_k)
    floor = max(top_k * (5 if deep else 4), 20 if deep else 14)
    cap = int(config_value(config, "deep_reranker_cap", 36 if deep else 24) or (36 if deep else 24))
    return min(max(configured, floor), cap)


def query_variant_limit(config: object, plan: object) -> int:
    semantics = getattr(plan, "semantics", None)
    operation = str(getattr(semantics, "operation", "lookup") or "lookup")
    # Planner-supplied evidence views are used only for an enumerative contract.
    return 4 if operation == "enumerate" else (2 if plan_requires_deep_read(plan) else 1)


def _source_channels(result: object) -> set[str]:
    return {
        str(getattr(item, "channel", "") or "")
        for item in (getattr(result, "contributions", []) or [])
        if str(getattr(item, "channel", "") or "")
    }


def has_strong_local_evidence(plan: object, candidates: Sequence[object]) -> bool:
    if not candidates or is_coverage_fallback(plan):
        return False
    top = candidates[0]
    channels = _source_channels(top)
    metadata = getattr(top, "metadata", {}) or {}
    chunk_kind = str(metadata.get("chunk_kind") or getattr(top, "chunk_kind", "") or "")
    link_confidence = float(getattr(plan, "entity_linking_confidence", 0.0) or 0.0)
    local_channels = {"exact_entity", "structured", "lexical"}
    agreement = len(channels.intersection(local_channels)) >= 2
    durable_card = "exact_entity" in channels and chunk_kind in {
        "entity_card", "document_card", "section_card", "metadata", "table_card", "code_card"
    }
    return agreement or (durable_card and link_confidence >= 0.92)


def should_run_reranker(
    config: object,
    plan: object,
    candidates: Sequence[object],
    entity_coverage_failed: bool,
) -> bool:
    if not as_bool(config_value(config, "enable_reranker", False), False):
        return False
    mode = normalized_mode(config, "reranker_mode", "adaptive")
    if mode == "never":
        return False
    if not candidates:
        return False
    if mode == "always":
        return True
    if entity_coverage_failed or is_coverage_fallback(plan) or plan_requires_deep_read(plan):
        return True
    return not has_strong_local_evidence(plan, candidates)


def should_run_evidence_judge(
    config: object,
    plan: object,
    candidates: Sequence[object],
    entity_coverage_failed: bool,
) -> bool:
    if not candidates:
        return False
    # A semantic-planner outage is precisely the case in which a source-aware
    # judge prevents arbitrary high-scoring mentions from being treated as an
    # answer. This transport fallback is independent of the question wording.
    if is_coverage_fallback(plan):
        return True
    semantics = getattr(plan, "semantics", None)
    operation = str(getattr(semantics, "operation", "lookup") or "lookup")
    # An enumeration must be judged at the atomic-evidence level. Otherwise a
    # generic similarity score can turn any embarrassing, nearby or co-mentioned
    # incident into a listed item.
    if operation == "enumerate":
        return True
    if not as_bool(config_value(config, "enable_evidence_judge", False), False):
        return False
    mode = normalized_mode(config, "evidence_judge_mode", "adaptive")
    if mode == "never":
        return False
    if mode == "always":
        return True
    answer_mode = str(getattr(semantics, "answer_mode", "direct") or "direct")
    return bool(
        entity_coverage_failed
        or is_coverage_fallback(plan)
        or plan_requires_deep_read(plan)
        or answer_mode == "inferred"
    )


def local_channels_agree(channel_results: dict[str, Sequence[object]], plan: object) -> bool:
    """Allow Dense skipping only for a very high-confidence simple lookup.

    Coverage fallback never skips Dense retrieval because its primary purpose is
    to escape the limitations of exact and lexical matching.
    """

    if is_coverage_fallback(plan) or plan_requires_deep_read(plan):
        return False
    exact = list(channel_results.get("exact_entity", []) or [])
    lexical = list(channel_results.get("lexical", []) or [])
    structured = list(channel_results.get("structured", []) or [])
    if not exact:
        return False
    exact_ids = {str(getattr(item, "chunk_id", "") or "") for item in exact[:8]}
    corroborating_ids = {
        str(getattr(item, "chunk_id", "") or "")
        for item in [*lexical[:12], *structured[:12]]
    }
    if exact_ids.intersection(corroborating_ids):
        return True
    top = exact[0]
    metadata = getattr(top, "metadata", {}) or {}
    kind = str(metadata.get("chunk_kind") or getattr(top, "chunk_kind", "") or "")
    return kind in {"entity_card", "document_card", "metadata"} and float(
        getattr(plan, "entity_linking_confidence", 0.0) or 0.0
    ) >= 0.96
