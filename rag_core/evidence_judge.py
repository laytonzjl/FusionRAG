from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence

from config import RagConfig
from rag_core.models import AtomicClaim, EvidenceResult, QueryPlan, SearchResult
from rag_core.multilingual import normalize_entity_name


logger = logging.getLogger(__name__)


class EvidenceJudge:
    """Validate source snippets before they are exposed to answer generation.

    The judge is intentionally source-first.  It may decide semantic
    qualification, but every retained atomic item must point back to a visible
    source excerpt and source-local role mentions.
    """

    def __init__(self, rag_config: RagConfig, client: object) -> None:
        self.config = rag_config.normalized()
        self.client = client

    def judge(
        self,
        query: str,
        results: Sequence[SearchResult],
        query_language: str,
        plan: Optional[QueryPlan] = None,
    ) -> List[SearchResult]:
        semantics = getattr(plan, "semantics", None)
        operation = str(getattr(semantics, "operation", "lookup") or "lookup")
        enumeration_mode = operation == "enumerate"
        coverage_mode = str(getattr(plan, "planner_source", "") or "").casefold() == "coverage_fallback"
        force_verification = enumeration_mode or coverage_mode
        if (not self.config.enable_evidence_judge and not force_verification) or not results:
            return list(results)

        if enumeration_mode:
            max_candidates = max(
                8,
                min(int(getattr(self.config, "enumeration_evidence_judge_max_candidates", 18) or 18), 24),
            )
            # Navigation windows may contain several pages.  The former 850
            # character head/tail truncation regularly removed the interaction
            # that made a candidate relevant.  This remains bounded but samples
            # the whole source window.
            configured_chars = int(getattr(self.config, "enumeration_evidence_chars", 850) or 850)
            evidence_chars = max(1200, min(max(configured_chars, 1200), 2200))
            max_tokens = max(
                1000,
                min(int(getattr(self.config, "enumeration_evidence_judge_max_tokens", 1600) or 1600), 2600),
            )
        else:
            max_candidates = max(3, min(int(getattr(self.config, "evidence_judge_max_candidates", 10) or 10), 12))
            evidence_chars = max(420, min(int(getattr(self.config, "evidence_judge_evidence_chars", 1100) or 1100), 1800))
            max_tokens = max(300, min(int(getattr(self.config, "evidence_judge_max_tokens", 700) or 700), 1200))

        candidates = list(results[: min(len(results), max_candidates)])
        anchor_terms = _plan_anchor_terms(plan)
        payload = [
            {
                "chunk_id": result.chunk_id,
                "evidence_language": (result.metadata or {}).get("chunk_language", "unknown"),
                "text_quality_score": (result.metadata or {}).get("text_quality_score", 0.75),
                "source_position": (result.metadata or {}).get("document_position_ratio"),
                "section_path": (result.metadata or {}).get("section_path", ""),
                # The content is a source-preserving evidence bundle selected
                # from the full source window, not a synthetic summary.
                "content": _focused_source_bundle(
                    result.content,
                    max_chars=evidence_chars,
                    anchor_terms=[*anchor_terms, *(result.matched_terms or [])],
                ),
            }
            for result in candidates
        ]

        contract = getattr(semantics, "evidence_contract", None)
        plan_payload = {
            "intent": getattr(plan, "intent", "fact_lookup"),
            "entities": [
                {
                    "mention": entity.surface,
                    "canonical": entity.canonical,
                    "aliases": list(getattr(entity, "aliases", []) or []),
                    "link_confidence": entity.link_confidence,
                }
                for entity in (getattr(plan, "entities", []) or [])
            ],
            "requested_property": getattr(semantics, "requested_property", ""),
            "operation": operation,
            "answer_mode": getattr(semantics, "answer_mode", "direct"),
            "constraints": getattr(semantics, "constraints", []),
            "evidence_contract": {
                "answer_unit": getattr(contract, "answer_unit", "fact"),
                "include_when": getattr(contract, "include_when", ""),
                "exclude_when": getattr(contract, "exclude_when", ""),
                "required_roles": getattr(contract, "required_roles", []),
            },
            "read_strategy": {
                "scope": getattr(semantics, "scope", "entity_or_document"),
                "regions": getattr(semantics, "regions", ["all"]),
                "position_bias": getattr(semantics, "position_bias", "none"),
                "need_timeline": getattr(semantics, "need_timeline", False),
            },
            "required_evidence": getattr(plan, "required_evidence", []),
        }

        prompt = (
            "You are an evidence validator in a document-grounded RAG system. "
            "Do not answer the user. Do not use external knowledge. Do not invent "
            "events, names, motivations, causes, responsibility, or omissions. "
            "Evaluate each candidate only against the supplied query plan and source excerpts. "
            "The candidate content may contain several source-preserving spans from one source window. "
            "For operation=enumerate, extract atomic source facts and set qualifies=true only when the "
            "fact satisfies the evidence contract. Do not replace the requested category with a broader "
            "or narrower category. "
            "A nearby event, a co-mention, or an unsupported consequence is not enough. "
            "source_excerpt must be copied exactly or nearly exactly from a supplied source span; otherwise "
            "leave it empty. "
            "For answer_unit=relation: roles must include the requested role keys. Use the wording that "
            "actually appears in the source for roles.target_entity and roles.object. A source window may "
            "identify the target through local discourse or a short form, so the two endpoints do not need "
            "to occur in the same quoted sentence. Set target_matches_query=true only when the source-local "
            "context supports that roles.target_entity refers to the target entity in the user question. "
            "Set it false when that link is not supported. "
            "A relation may be supported by an explicit relationship statement or by direct source-local "
            "interaction when that interaction satisfies the evidence contract; do not require the exact "
            "predicate word from the user, and do not infer a relation from mere co-mention. "
            "Treat the planner's include_when/exclude_when as a semantic qualification hint, not as a "
            "literal keyword requirement. For broad relation questions, direct address, conversation, "
            "exchange, joint action, or a source-defined role can support the endpoint relation when the "
            "source makes both endpoints and their interaction visible. "
            "A statement may be a faithful paraphrase but must not add information. "
            "For non-enumerative plans, atomic_claims may be empty and supported_claims may contain directly "
            "supported facts. Return only a JSON array. Each item must contain: chunk_id, relevance(0-1), "
            "answerability(0-1), entity_match(boolean), evidence_language, query_language, "
            "evidence_type(direct/inference_premise/indirect/irrelevant/conflict/ocr_noise/translation_only), "
            "supported_claims(list), atomic_claims(list of {statement,qualifies,classification,source_excerpt,"
            "roles,target_matches_query,reason,confidence}), reject_reason."
        )

        try:
            raw = self.client.complete_chat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": query,
                                "query_language": query_language,
                                "query_plan": plan_payload,
                                "candidates": payload,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )
            parsed = json.loads(_extract_json_array(raw))
            evidence_by_id: Dict[str, EvidenceResult] = {}
            source_by_id = {str(candidate.get("chunk_id") or ""): str(candidate.get("content") or "") for candidate in payload}
            for item in parsed if isinstance(parsed, list) else []:
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id") or "")
                atomic_claims = _validated_atomic_claims(
                    raw=item.get("atomic_claims"),
                    source_content=source_by_id.get(chunk_id, ""),
                    contract=contract,
                    plan=plan,
                    enumeration_mode=enumeration_mode,
                )
                supported = [_bounded_text(value, 320) for value in _as_list(item.get("supported_claims"))[:8]]
                supported = [value for value in supported if value]
                if enumeration_mode:
                    supported = list(dict.fromkeys(claim.statement for claim in atomic_claims if claim.qualifies and claim.statement))[:10]
                evidence = EvidenceResult(
                    chunk_id=chunk_id,
                    relevance=_bounded_score(item.get("relevance")),
                    answerability=_bounded_score(item.get("answerability")),
                    entity_match=_as_bool(item.get("entity_match")),
                    evidence_language=str(item.get("evidence_language") or "unknown"),
                    query_language=str(item.get("query_language") or query_language),
                    evidence_type=str(item.get("evidence_type") or "indirect"),
                    supported_claims=supported,
                    atomic_claims=atomic_claims,
                    reject_reason=_bounded_text(item.get("reject_reason"), 360),
                )
                if evidence.chunk_id:
                    evidence_by_id[evidence.chunk_id] = evidence

            judged: List[SearchResult] = []
            rejected_reasons: Dict[str, int] = {}
            for result in candidates:
                result.evidence = evidence_by_id.get(result.chunk_id)
                if result.evidence:
                    qualifying = sum(1 for claim in result.evidence.atomic_claims if claim.qualifies)
                    rejected = [claim for claim in result.evidence.atomic_claims if not claim.qualifies]
                    for claim in rejected:
                        for reason in str(claim.reason or "").split(";"):
                            clean = reason.strip()
                            if clean:
                                rejected_reasons[clean] = rejected_reasons.get(clean, 0) + 1
                    result.diagnostics["atomic_claims_total"] = len(result.evidence.atomic_claims)
                    result.diagnostics["atomic_claims_qualified"] = qualifying
                    result.final_score = result.final_score + result.evidence.answerability * 0.02
                    if enumeration_mode and qualifying:
                        result.final_score += min(qualifying, 3) * 0.015
                judged.append(result)

            if enumeration_mode:
                qualified_total = sum(int(item.diagnostics.get("atomic_claims_qualified", 0) or 0) for item in judged)
                if qualified_total == 0 and rejected_reasons:
                    logger.info(
                        "Enumeration evidence yielded zero qualified claims; rejection_reasons=%s",
                        dict(sorted(rejected_reasons.items(), key=lambda pair: (-pair[1], pair[0]))[:8]),
                    )
                judged.sort(
                    key=lambda item: (
                        int(item.diagnostics.get("atomic_claims_qualified", 0) or 0),
                        item.evidence.answerability if item.evidence else 0.0,
                        item.evidence.relevance if item.evidence else 0.0,
                        item.final_score,
                    ),
                    reverse=True,
                )
            else:
                judged.sort(
                    key=lambda item: (
                        item.evidence.answerability if item.evidence else 0.0,
                        item.evidence.relevance if item.evidence else 0.0,
                        item.final_score,
                    ),
                    reverse=True,
                )
            return judged + list(results[len(candidates) :])
        except Exception as exc:
            logger.warning("LLM Evidence Judge 失败，保留 rerank/RRF 结果: %s", exc)
            return list(results)


def _validated_atomic_claims(
    raw: object,
    source_content: str,
    contract: object,
    plan: Optional[QueryPlan],
    enumeration_mode: bool,
) -> List[AtomicClaim]:
    """Apply provenance and schema checks without lexical full-name lock-in.

    A relation has two source-grounded endpoints.  The mapping from a local
    source form (for example, a short name) to the query entity is a semantic,
    source-local decision that the judge must make explicitly via
    ``target_matches_query``.  Rejecting every local form that is not a string
    subset of the user wording creates systematic false negatives in literature,
    meeting notes, and multilingual documents.
    """

    claims = _atomic_claims_from_payload(raw)
    if not enumeration_mode:
        return claims

    required_roles = {str(role).strip() for role in (getattr(contract, "required_roles", []) or []) if str(role).strip()}
    answer_unit = str(getattr(contract, "answer_unit", "fact") or "fact")
    target_aliases = _plan_target_aliases(plan)

    for claim in claims:
        if not claim.qualifies:
            continue
        reasons: List[str] = []
        if not is_grounded_excerpt(claim.source_excerpt, source_content):
            reasons.append("source_excerpt_not_grounded")

        missing = [role for role in required_roles if not str(claim.roles.get(role, "") or "").strip()]
        if missing:
            reasons.append("missing_roles=" + ",".join(sorted(missing)))

        if answer_unit == "relation":
            target = str(claim.roles.get("target_entity", "") or "").strip()
            other = str(claim.roles.get("object", "") or "").strip()
            if not target or not other:
                reasons.append("relation_endpoints_missing")
            elif _same_entity(target, other):
                reasons.append("relation_endpoints_identical")
            target_visible = bool(target and _text_mentions(source_content, target))
            if not target_visible and claim.target_matches_query:
                target_visible = _text_mentions_any(source_content, target_aliases)
            if target and not target_visible:
                reasons.append("relation_target_not_in_source")
            if other and not _text_mentions(source_content, other):
                reasons.append("relation_object_not_in_source")
            lexical_target_match = bool(target_aliases and target and any(_entity_matches(target, alias) for alias in target_aliases))
            if target_aliases and not (claim.target_matches_query or lexical_target_match):
                reasons.append("target_not_resolved_to_query")

        if reasons:
            claim.qualifies = False
            claim.classification = "rejected"
            prior = str(claim.reason or "").strip()
            claim.reason = "; ".join([*reasons, prior]).strip("; ")
            claim.confidence = 0.0
    return claims


def _plan_target_aliases(plan: Optional[QueryPlan]) -> List[str]:
    aliases: List[str] = []
    for entity in getattr(plan, "entities", []) or []:
        aliases.extend(
            [
                str(getattr(entity, "surface", "") or ""),
                str(getattr(entity, "canonical", "") or ""),
                str(getattr(entity, "linked_alias", "") or ""),
                *[str(value or "") for value in (getattr(entity, "aliases", []) or [])],
            ]
        )
    return list(dict.fromkeys(value for value in aliases if normalize_entity_name(value)))


def _plan_anchor_terms(plan: Optional[QueryPlan]) -> List[str]:
    values: List[str] = []
    for entity in getattr(plan, "entities", []) or []:
        values.extend([
            str(getattr(entity, "surface", "") or ""),
            str(getattr(entity, "canonical", "") or ""),
            str(getattr(entity, "linked_alias", "") or ""),
            *[str(value or "") for value in (getattr(entity, "aliases", []) or [])],
        ])
    return list(dict.fromkeys(value for value in values if len(value.strip()) >= 2))[:12]


def _compact(value: str) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def is_grounded_excerpt(excerpt: str, source: str) -> bool:
    """Validate an exact or near-exact quote from a source-preserving span."""

    left = _compact(excerpt)
    right = _compact(source)
    if len(left) < 8 or not right:
        return False
    if left in right:
        return True
    # OCR and PDF extraction can alter a small number of characters.  Permit a
    # long contiguous near-match, not a bag-of-words paraphrase.
    match = SequenceMatcher(a=left, b=right, autojunk=False).find_longest_match(0, len(left), 0, len(right)).size
    return match >= max(10, int(round(len(left) * 0.84)))


def _text_mentions(source: str, value: str) -> bool:
    needle = _compact(value)
    haystack = _compact(source)
    return bool(needle and haystack and needle in haystack)


def _text_mentions_any(source: str, values: Sequence[str]) -> bool:
    return any(_text_mentions(source, value) for value in values if str(value or "").strip())


def _entity_matches(left: str, right: str) -> bool:
    a = normalize_entity_name(left)
    b = normalize_entity_name(right)
    return bool(a and b and (a == b or a in b or b in a))


def _same_entity(left: str, right: str) -> bool:
    return _entity_matches(left, right)


def _focused_source_bundle(text: str, max_chars: int, anchor_terms: Iterable[str]) -> str:
    """Return source-preserving spans across a long window.

    It combines explicit anchor neighborhoods with even source-position
    coverage.  No relation labels, language-specific lists, or question
    templates are used.
    """

    value = (text or "").strip()
    budget = max(240, int(max_chars or 0))
    if len(value) <= budget:
        return value

    per_span = max(150, min(420, budget // 4))
    ranges: List[tuple[int, int]] = []
    lower = value.casefold()
    for term in anchor_terms:
        clean = str(term or "").strip()
        if len(clean) < 2:
            continue
        start = lower.find(clean.casefold())
        if start < 0:
            continue
        center = start + len(clean) // 2
        left = max(0, center - per_span // 2)
        right = min(len(value), left + per_span)
        left = max(0, right - per_span)
        ranges.append((left, right))
        if len(ranges) >= 3:
            break

    # Ensure evidence in the middle of a long navigation window is not removed
    # merely because it is neither the first nor final paragraph.
    uniform_count = 4
    for index in range(uniform_count):
        if uniform_count == 1:
            left = 0
        else:
            left = int((len(value) - per_span) * index / (uniform_count - 1))
        right = min(len(value), left + per_span)
        left = max(0, right - per_span)
        ranges.append((left, right))

    ranges.sort()
    merged: List[tuple[int, int]] = []
    for left, right in ranges:
        if not merged or left > merged[-1][1] + 24:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))

    snippets: List[str] = []
    used = 0
    for left, right in merged:
        snippet = value[left:right].strip()
        if not snippet:
            continue
        addition = len(snippet) + (7 if snippets else 0)
        if snippets and used + addition > budget:
            remaining = budget - used - 7
            if remaining >= 100:
                snippets.append(snippet[:remaining].rstrip())
            break
        if not snippets and len(snippet) > budget:
            snippets.append(snippet[:budget].rstrip())
            break
        snippets.append(snippet)
        used += addition
        if used >= budget:
            break
    return "\n[…]\n".join(snippets).strip() or value[:budget].rstrip()


def _atomic_claims_from_payload(raw: object) -> List[AtomicClaim]:
    claims: List[AtomicClaim] = []
    seen: set[tuple[str, str]] = set()
    for item in _as_list(raw)[:8]:
        if not isinstance(item, dict):
            continue
        statement = _bounded_text(item.get("statement"), 420)
        source_excerpt = _bounded_text(item.get("source_excerpt"), 360)
        key = (statement.casefold(), source_excerpt.casefold())
        if not statement or key in seen:
            continue
        seen.add(key)
        roles_raw = item.get("roles")
        roles: Dict[str, str] = {}
        if isinstance(roles_raw, dict):
            for role, value in list(roles_raw.items())[:8]:
                clean_role = _bounded_text(role, 48)
                clean_value = _bounded_text(value, 160)
                if clean_role and clean_value:
                    roles[clean_role] = clean_value
        claims.append(
            AtomicClaim(
                statement=statement,
                qualifies=_as_bool(item.get("qualifies")),
                classification=_bounded_text(item.get("classification"), 48) or "irrelevant",
                source_excerpt=source_excerpt,
                roles=roles,
                target_matches_query=_as_bool(item.get("target_matches_query")),
                reason=_bounded_text(item.get("reason"), 300),
                confidence=_bounded_score(item.get("confidence")),
            )
        )
    return claims


def _as_list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _bounded_text(value: object, limit: int) -> str:
    return str(value or "").strip()[: max(0, int(limit))]


def _extract_json_array(text: str) -> str:
    value = (text or "").strip()
    try:
        parsed = json.loads(value)
        return value if isinstance(parsed, list) else "[]"
    except Exception:
        pass
    start = value.find("[")
    if start < 0:
        return "[]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                candidate = value[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                    return candidate if isinstance(parsed, list) else "[]"
                except Exception:
                    return "[]"
    return "[]"


def _bounded_score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
