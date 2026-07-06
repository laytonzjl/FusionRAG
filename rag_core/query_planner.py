from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

from config import RagConfig
from rag_core.adaptive_policy import should_use_llm_planner
from rag_core.models import EvidenceContract, QueryEntity, QueryPlan, QuerySemantics, QueryVariant
from rag_core.multilingual import detect_language


logger = logging.getLogger(__name__)

_ALLOWED_INTENTS = {
    "entity_definition",
    "entity_relation",
    "author_or_source",
    "document_summary",
    "section_summary",
    "fact_lookup",
    "procedure",
    "comparison",
    "numeric_lookup",
    "location_lookup",
    "table_lookup",
    "definition_lookup",
    "code_question",
    "policy_clause",
    "cross_document_conflict",
}
_ALLOWED_OPERATIONS = {"lookup", "enumerate", "compare", "summarize", "explain", "estimate", "trace", "verify"}
_ALLOWED_ANSWER_MODES = {"direct", "inferred"}
_ALLOWED_READ_SCOPES = {"collection", "document", "section", "entity_neighborhood", "entity_or_document"}
_ALLOWED_REGIONS = {"all", "front", "middle", "terminal"}
_ALLOWED_POSITION_BIASES = {"none", "chronological", "front", "terminal"}
_ALLOWED_ANSWER_UNITS = {
    "fact",
    "event",
    "action",
    "object",
    "relation",
    "attribute",
    "value",
    "procedure_step",
    "claim",
}
_ALLOWED_EVIDENCE_ROLES = {
    "target_entity",
    "actor",
    "action",
    "object",
    "outcome",
    "recipient",
    "time",
    "location",
    "responsibility",
    "comparison_side",
}

# A planner outage must not add the same network timeout to every user query.
# This is a transport-health circuit, not a semantic or wording decision.
_PLANNER_RETRY_AFTER_SECONDS = 45.0
_PLANNER_CIRCUIT_OPEN_UNTIL: Dict[str, float] = {}


def build_query_plan(
    query: str,
    rag_config: RagConfig,
    client: Optional[object] = None,
    alias_terms: Optional[List[str]] = None,
) -> QueryPlan:
    """Build a semantic read plan, with a structure-first fallback.

    The fallback deliberately does not classify intent from wording. It requests
    a bounded evidence exploration over the document topology, allowing the
    retrieval executor to select a document and structural span from source
    evidence rather than a phrase-specific rule.
    """

    aliases = alias_terms or []
    planner_key = _planner_key(client)
    planner_allowed = client is not None and should_use_llm_planner(query, rag_config)
    if planner_allowed and not _planner_circuit_is_open(planner_key):
        plan, failure = _try_llm_structured_plan(
            query=query,
            rag_config=rag_config,
            client=client,
            alias_terms=aliases,
        )
        if plan is not None:
            _reset_planner_circuit(planner_key)
            plan.warnings.append("planner_execution=llm")
            return plan
        _open_planner_circuit(planner_key)
        fallback = build_coverage_fallback_plan(query=query)
        fallback.warnings.extend(["planner_execution=coverage_fallback", f"planner_failure={failure or 'unknown'}"])
        return fallback

    fallback = build_coverage_fallback_plan(query=query)
    fallback.warnings.append("planner_execution=coverage_fallback")
    if planner_allowed and _planner_circuit_is_open(planner_key):
        fallback.warnings.append("planner_circuit_open")
    return fallback


def _planner_key(client: object | None) -> str:
    if client is None:
        return "none"
    api_config = getattr(client, "api_config", None)
    provider = str(getattr(api_config, "chat_provider", "") or getattr(client, "chat_provider", "") or "default")
    base = str(getattr(api_config, "chat_api_base", "") or "")
    return f"{provider}:{base}"


def _planner_circuit_is_open(key: str) -> bool:
    return float(_PLANNER_CIRCUIT_OPEN_UNTIL.get(key, 0.0) or 0.0) > time.monotonic()


def _open_planner_circuit(key: str) -> None:
    _PLANNER_CIRCUIT_OPEN_UNTIL[key] = time.monotonic() + _PLANNER_RETRY_AFTER_SECONDS


def _reset_planner_circuit(key: str) -> None:
    _PLANNER_CIRCUIT_OPEN_UNTIL.pop(key, None)


def build_coverage_fallback_plan(query: str) -> QueryPlan:
    """Return a question-agnostic plan for bounded structural exploration."""

    profile = detect_language(query)
    semantics = QuerySemantics(
        requested_property="",
        operation="verify",
        answer_mode="direct",
        constraints=[],
        answer_shape="",
        evidence_contract=EvidenceContract(answer_unit="fact"),
        scope="collection",
        regions=["all"],
        position_bias="none",
        need_timeline=False,
        need_entity_neighborhood=False,
        allow_partial=True,
        planner_confidence=0.0,
    )
    return QueryPlan(
        intent="fact_lookup",
        query_language=profile.language,
        language_confidence=profile.confidence,
        script_distribution=profile.script_distribution,
        entities=[],
        retrieval_queries=[QueryVariant(text=(query or "").strip(), language=profile.language, origin="original")],
        preferred_chunk_kinds=preferred_chunk_kinds("fact_lookup", semantics),
        required_evidence=required_evidence("fact_lookup", semantics),
        planner_source="coverage_fallback",
        warnings=[] if profile.language != "unknown" else ["query_language_unknown"],
        relation_type="",
        answer_type="",
        semantics=semantics,
    )


def build_rule_query_plan(
    query: str,
    rag_config: RagConfig,
    alias_terms: Optional[List[str]] = None,
    intent_override: Optional[str] = None,
    planner_source: str = "coverage_fallback",
    extra_warnings: Optional[List[str]] = None,
) -> QueryPlan:
    """Compatibility wrapper retained for callers of the older public helper."""

    plan = build_coverage_fallback_plan(query)
    plan.planner_source = planner_source or "coverage_fallback"
    plan.warnings.extend(list(extra_warnings or []))
    return plan


def _try_llm_structured_plan(
    query: str,
    rag_config: RagConfig,
    client: object,
    alias_terms: List[str],
) -> tuple[Optional[QueryPlan], str]:
    profile = detect_language(query)
    max_variants = max(1, int(getattr(rag_config, "query_rewrite_count", 1) or 1))
    prompt = (
        "You are a document-grounded retrieval planner. Return exactly one JSON object and no prose. "
        "Do not answer the user question and do not introduce facts, names, aliases, dates, or assumptions. "
        "Describe the evidence needed and how a source-ordered document should be read. "
        "`entity_mentions` may contain only literal entity strings present in the user question. "
        "`retrieval_queries` may paraphrase the question but must preserve all constraints and may not add proper nouns. "
        "Choose fields from these closed sets: "
        "intent=[entity_definition,entity_relation,author_or_source,document_summary,section_summary,fact_lookup,procedure,comparison,numeric_lookup,location_lookup,table_lookup,definition_lookup,code_question,policy_clause,cross_document_conflict]; "
        "operation=[lookup,enumerate,compare,summarize,explain,estimate,trace,verify]; "
        "answer_mode=[direct,inferred]; "
        "scope=[collection,document,section,entity_neighborhood,entity_or_document]; "
        "regions is a nonempty array drawn from [all,front,middle,terminal]; "
        "position_bias=[none,chronological,front,terminal]. "
        "Use source topology rather than wording: choose a structural region only when the requested evidence logically depends on that region. "
        "For operation=enumerate, return an evidence_contract that defines the answer unit and the source-grounded test for whether a candidate may be listed. "
        "The contract must be abstract and question-derived: it must not mention document facts, example incidents, or assumed answers. "
        "When the requested category logically depends on agency, responsibility, causality, or a relation, "
        "include_when must require the corresponding source-grounded role and exclude events merely experienced by the target. "
        "answer_unit=[fact,event,action,object,relation,attribute,value,procedure_step,claim]. "
        "required_roles is an array drawn from [target_entity,actor,action,object,outcome,recipient,time,location,responsibility,comparison_side]. "
        "retrieval_views is an array of at most three semantic paraphrases that search different evidence roles while preserving the original question and requested category; it may not add names, facts, or replace the category with a narrower or broader one. "
        "JSON schema: {intent,operation,answer_mode,entity_mentions:[{text,entity_type}],retrieval_queries:[{text,language,origin}],requested_property,constraints:[string],answer_shape,evidence_contract:{answer_unit,include_when,exclude_when,required_roles,retrieval_views},read_strategy:{scope,regions,position_bias,need_timeline,need_entity_neighborhood,allow_partial},planner_confidence}."
    )
    try:
        raw = client.complete_chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=max(320, min(int(getattr(rag_config, "planner_max_tokens", 420) or 420), 900)),
        )
    except Exception as exc:
        logger.warning("Structured planner request failed: %s", exc)
        return None, _failure_label(exc)

    data = _parse_json_object(raw)
    if data is None:
        logger.warning("Structured planner returned no recoverable JSON object")
        return None, "invalid_json"

    try:
        plan = _plan_from_payload(data=data, query=query, profile=profile, max_variants=max_variants)
        return plan, ""
    except Exception as exc:
        logger.warning("Structured planner payload validation failed: %s", exc)
        return None, _failure_label(exc)


def _plan_from_payload(
    data: Dict[str, object],
    query: str,
    profile: object,
    max_variants: int,
) -> QueryPlan:
    intent = _choice(data.get("intent"), _ALLOWED_INTENTS, "fact_lookup")
    operation = _choice(data.get("operation"), _ALLOWED_OPERATIONS, "lookup")
    answer_mode = _choice(data.get("answer_mode"), _ALLOWED_ANSWER_MODES, "direct")
    entities = _llm_entities(data.get("entity_mentions"), query)

    variants = [QueryVariant(text=query, language=getattr(profile, "language", "unknown"), origin="original")]
    for item in _as_list(data.get("retrieval_queries"))[:max_variants]:
        if isinstance(item, dict):
            text = _bounded_text(item.get("text"), 240)
            language = _bounded_text(item.get("language"), 24) or getattr(profile, "language", "unknown")
            origin = _bounded_text(item.get("origin"), 32) or "llm_structured"
        else:
            text = _bounded_text(item, 240)
            language = getattr(profile, "language", "unknown")
            origin = "llm_structured"
        if text and text.casefold() != query.casefold():
            variants.append(QueryVariant(text=text, language=language, origin=origin))

    read_strategy = _validated_read_strategy(data.get("read_strategy"))
    evidence_contract = _validated_evidence_contract(
        data.get("evidence_contract"),
        operation=operation,
        intent=intent,
    )
    # Keep planner-generated views inside the evidence contract rather than
    # turning them into direct retrieval queries.  A view is an optional reading
    # perspective, not an authorization to narrow or broaden the user's answer
    # category (for example, replacing a general relation request with one
    # specific relation subtype).  Enumerative recall is instead widened by
    # source-window coverage around the linked target entity.
    semantics = QuerySemantics(
        requested_property=_bounded_text(data.get("requested_property"), 100),
        operation=operation,
        answer_mode=answer_mode,
        constraints=[item for item in (_bounded_text(value, 120) for value in _as_list(data.get("constraints"))[:6]) if item],
        answer_shape=_bounded_text(data.get("answer_shape"), 80),
        evidence_contract=evidence_contract,
        scope=read_strategy["scope"],
        regions=read_strategy["regions"],
        position_bias=read_strategy["position_bias"],
        need_timeline=read_strategy["need_timeline"],
        need_entity_neighborhood=read_strategy["need_entity_neighborhood"],
        allow_partial=read_strategy["allow_partial"],
        planner_confidence=_bounded_score(data.get("planner_confidence")),
    )
    return QueryPlan(
        intent=intent,
        query_language=getattr(profile, "language", "unknown"),
        language_confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
        script_distribution=dict(getattr(profile, "script_distribution", {}) or {}),
        entities=entities,
        retrieval_queries=_dedupe_variants(variants, max_count=max(2, 1 + max_variants)),
        preferred_chunk_kinds=preferred_chunk_kinds(intent, semantics),
        required_evidence=required_evidence(intent, semantics),
        planner_source="llm_structured",
        warnings=[] if getattr(profile, "language", "unknown") != "unknown" else ["query_language_unknown"],
        relation_type="",
        answer_type=semantics.answer_shape,
        semantics=semantics,
    )


def _validated_read_strategy(raw: object) -> Dict[str, object]:
    data = raw if isinstance(raw, dict) else {}
    scope = _choice(data.get("scope"), _ALLOWED_READ_SCOPES, "entity_or_document")
    regions = [str(value).strip().casefold() for value in _as_list(data.get("regions"))]
    regions = [value for value in regions if value in _ALLOWED_REGIONS]
    if not regions:
        regions = ["all"]
    if len(regions) > 1 and "all" in regions:
        regions = [value for value in regions if value != "all"]
    position_bias = _choice(data.get("position_bias"), _ALLOWED_POSITION_BIASES, "none")
    return {
        "scope": scope,
        "regions": list(dict.fromkeys(regions))[:3],
        "position_bias": position_bias,
        "need_timeline": bool(data.get("need_timeline", False)),
        "need_entity_neighborhood": bool(data.get("need_entity_neighborhood", False)),
        "allow_partial": bool(data.get("allow_partial", False)),
    }




def _planner_text(value: object, limit: int) -> str:
    """Read text from either a scalar or a planner-style object.

    Some OpenAI-compatible models return evidence views in the same object shape
    used by ``retrieval_queries``.  Converting that dictionary with ``str``
    produces a Python literal such as ``{'text': ...}``, which is not a useful
    embedding query.  This parser accepts both schema-compatible shapes while
    discarding any fields other than text.
    """

    if isinstance(value, dict):
        value = value.get("text", "")
    return _bounded_text(value, limit)


def _validated_evidence_contract(raw: object, operation: str, intent: str = "") -> EvidenceContract:
    """Validate a source-neutral enumeration contract from the planner.

    Relation enumeration has a schema-level minimum: a relation cannot be
    source-audited without both endpoints.  This is not a lexical rule about
    a relation type; it applies uniformly to every requested relation.
    """

    data = raw if isinstance(raw, dict) else {}
    answer_unit = _choice(data.get("answer_unit"), _ALLOWED_ANSWER_UNITS, "fact")
    required_roles = [
        _choice(value, _ALLOWED_EVIDENCE_ROLES, "")
        for value in _as_list(data.get("required_roles"))[:6]
    ]
    required_roles = [value for value in required_roles if value]
    retrieval_views = [
        _planner_text(value, 240)
        for value in _as_list(data.get("retrieval_views"))[:3]
    ]
    retrieval_views = [value for value in retrieval_views if value]

    if operation == "enumerate" and intent == "entity_relation":
        answer_unit = "relation"

    if operation == "enumerate" and answer_unit == "relation":
        required_roles = [*required_roles, "target_entity", "object"]

    if operation != "enumerate":
        # The contract is harmless for non-enumerative tasks, but only
        # enumerations may cause extra semantic retrieval views.
        retrieval_views = []

    return EvidenceContract(
        answer_unit=answer_unit,
        include_when=_planner_text(data.get("include_when"), 260),
        exclude_when=_planner_text(data.get("exclude_when"), 260),
        required_roles=list(dict.fromkeys(required_roles)),
        retrieval_views=list(dict.fromkeys(retrieval_views)),
    )

def _llm_entities(raw_entities: object, query: str) -> List[QueryEntity]:
    entities: List[QueryEntity] = []
    query_key = "".join((query or "").split()).casefold()
    seen: set[str] = set()
    for item in _as_list(raw_entities)[:4]:
        if isinstance(item, dict):
            text = _bounded_text(item.get("text"), 120)
            entity_type = _bounded_text(item.get("entity_type"), 48) or "unknown"
        else:
            text = _bounded_text(item, 120)
            entity_type = "unknown"
        key = "".join(text.split()).casefold()
        if not text or key in seen or (key and key not in query_key):
            continue
        seen.add(key)
        entities.append(
            QueryEntity(
                surface=text,
                canonical=text,
                aliases=[text],
                alias_sources=["llm_structured"],
                entity_type=entity_type,
            )
        )
    return entities


def preferred_chunk_kinds(intent: str, semantics: QuerySemantics) -> List[str]:
    kinds = ["document_card", "section_card", "entity_card", "navigation_window", "terminal_window", "child", "parent"]
    if semantics.operation == "enumerate":
        # Enumerations are evidence-coverage tasks. Source windows and section
        # topology come before generic entity mentions, independent of topic.
        return ["navigation_window", "section_card", "terminal_window", "child", "parent", "entity_card", "document_card"]
    if semantics.scope in {"document", "section", "entity_neighborhood", "collection"}:
        return ["navigation_window", "terminal_window", "section_card", "document_card", "entity_card", "child", "parent"]
    return kinds


def required_evidence(intent: str, semantics: QuerySemantics) -> List[str]:
    if semantics.operation == "enumerate":
        return ["qualified_atomic_evidence", "source_backed_answer_statement"]
    if semantics.answer_mode == "inferred":
        return ["independent_source_evidence"]
    return ["source_backed_answer_statement"]


def _parse_json_object(raw: object) -> Optional[Dict[str, object]]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
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
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                    return value if isinstance(value, dict) else None
                except Exception:
                    return None
    return None


def _as_list(value: object) -> List[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _choice(value: object, allowed: set[str], default: str) -> str:
    item = str(value or "").strip().casefold()
    return item if item in allowed else default


def _bounded_text(value: object, max_length: int) -> str:
    return str(value or "").strip()[:max_length]


def _bounded_score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dedupe_variants(variants: List[QueryVariant], max_count: int) -> List[QueryVariant]:
    output: List[QueryVariant] = []
    seen: set[str] = set()
    for variant in variants:
        text = str(variant.text or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(QueryVariant(text=text, language=variant.language or "unknown", origin=variant.origin or "unknown"))
        if len(output) >= max_count:
            break
    return output


def _failure_label(exc: Exception) -> str:
    label = type(exc).__name__.strip() or "unknown"
    return label[:80]
