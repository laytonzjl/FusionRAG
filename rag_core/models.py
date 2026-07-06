from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParsedChunk:
    """一个可入库的文本块及其元数据。"""

    content: str
    metadata: Dict[str, object]


@dataclass
class LanguageProfile:
    """语言识别结果，既保留标签，也保留脚本分布与置信度。"""

    language: str = "unknown"
    confidence: float = 0.0
    script_distribution: Dict[str, float] = field(default_factory=dict)
    is_mixed: bool = False


@dataclass
class QueryEntity:
    """用户问题中的实体提及，以及其在本地实体索引中的链接结果。"""

    surface: str
    canonical: str
    aliases: List[str] = field(default_factory=list)
    alias_sources: List[str] = field(default_factory=list)
    entity_type: str = "unknown"
    linked_alias: str = ""
    link_confidence: float = 0.0


@dataclass
class QueryVariant:
    text: str
    language: str = "unknown"
    origin: str = "original"


@dataclass
class EvidenceContract:
    """Question-specific but source-neutral test for evidence qualification.

    The planner may describe what a qualifying item must contain, but it may
    never introduce a factual answer.  The contract is consumed by the evidence
    judge for every enumerative question, regardless of document genre or
    surface wording.
    """

    answer_unit: str = "fact"
    include_when: str = ""
    exclude_when: str = ""
    required_roles: List[str] = field(default_factory=list)
    retrieval_views: List[str] = field(default_factory=list)


@dataclass
class QuerySemantics:
    """Validated semantic and reading plan produced for one user query.

    The planner never supplies facts or answers.  It can only request a generic
    reading strategy over source-backed document structure that was created
    during offline indexing.
    """

    requested_property: str = ""
    operation: str = "lookup"
    answer_mode: str = "direct"
    constraints: List[str] = field(default_factory=list)
    answer_shape: str = ""
    evidence_contract: EvidenceContract = field(default_factory=EvidenceContract)
    # ReadPlan fields. They are intentionally generic rather than a growing
    # list of task-specific question templates.
    scope: str = "entity_or_document"
    regions: List[str] = field(default_factory=lambda: ["all"])
    position_bias: str = "none"
    need_timeline: bool = False
    need_entity_neighborhood: bool = False
    allow_partial: bool = False
    planner_confidence: float = 0.0


@dataclass
class QueryPlan:
    intent: str
    query_language: str
    language_confidence: float
    script_distribution: Dict[str, float]
    entities: List[QueryEntity] = field(default_factory=list)
    retrieval_queries: List[QueryVariant] = field(default_factory=list)
    preferred_chunk_kinds: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    planner_source: str = "rules"
    entity_coverage_failed: bool = False
    warnings: List[str] = field(default_factory=list)
    relation_type: str = ""
    answer_type: str = ""
    semantics: QuerySemantics = field(default_factory=QuerySemantics)
    entity_linking_confidence: float = 0.0


@dataclass
class RRFContribution:
    channel: str
    rank: int
    weight: float
    contribution: float
    query_text: str = ""
    query_language: str = "unknown"
    query_origin: str = "original"


@dataclass
class AtomicClaim:
    """One source-grounded candidate answer unit emitted by the evidence judge."""

    statement: str = ""
    qualifies: bool = False
    classification: str = "irrelevant"
    source_excerpt: str = ""
    roles: Dict[str, str] = field(default_factory=dict)
    # For relation-shaped enumerations, this is the judge's source-local
    # coreference decision: the target mention used in `roles` refers to the
    # entity asked about by the user. It is not an answer fact by itself; it
    # only prevents a literal full-name mismatch from discarding a grounded
    # source claim when the document uses a short form or a local reference.
    target_matches_query: bool = False
    reason: str = ""
    confidence: float = 0.0


@dataclass
class EvidenceResult:
    chunk_id: str
    relevance: float = 0.0
    answerability: float = 0.0
    entity_match: bool = False
    evidence_language: str = "unknown"
    query_language: str = "unknown"
    evidence_type: str = "indirect"
    supported_claims: List[str] = field(default_factory=list)
    atomic_claims: List[AtomicClaim] = field(default_factory=list)
    reject_reason: str = ""


@dataclass
class RetrievalDiagnostics:
    query_plan: Optional[QueryPlan] = None
    candidates_by_channel: Dict[str, List[str]] = field(default_factory=dict)
    final_chunk_ids: List[str] = field(default_factory=list)
    reranker_enabled: bool = False
    reranker_status: str = "disabled"
    evidence_judge_enabled: bool = False
    entity_coverage_failed: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """检索返回结果，包含内容、来源与融合后的评分。"""

    content: str
    metadata: Dict[str, object]
    vector_score: float
    keyword_score: float
    final_score: float
    chunk_id: str = ""
    parent_chunk_id: str = ""
    chunk_kind: str = "child"
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    evidence: Optional[EvidenceResult] = None
    contributions: List[RRFContribution] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)
