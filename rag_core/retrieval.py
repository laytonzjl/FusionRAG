from __future__ import annotations

import logging
import math
import re
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Generator, List, Sequence, Tuple

from chromadb import Collection

from config import ApiConfig, RagConfig
from rag_core.evidence_judge import EvidenceJudge, is_grounded_excerpt
from rag_core.hybrid_index import HybridHit, HybridIndex
from rag_core.models import QueryPlan, QueryVariant, RetrievalDiagnostics, RRFContribution, SearchResult
from rag_core.multilingual import normalize_entity_name, query_focus_terms, surface_variants, tokenize_for_search
from rag_core.providers import MultiProviderClient
from rag_core.query_planner import build_query_plan
from rag_core.reranker import CrossEncoderReranker
from rag_core.adaptive_policy import (
    candidate_budget as adaptive_candidate_budget,
    is_coverage_fallback,
    local_channels_agree,
    plan_requires_deep_read,
    query_variant_limit,
    rerank_candidate_budget as adaptive_rerank_candidate_budget,
    should_run_evidence_judge,
    should_run_reranker,
)


logger = logging.getLogger(__name__)


@dataclass
class LexicalDocument:
    content: str
    metadata: Dict[str, object]
    term_freq: Dict[str, int]
    length: int


@dataclass
class LexicalIndex:
    documents: List[LexicalDocument]
    document_frequency: Dict[str, int]
    average_length: float
    collection_count: int


class RetrievalService:
    """负责检索、融合排序、上下文拼装与问答提示词构建。"""

    def __init__(
        self,
        collection: Collection,
        client: MultiProviderClient,
        api_config: ApiConfig,
        rag_config: RagConfig,
    ) -> None:
        self.collection = collection
        self.client = client
        self.api_config = api_config
        self.rag_config = rag_config.normalized()
        self._lexical_index: LexicalIndex | None = None
        self.hybrid_index = HybridIndex(collection_name=self.rag_config.collection_name)
        self.reranker = CrossEncoderReranker(self.rag_config)
        self.evidence_judge = EvidenceJudge(self.rag_config, self.client)
        self.last_diagnostics = RetrievalDiagnostics()
        self._last_results: List[SearchResult] = []

    def _is_deep_read_plan(self, plan: QueryPlan) -> bool:
        """Return whether a plan genuinely needs source navigation.

        ``entity_or_document`` with ``regions=['all']`` is the neutral planner
        default.  Treating that default as a navigation request made every
        simple question pay for document-window reads.
        """

        semantics = getattr(plan, "semantics", None)
        regions = {str(item).strip().lower() for item in (getattr(semantics, "regions", []) or [])}
        explicit_region = bool(regions.intersection({"front", "middle", "terminal"}))
        scope = str(getattr(semantics, "scope", "entity_or_document") or "entity_or_document")
        return bool(
            explicit_region
            or getattr(semantics, "need_timeline", False)
            or getattr(semantics, "need_entity_neighborhood", False)
            or scope in {"document", "section", "entity_neighborhood"}
        )

    def _candidate_budget(self, plan: QueryPlan, final_top_k: int, collection_count: int) -> int:
        """Use an adaptive candidate budget rather than a fixed high floor."""

        return adaptive_candidate_budget(
            self.rag_config,
            plan=plan,
            final_top_k=final_top_k,
            collection_count=collection_count,
        )

    def _rerank_candidate_budget(self, plan: QueryPlan, final_top_k: int) -> int:
        """Keep costly Cross-Encoder work proportional to question ambiguity."""

        return adaptive_rerank_candidate_budget(self.rag_config, plan=plan, final_top_k=final_top_k)

    def _select_query_variants(self, plan: QueryPlan, limit: int = 2) -> List[object]:
        """Select at most two high-value Dense / lexical query variants.

        An entity-only vector query usually retrieves every mention of that
        entity, adds substantial noise, and duplicates the exact-entity channel.
        Exact search retains entity recall; Dense retrieval uses the full user
        question plus at most one semantic rewrite.
        """

        variants = list(getattr(plan, "retrieval_queries", []) or [])
        if not variants:
            return []
        limit = max(1, int(limit or 1))
        selected: List[object] = []
        seen: set[str] = set()

        def add(variant: object) -> None:
            text = str(getattr(variant, "text", "") or "").strip()
            key = text.casefold()
            if not text or key in seen or len(selected) >= limit:
                return
            seen.add(key)
            selected.append(variant)

        for variant in variants:
            if str(getattr(variant, "origin", "")) == "original":
                add(variant)
                break
        if not selected:
            add(variants[0])

        entity_only_origins = {"llm_entity", "entity_surface_variant", "entity_alias", "relation_expansion"}
        for variant in variants:
            if str(getattr(variant, "origin", "")) in entity_only_origins:
                continue
            add(variant)
            if len(selected) >= limit:
                return selected

        # A malformed/very small plan can contain only entity variants.  Keep a
        # single fallback rather than failing Dense retrieval altogether.
        for variant in variants:
            add(variant)
            if len(selected) >= limit:
                break
        return selected

    def _entity_search_terms(self, plan: QueryPlan | None, limit: int = 24) -> List[str]:
        """Return source-visible entity aliases for local retrieval.

        Terms come from the planner entity plus aliases learned from the local
        index.  Short forms are derived only from already visible name surfaces
        such as dotted or whitespace-separated personal names; no answer names
        or relation labels are introduced here.
        """

        values: List[str] = []
        for entity in getattr(plan, "entities", []) or []:
            values.extend(
                [
                    str(getattr(entity, "surface", "") or ""),
                    str(getattr(entity, "canonical", "") or ""),
                    str(getattr(entity, "linked_alias", "") or ""),
                    *[str(value or "") for value in (getattr(entity, "aliases", []) or [])],
                ]
            )
        return self._expand_entity_surface_terms(values, limit=limit)

    def _expand_entity_surface_terms(self, terms: Sequence[str], limit: int = 24) -> List[str]:
        expanded: List[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            text = str(value or "").strip()
            key = normalize_entity_name(text)
            if len(key) < 2 or key in seen or len(expanded) >= limit:
                return
            seen.add(key)
            expanded.append(text)

        for term in terms:
            for variant in surface_variants(str(term or "")):
                add(variant)
                for short in self._source_visible_short_forms(variant):
                    add(short)
                if len(expanded) >= limit:
                    return expanded
        return expanded

    @staticmethod
    def _source_visible_short_forms(term: str) -> List[str]:
        """Derive cautious short forms from a visible multi-part name."""

        text = str(term or "").strip()
        if not text:
            return []
        separators = ["·", "・", "•", "路", " "]
        parts = [text]
        for separator in separators:
            next_parts: List[str] = []
            for part in parts:
                next_parts.extend(item.strip() for item in part.split(separator) if item.strip())
            parts = next_parts or parts
        if len(parts) < 2:
            return []
        first = parts[0].strip()
        return [first] if len(normalize_entity_name(first)) >= 2 else []

    def _augment_plan_entities_from_source_aliases(
        self,
        plan: QueryPlan,
        exact_results: Sequence[SearchResult],
    ) -> None:
        """Add source-observed aliases that exactly normalize to the target.

        The local index may know that a query surface such as a dotless name
        corresponds to a dotted source form.  Absorbing only exact normalized
        aliases keeps this generic and conservative while enabling later
        source-window searches to use the document's own short forms.
        """

        if not getattr(plan, "entities", None) or not exact_results:
            return
        for entity in plan.entities:
            target_keys = {
                normalize_entity_name(value)
                for value in [
                    str(getattr(entity, "surface", "") or ""),
                    str(getattr(entity, "canonical", "") or ""),
                    str(getattr(entity, "linked_alias", "") or ""),
                    *[str(item or "") for item in (getattr(entity, "aliases", []) or [])],
                ]
            }
            target_keys = {value for value in target_keys if value}
            if not target_keys:
                continue
            additions: List[str] = []
            for result in exact_results:
                metadata = result.metadata or {}
                aliases = metadata.get("aliases") or []
                if isinstance(aliases, str):
                    aliases = [aliases]
                aliases = [*list(aliases), *list(result.matched_terms or [])]
                for alias in aliases:
                    alias_text = str(alias or "").strip()
                    if not alias_text or normalize_entity_name(alias_text) not in target_keys:
                        continue
                    additions.extend(self._expand_entity_surface_terms([alias_text], limit=8))
            if additions:
                entity.aliases = list(dict.fromkeys([*entity.aliases, *additions]))[:16]

    def search(self, query: str, top_k: int | None = None, keyword_filter: str = "") -> List[SearchResult]:
        """Execute the evidence-first retrieval cascade.

        Enumerative plans use a distinct, generic branch: recall is widened by
        planner-provided semantic views, source windows are position-diversified,
        and only atomic claims that pass the evidence contract may reach answer
        generation.
        """

        started_at = perf_counter()
        stage_started = started_at
        stage_ms: Dict[str, float] = {}
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []
        collection_count = self._safe_collection_count()
        if collection_count <= 0:
            return []

        plan = build_query_plan(cleaned_query, rag_config=self.rag_config, client=self.client)
        stage_ms["planner"] = round((perf_counter() - stage_started) * 1000, 1)
        stage_started = perf_counter()
        self._link_plan_entities(plan)
        stage_ms["entity_linking"] = round((perf_counter() - stage_started) * 1000, 1)

        enumeration_mode = self._is_enumeration_plan(plan)
        final_top_k = max(int(top_k or self.rag_config.top_k), 1)
        candidate_k = self._candidate_budget(plan, final_top_k, collection_count)
        query_variants = self._select_query_variants(
            plan,
            limit=query_variant_limit(self.rag_config, plan),
        )

        stage_started = perf_counter()
        channel_results = self._run_retrieval_channels(
            plan,
            keyword_filter=keyword_filter,
            candidate_k=candidate_k,
            query_variants=query_variants,
        )
        self._augment_plan_entities_from_source_aliases(plan, channel_results.get("exact_entity", []))
        stage_ms["hybrid_recall"] = round((perf_counter() - stage_started) * 1000, 1)

        focus_terms = self._plan_focus_terms(plan, cleaned_query)
        anchor_documents = self._anchor_documents(channel_results.get("exact_entity", []), focus_terms)
        anchor_document_ids = self._anchor_document_ids(channel_results.get("exact_entity", []), focus_terms)
        anchor_segment_ids: List[str] = []
        coverage_anchor_note = ""

        # Enumerations need a concrete structural span even when exact entity
        # retrieval succeeds. Consensus across channels is stronger than treating
        # every mention of one entity as equally relevant.
        if is_coverage_fallback(plan) or enumeration_mode or not anchor_document_ids:
            discovered_document_ids, discovered_segment_ids, discovered_anchors = self._discover_navigation_anchors(channel_results)
            if discovered_document_ids:
                anchor_document_ids = discovered_document_ids
                anchor_segment_ids = discovered_segment_ids
                anchor_documents = discovered_anchors
                coverage_anchor_note = (
                    f"source_anchor documents={len(anchor_document_ids)} "
                    f"segments={len(anchor_segment_ids)}"
                )

        stage_started = perf_counter()
        navigation_results = self._run_read_plan_navigation(
            plan=plan,
            anchor_document_ids=anchor_document_ids,
            query=cleaned_query,
            candidate_k=candidate_k,
            anchor_segment_ids=anchor_segment_ids,
        )
        if navigation_results:
            channel_results["navigation"] = navigation_results
            dense_navigation = self._dense_navigation_windows(
                plan=plan,
                document_ids=anchor_document_ids,
                segment_ids=anchor_segment_ids,
                candidate_k=candidate_k,
            )
            if dense_navigation:
                channel_results["coverage_dense_navigation"] = dense_navigation
                navigation_results = self._merge_navigation_results(
                    navigation_results,
                    dense_navigation,
                )
        stage_ms["navigation"] = round((perf_counter() - stage_started) * 1000, 1)

        stage_started = perf_counter()
        fused = self._weighted_rrf(plan.intent, channel_results, plan=plan)
        focused_candidates = self._apply_focus_guard(fused, focus_terms, anchor_documents)
        rerank_candidates = list(focused_candidates)
        stage_ms["fusion"] = round((perf_counter() - stage_started) * 1000, 1)

        entity_coverage_failed = bool(focus_terms) and not any(
            self._matches_focus_terms(result, focus_terms) for result in rerank_candidates[:10]
        )
        plan.entity_coverage_failed = entity_coverage_failed

        rerank_budget = self._rerank_candidate_budget(plan, final_top_k)
        if enumeration_mode:
            rerank_input = self._enumeration_candidate_pool(
                ranked=rerank_candidates,
                navigation_results=navigation_results,
                budget=rerank_budget,
            )
        else:
            rerank_input = self._diversify_results(rerank_candidates[:rerank_budget], rerank_budget)

        stage_started = perf_counter()
        if should_run_reranker(
            self.rag_config,
            plan=plan,
            candidates=rerank_input,
            entity_coverage_failed=entity_coverage_failed,
        ):
            reranked, reranker_status = self.reranker.rerank(cleaned_query, rerank_input)
        else:
            reranked, reranker_status = list(rerank_input), "adaptive_skip_strong_local_evidence"
        stage_ms["rerank"] = round((perf_counter() - stage_started) * 1000, 1)

        stage_started = perf_counter()
        if enumeration_mode:
            judge_limit = min(len(reranked), max(12, min(24, rerank_budget)))
            judge_input = reranked[:judge_limit]
        else:
            judge_limit = 10 if is_coverage_fallback(plan) else 6
            judge_input = reranked[: min(judge_limit, max(final_top_k + 3, 5))]
        enumeration_rescue_count = 0
        if should_run_evidence_judge(
            self.rag_config,
            plan=plan,
            candidates=judge_input,
            entity_coverage_failed=entity_coverage_failed,
        ):
            judged_head = self.evidence_judge.judge(
                cleaned_query,
                judge_input,
                query_language=plan.query_language,
                plan=plan,
            )
            if enumeration_mode and not any(self._qualified_claims(item, plan=plan) for item in judged_head):
                judged_head, enumeration_rescue_count = self._run_enumeration_rescue_judge(
                    query=cleaned_query,
                    plan=plan,
                    judged_head=judged_head,
                    reranked=reranked,
                    navigation_results=navigation_results,
                    judge_limit=judge_limit,
                )
            judged_ids = {item.chunk_id for item in judged_head if item.chunk_id}
            judged = [
                *judged_head,
                *[
                    item
                    for item in reranked
                    if not item.chunk_id or item.chunk_id not in judged_ids
                ],
            ]
            evidence_judge_status = "executed"
        else:
            judged = list(reranked)
            evidence_judge_status = "adaptive_skip_high_confidence"
        stage_ms["evidence_judge"] = round((perf_counter() - stage_started) * 1000, 1)

        final_candidates = self._apply_focus_guard(judged, focus_terms, anchor_documents)
        if enumeration_mode:
            # Allow several independent source windows when the question asks
            # for a list. This does not assert completeness; it merely prevents
            # the UI's generic Top-K from hiding validated incidents.
            enumeration_top_k = max(final_top_k, min(10, final_top_k + 2))
            final_results = self._select_enumerated_evidence(
                candidates=final_candidates,
                navigation_results=navigation_results,
                top_k=enumeration_top_k,
                plan=plan,
            )
        else:
            final_results = self._prioritize_read_plan_results(
                candidates=final_candidates,
                navigation_results=navigation_results,
                plan=plan,
                top_k=final_top_k,
            )
        stage_ms["total"] = round((perf_counter() - started_at) * 1000, 1)

        qualified_atomic_count = sum(
            len(self._qualified_claims(result, plan=plan))
            for result in final_results
        ) if enumeration_mode else 0
        entity_window_count = sum(
            1
            for result in navigation_results
            if str((result.diagnostics or {}).get("navigation_channel") or "") == "entity_coverage"
        )
        performance_note = (
            "perf_ms "
            + " ".join(f"{name}={value}" for name, value in stage_ms.items())
            + f"; budget candidates={candidate_k} dense_queries={len(query_variants)} "
            + f"rerank={len(rerank_input)}({reranker_status}) judge={evidence_judge_status} "
            + f"navigation={bool(navigation_results)} enumerate={enumeration_mode} "
            + f"entity_windows={entity_window_count} qualified_atomic={qualified_atomic_count} "
            + (f"enumeration_rescue={enumeration_rescue_count} " if enumeration_rescue_count else "")
            + coverage_anchor_note
        ).rstrip()
        logger.info("RAG retrieval %s", performance_note)
        warnings = list(dict.fromkeys([*list(plan.warnings), performance_note]))
        self.last_diagnostics = RetrievalDiagnostics(
            query_plan=plan,
            candidates_by_channel={key: [item.chunk_id for item in value[:10]] for key, value in channel_results.items()},
            final_chunk_ids=[result.chunk_id for result in final_results],
            reranker_enabled=self.rag_config.enable_reranker,
            reranker_status=reranker_status,
            evidence_judge_enabled=bool(
                self.rag_config.enable_evidence_judge
                or is_coverage_fallback(plan)
                or enumeration_mode
            ),
            entity_coverage_failed=entity_coverage_failed,
            warnings=warnings,
        )
        for result in final_results:
            result.diagnostics["query_plan_intent"] = plan.intent
            result.diagnostics["query_language"] = plan.query_language
            result.diagnostics["entity_coverage_failed"] = entity_coverage_failed
            result.diagnostics["planner_source"] = plan.planner_source
            result.diagnostics["answer_mode"] = getattr(plan.semantics, "answer_mode", "direct")
            result.diagnostics["enumeration_mode"] = enumeration_mode
            result.diagnostics["entity_linking_confidence"] = plan.entity_linking_confidence
            result.diagnostics["retrieval_stage_ms"] = dict(stage_ms)
            result.diagnostics["reranker_status"] = reranker_status
            result.diagnostics["evidence_judge_status"] = evidence_judge_status
        self._last_results = final_results
        return final_results
    def _anchor_document_ids(

        self,
        exact_results: Sequence[SearchResult],
        focus_terms: Sequence[str],
    ) -> List[str]:
        """Return concrete document ids for the locally linked entity/document.

        The document id is an execution anchor, not an answer.  It lets a
        ReadPlan navigate source order inside the right document rather than
        asking a global vector search to infer where a conclusion lives.
        """

        ids: List[str] = []
        for result in exact_results:
            if focus_terms and not self._matches_focus_terms(result, focus_terms):
                continue
            document_id = str((result.metadata or {}).get("document_id") or "")
            if document_id:
                ids.append(document_id)
        return list(dict.fromkeys(ids))[:4]

    def _discover_navigation_anchors(
        self,
        channel_results: Dict[str, List[SearchResult]],
    ) -> Tuple[List[str], List[str], set[str]]:
        """Select a document/span from independent first-pass evidence.

        The score uses only channel agreement, rank, and structural record type;
        it never inspects question-specific tokens or document-topic vocabulary.
        """

        scores: Dict[Tuple[str, str], float] = {}
        support: Dict[Tuple[str, str], set[str]] = {}
        kind_weight = {
            "child": 1.0,
            "parent": 0.95,
            "navigation_window": 0.92,
            "terminal_window": 0.92,
            "section_card": 0.78,
            "entity_card": 0.72,
            "document_card": 0.55,
        }
        for channel, results in channel_results.items():
            for rank, result in enumerate(results[:24], start=1):
                metadata = result.metadata or {}
                document_id = str(metadata.get("document_id") or "")
                if not document_id:
                    continue
                segment_id = str(metadata.get("document_segment_id") or "")
                key = (document_id, segment_id)
                chunk_kind = str(metadata.get("chunk_kind") or result.chunk_kind or "")
                scores[key] = scores.get(key, 0.0) + kind_weight.get(chunk_kind, 0.65) / (rank + 3.0)
                support.setdefault(key, set()).add(channel)

        if not scores:
            return [], [], set()
        # Prefer a concrete source span whenever first-pass evidence contains
        # one. A document-card has no span and is retained only as the generic
        # fallback for unsegmented documents.
        candidate_keys = [key for key in scores if key[1]] or list(scores)
        ordered = sorted(
            candidate_keys,
            key=lambda key: (scores[key] + 0.08 * len(support.get(key, set())), len(support.get(key, set()))),
            reverse=True,
        )
        selected = ordered[:1]
        document_ids = [key[0] for key in selected]
        segment_ids = [key[1] for key in selected if key[1]]
        anchors = {f"document:{document_id}" for document_id in document_ids}
        return document_ids, segment_ids, anchors

    def _dense_navigation_windows(
        self,
        plan: QueryPlan,
        document_ids: Sequence[str],
        segment_ids: Sequence[str],
        candidate_k: int,
    ) -> List[SearchResult]:
        """Semantically rank raw source windows inside an anchored span.

        Enumerations use several planner-generated evidence views. These are
        source-neutral paraphrases of the current question, not a keyword list.
        """

        document_ids = [str(item) for item in document_ids if str(item)]
        if not document_ids:
            return []
        enumeration_mode = self._is_enumeration_plan(plan)
        variants = self._source_window_query_variants(plan, limit=4 if enumeration_mode else 1)
        if not variants:
            return []
        try:
            texts = [str(getattr(item, "text", "") or "").strip() for item in variants]
            if hasattr(self.client, "embed_queries"):
                embeddings = self.client.embed_queries(texts, batch_size=len(texts))
            else:
                embeddings = [self.client.embed_query(text) for text in texts]
        except Exception as exc:
            logger.warning("Source-window semantic ranking unavailable: %s", exc)
            return []
        if not embeddings:
            return []

        wanted_segments = {str(item) for item in segment_ids if str(item)}
        output: List[SearchResult] = []
        window_limit = min(
            max(candidate_k * (3 if enumeration_mode else 2), 36),
            160 if enumeration_mode else 96,
        )
        for document_id in document_ids:
            raw = None
            window_filter = {
                "$and": [
                    {"document_id": document_id},
                    {"chunk_kind": {"$in": ["navigation_window", "terminal_window"]}},
                ]
            }
            try:
                raw = self.collection.query(
                    query_embeddings=embeddings,
                    n_results=window_limit,
                    where=window_filter,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                try:
                    raw = self.collection.query(
                        query_embeddings=embeddings,
                        n_results=window_limit,
                        where={"document_id": document_id},
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as exc:
                    logger.warning("Anchored source-window query failed: %s", exc)
                    continue
            for channel_results in self._dense_results_from_batch(raw, variants).values():
                for result in channel_results:
                    metadata = result.metadata or {}
                    chunk_kind = str(metadata.get("chunk_kind") or result.chunk_kind or "")
                    if chunk_kind not in {"navigation_window", "terminal_window"}:
                        continue
                    segment_id = str(metadata.get("document_segment_id") or "")
                    if wanted_segments and segment_id and segment_id not in wanted_segments:
                        continue
                    result.diagnostics["navigation_channel"] = "dense_navigation"
                    output.append(result)

        output.sort(key=lambda item: item.vector_score, reverse=True)
        if enumeration_mode:
            return self._position_diverse_results(
                output,
                top_k=min(max(24, candidate_k // 2), 48),
            )
        return self._diversify_results(output, min(max(10, candidate_k // 2), 24))
    def _run_read_plan_navigation(
        self,
        plan: QueryPlan,
        anchor_document_ids: Sequence[str],
        query: str,
        candidate_k: int,
        anchor_segment_ids: Sequence[str] | None = None,
    ) -> List[SearchResult]:
        """Execute topology-aware source-window reading for a semantic plan."""

        semantics = getattr(plan, "semantics", None)
        regions = list(getattr(semantics, "regions", []) or ["all"])
        scope = str(getattr(semantics, "scope", "entity_or_document") or "entity_or_document")
        enumeration_mode = self._is_enumeration_plan(plan)
        needs_navigation = (
            enumeration_mode
            or is_coverage_fallback(plan)
            or any(region in {"front", "middle", "terminal"} for region in regions)
            or bool(getattr(semantics, "need_timeline", False))
            or bool(getattr(semantics, "need_entity_neighborhood", False))
            or scope in {"collection", "document", "section", "entity_neighborhood"}
        )
        if not needs_navigation or not anchor_document_ids:
            return []

        if enumeration_mode:
            navigation_limit = min(max(24, candidate_k // 2), 48)
        else:
            navigation_limit = min(
                max(12 if is_coverage_fallback(plan) else 6, candidate_k // 3),
                32,
            )
        hits = self.hybrid_index.navigation_search(
            document_ids=anchor_document_ids,
            regions=regions,
            query=query,
            language=plan.query_language,
            limit=navigation_limit,
            position_bias=str(getattr(semantics, "position_bias", "none") or "none"),
            segment_ids=anchor_segment_ids,
            coverage=enumeration_mode,
        )

        entity_terms = self._entity_search_terms(plan)

        if enumeration_mode and entity_terms:
            entity_hits = self.hybrid_index.entity_window_search(
                terms=entity_terms,
                document_ids=anchor_document_ids,
                segment_ids=anchor_segment_ids,
                limit=min(max(36, candidate_k), 72),
            )
            hits = [*entity_hits, *hits]
        elif bool(getattr(semantics, "need_entity_neighborhood", False)) and not is_coverage_fallback(plan):
            # Non-enumerative timeline questions retain the earlier terminal
            # neighborhood behavior; it is not used to decide an entity list.
            entity_hits = self.hybrid_index.entity_terminal_neighborhood_search(
                terms=entity_terms,
                document_ids=anchor_document_ids,
                limit=min(max(4, candidate_k // 16), 8),
            )
            hits = [*entity_hits, *hits]

        deduped: List[SearchResult] = []
        seen_ids: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen_ids:
                continue
            seen_ids.add(hit.chunk_id)
            result = self._hit_to_result(hit)
            result.diagnostics["navigation_channel"] = hit.channel
            deduped.append(result)
        return deduped
    def _merge_navigation_results(
        self,
        first: Sequence[SearchResult],
        second: Sequence[SearchResult],
    ) -> List[SearchResult]:
        merged: List[SearchResult] = []
        seen: set[str] = set()
        for result in [*first, *second]:
            key = result.chunk_id or self._result_key(result, fallback=str(len(merged)))
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
        return merged

    def _is_enumeration_plan(self, plan: QueryPlan | None) -> bool:
        return str(
            getattr(getattr(plan, "semantics", None), "operation", "lookup") or "lookup"
        ) == "enumerate"

    def _source_window_query_variants(self, plan: QueryPlan, limit: int) -> List[QueryVariant]:
        """Return semantic query views for ranking windows inside an anchor.

        Evidence-contract views are recall hints only. They are used after a
        document/span has already been anchored, and the evidence judge still
        decides which atomic facts qualify for the requested category.
        """

        selected = list(self._select_query_variants(plan, limit=limit))
        if not self._is_enumeration_plan(plan):
            return selected

        semantics = getattr(plan, "semantics", None)
        contract = getattr(semantics, "evidence_contract", None)
        seen = {str(getattr(item, "text", "") or "").strip().casefold() for item in selected}
        for view in list(getattr(contract, "retrieval_views", []) or []):
            text = str(view or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            selected.append(QueryVariant(text=text, language=plan.query_language, origin="evidence_view"))
            seen.add(key)
            if len(selected) >= max(1, int(limit or 1)):
                break
        return selected

    def _result_position(self, result: SearchResult) -> float:
        metadata = result.metadata or {}
        value = metadata.get("document_position_ratio")
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    def _position_diverse_results(
        self,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Keep high-ranked results while seeding evidence across source order."""

        wanted = max(1, int(top_k))
        unique: List[SearchResult] = []
        seen: set[str] = set()
        for index, result in enumerate(results):
            key = result.chunk_id or self._result_key(result, fallback=str(index))
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
        if len(unique) <= wanted:
            return unique
        bin_count = min(8, max(3, wanted // 4))
        buckets: List[List[SearchResult]] = [[] for _ in range(bin_count)]
        for result in unique:
            bucket = min(bin_count - 1, int(self._result_position(result) * bin_count))
            buckets[bucket].append(result)

        output: List[SearchResult] = []
        picked: set[str] = set()
        for bucket in buckets:
            if not bucket:
                continue
            result = bucket[0]
            key = result.chunk_id or self._result_key(result, fallback=str(len(output)))
            if key not in picked:
                picked.add(key)
                output.append(result)
            if len(output) >= wanted:
                return output
        for result in unique:
            key = result.chunk_id or self._result_key(result, fallback=str(len(output)))
            if key in picked:
                continue
            picked.add(key)
            output.append(result)
            if len(output) >= wanted:
                break
        return output

    def _enumeration_candidate_pool(
        self,
        ranked: Sequence[SearchResult],
        navigation_results: Sequence[SearchResult],
        budget: int,
    ) -> List[SearchResult]:
        """Mix semantic ranking with topology coverage before atomic judging."""

        primary_limit = max(int(budget) * 2, 36)
        source_kinds = {"navigation_window", "terminal_window", "child", "parent"}
        merged = [
            result
            for result in [*list(ranked)[:primary_limit], *list(navigation_results)]
            if str((result.metadata or {}).get("chunk_kind") or result.chunk_kind or "") in source_kinds
        ]
        return self._position_diverse_results(merged, top_k=budget)

    def _run_enumeration_rescue_judge(
        self,
        query: str,
        plan: QueryPlan,
        judged_head: Sequence[SearchResult],
        reranked: Sequence[SearchResult],
        navigation_results: Sequence[SearchResult],
        judge_limit: int,
    ) -> tuple[List[SearchResult], int]:
        """Run a bounded second-pass judge when enumeration would be empty.

        This is a generic recall-safety valve: if the first judge batch finds no
        qualifying atomic evidence, inspect additional source windows from the
        already retrieved pool instead of letting the final answer collapse to a
        false "no evidence" conclusion.
        """

        judged_ids = {item.chunk_id for item in judged_head if item.chunk_id}
        rescue_budget = max(int(judge_limit), min(int(judge_limit) * 2, 48))
        rescue_pool = self._enumeration_candidate_pool(
            ranked=reranked,
            navigation_results=navigation_results,
            budget=rescue_budget,
        )
        unjudged = [
            item
            for item in rescue_pool
            if item.chunk_id and item.chunk_id not in judged_ids
        ][:rescue_budget]
        if not unjudged:
            return list(judged_head), 0

        batch_size = max(6, min(int(judge_limit or 12), 12))
        rescued: List[SearchResult] = []
        for start in range(0, len(unjudged), batch_size):
            batch = unjudged[start : start + batch_size]
            if not batch:
                continue
            rescued.extend(
                self.evidence_judge.judge(
                    query,
                    batch,
                    query_language=plan.query_language,
                    plan=plan,
                )
            )

        merged: List[SearchResult] = []
        seen: set[str] = set()
        for result in [*list(judged_head), *rescued]:
            key = result.chunk_id or self._result_key(result, fallback=str(len(merged)))
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)

        merged.sort(
            key=lambda item: (
                len(self._qualified_claims(item, plan=plan)),
                item.evidence.answerability if item.evidence else 0.0,
                item.evidence.relevance if item.evidence else 0.0,
                item.final_score,
            ),
            reverse=True,
        )
        return merged, len(rescued)

    def _qualified_claims(self, result: SearchResult, plan: QueryPlan | None = None) -> List[object]:
        """Return only source-grounded claims that survive defensive checks."""

        evidence = getattr(result, "evidence", None)
        contract = getattr(getattr(plan, "semantics", None), "evidence_contract", None)
        required = {
            str(role).strip()
            for role in (getattr(contract, "required_roles", []) or [])
            if str(role).strip()
        }
        answer_unit = str(getattr(contract, "answer_unit", "fact") or "fact")
        source = result.content or ""
        qualified: List[object] = []
        for claim in (getattr(evidence, "atomic_claims", []) or []):
            statement = str(getattr(claim, "statement", "") or "").strip()
            excerpt = str(getattr(claim, "source_excerpt", "") or "").strip()
            roles = getattr(claim, "roles", {}) or {}
            if not bool(getattr(claim, "qualifies", False)) or not statement or not excerpt:
                continue
            if not is_grounded_excerpt(excerpt, source):
                continue
            if any(not str(roles.get(role, "") or "").strip() for role in required):
                continue
            if answer_unit == "relation":
                left = str(roles.get("target_entity", "") or "").strip()
                right = str(roles.get("object", "") or "").strip()
                if not left or not right:
                    continue
                left_key = normalize_entity_name(left)
                right_key = normalize_entity_name(right)
                if left_key and right_key and (left_key == right_key or left_key in right_key or right_key in left_key):
                    continue
                compact_source = "".join(ch for ch in source.casefold() if ch.isalnum())
                compact_right = "".join(ch for ch in right.casefold() if ch.isalnum())
                compact_left = "".join(ch for ch in left.casefold() if ch.isalnum())
                target_aliases = [
                    alias
                    for entity in (getattr(plan, "entities", []) or [])
                    for alias in [
                        getattr(entity, "surface", ""),
                        getattr(entity, "canonical", ""),
                        getattr(entity, "linked_alias", ""),
                        *(getattr(entity, "aliases", []) or []),
                    ]
                    if normalize_entity_name(alias)
                ]
                source_has_target_alias = any(
                    "".join(ch for ch in str(alias).casefold() if ch.isalnum()) in compact_source
                    for alias in target_aliases
                    if "".join(ch for ch in str(alias).casefold() if ch.isalnum())
                )
                if not compact_right or compact_right not in compact_source:
                    continue
                if not compact_left or (
                    compact_left not in compact_source
                    and not (bool(getattr(claim, "target_matches_query", False)) and source_has_target_alias)
                ):
                    continue
                # The judge has already made an explicit source-local
                # coreference decision.  A literal full-name equality would
                # reject short forms used by the document itself.
                target_linked = bool(getattr(claim, "target_matches_query", False))
                if not target_linked:
                    target_linked = any(
                        normalize_entity_name(left) == normalize_entity_name(alias)
                        or normalize_entity_name(left) in normalize_entity_name(alias)
                        or normalize_entity_name(alias) in normalize_entity_name(left)
                        for alias in target_aliases
                    )
                if not target_linked:
                    continue
            qualified.append(claim)
        return qualified

    def _claim_key(self, claim: object, plan: QueryPlan | None = None) -> str:
        """Use the answer object as the de-duplication key for relation lists."""

        roles = getattr(claim, "roles", {}) or {}
        semantics = getattr(plan, "semantics", None)
        contract = getattr(semantics, "evidence_contract", None)
        unit = str(getattr(contract, "answer_unit", "fact") or "fact")
        if unit == "relation":
            target = normalize_entity_name(str(roles.get("target_entity", "") or ""))
            other = normalize_entity_name(str(roles.get("object", "") or ""))
            if target and other:
                return f"relation:{target}:{other}"
        statement = " ".join(str(getattr(claim, "statement", "") or "").casefold().split())
        role_text = " ".join(
            f"{str(key).casefold()}:{str(value).casefold()}"
            for key, value in sorted(roles.items())
        )
        return statement + "|" + role_text

    def _select_enumerated_evidence(
        self,
        candidates: Sequence[SearchResult],
        navigation_results: Sequence[SearchResult],
        top_k: int,
        plan: QueryPlan | None = None,
    ) -> List[SearchResult]:
        """Select raw source windows that contribute distinct qualified units."""

        source_kinds = {"navigation_window", "terminal_window", "child", "parent"}
        selected: List[SearchResult] = []
        seen_claims: set[str] = set()
        for result in candidates:
            kind = str((result.metadata or {}).get("chunk_kind") or result.chunk_kind or "")
            if kind not in source_kinds:
                continue
            claims = self._qualified_claims(result, plan=plan)
            if not claims:
                continue
            new_claims = [claim for claim in claims if self._claim_key(claim, plan=plan) not in seen_claims]
            if not new_claims:
                continue
            for claim in new_claims:
                seen_claims.add(self._claim_key(claim, plan=plan))
            selected.append(result)
            if len(selected) >= top_k:
                break

        if not selected:
            fallback_pool = [
                result
                for result in [*list(candidates), *list(navigation_results)]
                if str((result.metadata or {}).get("chunk_kind") or result.chunk_kind or "") in source_kinds
            ]
            fallback = self._position_diverse_results(fallback_pool, top_k=top_k)
            for result in fallback:
                result.diagnostics["enumeration_source_fallback"] = True
            return fallback

        selected.sort(key=self._result_position)
        return selected
    def _prioritize_read_plan_results(
        self,
        candidates: Sequence[SearchResult],
        navigation_results: Sequence[SearchResult],
        plan: QueryPlan,
        top_k: int,
    ) -> List[SearchResult]:
        """Reserve context budget for the requested source region.

        RRF is useful for broad recall but can drown terminal windows in many
        mid-document mentions of an entity.  This function pins a small number
        of *already retrieved source windows* for an explicit ReadPlan; it does
        not create or infer additional content.
        """

        if not candidates:
            return []
        semantics = getattr(plan, "semantics", None)
        regions = set(getattr(semantics, "regions", []) or [])
        navigation_ids = {result.chunk_id for result in navigation_results if result.chunk_id}
        coverage_mode = is_coverage_fallback(plan)
        if not navigation_ids or (not coverage_mode and not regions.intersection({"front", "middle", "terminal"})):
            return self._diversify_results(candidates, top_k)

        region_candidates = [result for result in candidates if result.chunk_id in navigation_ids]
        others = [result for result in candidates if result.chunk_id not in navigation_ids]
        # An explicit ReadPlan is a retrieval instruction, not merely another
        # weak ranking feature. If reranking or evidence filtering removes all
        # requested source windows, retain those raw windows as a bounded
        # fallback instead of silently substituting mid-document candidates.
        if not region_candidates:
            region_candidates = list(navigation_results)
        reserve = min(max(3 if coverage_mode else 2, (top_k + 1) // 2), top_k)
        pinned = self._diversify_results(region_candidates, reserve)
        remaining = self._diversify_results(others, max(0, top_k - len(pinned)))
        output: List[SearchResult] = []
        seen_ids: set[str] = set()
        for result in [*pinned, *remaining]:
            key = result.chunk_id or self._result_key(result, fallback=str(len(output)))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            output.append(result)
            if len(output) >= top_k:
                break
        return output

    def _link_plan_entities(self, plan: QueryPlan) -> None:
        """将 LLM 抽取的实体提及链接到离线 aliases 索引。

        LLM 只给出问题语义和表面提及；canonical 名称、可用于 exact search 的别名
        均由本地索引返回，避免 LLM 补造实体或针对某个问法编写清洗规则。
        """

        link_scores: List[float] = []
        for entity in plan.entities:
            candidates = self.hybrid_index.resolve_entity_candidates(entity.surface, limit=4)
            if not candidates:
                continue
            top = candidates[0]
            canonical = str(top.get("canonical") or entity.canonical)
            indexed_alias = str(top.get("alias") or canonical)
            entity.canonical = canonical
            entity.linked_alias = indexed_alias
            entity.link_confidence = float(top.get("score") or 0.0)
            entity.entity_type = str(top.get("entity_type") or entity.entity_type)
            entity.aliases = list(
                dict.fromkeys(
                    [entity.surface, canonical, indexed_alias, *entity.aliases]
                )
            )[:10]
            entity.alias_sources = list(dict.fromkeys([*entity.alias_sources, "hybrid_entity_index"]))
            link_scores.append(entity.link_confidence)

        if link_scores:
            plan.entity_linking_confidence = round(sum(link_scores) / len(link_scores), 4)
        elif plan.entities:
            plan.entity_linking_confidence = 0.0
            plan.warnings.append("entity_link_not_found")

    def _run_retrieval_channels(
        self,
        plan,
        keyword_filter: str,
        candidate_k: int,
        query_variants: Sequence[object] | None = None,
    ) -> Dict[str, List[SearchResult]]:
        """Run bounded first-pass channels.

        Exact retrieval preserves entity coverage.  Dense and lexical channels
        deliberately avoid an entity-only expansion that otherwise retrieves a
        large amount of generic entity context.
        """

        channels: Dict[str, List[SearchResult]] = {}
        selected_variants = list(query_variants or self._select_query_variants(plan, limit=2))
        entity_terms = self._entity_search_terms(plan)

        if self.rag_config.enable_exact_retrieval and entity_terms:
            hits = self.hybrid_index.exact_search(entity_terms, limit=min(candidate_k, 24))
            channels["exact_entity"] = [self._hit_to_result(hit) for hit in hits]

        if self.rag_config.enable_structured_retrieval and selected_variants:
            hits = self.hybrid_index.structured_search(
                selected_variants[0].text,
                preferred_chunk_kinds=plan.preferred_chunk_kinds,
                language=plan.query_language,
                limit=min(candidate_k, 24),
            )
            channels["structured"] = [self._hit_to_result(hit) for hit in hits]

        lexical_query = " ".join([variant.text for variant in selected_variants] + [keyword_filter])
        hits = self.hybrid_index.lexical_search(lexical_query, language=plan.query_language, limit=min(candidate_k, 36))
        channels["lexical"] = [self._hit_to_result(hit) for hit in hits]

        # Exact/structured/lexical agreement is a stronger and cheaper signal
        # than a redundant remote embedding call for simple anchored lookups.
        if local_channels_agree(channels, plan):
            logger.debug("跳过 Dense 通道：本地精确、结构化与词法通道已形成强一致证据。")
        else:
            dense_channels = self._dense_retrieve(
                plan,
                candidate_k=candidate_k,
                query_variants=selected_variants,
            )
            channels.update(dense_channels)
        return {key: value for key, value in channels.items() if value}

    def _dense_retrieve(
        self,
        plan,
        candidate_k: int,
        query_variants: Sequence[object] | None = None,
    ) -> Dict[str, List[SearchResult]]:
        """Embed/query selected variants in one batch when the client supports it.

        Batched query embeddings eliminate repeated local E5 inference and make
        remote embedding providers use a single request.  A sequential fallback
        keeps the patch compatible with older provider clients.
        """

        variants = list(query_variants or self._select_query_variants(plan, limit=2))
        if not variants:
            return {}
        try:
            query_texts = [str(getattr(variant, "text", "") or "").strip() for variant in variants]
            if hasattr(self.client, "embed_queries"):
                query_embeddings = self.client.embed_queries(query_texts, batch_size=len(query_texts))
            else:
                query_embeddings = [self.client.embed_query(text) for text in query_texts]
            raw = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=candidate_k,
                include=["documents", "metadatas", "distances"],
            )
            return self._dense_results_from_batch(raw, variants)
        except Exception as exc:
            logger.warning("批量 Dense 检索失败，回退顺序检索: %s", exc)
            output: Dict[str, List[SearchResult]] = {}
            for query_index, variant in enumerate(variants):
                channel = "dense_original_query" if getattr(variant, "origin", "") == "original" else f"dense_{getattr(variant, 'origin', 'variant')}_{query_index}"
                try:
                    query_embedding = self.client.embed_query(variant.text)
                    raw = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=candidate_k,
                        include=["documents", "metadatas", "distances"],
                    )
                    output.update(self._dense_results_from_batch(raw, [variant]))
                except Exception as inner_exc:
                    message = str(inner_exc)
                    if "dimension" in message.lower():
                        logger.warning("跳过向量检索：Chroma 向量维度与当前 Embedding 模型不一致；继续使用 SQLite 通道。")
                    else:
                        logger.warning("跳过 Dense 检索变体 %r: %s", str(variant.text)[:80], message)
            return output

    def _dense_results_from_batch(self, raw: Dict[str, object], variants: Sequence[object]) -> Dict[str, List[SearchResult]]:
        output: Dict[str, List[SearchResult]] = {}
        all_documents = raw.get("documents") or []
        all_metadatas = raw.get("metadatas") or []
        all_distances = raw.get("distances") or []
        all_ids = raw.get("ids") or []
        for query_index, variant in enumerate(variants):
            channel = "dense_original_query" if getattr(variant, "origin", "") == "original" else f"dense_{getattr(variant, 'origin', 'variant')}_{query_index}"
            documents = all_documents[query_index] if query_index < len(all_documents) else []
            metadatas = all_metadatas[query_index] if query_index < len(all_metadatas) else []
            distances = all_distances[query_index] if query_index < len(all_distances) else []
            ids = all_ids[query_index] if query_index < len(all_ids) else []
            results: List[SearchResult] = []
            for index, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                metadata = dict(metadata or {})
                metadata.setdefault("chunk_id", metadata.get("chunk_id") or (ids[index] if index < len(ids) else ""))
                results.append(
                    SearchResult(
                        content=document,
                        metadata=metadata,
                        vector_score=self._distance_to_score(distance),
                        keyword_score=0.0,
                        final_score=0.0,
                        chunk_id=str(metadata.get("chunk_id") or ""),
                        parent_chunk_id=str(metadata.get("parent_chunk_id") or ""),
                        chunk_kind=str(metadata.get("chunk_kind") or "child"),
                    )
                )
            output[channel] = results
        return output

    def _weighted_rrf(
        self,
        intent: str,
        channel_results: Dict[str, List[SearchResult]],
        plan: QueryPlan | None = None,
    ) -> List[SearchResult]:
        operation = str(getattr(getattr(plan, "semantics", None), "operation", "lookup") or "lookup")
        weights = self._channel_weights(intent, operation=operation)
        fused: Dict[str, SearchResult] = {}
        for channel, results in channel_results.items():
            weight = self._weight_for_channel(channel, weights)
            for rank, result in enumerate(results, start=1):
                key = result.chunk_id or self._result_key(result, fallback=f"{channel}:{rank}")
                contribution = weight / (self.rag_config.rrf_k + rank)
                existing = fused.get(key)
                if existing is None:
                    existing = result
                    existing.chunk_id = key
                    existing.rrf_score = 0.0
                    existing.contributions = []
                    fused[key] = existing
                existing.rrf_score += contribution
                existing.final_score = existing.rrf_score
                existing.contributions.append(
                    RRFContribution(
                        channel=channel,
                        rank=rank,
                        weight=weight,
                        contribution=contribution,
                    )
                )
                existing.matched_terms = list(
                    dict.fromkeys(existing.matched_terms + result.matched_terms)
                )
        output = list(fused.values())
        output.sort(key=lambda item: item.rrf_score, reverse=True)
        return output[:96]

    def _channel_weights(self, intent: str, operation: str = "lookup") -> Dict[str, float]:
        if operation == "enumerate":
            # Subject/entity matches are useful for anchoring, but must not
            # dominate event recall. Source-window and semantic channels supply
            # independent candidate incidents for atomic validation.
            return {
                "exact_entity": 1.0,
                "structured": 1.0,
                "lexical": 1.0,
                "dense_original_query": 1.25,
                "dense_evidence_view": 1.2,
                "navigation": 1.45,
                "coverage_dense_navigation": 1.5,
            }
        if intent in {"entity_definition", "entity_relation"}:
            return {
                "exact_entity": 2.0,
                "structured": 1.6,
                "lexical": 1.2,
                "dense_original_query": 1.0,
                "dense_entity_alias": 0.8,
                "dense_generic_intent": 0.7,
                "dense_llm_variant": 0.7,
                "navigation": 2.2,
            }
        if intent in {"document_summary", "comparison", "procedure"}:
            return {
                "dense_original_query": 1.3,
                "structured": 1.2,
                "lexical": 0.9,
                "exact_entity": 0.6,
                "navigation": 2.0,
            }
        if intent == "location_lookup":
            return {
                "exact_entity": 1.7,
                "structured": 1.3,
                "lexical": 1.2,
                "dense_original_query": 1.0,
                "dense_generic_intent": 0.8,
                "dense_llm_variant": 0.7,
                "navigation": 1.7,
            }
        if intent == "code_question":
            return {
                "exact_entity": 2.0,
                "structured": 1.6,
                "lexical": 1.3,
                "dense_original_query": 0.8,
                "navigation": 1.7,
            }
        return {
            "exact_entity": 1.4,
            "structured": 1.2,
            "lexical": 1.1,
            "dense_original_query": 1.0,
            "dense_generic_intent": 0.8,
            "dense_llm_variant": 0.7,
            "navigation": 2.3,
        }
    def _weight_for_channel(self, channel: str, weights: Dict[str, float]) -> float:
        if channel in weights:
            return weights[channel]
        for prefix, weight in weights.items():
            if channel.startswith(prefix):
                return weight
        if "navigation" in channel:
            return weights.get("navigation", 1.9)
        if channel.startswith("dense"):
            return 0.8
        return 1.0

    def _hit_to_result(self, hit: HybridHit) -> SearchResult:
        return SearchResult(
            content=hit.content,
            metadata=dict(hit.metadata or {}),
            vector_score=0.0,
            keyword_score=hit.score,
            final_score=0.0,
            chunk_id=hit.chunk_id,
            parent_chunk_id=str(hit.metadata.get("parent_chunk_id") or ""),
            chunk_kind=str(hit.metadata.get("chunk_kind") or "child"),
            matched_terms=list(hit.matched_terms),
        )

    def _plan_focus_terms(self, plan, query: str) -> List[str]:
        if is_coverage_fallback(plan):
            return []
        terms: List[str] = []
        for entity in getattr(plan, "entities", []) or []:
            terms.extend([entity.surface, entity.canonical, *entity.aliases])
        if not terms:
            terms.extend(self._focus_terms(query))
        compact_terms = []
        for term in terms:
            compact = self._compact_text(str(term))
            if len(compact) >= 2:
                compact_terms.append(compact)
        return list(dict.fromkeys(compact_terms))[:8]

    def _anchor_documents(
        self,
        exact_results: Sequence[SearchResult],
        focus_terms: Sequence[str],
    ) -> set[str]:
        anchors: set[str] = set()
        for result in exact_results:
            if focus_terms and not self._matches_focus_terms(result, focus_terms):
                continue
            metadata = result.metadata or {}
            document_id = str(metadata.get("document_id") or "")
            file_name = str(metadata.get("file_name") or metadata.get("source_ref") or "")
            if document_id:
                anchors.add(f"document:{document_id}")
            if file_name:
                anchors.add(f"file:{file_name}")
        return anchors

    def _apply_focus_guard(
        self,
        results: Sequence[SearchResult],
        focus_terms: Sequence[str],
        anchor_documents: set[str],
    ) -> List[SearchResult]:
        if not results or not focus_terms:
            return list(results)

        anchored: List[SearchResult] = []
        focused: List[SearchResult] = []
        fallback: List[SearchResult] = []
        for result in results:
            if anchor_documents and self._result_in_anchor(result, anchor_documents):
                anchored.append(result)
            elif self._matches_focus_terms(result, focus_terms):
                focused.append(result)
            else:
                fallback.append(result)

        guarded = anchored + focused
        if guarded:
            # Keep a small tail only for context diversity, never enough to dominate evidence.
            tail_limit = min(3, max(0, self.rag_config.top_k // 2))
            return guarded + fallback[:tail_limit]
        return list(results)

    def _result_in_anchor(self, result: SearchResult, anchor_documents: set[str]) -> bool:
        metadata = result.metadata or {}
        document_id = str(metadata.get("document_id") or "")
        file_name = str(metadata.get("file_name") or metadata.get("source_ref") or "")
        return f"document:{document_id}" in anchor_documents or f"file:{file_name}" in anchor_documents

    def _matches_focus_terms(self, result: SearchResult, focus_terms: Sequence[str]) -> bool:
        metadata = result.metadata or {}
        source = " ".join(
            [
                str(metadata.get("file_name") or metadata.get("source_ref") or ""),
                str(metadata.get("document_title") or ""),
                str(metadata.get("section_path") or ""),
                result.content or "",
            ]
        )
        normalized_source = self._compact_text(source)
        return any(term and term in normalized_source for term in focus_terms)

    def _apply_evidence_requirements(
        self,
        results: Sequence[SearchResult],
        plan,
        focus_terms: Sequence[str],
    ) -> List[SearchResult]:
        requirements = list(dict.fromkeys(getattr(plan, "required_evidence", []) or []))
        if not results or not requirements:
            return list(results)

        scored: List[tuple[float, SearchResult]] = []
        for result in results:
            covered = [
                requirement
                for requirement in requirements
                if self._requirement_matches(requirement, result, focus_terms)
            ]
            coverage = len(covered) / max(len(requirements), 1)
            result.diagnostics["evidence_requirement_score"] = round(coverage, 4)
            result.diagnostics["covered_evidence_requirements"] = covered
            result.diagnostics["missing_evidence_requirements"] = [
                requirement for requirement in requirements if requirement not in covered
            ]

            # Requirement coverage is a rerank prior, not a hard gate. We keep
            # RRF/reranker order as the backbone and only lift evidence-rich
            # candidates above weak semantic neighbors.
            existing_score = result.rerank_score if result.rerank_score is not None else result.rrf_score
            focus_bonus = 0.08 if self._matches_focus_terms(result, focus_terms) else 0.0
            scored.append((coverage + focus_bonus + min(float(existing_score or 0.0), 1.0) * 0.05, result))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored]

    def _rank_relation_evidence(
        self,
        plan,
        results: Sequence[SearchResult],
        focus_terms: Sequence[str],
    ) -> List[SearchResult]:
        if not results:
            return []
        scored: List[tuple[float, SearchResult]] = []
        subject_aliases = self._subject_aliases(plan, focus_terms)
        for result in results:
            info = self._relation_evidence_info(result, subject_aliases, getattr(plan, "relation_type", ""))
            result.diagnostics["relation_evidence"] = info
            base_score = result.rerank_score if result.rerank_score is not None else result.rrf_score
            score = float(info["score"]) + min(float(base_score or 0.0), 1.0) * 0.08
            scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored]

    def _subject_aliases(self, plan, focus_terms: Sequence[str]) -> List[str]:
        aliases: List[str] = []
        for entity in getattr(plan, "entities", []) or []:
            aliases.extend([entity.surface, entity.canonical, *entity.aliases])
        aliases.extend(focus_terms)
        output = []
        for alias in aliases:
            compact = self._compact_text(str(alias))
            if len(compact) >= 2:
                output.append(compact)
        return list(dict.fromkeys(output))

    def _relation_evidence_info(
        self,
        result: SearchResult,
        subject_aliases: Sequence[str],
        relation_type: str,
    ) -> Dict[str, object]:
        text = result.content or ""
        metadata = result.metadata or {}
        normalized = self._compact_text(" ".join([text, str(metadata.get("section_path") or "")]))
        subject_matches = [alias for alias in subject_aliases if alias and alias in normalized]
        candidates = self._extract_candidate_people(text, subject_aliases)
        relation_signal = self._relation_signal_score(text, relation_type)
        source_score, source_type = self._relation_source_score(result)
        has_candidate = bool(candidates)
        complete = bool(subject_matches and has_candidate and relation_signal > 0 and source_score > 0)
        score = 0.0
        if subject_matches:
            score += 0.34
        if has_candidate:
            score += 0.26
        score += min(relation_signal, 0.24)
        score += source_score
        if source_type == "supplemental":
            score *= 0.25
        if not complete:
            score *= 0.55
        return {
            "subject_entity": subject_matches[0] if subject_matches else "",
            "candidate_people": candidates[:8],
            "relation_type": relation_type or "general_interaction",
            "relation_signal_score": round(relation_signal, 4),
            "source_type": source_type,
            "score": round(min(score, 1.0), 4),
            "judge_passed": complete and source_score > 0,
            "judge_reason": self._relation_judge_reason(subject_matches, candidates, relation_signal, source_type),
        }

    def _extract_candidate_people(self, text: str, subject_aliases: Sequence[str]) -> List[str]:
        candidates: List[str] = []
        compact_subjects = {normalize_entity_name(alias) for alias in subject_aliases if alias}
        relation_name_patterns = [
            r"(?:和|与|跟|同|陪|带着|帮助|遇见|碰见|叫住|问|对|向)([\u4e00-\u9fff]{2,4}?(?:老师|先生|太太|小姐|姨妈|叔叔)?)(?=一起|一同|去了|去|来|说|问|回答|交谈|争执|争吵|，|。|、|\s|$)",
            r"([\u4e00-\u9fff]{2,5})(?:这会儿|说|问|回答|喊|叫|哭|笑|争吵|争执|一起|一同|也)",
            r"[“\"']([\u4e00-\u9fff]{2,5})[”\"']",
        ]
        for pattern in relation_name_patterns:
            candidates.extend(re.findall(pattern, text or ""))
        candidates.extend(re.findall(r"[\u4e00-\u9fff]{1,6}[·・•][\u4e00-\u9fff]{1,4}", text or ""))
        candidates.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text or ""))
        filtered: List[str] = []
        for candidate in candidates:
            clean = candidate.strip(" \t\r\n，。！？；：、“”‘’（）()[]【】")
            if any(separator in clean for separator in ("和", "与", "跟", "同")):
                continue
            if not self._looks_like_person_name(clean):
                continue
            normalized = normalize_entity_name(clean)
            if len(normalized) < 2 or normalized in compact_subjects:
                continue
            if any(subject and subject in normalized for subject in compact_subjects):
                continue
            if self._is_relation_candidate_noise(clean):
                continue
            filtered.append(clean)
        return list(dict.fromkeys(filtered))

    def _is_relation_candidate_noise(self, text: str) -> bool:
        return re.search(
            r"^(我|你|他|她|它|我们|你们|他们|她们|这个|那个|哪些|什么|两人|大人|小孩|孩子|读者)$|"
            r"(关系|人物|正文|片段|来源|文档|标题|作者|出版|目录|序言|前言|说明|原型|故事|章节|学校里|学校|孩子们|听众|经历|证据|一起|交谈|帮助|处理|争执|硬咽|小声|俏声|老油子|机会|国外|地方|游历|嘲讽|演习|识别|unknown|chapter|document|source|author|publisher)",
            text,
            re.I,
        ) is not None

    def _looks_like_person_name(self, text: str) -> bool:
        clean = (text or "").strip()
        if not clean:
            return False
        if re.search(r"[·・•]", clean):
            return True
        if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}", clean):
            return True
        if re.search(r"(老师|先生|太太|小姐|姨妈|叔叔|阿[\u4e00-\u9fff]|小[\u4e00-\u9fff]|老[\u4e00-\u9fff])", clean):
            return True
        if re.search(
            r"(我|你|他|她|它|咱|再|想|可|给|往|在|的|得|地|很|可能|认为|指出|成为|成熟|读者|成人|少年|年人|站到|体面|拼写|竞赛|语气|停顿|作文|友谊|回忆|往昔|缎带|演习|上场|宗教|梦幻|伤感|儿女|心愿|猪|睡|刚才|抽咽|小声|俏声|机会|国外|地方|游历|嘲讽|说明|理由|盘子|杯子|帽|插着|穿|靴|裤|衣)",
            clean,
        ):
            return False
        return bool(re.fullmatch(r"[\u4e00-\u9fff]{3,5}", clean))

    def _relation_signal_score(self, text: str, relation_type: str) -> float:
        patterns = [
            r"(认识|相识|来往|互动|交谈|说话|对话|一起|一同|共同|陪同|帮助|遇见|碰见|跟着|带着|结伴|冒险|同学|朋友|伙伴|家人|亲人|父亲|母亲|兄弟|姐妹|姨妈|老师|法官|对手|冲突|争执|打架|争吵)",
            r"(friend|companion|classmate|family|relative|mother|father|aunt|uncle|brother|sister|talked|spoke|met|went with|together|helped|argued|conflict|opponent)",
        ]
        score = 0.0
        for pattern in patterns:
            if re.search(pattern, text or "", flags=re.I):
                score += 0.12
        typed_patterns = {
            "family": r"(家人|亲人|父亲|母亲|兄弟|姐妹|姨妈|叔叔|family|relative|mother|father|aunt|uncle|brother|sister)",
            "conflict": r"(冲突|争执|争吵|打架|对手|敌人|conflict|argued|opponent|enemy|rival)",
            "friend": r"(朋友|伙伴|同伴|同学|friend|companion|classmate)",
        }
        typed = typed_patterns.get(relation_type)
        if typed and re.search(typed, text or "", flags=re.I):
            score += 0.12
        return min(score, 0.28)

    def _relation_source_score(self, result: SearchResult) -> tuple[float, str]:
        metadata = result.metadata or {}
        chunk_kind = str(metadata.get("chunk_kind") or result.chunk_kind or "")
        text = result.content or ""
        source_type = "body" if chunk_kind in {"child", "parent"} else chunk_kind or "unknown"
        if self._looks_like_front_matter(text) or re.search(r"(人物原型|原型说明|附录|序言|前言|译者|评论|目录|读者认为|作品|本书|编者|文库|copyright|preface|appendix)", text, re.I):
            return 0.0, "supplemental"
        if chunk_kind in {"child", "parent"}:
            return 0.16, source_type
        if chunk_kind == "section_card":
            return 0.06, source_type
        return 0.02, source_type

    def _relation_judge_reason(
        self,
        subject_matches: Sequence[str],
        candidates: Sequence[str],
        relation_signal: float,
        source_type: str,
    ) -> str:
        missing = []
        if not subject_matches:
            missing.append("missing_subject")
        if not candidates:
            missing.append("missing_candidate_person")
        if relation_signal <= 0:
            missing.append("missing_relation_or_interaction_signal")
        if source_type == "supplemental":
            missing.append("supplemental_source_not_core_body")
        return "passed" if not missing else ",".join(missing)

    def _requirement_matches(
        self,
        requirement: str,
        result: SearchResult,
        focus_terms: Sequence[str],
    ) -> bool:
        text = result.content or ""
        metadata = result.metadata or {}
        searchable = " ".join(
            [
                str(metadata.get("file_name") or metadata.get("source_ref") or ""),
                str(metadata.get("document_title") or ""),
                str(metadata.get("section_path") or ""),
                str(metadata.get("chunk_kind") or result.chunk_kind or ""),
                text,
            ]
        )
        normalized = self._compact_text(searchable)
        lowered = searchable.casefold()

        if requirement == "target_entity":
            return not focus_terms or any(term and term in normalized for term in focus_terms)
        if requirement == "identity_statement":
            return self._has_identity_statement(searchable, focus_terms)
        if requirement == "related_entity":
            return self._has_related_entity(searchable, focus_terms)
        if requirement == "relation_statement":
            return self._has_relation_statement(searchable)
        if requirement == "numeric_value":
            return self._has_numeric_value(searchable)
        if requirement == "unit_or_context":
            return self._has_unit_or_context(searchable)
        if requirement == "location_name":
            return self._location_evidence_count(normalized) > 0
        if requirement == "movement_or_location_context":
            return self._has_location_context(searchable)
        if requirement == "source_metadata":
            return str(metadata.get("chunk_kind") or result.chunk_kind) in {"document_card", "metadata"} or self._looks_like_front_matter(searchable)
        if requirement == "author_or_publisher":
            return re.search(r"(作者|著|译|出版社|出版|发布|author|publisher|issued by|translated by)", lowered) is not None
        if requirement == "code_symbol":
            return re.search(r"(\b[A-Za-z_][A-Za-z0-9_]*\(|\bclass\s+\w+|\bdef\s+\w+|api|参数|函数|类)", searchable, re.I) is not None
        if requirement == "definition_or_callsite":
            return re.search(r"(def |class |function|调用|定义|参数|returns?|raises?)", searchable, re.I) is not None
        if requirement == "clause_reference":
            return re.search(r"(第\s*[一二三四五六七八九十百\d]+\s*条|\b\d+(?:\.\d+)+\b|clause|section)", searchable, re.I) is not None
        if requirement == "clause_text":
            return len(text.strip()) >= 60
        if requirement == "table_cell_or_row":
            return str(metadata.get("chunk_kind") or result.chunk_kind) == "table_card" or "|" in text or "\t" in text
        if requirement == "table_header_context":
            return re.search(r"(列名|表头|字段|header|column|row)", lowered) is not None or "|" in text
        if requirement == "answer_statement":
            return len(text.strip()) >= 40
        return False

    def build_context(
        self,
        results: Sequence[SearchResult],
        plan: QueryPlan | None = None,
        query: str = "",
    ) -> str:
        """Assemble citation-addressable evidence for answer generation.

        Enumerative plans prefer judge-validated atomic claims. If the judge
        yields none despite source-window recall, the context falls back to
        raw source candidates rather than presenting a false empty retrieval.
        """

        blocks: List[str] = []
        total_chars = 0
        max_chars = max(1200, int(getattr(self.rag_config, "max_context_chars", 5000) or 5000))
        per_source_limit = max(
            520,
            min(
                int(getattr(self.rag_config, "evidence_excerpt_chars", 1000) or 1000),
                max_chars // max(min(len(results), 4), 1),
            ),
        )
        parent_ids = [result.parent_chunk_id for result in results if result.parent_chunk_id]
        parent_records = self.hybrid_index.get_records(parent_ids)
        enumeration_mode = self._is_enumeration_plan(plan)

        for index, result in enumerate(results, start=1):
            metadata = result.metadata or {}
            file_name = str(metadata.get("file_name") or metadata.get("source_ref") or "unknown")
            page = metadata.get("page_start") or metadata.get("page")
            row_index = metadata.get("row_index")
            section = str(metadata.get("section_path") or "").strip()
            source_text = result.content or ""
            if result.parent_chunk_id and result.parent_chunk_id in parent_records:
                parent = parent_records[result.parent_chunk_id]
                if parent.content:
                    source_text = parent.content

            excerpt = self._extract_evidence_excerpt(
                source_text=source_text,
                retrieved_text=result.content or "",
                query=query,
                plan=plan,
                max_chars=per_source_limit,
                result_metadata=metadata,
            )
            if not excerpt:
                continue

            location_parts: List[str] = []
            if page is not None:
                location_parts.append(f"页码 {page}")
            if row_index is not None:
                location_parts.append(f"行 {row_index}")
            location = " / ".join(location_parts) if location_parts else "定位未知"
            section_hint = f" | 章节: {section}" if section else ""
            header = f"[S{index}] 来源: {file_name} | {location}{section_hint}"

            if enumeration_mode:
                atomic = self._format_atomic_claims(result, source_text, plan=plan)
                if not atomic:
                    if not bool((result.diagnostics or {}).get("enumeration_source_fallback")):
                        continue
                    block = f"{header}\n[源文候选 | 原子验证未形成]\n{excerpt}"
                else:
                    block = f"{header}\n[已验证原子事实]\n{atomic}"
            else:
                block = f"{header}\n{excerpt}"

            if total_chars + len(block) > max_chars:
                remaining = max_chars - total_chars
                if remaining >= 160:
                    blocks.append(block[:remaining].rstrip() + "…")
                break
            blocks.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(blocks).strip()

    def _format_atomic_claims(self, result: SearchResult, source_text: str, plan: QueryPlan | None = None) -> str:
        claims = self._qualified_claims(result, plan=plan)
        if not claims:
            return ""
        lines: List[str] = []
        for claim in claims[:4]:
            statement = str(getattr(claim, "statement", "") or "").strip()
            if not statement:
                continue
            classification = str(getattr(claim, "classification", "") or "").strip()
            label = f"（{classification}）" if classification else ""
            lines.append(f"- {statement}{label}")
            cited = self._grounded_claim_excerpt(
                str(getattr(claim, "source_excerpt", "") or ""),
                source_text,
            )
            if cited:
                lines.append(f"  原文依据：{cited}")
        return "\n".join(lines)

    def _grounded_claim_excerpt(self, candidate: str, source_text: str) -> str:
        value = (candidate or "").strip()
        if len(value) < 12:
            return ""
        compact_value = "".join(value.casefold().split())
        compact_source = "".join((source_text or "").casefold().split())
        if compact_value and compact_value in compact_source:
            return value[:360]
        return ""
    def _extract_evidence_excerpt(
        self,
        source_text: str,
        retrieved_text: str,
        query: str,
        plan: QueryPlan | None,
        max_chars: int,
        result_metadata: Dict[str, object] | None = None,
    ) -> str:
        """Select source-backed paragraphs rather than blindly taking a parent."""

        text = (source_text or retrieved_text or "").strip()
        if not text:
            return ""
        max_chars = max(200, int(max_chars))
        if len(text) <= max_chars:
            return text

        metadata = result_metadata or {}
        kind = str(metadata.get("chunk_kind") or "")
        if kind in {"navigation_window", "terminal_window"}:
            region = str(metadata.get("navigation_region") or "middle")
            return self._source_window_excerpt(text, max_chars=max_chars, region=region)

        focus_terms: List[str] = []
        if plan is not None:
            focus_terms.extend(self._plan_focus_terms(plan, query))
        if not focus_terms:
            focus_terms.extend(self._focus_terms(query))
        # Retrieved children are already retrieval-grounded.  Their prefix acts
        # as an anchor when parent text has little literal query overlap.
        anchor = self._compact_text((retrieved_text or "")[:180])
        compact_terms = [self._compact_text(term) for term in focus_terms if self._compact_text(term)]
        compact_terms = list(dict.fromkeys(compact_terms))[:10]

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n|\n+", text) if part.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [
                part.strip()
                for part in re.split(r"(?<=[。！？!?；;])\s+|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])", text)
                if part.strip()
            ]
        if not paragraphs:
            return text[:max_chars].rstrip() + "…"

        query_tokens = [
            token for token in self._tokenize(query) if len(self._compact_text(token)) >= 2
        ][:12]
        scored: List[tuple[float, int, str]] = []
        for idx, paragraph in enumerate(paragraphs):
            compact = self._compact_text(paragraph)
            score = 0.0
            score += sum(3.0 for term in compact_terms if term and term in compact)
            score += sum(0.6 for token in query_tokens if self._compact_text(token) in compact)
            if anchor and anchor[: min(36, len(anchor))] in compact:
                score += 4.0
            # Prefer complete factual spans over headings / fragments.
            if len(paragraph) >= 40:
                score += 0.15
            scored.append((score, idx, paragraph))

        selected_ids: List[int] = []
        for score, idx, _ in sorted(scored, key=lambda item: (-item[0], item[1])):
            if score <= 0 and selected_ids:
                break
            if idx not in selected_ids:
                selected_ids.append(idx)
            if len(selected_ids) >= 2:
                break
        if not selected_ids:
            selected_ids = [0]

        # Preserve original source order to avoid joining statements into a
        # misleading timeline.
        selected_ids.sort()
        chosen = "\n".join(paragraphs[idx] for idx in selected_ids)
        if len(chosen) <= max_chars:
            return chosen

        # When the supporting paragraph itself is long, retain the leading and
        # trailing portions rather than cutting a number, code path, or citation
        # context from one side only.
        head = max_chars * 3 // 4
        tail = max_chars - head - 1
        if tail >= 80:
            return chosen[:head].rstrip() + "…" + chosen[-tail:].lstrip()
        return chosen[:max_chars].rstrip() + "…"

    def _source_window_excerpt(self, text: str, max_chars: int, region: str) -> str:
        """Preserve the source span selected by topology-aware retrieval.

        Terminal windows retain their closing source text; other windows retain
        both boundaries. This is structural handling, not a question-specific
        interpretation of the content.
        """

        if len(text) <= max_chars:
            return text
        if region == "terminal":
            return text[-max_chars:].lstrip()
        head = max_chars * 2 // 3
        tail = max_chars - head - 1
        return text[:head].rstrip() + "…" + text[-tail:].lstrip()

    def build_messages(
        self,
        query: str,
        history_messages: Sequence[Dict[str, str]],
        context: str,
        plan: QueryPlan | None = None,
    ) -> List[Dict[str, str]]:
        """Build a compact evidence-first answer prompt."""

        system_prompt = (
            "你是多语言企业知识库问答助手。只能使用 [证据] 中的内容，不得调用外部知识。"
            "默认用用户提问语言回答；数字、日期、代码、路径、版本号和专有名词必须与证据一致。"
            "每一项可验证结论的句末必须标注对应的 [S编号]；不要虚构来源或把改写当作原文引号。"

            "\n\n回答前，必须先进行证据约束的语义对齐："
            "\n1. 不得因为用户问题中的词语没有在证据中逐字出现，就直接判定证据不足。"
            "\n2. 将原文的具体事实映射到问题所要求的事件、行为、关系、属性、原因、结果或过程；"
            "但必须保留原文可验证的主体、行为和因果方向。"
            "\n3. 不得把他人的行为、人物遭遇、偶然结果或没有明确责任归属的事件，"
            "自动归因于某个主体的过错、动机、品质或责任。"
            "\n4. 若仅能确认部分内容，先给出已确认部分，不要把部分可答误判为完全不可答。"
            "\n5. 证据不足时，明确说明缺少什么事实；"
            "不得把未覆盖来源范围的缺失误写成高置信结论。"
            "\n6. 在枚举任务中，结论、证据、限制、反例和举例中出现的每个专有名词都必须"
            "出现在对应 [S编号] 的[已验证原子事实]或其原文依据中；不得从记忆、"
            "常识、候选窗口或未验证片段补充任何名字。"
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        max_history = max(
            0,
            int(getattr(self.rag_config, "answer_history_messages", 4) or 4),
        )
        # Previous assistant messages are untrusted generated text.  Feeding
        # them back into a new grounded answer can repeat greetings, stale
        # conclusions, or unsupported names.  Preserve prior user turns only;
        # current retrieval remains the exclusive factual context.
        prior_users = [
            message
            for message in list(history_messages)
            if message.get("role") == "user" and str(message.get("content", "") or "").strip()
        ]
        for message in prior_users[-max_history:]:
            messages.append({"role": "user", "content": str(message.get("content", "") or "")})

        semantics = getattr(plan, "semantics", None)
        operation = str(getattr(semantics, "operation", "lookup") or "lookup")
        answer_mode = str(getattr(semantics, "answer_mode", "direct") or "direct")
        contract = getattr(semantics, "evidence_contract", None)

        task_hint = (
            "从证据中提取与问题语义相符的事实；"
            "允许对原文具体事实做受控概括，但不得补充未出现的事实"
        )
        if operation == "enumerate":
            unit = str(getattr(contract, "answer_unit", "fact") or "fact")
            task_hint = (
                "这是枚举任务。只列出 [已验证原子事实] 中已经通过证据判别的独立项目；"
                "不要自行把未标为已验证的原文窗口、失败经历、共现信息或结果补充进名单。"
                f"当前答案单元类型为：{unit}。"
                "每一项都应简洁说明原文事实与其受控归类，并保留对应 [S编号]。"
                "不要把不同性质的关系强行合并为同一标签，也不要把用户要求的关系类别"
                "替换为更窄或更宽的类别；应按已验证原子事实中的关系依据表述。"
                "若存在多个项目，逐项列出；若无法确认完整名单，说明“现有证据可确认的包括”。"
            )
        elif operation in {"compare", "summarize", "trace", "verify"}:
            task_hint = (
                "综合多个来源时清楚区分各来源；"
                "允许对证据中的具体事实做受控概括，但不得补充未出现的事实"
            )

        if answer_mode == "inferred":
            task_hint += "；推断必须标注“谨慎推断”并同时引用至少两条独立证据"

        user_prompt = (
            f"[证据]\n{context if context else '本轮检索未形成可用于回答的已验证证据；这不等于文档中不存在相关事实。'}\n\n"
            f"[问题]\n{query}\n\n"
            f"[任务]\n{task_hint}\n\n"
            "请采用以下简洁格式：\n"
            "结论：\n"
            "- 先直接回答；列举类问题优先写“现有证据可确认的包括：”。\n"
            "证据：\n"
            "- 对每个结论项目用一句话说明对应原文依据 [S编号]。\n"
            "限制：\n"
            "- 仅在确有不足、冲突、覆盖不完整或无法确认完整名单时写出。"
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages
    def answer_stream(
        self,
        query: str,
        history_messages: Sequence[Dict[str, str]],
        top_k: int | None = None,
        keyword_filter: str = "",
    ) -> Tuple[List[SearchResult], Generator[str, None, None]]:
        """Retrieve first, then return the streaming answer generator."""

        results = self.search(query=query, top_k=top_k, keyword_filter=keyword_filter)
        plan = self.last_diagnostics.query_plan
        context = self.build_context(results, plan=plan, query=query)
        messages = self.build_messages(query=query, history_messages=history_messages, context=context, plan=plan)
        return results, self.client.stream_chat(messages)

    def _distance_to_score(self, distance: float) -> float:
        if distance is None:
            return 0.0
        try:
            return max(0.0, 1.0 - float(distance))
        except Exception:
            return 0.0

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in tokenize_for_search(text) if token not in self._stopwords()]

    def _stopwords(self) -> set[str]:
        return {
            "请问",
            "一下",
            "这个",
            "那个",
            "什么",
            "如何",
            "怎么",
            "为什么",
            "多少",
            "哪些",
            "哪里",
            "哪儿",
            "地方",
            "去过",
            "以及",
            "the",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
        }

    def _generic_phrase_tokens(self, cjk_text: str) -> List[str]:
        """Legacy hook kept for compatibility; generic tokenization lives in multilingual.py."""

        return []

    def _build_search_queries(self, query: str) -> List[str]:
        """构建多路召回查询：原问题 + 通用本地扩展 + 可选 LLM 改写。"""

        queries = [query]
        expanded_query = self._expand_query(query)
        if expanded_query and expanded_query != query:
            queries.append(expanded_query)
        queries.extend(self._llm_rewrite_queries(query))
        return self._dedupe_queries(queries)

    def _expand_query(self, query: str) -> str:
        """Generate surface-form variants only; intent expansion belongs to QueryPlan."""

        additions: List[str] = []
        additions.extend(self._surface_form_variants(query))
        if not additions:
            return query
        return f"{query} {' '.join(dict.fromkeys(additions))}"

    def _surface_form_variants(self, query: str) -> List[str]:
        """生成用户原词的表层变体，例如去掉中点、空格、书名号。"""

        variants: List[str] = []
        text = (query or "").strip()
        compact = re.sub(r"[\s·・•《》「」『』“”\"'：:，,。？?！!、\(\)（）\[\]【】]+", "", text)
        if len(compact) >= 2 and compact != text:
            variants.append(compact)
        middle_dot_terms = re.findall(r"[\u4e00-\u9fffA-Za-z]+[·・•][\u4e00-\u9fffA-Za-z·・•]+", text)
        for term in middle_dot_terms:
            variants.append(re.sub(r"[·・•]", "", term))
            variants.append(re.sub(r"[·・•]", " ", term))
        return variants

    def _llm_rewrite_queries(self, query: str) -> List[str]:
        if not self.rag_config.enable_query_rewrite or self.rag_config.query_rewrite_count <= 0:
            return []

        prompt = (
            "你是企业知识库 RAG 检索查询改写器。"
            "请把用户问题改写成更适合全文检索和向量检索的查询。"
            "要求：只基于用户问题本身，不编造具体答案，不添加用户没有提到的专有名词。"
            f"最多输出 {self.rag_config.query_rewrite_count} 行，每行一条查询，不要编号。"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        raw_text = self.client.complete_chat(messages, max_tokens=160)
        rewrites: List[str] = []
        for line in (raw_text or "").splitlines():
            cleaned = re.sub(r"^\s*[-*\d.、)）]+", "", line).strip()
            if cleaned:
                rewrites.append(cleaned)
            if len(rewrites) >= self.rag_config.query_rewrite_count:
                break
        return rewrites

    def _dedupe_queries(self, queries: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        output: List[str] = []
        for query in queries:
            cleaned = re.sub(r"\s+", " ", (query or "").strip())
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            output.append(cleaned)
        return output[: max(1, 2 + self.rag_config.query_rewrite_count)]

    def _keyword_overlap_score(self, text: str, query_tokens: Sequence[str]) -> float:
        if not text or not query_tokens:
            return 0.0
        document_tokens = set(self._tokenize(text))
        if not document_tokens:
            return 0.0
        query_set = set(query_tokens)
        return len(query_set & document_tokens) / max(len(query_set), 1)

    def _bm25_candidates(self, query_tokens: Sequence[str], limit: int) -> List[SearchResult]:
        if not query_tokens:
            return []

        index = self._get_lexical_index()
        if not index.documents:
            return []

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        raw_scores: List[Tuple[float, LexicalDocument]] = []
        for document in index.documents:
            score = self._bm25_score(unique_query_tokens, document, index)
            if score > 0:
                raw_scores.append((score, document))

        raw_scores.sort(key=lambda item: item[0], reverse=True)
        top_scores = raw_scores[:limit]
        max_score = top_scores[0][0] if top_scores else 0.0
        candidates: List[SearchResult] = []
        for score, document in top_scores:
            keyword_score = score / max_score if max_score > 0 else 0.0
            final_score = self.rag_config.keyword_weight * keyword_score
            candidates.append(
                SearchResult(
                    content=document.content,
                    metadata=dict(document.metadata or {}),
                    vector_score=0.0,
                    keyword_score=keyword_score,
                    final_score=final_score,
                )
            )
        return candidates

    def _hybrid_score(
        self,
        result: SearchResult,
        query: str,
        query_tokens: Sequence[str],
        vector_rank: int | None,
        keyword_rank: int | None,
    ) -> float:
        evidence_score = self._evidence_score(query=query, text=result.content, query_tokens=query_tokens)
        rrf_score = self._rrf_score(vector_rank) + self._rrf_score(keyword_rank)
        weighted_score = (
            0.42 * result.vector_score
            + 0.34 * result.keyword_score
            + 0.18 * evidence_score
            + 0.06 * min(rrf_score * 30.0, 1.0)
        )
        return max(0.0, min(1.0, weighted_score * self._content_quality_multiplier(query, result)))

    def _rrf_score(self, rank: int | None, k: int = 60) -> float:
        if not rank or rank <= 0:
            return 0.0
        return 1.0 / (k + rank)

    def _best_rank(self, ranks: Dict[str, int], prefix: str) -> int | None:
        values = [value for key, value in ranks.items() if key.startswith(prefix) and value > 0]
        return min(values) if values else None

    def _evidence_score(self, query: str, text: str, query_tokens: Sequence[str]) -> float:
        if not text:
            return 0.0
        return max(0.0, min(self._keyword_overlap_score(text, query_tokens), 1.0))

    def _content_quality_multiplier(self, query: str, result: SearchResult) -> float:
        text = result.content or ""
        metadata = result.metadata or {}
        file_name = str(metadata.get("file_name") or metadata.get("source_ref") or "")
        page = metadata.get("page")
        multiplier = 1.0
        normalized_source = self._compact_text(f"{file_name} {text}")

        focus_terms = self._focus_terms(query)
        if focus_terms:
            if any(term in normalized_source for term in focus_terms):
                multiplier *= 1.12
            else:
                multiplier *= 0.28

        if len(text.strip()) < 80:
            multiplier *= 0.65

        front_matter_patterns = (
            "图书在版",
            "CIP",
            "ISBN",
            "责任编辑",
            "出版社",
            "作者介绍",
            "目录",
            "目次",
            "CONTENTS",
            "版权",
            "前言",
            "序言",
            "附录",
            "编者",
            "本书人物原型",
        )
        is_front_page = False
        try:
            is_front_page = page is not None and int(page) <= 12
        except Exception:
            is_front_page = False

        bibliographic_query = re.search(r"(作者|出版社|出版|ISBN|目录|前言|序|附录|编者)", query or "") is not None
        if not bibliographic_query and any(pattern.lower() in text.lower() for pattern in front_matter_patterns):
            multiplier *= 0.45 if is_front_page else 0.7
        elif not bibliographic_query and is_front_page:
            multiplier *= 0.75

        if re.search(r"第\s*[一二三四五六七八九十百\d]+\s*[章节回]|chapter", text, flags=re.IGNORECASE):
            multiplier *= 1.04

        return multiplier

    def _focus_terms(self, query: str) -> List[str]:
        """Extract focus terms with the shared multilingual query analyzer."""

        return [self._compact_text(term) for term in query_focus_terms(query) if self._compact_text(term)][:4]

    def _matches_focus(self, result: SearchResult, focus_terms: Sequence[str]) -> bool:
        metadata = result.metadata or {}
        source = f"{metadata.get('file_name') or metadata.get('source_ref') or ''} {result.content or ''}"
        normalized_source = self._compact_text(source)
        return any(term in normalized_source for term in focus_terms)

    def _compact_text(self, text: str) -> str:
        return re.sub(r"[\s·・•《》「」『』“”\"'：:，,。？?！!、\(\)（）\[\]【】\-_—.]+", "", text or "")

    def _has_identity_statement(self, text: str, focus_terms: Sequence[str]) -> bool:
        normalized = self._compact_text(text)
        has_focus = not focus_terms or any(term in normalized for term in focus_terms)
        if not has_focus:
            return False
        return re.search(
            r"(是|为|叫|名叫|称为|被称为|指的是|属于|protagonist|character|is a|was a|known as|refers to)",
            text,
            re.I,
        ) is not None

    def _has_related_entity(self, text: str, focus_terms: Sequence[str]) -> bool:
        normalized = self._compact_text(text)
        if focus_terms and not any(term in normalized for term in focus_terms):
            return False
        cjk_names = re.findall(r"[\u4e00-\u9fff]{2,8}(?:·[\u4e00-\u9fff]{1,8})?", text or "")
        latin_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text or "")
        candidate_count = len(set(cjk_names + latin_names))
        return candidate_count >= 2

    def _has_relation_statement(self, text: str) -> bool:
        return re.search(
            r"(朋友|伙伴|同伴|父亲|母亲|兄弟|姐妹|同学|同事|关系|一起|共同|陪同|friend|companion|partner|father|mother|relationship|with)",
            text,
            re.I,
        ) is not None

    def _has_numeric_value(self, text: str) -> bool:
        return re.search(
            r"(\b\d+(?:[.,]\d+)?\b|[一二三四五六七八九十百千万两]+)(?:\s*(?:岁|年|月|日|%|％|元|美元|公里|千米|米|kg|g|mb|gb|token|页|章|次))?",
            text,
            re.I,
        ) is not None

    def _has_unit_or_context(self, text: str) -> bool:
        return re.search(
            r"(岁|年龄|年纪|年|月|日|%|％|元|美元|公里|千米|米|数量|金额|比例|percent|percentage|age|years?|amount|count|date)",
            text,
            re.I,
        ) is not None

    def _has_location_context(self, text: str) -> bool:
        normalized = self._compact_text(text)
        if self._location_evidence_count(normalized) > 0:
            return True
        return re.search(
            r"(在|到|去|来|抵达|前往|离开|旅行|游历|位于|发生于|where|located|arrived|travel|visited|went to)",
            text,
            re.I,
        ) is not None

    def _looks_like_front_matter(self, text: str) -> bool:
        return re.search(
            r"(图书在版|CIP|ISBN|版权|出版社|出版|作者|译者|发布方|目录|前言|序言|metadata|copyright|publisher|author)",
            text,
            re.I,
        ) is not None

    def _location_evidence_count(self, normalized_text: str) -> int:
        """通用地点证据：旅行动词 + 地理/行政区划形态，不绑定具体语料。"""

        count = 0
        if re.search(r"(去|到|来到|抵达|前往|驶向|进入|离开|旅行|游历|周游|访问|经过)", normalized_text):
            count += 1
        geo_patterns = (
            r"[\u4e00-\u9fff]{1,8}(国|州|省|市|县|镇|村|岛|港|湾|海|洋|河|湖|山|谷|洞|城|堡|宫|馆|营|要塞|战壕|工场)",
            r"(国家|地区|城市|小镇|村庄|岛屿|港口|海岸|海峡|大陆|山洞|月亮|月球|太阳|地下|海底)",
            r"[A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,}){0,3}",
        )
        for pattern in geo_patterns:
            matches = re.findall(pattern, normalized_text)
            if matches:
                count += min(len(matches), 4)
        return count

    def _diversify_results(self, results: Sequence[SearchResult], top_k: int) -> List[SearchResult]:
        selected: List[SearchResult] = []
        seen_locations: set[tuple[str, object]] = set()

        for result in results:
            metadata = result.metadata or {}
            location_key = (
                str(metadata.get("file_name") or metadata.get("source_ref") or ""),
                metadata.get("page", metadata.get("row_index")),
            )
            if location_key in seen_locations:
                continue
            if any(self._too_similar(result.content, item.content) for item in selected):
                continue
            selected.append(result)
            seen_locations.add(location_key)
            if len(selected) >= top_k:
                return selected

        for result in results:
            if result not in selected:
                selected.append(result)
            if len(selected) >= top_k:
                break
        return selected

    def _too_similar(self, left: str, right: str) -> bool:
        left_tokens = set(self._tokenize((left or "")[:500]))
        right_tokens = set(self._tokenize((right or "")[:500]))
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens) / max(min(len(left_tokens), len(right_tokens)), 1)
        return overlap >= 0.82

    def _get_lexical_index(self) -> LexicalIndex:
        current_count = self._safe_collection_count()
        if self._lexical_index and self._lexical_index.collection_count == current_count:
            return self._lexical_index

        try:
            raw = self.collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("词法索引构建失败: %s", exc)
            self._lexical_index = LexicalIndex([], {}, 0.0, current_count)
            return self._lexical_index

        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        lexical_documents: List[LexicalDocument] = []
        document_frequency: Dict[str, int] = {}
        total_length = 0
        for content, metadata in zip(documents, metadatas):
            tokens = self._tokenize(content)
            if not tokens:
                continue
            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            for token in term_freq:
                document_frequency[token] = document_frequency.get(token, 0) + 1
            total_length += len(tokens)
            lexical_documents.append(
                LexicalDocument(
                    content=content,
                    metadata=dict(metadata or {}),
                    term_freq=term_freq,
                    length=len(tokens),
                )
            )

        average_length = total_length / max(len(lexical_documents), 1)
        self._lexical_index = LexicalIndex(
            documents=lexical_documents,
            document_frequency=document_frequency,
            average_length=average_length,
            collection_count=current_count,
        )
        return self._lexical_index

    def _bm25_score(self, query_tokens: Sequence[str], document: LexicalDocument, index: LexicalIndex) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        total_documents = max(len(index.documents), 1)
        average_length = max(index.average_length, 1.0)
        for token in query_tokens:
            frequency = document.term_freq.get(token, 0)
            if frequency <= 0:
                continue
            df = index.document_frequency.get(token, 0)
            idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * document.length / average_length)
            score += idf * (frequency * (k1 + 1)) / denominator
        return score

    def _safe_collection_count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return -1

    def _result_key(self, result: SearchResult, fallback: str) -> str:
        metadata = result.metadata or {}
        file_name = str(metadata.get("file_name") or metadata.get("source_ref") or "")
        chunk_index = str(metadata.get("chunk_index") or metadata.get("local_chunk_index") or "")
        if file_name or chunk_index:
            return f"{file_name}:{chunk_index}"
        return fallback
