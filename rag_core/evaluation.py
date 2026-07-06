from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(relevant_ids: Iterable[str], retrieved_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved_ids[:k])) / len(relevant)


def mrr_at_k(relevant_ids: Iterable[str], retrieved_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    for index, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance_by_id: dict[str, float], retrieved_ids: Sequence[str], k: int) -> float:
    def dcg(ids: Sequence[str]) -> float:
        score = 0.0
        for index, chunk_id in enumerate(ids[:k], start=1):
            relevance = float(relevance_by_id.get(chunk_id, 0.0))
            score += (2**relevance - 1.0) / math.log2(index + 1)
        return score

    ideal_ids = sorted(relevance_by_id, key=lambda key: relevance_by_id[key], reverse=True)
    ideal = dcg(ideal_ids)
    return dcg(retrieved_ids) / ideal if ideal > 0 else 0.0


def entity_coverage_rate(required_entities: Iterable[str], evidence_texts: Sequence[str]) -> float:
    entities = [entity for entity in required_entities if entity]
    if not entities:
        return 1.0
    joined = "\n".join(evidence_texts).casefold()
    covered = sum(1 for entity in entities if entity.casefold() in joined)
    return covered / len(entities)


def unsupported_claim_rate(total_claims: int, supported_claims: int) -> float:
    if total_claims <= 0:
        return 0.0
    return max(0.0, min(1.0, (total_claims - supported_claims) / total_claims))
