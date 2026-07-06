from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Sequence, Tuple

from config import RagConfig
from rag_core.models import SearchResult


logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    kwargs = {}
    if device and device != "auto":
        kwargs["device"] = device
    return CrossEncoder(model_name, **kwargs)


class CrossEncoderReranker:
    def __init__(self, rag_config: RagConfig) -> None:
        self.config = rag_config.normalized()
        self.status = "disabled"
        self._model = None

    def rerank(self, query: str, results: Sequence[SearchResult]) -> Tuple[List[SearchResult], str]:
        if not self.config.enable_reranker:
            return list(results), "disabled"
        if not results:
            return [], "no_candidates"
        try:
            model = _load_cross_encoder(self.config.reranker_model, self.config.reranker_device)
            candidate_limit = min(len(results), self.config.reranker_candidate_k)
            pairs = [(query, self._format_passage(result)) for result in results[:candidate_limit]]
            scores = model.predict(
                pairs,
                batch_size=self.config.reranker_batch_size,
                show_progress_bar=False,
            )
            scored = []
            for result, score in zip(results[:candidate_limit], scores):
                result.rerank_score = float(score)
                scored.append(result)
            scored.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else -999.0, reverse=True)
            tail = list(results[candidate_limit:])
            # Keep the full scored order.  ``reranker_top_k`` is a UI-facing
            # presentation budget in older versions; truncating here silently
            # starves downstream evidence aggregation, especially for
            # enumerate operations that need several independent incidents.
            return scored + tail, "ok"
        except Exception as exc:
            logger.warning("Cross-Encoder reranker 加载或执行失败，回退 RRF 排序: %s", exc)
            return list(results), f"fallback_rrf: {exc}"

    def _format_passage(self, result: SearchResult) -> str:
        metadata = result.metadata or {}
        max_chars = max(
            500,
            min(
                int(getattr(self.config, "reranker_passage_chars", 1100) or 1100),
                max(700, int(getattr(self.config, "reranker_max_length", 384) or 384) * 4),
            ),
        )
        text = (result.content or "").strip()
        if len(text) > max_chars:
            head = max_chars * 3 // 4
            tail = max_chars - head - 1
            text = text[:head].rstrip() + "…" + text[-tail:].lstrip()
        return (
            f"文档标题: {metadata.get('document_title') or metadata.get('file_name') or ''}\n"
            f"章节路径: {metadata.get('section_path') or ''}\n"
            f"页码: {metadata.get('page_start') or metadata.get('page') or ''}\n"
            f"类型: {metadata.get('chunk_kind') or result.chunk_kind}\n"
            f"正文:\n{text}"
        )
