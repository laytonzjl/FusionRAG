from __future__ import annotations

import logging
import os
import re
import time
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
from uuid import uuid4

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings

from config import ApiConfig, RagConfig, sanitize_collection_name, settings
from rag_core.cards import DOCUMENT_STRUCTURE_VERSION, analyze_document, build_cards, build_document_map, enrich_chunk_metadata
from rag_core.chunking import split_text_with_overlap
from rag_core.hybrid_index import HybridIndex, HybridRecord, INDEX_SCHEMA_VERSION, LEXICAL_INDEX_VERSION
from rag_core.logging_utils import setup_logging
from rag_core.multilingual import LANGUAGE_PROCESSING_VERSION, NORMALIZATION_VERSION
from rag_core.models import ParsedChunk, SearchResult
from rag_core.parsers import SUPPORTED_EXTENSIONS, parse_file
from rag_core.providers import MultiProviderClient
from rag_core.retrieval import RetrievalService


logger = logging.getLogger(__name__)
setup_logging()

CHROMA_COLLECTION_METADATA = {
    "hnsw:batch_size": 100000,
    "hnsw:sync_threshold": 1000000,
}


class RAGEngine:
    """本地轻量级企业知识库引擎。"""

    def __init__(self, api_config: Optional[ApiConfig] = None, rag_config: Optional[RagConfig] = None) -> None:
        self.api_config = api_config or ApiConfig()
        self.rag_config = (rag_config or RagConfig()).normalized()
        self.client = MultiProviderClient(self.api_config)
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = sanitize_collection_name(self.rag_config.collection_name)
        self.distance_metric = self.rag_config.distance_metric

        self.chroma_client = self._create_chroma_client(self.persist_dir)
        self.hybrid_index = HybridIndex(collection_name=self.collection_name)
        try:
            self.collection = self._get_or_create_collection()
        except Exception as exc:
            if self._is_chroma_compatibility_error(exc):
                fallback_dir = Path(self.persist_dir).with_name(f"{Path(self.persist_dir).name}_v06")
                fallback_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    "当前 Chroma 目录与稳定版 Chroma 不兼容，自动切换到新目录: %s",
                    fallback_dir,
                )
                self.persist_dir = str(fallback_dir)
                self.chroma_client = self._create_chroma_client(self.persist_dir)
                self.collection = self._get_or_create_collection()
            else:
                raise
        self.retrieval = RetrievalService(self.collection, self.client, self.api_config, self.rag_config)

    def _create_chroma_client(self, persist_dir: str) -> PersistentClient:
        return PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _get_or_create_collection(self):
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric, **CHROMA_COLLECTION_METADATA},
        )

    def _is_chroma_compatibility_error(self, exc: Exception) -> bool:
        error_text = f"{type(exc).__name__}: {exc}"
        return "_type" in error_text or "configuration" in error_text.lower()

    def ingest_uploaded_file(self, uploaded_file) -> Dict[str, object]:
        """接收 Streamlit 上传文件并写入知识库。"""

        original_name = self._sanitize_filename(uploaded_file.name)
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"暂不支持该文件类型：{suffix}，仅支持 TXT / PDF / CSV。")

        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stored_name = f"{timestamp}_{uuid4().hex[:8]}_{original_name}"
        saved_path = Path(settings.upload_dir) / stored_name

        try:
            saved_path.write_bytes(uploaded_file.getbuffer())
        except Exception as exc:
            logger.exception("保存上传文件失败: %s", exc)
            raise RuntimeError(f"保存上传文件失败：{exc}") from exc

        return self._index_file(
            saved_path=saved_path,
            original_name=original_name,
            stored_name=stored_name,
            upload_time=upload_time,
        )

    def ingest_existing_file(self, saved_path: Path | str) -> Dict[str, object]:
        """把已经保存在 data/uploads 下的文件重新解析并写入索引。"""

        path = Path(saved_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"暂不支持该文件类型：{suffix}，仅支持 TXT / PDF / CSV。")

        original_name = self._original_name_from_stored_name(path.name)
        upload_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        return self._index_file(
            saved_path=path,
            original_name=original_name,
            stored_name=path.name,
            upload_time=upload_time,
        )

    def rebuild_from_uploads(self) -> Dict[str, object]:
        """清空当前 collection，并用 data/uploads 中已有文件重建索引。"""

        upload_dir = Path(settings.upload_dir)
        if not upload_dir.exists():
            return {"indexed": [], "errors": []}

        self.reset_collection()
        indexed: List[Dict[str, object]] = []
        errors: List[Dict[str, str]] = []
        seen_hashes: set[str] = set()
        for path in sorted(upload_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                file_hash = self._file_sha256(path)
                if file_hash in seen_hashes:
                    indexed.append(
                        {
                            "file_name": self._original_name_from_stored_name(path.name),
                            "stored_file_name": path.name,
                            "chunk_count": 0,
                            "document_count": 0,
                            "collection_name": self.collection_name,
                            "skipped": True,
                            "skip_reason": "重复文件，已跳过",
                            "file_sha256": file_hash,
                        }
                    )
                    logger.info("跳过重复上传文件: %s", path.name)
                    continue
                seen_hashes.add(file_hash)
                indexed.append(self.ingest_existing_file(path))
            except Exception as exc:
                logger.exception("重建索引失败: %s", path)
                errors.append({"file_name": path.name, "error": str(exc)})

        return {"indexed": indexed, "errors": errors}

    def _index_file(
        self,
        saved_path: Path,
        original_name: str,
        stored_name: str,
        upload_time: str,
    ) -> Dict[str, object]:
        """解析、切片、向量化并写入 Chroma。"""

        total_start = time.perf_counter()
        file_sha256 = self._file_sha256(saved_path)

        parse_start = time.perf_counter()
        parsed_documents = self._parse_file(saved_path, original_name=original_name)
        parse_seconds = time.perf_counter() - parse_start

        chunk_start = time.perf_counter()
        chunks = self._prepare_index_chunks(
            parsed_documents=parsed_documents,
            original_name=original_name,
            stored_name=stored_name,
            upload_time=upload_time,
            file_sha256=file_sha256,
        )
        chunk_seconds = time.perf_counter() - chunk_start

        if not chunks:
            raise ValueError(f"文件 {original_name} 未解析出可入库内容。")

        contents = [chunk.content for chunk in chunks]
        embedding_start = time.perf_counter()
        embeddings = self.client.embed_documents(contents)
        embedding_seconds = time.perf_counter() - embedding_start
        ids: List[str] = []
        metadatas: List[Dict[str, object]] = []

        for index, chunk in enumerate(chunks):
            chunk_id = uuid4().hex
            ids.append(chunk_id)
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "file_name": original_name,
                    "stored_file_name": stored_name,
                    "upload_time": upload_time,
                    "file_sha256": file_sha256,
                    "chunk_index": index,
                    "collection_name": self.collection_name,
                }
            )
            metadatas.append(metadata)

        try:
            upsert_start = time.perf_counter()
            self._upsert_chunks(ids=ids, contents=contents, embeddings=embeddings, metadatas=metadatas)
            self._upsert_hybrid_records(contents=contents, metadatas=metadatas)
            self._write_index_meta()
            upsert_seconds = time.perf_counter() - upsert_start
        except Exception as exc:
            logger.exception("写入 Chroma 失败: %s", exc)
            error_text = str(exc)
            if "dimension" in error_text.lower():
                raise RuntimeError(
                    "写入向量数据库失败：当前 Chroma collection 里已有不同维度的向量。"
                    "如果刚从 OpenAI 向量切换到本地向量，请新建知识库集合名，或点击“清空当前知识库”后重新导入文件。"
                ) from exc
            raise RuntimeError(f"写入向量数据库失败：{exc}") from exc

        total_seconds = time.perf_counter() - total_start
        logger.info(
            "文件入库完成: %s, docs=%s, chunks=%s, parse=%.2fs, chunk=%.2fs, embed=%.2fs, upsert=%.2fs, total=%.2fs",
            original_name,
            len(parsed_documents),
            len(chunks),
            parse_seconds,
            chunk_seconds,
            embedding_seconds,
            upsert_seconds,
            total_seconds,
        )

        return {
            "file_name": original_name,
            "stored_file_name": stored_name,
            "upload_time": upload_time,
            "file_sha256": file_sha256,
            "chunk_count": len(chunks),
            "document_count": len(parsed_documents),
            "collection_name": self.collection_name,
            "timing": {
                "parse_seconds": round(parse_seconds, 2),
                "chunk_seconds": round(chunk_seconds, 2),
                "embedding_seconds": round(embedding_seconds, 2),
                "upsert_seconds": round(upsert_seconds, 2),
                "total_seconds": round(total_seconds, 2),
            },
        }

    def _upsert_chunks(
        self,
        ids: Sequence[str],
        contents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, object]],
        batch_size: int = 64,
    ) -> None:
        """分批写入 Chroma，降低 Windows 本地 HNSW 大批量写入的不稳定性。"""

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=list(ids[start:end]),
                documents=list(contents[start:end]),
                embeddings=[list(vector) for vector in embeddings[start:end]],
                metadatas=[self._sanitize_chroma_metadata(metadata) for metadata in metadatas[start:end]],
            )

    @staticmethod
    def _sanitize_chroma_metadata(metadata: Dict[str, object]) -> Dict[str, str | int | float | bool]:
        """Convert rich internal metadata into Chroma-compatible scalar metadata.

        Chroma only accepts str/int/float/bool values. Our hybrid retrieval layer
        intentionally keeps richer values such as aliases lists and language
        distribution dicts, so the conversion must happen only at the Chroma
        boundary instead of mutating the original metadata.
        """

        sanitized: Dict[str, str | int | float | bool] = {}
        for key, value in (metadata or {}).items():
            if not isinstance(key, str) or value is None:
                continue
            if isinstance(value, bool):
                sanitized[key] = value
                continue
            if isinstance(value, int):
                sanitized[key] = value
                continue
            if isinstance(value, float):
                if math.isfinite(value):
                    sanitized[key] = value
                continue
            if isinstance(value, str):
                sanitized[key] = value
                continue
            sanitized[key] = RAGEngine._stringify_chroma_metadata_value(value)
        return sanitized

    @staticmethod
    def _stringify_chroma_metadata_value(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)

    def answer_stream(
        self,
        query: str,
        history_messages: Sequence[Dict[str, str]],
        top_k: int | None = None,
        keyword_filter: str = "",
    ) -> Tuple[List[SearchResult], Generator[str, None, None]]:
        return self.retrieval.answer_stream(
            query=query,
            history_messages=history_messages,
            top_k=top_k or self.rag_config.top_k,
            keyword_filter=keyword_filter,
        )

    def collection_count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def retrieval_diagnostics(self) -> Dict[str, object]:
        diagnostics = getattr(self.retrieval, "last_diagnostics", None)
        if diagnostics is None:
            return {}
        plan = diagnostics.query_plan
        return {
            "query_plan": {
                "intent": getattr(plan, "intent", ""),
                "query_language": getattr(plan, "query_language", ""),
                "language_confidence": getattr(plan, "language_confidence", 0.0),
                "script_distribution": getattr(plan, "script_distribution", {}),
                "entities": [
                    {
                        "surface": entity.surface,
                        "canonical": entity.canonical,
                        "aliases": entity.aliases,
                        "alias_sources": entity.alias_sources,
                        "linked_alias": getattr(entity, "linked_alias", ""),
                        "link_confidence": getattr(entity, "link_confidence", 0.0),
                    }
                    for entity in (getattr(plan, "entities", []) or [])
                ],
                "retrieval_queries": [
                    {"text": item.text, "language": item.language, "origin": item.origin}
                    for item in (getattr(plan, "retrieval_queries", []) or [])
                ],
                "preferred_chunk_kinds": getattr(plan, "preferred_chunk_kinds", []),
                "required_evidence": getattr(plan, "required_evidence", []),
                "planner_source": getattr(plan, "planner_source", ""),
                "entity_coverage_failed": getattr(plan, "entity_coverage_failed", False),
                "warnings": getattr(plan, "warnings", []),
                "relation_type": getattr(plan, "relation_type", ""),
                "answer_type": getattr(plan, "answer_type", ""),
                "semantics": {
                    "requested_property": getattr(getattr(plan, "semantics", None), "requested_property", ""),
                    "operation": getattr(getattr(plan, "semantics", None), "operation", "lookup"),
                    "answer_mode": getattr(getattr(plan, "semantics", None), "answer_mode", "direct"),
                    "constraints": getattr(getattr(plan, "semantics", None), "constraints", []),
                    "answer_shape": getattr(getattr(plan, "semantics", None), "answer_shape", ""),
                    "evidence_contract": {
                        "answer_unit": getattr(
                            getattr(getattr(plan, "semantics", None), "evidence_contract", None),
                            "answer_unit",
                            "fact",
                        ),
                        "include_when": getattr(
                            getattr(getattr(plan, "semantics", None), "evidence_contract", None),
                            "include_when",
                            "",
                        ),
                        "exclude_when": getattr(
                            getattr(getattr(plan, "semantics", None), "evidence_contract", None),
                            "exclude_when",
                            "",
                        ),
                        "required_roles": getattr(
                            getattr(getattr(plan, "semantics", None), "evidence_contract", None),
                            "required_roles",
                            [],
                        ),
                        "retrieval_views": getattr(
                            getattr(getattr(plan, "semantics", None), "evidence_contract", None),
                            "retrieval_views",
                            [],
                        ),
                    },
                    "scope": getattr(getattr(plan, "semantics", None), "scope", "entity_or_document"),
                    "regions": getattr(getattr(plan, "semantics", None), "regions", ["all"]),
                    "position_bias": getattr(getattr(plan, "semantics", None), "position_bias", "none"),
                    "need_timeline": getattr(getattr(plan, "semantics", None), "need_timeline", False),
                    "need_entity_neighborhood": getattr(getattr(plan, "semantics", None), "need_entity_neighborhood", False),
                    "allow_partial": getattr(getattr(plan, "semantics", None), "allow_partial", False),
                    "planner_confidence": getattr(getattr(plan, "semantics", None), "planner_confidence", 0.0),
                },
                "entity_linking_confidence": getattr(plan, "entity_linking_confidence", 0.0),
            }
            if plan
            else {},
            "candidates_by_channel": diagnostics.candidates_by_channel,
            "final_chunk_ids": diagnostics.final_chunk_ids,
            "reranker_enabled": diagnostics.reranker_enabled,
            "reranker_status": diagnostics.reranker_status,
            "evidence_judge_enabled": diagnostics.evidence_judge_enabled,
            "entity_coverage_failed": diagnostics.entity_coverage_failed,
            "warnings": diagnostics.warnings,
            "final_evidence": [
                {
                    "chunk_id": result.chunk_id,
                    "file": (result.metadata or {}).get("file_name") or (result.metadata or {}).get("source_ref"),
                    "page": (result.metadata or {}).get("page_start") or (result.metadata or {}).get("page"),
                    "chunk_kind": result.chunk_kind,
                    "evidence_judge": {
                        "answerability": getattr(getattr(result, "evidence", None), "answerability", None),
                        "relevance": getattr(getattr(result, "evidence", None), "relevance", None),
                        "entity_match": getattr(getattr(result, "evidence", None), "entity_match", None),
                        "evidence_type": getattr(getattr(result, "evidence", None), "evidence_type", None),
                        "supported_claims": getattr(getattr(result, "evidence", None), "supported_claims", []),
                        "atomic_claims": [
                            {
                                "statement": getattr(claim, "statement", ""),
                                "qualifies": getattr(claim, "qualifies", False),
                                "classification": getattr(claim, "classification", ""),
                                "source_excerpt": getattr(claim, "source_excerpt", ""),
                                "roles": getattr(claim, "roles", {}),
                                "reason": getattr(claim, "reason", ""),
                                "confidence": getattr(claim, "confidence", 0.0),
                            }
                            for claim in (
                                getattr(getattr(result, "evidence", None), "atomic_claims", [])
                                or []
                            )
                        ],
                        "reject_reason": getattr(getattr(result, "evidence", None), "reject_reason", ""),
                    },
                    "rerank_score": result.rerank_score,
                    "rrf_score": result.rrf_score,
                }
                for result in getattr(self.retrieval, "_last_results", [])[:10]
            ],
        }

    def list_files(self) -> List[Dict[str, object]]:
        try:
            raw = self.collection.get(include=["metadatas"])
        except Exception as exc:
            logger.exception("读取文件列表失败: %s", exc)
            raise RuntimeError(f"读取知识库文件列表失败：{exc}") from exc

        metadatas = raw.get("metadatas") or []
        grouped: Dict[str, Dict[str, object]] = {}

        for metadata in metadatas:
            if not metadata:
                continue
            file_name = str(metadata.get("file_name") or metadata.get("source_ref") or "unknown")
            item = grouped.setdefault(
                file_name,
                {
                    "file_name": file_name,
                    "chunk_count": 0,
                    "source_type": metadata.get("source_type", "unknown"),
                    "latest_upload_time": "",
                    "pages": set(),
                    "rows": set(),
                    "stored_file_names": set(),
                },
            )
            item["chunk_count"] = int(item["chunk_count"]) + 1
            upload_time = str(metadata.get("upload_time") or "")
            if upload_time and upload_time > str(item["latest_upload_time"]):
                item["latest_upload_time"] = upload_time

            page = metadata.get("page")
            row_index = metadata.get("row_index")
            stored_file_name = metadata.get("stored_file_name")
            if page is not None:
                item["pages"].add(page)
            if row_index is not None:
                item["rows"].add(row_index)
            if stored_file_name:
                item["stored_file_names"].add(str(stored_file_name))

        items: List[Dict[str, object]] = []
        for item in grouped.values():
            items.append(
                {
                    "file_name": item["file_name"],
                    "chunk_count": item["chunk_count"],
                    "source_type": item["source_type"],
                    "latest_upload_time": item["latest_upload_time"],
                    "page_count": len(item["pages"]),
                    "row_count": len(item["rows"]),
                    "stored_file_count": len(item["stored_file_names"]),
                }
            )
        items.sort(key=lambda value: (str(value["latest_upload_time"]), str(value["file_name"])), reverse=True)
        return items

    def delete_file(self, file_name: str, delete_uploads: bool = True) -> int:
        cleaned_name = self._sanitize_filename(file_name)
        if not cleaned_name:
            raise ValueError("文件名不能为空。")

        try:
            before = int(self.collection.count())
            self.collection.delete(where={"file_name": cleaned_name})
            after = int(self.collection.count())
            deleted = max(before - after, 0)
            if deleted == 0:
                self.collection.delete(where={"source_ref": cleaned_name})
                after = int(self.collection.count())
                deleted = max(before - after, 0)
            if delete_uploads:
                self.delete_uploaded_files_by_original_name(cleaned_name)
            self.hybrid_index.delete_file(cleaned_name)
            return deleted
        except Exception as exc:
            logger.exception("删除文件失败: %s", exc)
            raise RuntimeError(f"删除文件失败：{exc}") from exc

    def reset_collection(self, delete_uploads: bool = False) -> Dict[str, int]:
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric, **CHROMA_COLLECTION_METADATA},
        )
        self.hybrid_index.reset_collection()
        self.retrieval = RetrievalService(self.collection, self.client, self.api_config, self.rag_config)
        deleted_uploads = self.clear_uploaded_files() if delete_uploads else 0
        return {"deleted_upload_files": deleted_uploads}

    def clear_uploaded_files(self) -> int:
        """删除本地上传缓存，防止清库后重建索引又把旧文件导回来。"""

        upload_dir = Path(settings.upload_dir)
        if not upload_dir.exists():
            return 0

        deleted = 0
        for path in upload_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                path.unlink()
                deleted += 1
            except Exception as exc:
                logger.warning("删除上传缓存失败: %s, error=%s", path, exc)
        return deleted

    def delete_uploaded_files_by_original_name(self, file_name: str) -> int:
        """按原始文件名删除上传缓存；同名重复上传会一起清理。"""

        cleaned_name = self._sanitize_filename(file_name)
        upload_dir = Path(settings.upload_dir)
        if not cleaned_name or not upload_dir.exists():
            return 0

        deleted = 0
        for path in upload_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if self._original_name_from_stored_name(path.name) != cleaned_name:
                continue
            try:
                path.unlink()
                deleted += 1
            except Exception as exc:
                logger.warning("删除上传缓存失败: %s, error=%s", path, exc)
        return deleted

    def _parse_file(self, path: Path, original_name: str) -> List[Dict[str, object]]:
        try:
            return parse_file(
                path,
                original_name,
                enable_pdf_ocr=self.rag_config.enable_pdf_ocr,
                pdf_ocr_dpi=self.rag_config.pdf_ocr_dpi,
                pdf_ocr_min_text_chars=self.rag_config.pdf_ocr_min_text_chars,
                pdf_ocr_device=self.rag_config.pdf_ocr_device,
                pdf_ocr_threads=self.rag_config.pdf_ocr_threads,
                pdf_ocr_max_side_len=self.rag_config.pdf_ocr_max_side_len,
            )
        except Exception as exc:
            logger.exception("解析文件失败: %s", exc)
            raise

    def _split_document(self, document: Dict[str, object]) -> List[ParsedChunk]:
        text = str(document.get("text", "")).strip()
        metadata = dict(document.get("metadata") or {})
        if not text:
            return []

        chunks = split_text_with_overlap(
            text,
            chunk_size=self.rag_config.chunk_size,
            overlap=self.rag_config.chunk_overlap,
        )

        parsed_chunks: List[ParsedChunk] = []
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            parsed_chunks.append(
                ParsedChunk(content=chunk.strip(), metadata={**metadata, "local_chunk_index": index})
            )
        return parsed_chunks

    def _prepare_index_chunks(
        self,
        parsed_documents: Sequence[Dict[str, object]],
        original_name: str,
        stored_name: str,
        upload_time: str,
        file_sha256: str,
    ) -> List[ParsedChunk]:
        document_id = hashlib.sha1(f"{file_sha256}:{original_name}".encode("utf-8")).hexdigest()[:16]
        signals = analyze_document(original_name, parsed_documents)
        # Build the navigation map once and reuse it for cards and body chunks.
        # The map contains no generated facts: it only records source order,
        # headings, regions, pages, and entities observed in the raw document.
        document_map = build_document_map(original_name, parsed_documents, signals=signals)
        output: List[ParsedChunk] = []
        card_chunks = build_cards(
            original_name=original_name,
            stored_name=stored_name,
            document_id=document_id,
            upload_time=upload_time,
            file_sha256=file_sha256,
            documents=parsed_documents,
            document_map=document_map,
        )
        for card_index, card in enumerate(card_chunks):
            chunk_id = f"{document_id}:card:{card_index}"
            content_hash = hashlib.sha1(card.content.encode("utf-8", errors="ignore")).hexdigest()
            output.append(
                enrich_chunk_metadata(
                    card,
                    document_id=document_id,
                    document_title=signals.document_title,
                    document_type=signals.document_type,
                    parent_chunk_id="",
                    chunk_id=chunk_id,
                    chunk_kind=str(card.metadata.get("chunk_kind", "document_card")),
                    sequence_no=card_index,
                    content_hash=content_hash,
                )
            )

        sequence_no = len(output)
        source_unit_index = -1
        for document_index, document in enumerate(parsed_documents):
            text = str(document.get("text", "")).strip()
            metadata = dict(document.get("metadata") or {})
            if not text:
                continue
            source_unit_index += 1
            # Preserve navigable source structure on every Parent/Child chunk.
            # Exact entity retrieval can now return source body evidence rather
            # than only an entity-card alias record.
            metadata.update(document_map.metadata_for_unit(source_unit_index))
            parent_chunks = split_text_with_overlap(
                text,
                chunk_size=self.rag_config.parent_chunk_size,
                overlap=min(self.rag_config.child_chunk_overlap, 120),
            )
            for parent_index, parent_text in enumerate(parent_chunks):
                parent_id = f"{document_id}:p:{document_index}:{parent_index}"
                parent_hash = hashlib.sha1(parent_text.encode("utf-8", errors="ignore")).hexdigest()
                parent_chunk = ParsedChunk(content=parent_text, metadata=dict(metadata))
                output.append(
                    enrich_chunk_metadata(
                        parent_chunk,
                        document_id=document_id,
                        document_title=signals.document_title,
                        document_type=signals.document_type,
                        parent_chunk_id="",
                        chunk_id=parent_id,
                        chunk_kind="parent",
                        sequence_no=sequence_no,
                        content_hash=parent_hash,
                    )
                )
                sequence_no += 1
                child_chunks = split_text_with_overlap(
                    parent_text,
                    chunk_size=self.rag_config.child_chunk_size,
                    overlap=self.rag_config.child_chunk_overlap,
                )
                for child_index, child_text in enumerate(child_chunks):
                    child_id = f"{document_id}:c:{document_index}:{parent_index}:{child_index}"
                    child_hash = hashlib.sha1(child_text.encode("utf-8", errors="ignore")).hexdigest()
                    child_chunk = ParsedChunk(content=child_text, metadata=dict(metadata))
                    output.append(
                        enrich_chunk_metadata(
                            child_chunk,
                            document_id=document_id,
                            document_title=signals.document_title,
                            document_type=signals.document_type,
                            parent_chunk_id=parent_id,
                            chunk_id=child_id,
                            chunk_kind="child",
                            sequence_no=sequence_no,
                            content_hash=child_hash,
                        )
                    )
                    sequence_no += 1
        return output

    def _upsert_hybrid_records(self, contents: Sequence[str], metadatas: Sequence[Dict[str, object]]) -> None:
        records: List[HybridRecord] = []
        for content, metadata in zip(contents, metadatas):
            aliases = metadata.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            exact_terms = [
                str(metadata.get("document_title") or ""),
                str(metadata.get("file_name") or ""),
                str(metadata.get("section_path") or ""),
            ]
            records.append(
                HybridRecord(
                    chunk_id=str(metadata.get("chunk_id") or ""),
                    document_id=str(metadata.get("document_id") or ""),
                    parent_chunk_id=str(metadata.get("parent_chunk_id") or ""),
                    chunk_kind=str(metadata.get("chunk_kind") or "child"),
                    language=str(metadata.get("chunk_language") or metadata.get("document_language") or "unknown"),
                    title=str(metadata.get("document_title") or metadata.get("file_name") or ""),
                    section_path=str(metadata.get("section_path") or ""),
                    content=content,
                    metadata=dict(metadata),
                    aliases=[str(alias) for alias in aliases if alias],
                    exact_terms=[term for term in exact_terms if term],
                )
            )
        self.hybrid_index.upsert_records(records)

    def _write_index_meta(self) -> None:
        self.hybrid_index.set_meta(
            {
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "embedding_model": self.api_config.embedding_model,
                "embedding_provider": self.api_config.embedding_provider,
                "reranker_model": self.rag_config.reranker_model,
                "chunk_size": self.rag_config.chunk_size,
                "chunk_overlap": self.rag_config.chunk_overlap,
                "parent_chunk_size": self.rag_config.parent_chunk_size,
                "child_chunk_size": self.rag_config.child_chunk_size,
                "normalization_version": NORMALIZATION_VERSION,
                "language_processing_version": LANGUAGE_PROCESSING_VERSION,
                "lexical_index_version": LEXICAL_INDEX_VERSION,
                "document_structure_version": DOCUMENT_STRUCTURE_VERSION,
            }
        )

    def index_compatibility(self) -> Dict[str, object]:
        meta = self.hybrid_index.get_meta()
        expected = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": self.api_config.embedding_model,
            "embedding_provider": self.api_config.embedding_provider,
            "parent_chunk_size": self.rag_config.parent_chunk_size,
            "child_chunk_size": self.rag_config.child_chunk_size,
            "normalization_version": NORMALIZATION_VERSION,
            "language_processing_version": LANGUAGE_PROCESSING_VERSION,
            "lexical_index_version": LEXICAL_INDEX_VERSION,
            "document_structure_version": DOCUMENT_STRUCTURE_VERSION,
        }
        mismatches = {key: {"current": meta.get(key), "expected": value} for key, value in expected.items() if meta.get(key) != value}
        return {
            "compatible": not mismatches,
            "mismatches": mismatches,
            "meta": meta,
            "requires_rebuild": bool(mismatches),
            "rebuild_reason": "离线文档导航结构已升级；旧索引不包含页面顺序、区段和导航窗口。" if mismatches else "",
        }

    def _sanitize_filename(self, filename: str) -> str:
        return os.path.basename(filename).replace("..", "_").strip()

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file_obj:
            for block in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _original_name_from_stored_name(self, filename: str) -> str:
        cleaned = self._sanitize_filename(filename)
        match = re.match(r"^\d{8}_\d{6}_[0-9a-fA-F]{8}_(.+)$", cleaned)
        if match:
            return match.group(1).strip() or cleaned
        return cleaned


def build_engine(api_config: Optional[ApiConfig] = None, rag_config: Optional[RagConfig] = None) -> RAGEngine:
    return RAGEngine(api_config=api_config, rag_config=rag_config)
