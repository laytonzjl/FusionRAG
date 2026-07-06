from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_semantic"
TEMP_DIR = DATA_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"

API_CONFIG_FILE = DATA_DIR / "api_config.json"
RAG_CONFIG_FILE = DATA_DIR / "rag_config.json"
UI_CONFIG_FILE = DATA_DIR / "ui_config.json"
CONVERSATION_STORE_FILE = DATA_DIR / "conversations.json"
CONFIG_STORE_VERSION = 2
CONFIG_PAYLOAD_KEY = "payload"
DEFAULT_WORKSPACE_SCOPE_ID = os.getenv(
    "RAG_WORKSPACE_ID",
    hashlib.sha1(str(BASE_DIR).encode("utf-8", errors="ignore")).hexdigest()[:16],
)

TConfig = TypeVar("TConfig")


PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "local": {
        "label": "本地向量",
        "api_key_hint": "本地模型无需 API Key",
        "chat_api_base": "",
        "chat_model": "",
        "embedding_api_base": "local",
        "embedding_model": "intfloat/multilingual-e5-small",
        "recommended_embedding_provider": "local",
    },
    "openai": {
        "label": "OpenAI",
        "api_key_hint": "sk-...",
        "chat_api_base": "https://api.openai.com/v1",
        "chat_model": "gpt-4o-mini",
        "embedding_api_base": "https://api.openai.com/v1",
        "embedding_model": "text-embedding-3-small",
        "recommended_embedding_provider": "local",
    },
    "claude": {
        "label": "Claude",
        "api_key_hint": "sk-ant-...",
        "chat_api_base": "https://api.anthropic.com",
        "chat_model": "claude-3-5-sonnet-latest",
        "embedding_api_base": "local",
        "embedding_model": "intfloat/multilingual-e5-small",
        "recommended_embedding_provider": "local",
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_key_hint": "sk-...",
        "chat_api_base": "https://api.deepseek.com/v1",
        "chat_model": "deepseek-chat",
        "embedding_api_base": "local",
        "embedding_model": "intfloat/multilingual-e5-small",
        "recommended_embedding_provider": "local",
    },
    "qwen": {
        "label": "Qwen",
        "api_key_hint": "sk-...",
        "chat_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "embedding_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embedding_model": "text-embedding-v4",
        "recommended_embedding_provider": "local",
    },
    "custom": {
        "label": "自定义",
        "api_key_hint": "your-api-key",
        "chat_api_base": "https://api.openai.com/v1",
        "chat_model": "gpt-4o-mini",
        "embedding_api_base": "https://api.openai.com/v1",
        "embedding_model": "text-embedding-3-small",
        "recommended_embedding_provider": "local",
    },
}


def _safe_provider(provider: str, fallback: str = "openai") -> str:
    key = (provider or fallback).strip().lower()
    return key if key in PROVIDER_PRESETS else fallback


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def provider_label(provider: str) -> str:
    return str(PROVIDER_PRESETS[_safe_provider(provider)].get("label", provider)).strip()


def provider_api_key_hint(provider: str) -> str:
    return str(PROVIDER_PRESETS[_safe_provider(provider)].get("api_key_hint", "your-api-key")).strip()


def provider_default_chat_fields(provider: str) -> Dict[str, str]:
    preset = PROVIDER_PRESETS[_safe_provider(provider)]
    return {
        "chat_api_base": str(preset["chat_api_base"]),
        "chat_model": str(preset["chat_model"]),
    }


def provider_default_embedding_fields(provider: str) -> Dict[str, str]:
    preset = PROVIDER_PRESETS[_safe_provider(provider)]
    return {
        "embedding_api_base": str(preset["embedding_api_base"]),
        "embedding_model": str(preset["embedding_model"]),
    }


def recommended_embedding_provider(chat_provider: str) -> str:
    preset = PROVIDER_PRESETS[_safe_provider(chat_provider)]
    return _safe_provider(str(preset.get("recommended_embedding_provider", "openai")), fallback="openai")


def infer_provider_from_base_url(base_url: str) -> str:
    lowered = (base_url or "").strip().lower()
    if lowered in {"local", "sentence-transformers", "sentence_transformers"}:
        return "local"
    if "anthropic" in lowered:
        return "claude"
    if "deepseek" in lowered:
        return "deepseek"
    if "dashscope" in lowered or "aliyuncs" in lowered or "qwen" in lowered:
        return "qwen"
    if "openai" in lowered:
        return "openai"
    return "custom"


def _known_default_base_urls() -> set[str]:
    urls = set()
    for preset in PROVIDER_PRESETS.values():
        urls.add(str(preset["chat_api_base"]).rstrip("/"))
        urls.add(str(preset["embedding_api_base"]).rstrip("/"))
    return urls


def sanitize_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip())
    return cleaned[:63].strip("_") or "enterprise_knowledge_base"


@dataclass(frozen=True)
class Settings:
    """统一管理目录与少量全局默认值。"""

    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
    upload_dir: str = os.getenv("UPLOAD_DIR", str(UPLOAD_DIR))
    temp_dir: str = os.getenv("TEMP_DIR", str(TEMP_DIR))
    log_dir: str = os.getenv("LOG_DIR", str(LOG_DIR))
    default_theme_mode: str = os.getenv("UI_THEME_MODE", "system")
    hybrid_index_path: str = os.getenv("HYBRID_INDEX_PATH", str(DATA_DIR / "rag_hybrid.sqlite"))


settings = Settings()


@dataclass
class ApiConfig:
    """聊天与向量化接口配置。"""

    chat_provider: str = os.getenv("CHAT_PROVIDER", "openai")
    chat_api_key: str = os.getenv("CHAT_API_KEY", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    chat_api_base: str = os.getenv("CHAT_API_BASE", provider_default_chat_fields("openai")["chat_api_base"])
    chat_model: str = os.getenv("CHAT_MODEL", provider_default_chat_fields("openai")["chat_model"])

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_api_base: str = os.getenv(
        "EMBEDDING_API_BASE", provider_default_embedding_fields("local")["embedding_api_base"]
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", provider_default_embedding_fields("local")["embedding_model"]
    )

    temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_provider": _safe_provider(self.chat_provider),
            "chat_api_key": self.chat_api_key,
            "chat_api_base": self.chat_api_base,
            "chat_model": self.chat_model,
            "embedding_provider": _safe_provider(self.embedding_provider),
            "embedding_api_key": self.embedding_api_key,
            "embedding_api_base": self.embedding_api_base,
            "embedding_model": self.embedding_model,
            "temperature": float(self.temperature),
            "max_output_tokens": int(self.max_output_tokens),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ApiConfig":
        raw = dict(payload or {})
        chat_provider = _safe_provider(str(raw.get("chat_provider", "openai")))
        if chat_provider == "local":
            chat_provider = "openai"
        raw_chat_base = str(raw.get("chat_api_base") or "").strip()
        raw_chat_model = str(raw.get("chat_model") or "").strip()
        inferred_chat_provider = infer_provider_from_base_url(raw_chat_base)
        embedding_provider = raw.get("embedding_provider")
        if not embedding_provider:
            embedding_provider = infer_provider_from_base_url(str(raw.get("embedding_api_base", "")))
        embedding_provider = _safe_provider(str(embedding_provider), fallback=recommended_embedding_provider(chat_provider))

        chat_defaults = provider_default_chat_fields(chat_provider)
        known_base_urls = _known_default_base_urls()
        if (
            raw_chat_base.rstrip("/") in known_base_urls
            and inferred_chat_provider != "custom"
            and inferred_chat_provider != chat_provider
        ):
            raw_chat_base = chat_defaults["chat_api_base"]
            raw_chat_model = chat_defaults["chat_model"]

        chat_api_key = str(raw.get("chat_api_key", cls.chat_api_key)).strip()
        embedding_api_key = str(raw.get("embedding_api_key", cls.embedding_api_key)).strip()
        migrated_embedding_to_local = False
        if (
            chat_provider != "openai"
            and embedding_provider == "openai"
            and (not embedding_api_key or embedding_api_key == chat_api_key)
        ):
            # 避免把 DeepSeek/Claude/Qwen 的聊天 Key 当成 OpenAI Embedding Key 使用。
            embedding_provider = recommended_embedding_provider(chat_provider)
            migrated_embedding_to_local = embedding_provider == "local"
        elif embedding_provider == "openai" and not embedding_api_key:
            embedding_provider = "local"
            migrated_embedding_to_local = True

        embedding_defaults = provider_default_embedding_fields(embedding_provider)
        embedding_api_base = str(raw.get("embedding_api_base") or embedding_defaults["embedding_api_base"]).strip()
        embedding_model = str(raw.get("embedding_model") or embedding_defaults["embedding_model"]).strip()
        if embedding_provider == "local":
            embedding_api_key = ""
            embedding_api_base = embedding_defaults["embedding_api_base"]
            if (
                migrated_embedding_to_local
                or embedding_model.startswith("text-embedding")
                or embedding_model.startswith("local-hashing")
            ):
                embedding_model = embedding_defaults["embedding_model"]

        return cls(
            chat_provider=chat_provider,
            chat_api_key=chat_api_key,
            chat_api_base=str(raw_chat_base or chat_defaults["chat_api_base"]).strip(),
            chat_model=str(raw_chat_model or chat_defaults["chat_model"]).strip(),
            embedding_provider=embedding_provider,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
            temperature=float(raw.get("temperature", cls.temperature)),
            max_output_tokens=int(raw.get("max_output_tokens", cls.max_output_tokens)),
        )

    @classmethod
    def default_for_chat_provider(cls, provider: str) -> "ApiConfig":
        chat_provider = _safe_provider(provider)
        embedding_provider = recommended_embedding_provider(chat_provider)
        chat_defaults = provider_default_chat_fields(chat_provider)
        embedding_defaults = provider_default_embedding_fields(embedding_provider)
        return cls(
            chat_provider=chat_provider,
            chat_api_base=chat_defaults["chat_api_base"],
            chat_model=chat_defaults["chat_model"],
            embedding_provider=embedding_provider,
            embedding_api_base=embedding_defaults["embedding_api_base"],
            embedding_model=embedding_defaults["embedding_model"],
        )


@dataclass
class RagConfig:
    """知识库入库、检索与 Chroma 配置。"""

    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "enterprise_knowledge_base")
    distance_metric: str = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    retrieval_candidate_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "120"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "7000"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "1.0"))
    keyword_weight: float = float(os.getenv("KEYWORD_WEIGHT", "0.0"))
    enable_hybrid_retrieval: bool = _parse_bool(os.getenv("ENABLE_HYBRID_RETRIEVAL", "true"), True)
    enable_exact_retrieval: bool = _parse_bool(os.getenv("ENABLE_EXACT_RETRIEVAL", "true"), True)
    enable_structured_retrieval: bool = _parse_bool(os.getenv("ENABLE_STRUCTURED_RETRIEVAL", "true"), True)
    enable_query_planner: bool = _parse_bool(os.getenv("ENABLE_QUERY_PLANNER", "true"), True)
    enable_cross_lingual_variants: bool = _parse_bool(os.getenv("ENABLE_CROSS_LINGUAL_VARIANTS", "true"), True)
    enable_query_rewrite: bool = _parse_bool(os.getenv("ENABLE_QUERY_REWRITE", "false"), False)
    query_rewrite_count: int = int(os.getenv("QUERY_REWRITE_COUNT", "1"))
    default_answer_language: str = os.getenv("DEFAULT_ANSWER_LANGUAGE", "auto")
    language_mode: str = os.getenv("LANGUAGE_MODE", "auto")
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "1200"))
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "380"))
    child_chunk_overlap: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "80"))
    enable_reranker: bool = _parse_bool(os.getenv("ENABLE_RERANKER", "false"), False)
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_device: str = os.getenv("RERANKER_DEVICE", "auto")
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
    reranker_max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
    reranker_candidate_k: int = int(os.getenv("RERANKER_CANDIDATE_K", "60"))
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "12"))
    enable_evidence_judge: bool = _parse_bool(os.getenv("ENABLE_EVIDENCE_JUDGE", "true"), True)
    enable_pdf_ocr: bool = os.getenv("ENABLE_PDF_OCR", "true").strip().lower() not in {"0", "false", "no"}
    pdf_ocr_language_hint: str = os.getenv("PDF_OCR_LANGUAGE_HINT", "auto")
    pdf_ocr_dpi: int = int(os.getenv("PDF_OCR_DPI", "150"))
    pdf_ocr_min_text_chars: int = int(os.getenv("PDF_OCR_MIN_TEXT_CHARS", "80"))
    pdf_ocr_device: str = os.getenv("PDF_OCR_DEVICE", "cpu")
    pdf_ocr_threads: int = int(os.getenv("PDF_OCR_THREADS", "-1"))
    pdf_ocr_max_side_len: int = int(os.getenv("PDF_OCR_MAX_SIDE_LEN", "1400"))

    def normalized(self) -> "RagConfig":
        chunk_size = max(int(self.chunk_size), 200)
        chunk_overlap = max(0, min(int(self.chunk_overlap), chunk_size - 20))
        top_k = max(int(self.top_k), 1)
        retrieval_candidate_k = max(int(self.retrieval_candidate_k), top_k)
        max_context_chars = max(int(self.max_context_chars), 1000)
        metric = str(self.distance_metric or "cosine").strip().lower()
        if metric not in {"cosine", "l2", "ip"}:
            metric = "cosine"

        vector_weight = max(0.0, float(self.vector_weight))
        keyword_weight = max(0.0, float(self.keyword_weight))
        total_weight = vector_weight + keyword_weight
        if total_weight <= 0:
            vector_weight = 0.75
            keyword_weight = 0.25
        else:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight
        pdf_ocr_dpi = max(100, min(int(self.pdf_ocr_dpi), 300))
        query_rewrite_count = max(0, min(int(self.query_rewrite_count), 4))
        rrf_k = max(1, min(int(self.rrf_k), 200))
        parent_chunk_size = max(400, min(int(self.parent_chunk_size), 5000))
        child_chunk_size = max(120, min(int(self.child_chunk_size), 1600))
        child_chunk_overlap = max(0, min(int(self.child_chunk_overlap), child_chunk_size - 20))
        reranker_batch_size = max(1, min(int(self.reranker_batch_size), 64))
        reranker_max_length = max(128, min(int(self.reranker_max_length), 2048))
        reranker_candidate_k = max(1, min(int(self.reranker_candidate_k), 200))
        reranker_top_k = max(1, min(int(self.reranker_top_k), reranker_candidate_k))
        language_mode = str(self.language_mode or "auto").strip().lower()
        if language_mode not in {"auto", "force"}:
            language_mode = "auto"
        default_answer_language = str(self.default_answer_language or "auto").strip().lower()
        pdf_ocr_min_text_chars = max(0, min(int(self.pdf_ocr_min_text_chars), 1000))
        pdf_ocr_device = str(self.pdf_ocr_device or "cpu").strip().lower()
        if pdf_ocr_device not in {"cpu", "auto", "cuda", "directml"}:
            pdf_ocr_device = "cpu"
        pdf_ocr_threads = int(self.pdf_ocr_threads)
        if pdf_ocr_threads < 1:
            pdf_ocr_threads = -1
        pdf_ocr_threads = min(pdf_ocr_threads, 32) if pdf_ocr_threads > 0 else -1
        pdf_ocr_max_side_len = max(960, min(int(self.pdf_ocr_max_side_len), 3000))

        return RagConfig(
            collection_name=sanitize_collection_name(self.collection_name),
            distance_metric=metric,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            retrieval_candidate_k=retrieval_candidate_k,
            max_context_chars=max_context_chars,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            enable_hybrid_retrieval=_parse_bool(self.enable_hybrid_retrieval, True),
            enable_exact_retrieval=_parse_bool(self.enable_exact_retrieval, True),
            enable_structured_retrieval=_parse_bool(self.enable_structured_retrieval, True),
            enable_query_planner=_parse_bool(self.enable_query_planner, True),
            enable_cross_lingual_variants=_parse_bool(self.enable_cross_lingual_variants, True),
            enable_query_rewrite=_parse_bool(self.enable_query_rewrite, False),
            query_rewrite_count=query_rewrite_count,
            default_answer_language=default_answer_language,
            language_mode=language_mode,
            rrf_k=rrf_k,
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            enable_reranker=_parse_bool(self.enable_reranker, False),
            reranker_model=str(self.reranker_model or "BAAI/bge-reranker-v2-m3").strip(),
            reranker_device=str(self.reranker_device or "auto").strip().lower(),
            reranker_batch_size=reranker_batch_size,
            reranker_max_length=reranker_max_length,
            reranker_candidate_k=reranker_candidate_k,
            reranker_top_k=reranker_top_k,
            enable_evidence_judge=_parse_bool(self.enable_evidence_judge, True),
            enable_pdf_ocr=_parse_bool(self.enable_pdf_ocr, True),
            pdf_ocr_language_hint=str(self.pdf_ocr_language_hint or "auto").strip().lower(),
            pdf_ocr_dpi=pdf_ocr_dpi,
            pdf_ocr_min_text_chars=pdf_ocr_min_text_chars,
            pdf_ocr_device=pdf_ocr_device,
            pdf_ocr_threads=pdf_ocr_threads,
            pdf_ocr_max_side_len=pdf_ocr_max_side_len,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = self.normalized()
        return {
            "collection_name": data.collection_name,
            "distance_metric": data.distance_metric,
            "chunk_size": data.chunk_size,
            "chunk_overlap": data.chunk_overlap,
            "top_k": data.top_k,
            "retrieval_candidate_k": data.retrieval_candidate_k,
            "max_context_chars": data.max_context_chars,
            "vector_weight": data.vector_weight,
            "keyword_weight": data.keyword_weight,
            "enable_hybrid_retrieval": data.enable_hybrid_retrieval,
            "enable_exact_retrieval": data.enable_exact_retrieval,
            "enable_structured_retrieval": data.enable_structured_retrieval,
            "enable_query_planner": data.enable_query_planner,
            "enable_cross_lingual_variants": data.enable_cross_lingual_variants,
            "enable_query_rewrite": data.enable_query_rewrite,
            "query_rewrite_count": data.query_rewrite_count,
            "default_answer_language": data.default_answer_language,
            "language_mode": data.language_mode,
            "rrf_k": data.rrf_k,
            "parent_chunk_size": data.parent_chunk_size,
            "child_chunk_size": data.child_chunk_size,
            "child_chunk_overlap": data.child_chunk_overlap,
            "enable_reranker": data.enable_reranker,
            "reranker_model": data.reranker_model,
            "reranker_device": data.reranker_device,
            "reranker_batch_size": data.reranker_batch_size,
            "reranker_max_length": data.reranker_max_length,
            "reranker_candidate_k": data.reranker_candidate_k,
            "reranker_top_k": data.reranker_top_k,
            "enable_evidence_judge": data.enable_evidence_judge,
            "enable_pdf_ocr": data.enable_pdf_ocr,
            "pdf_ocr_language_hint": data.pdf_ocr_language_hint,
            "pdf_ocr_dpi": data.pdf_ocr_dpi,
            "pdf_ocr_min_text_chars": data.pdf_ocr_min_text_chars,
            "pdf_ocr_device": data.pdf_ocr_device,
            "pdf_ocr_threads": data.pdf_ocr_threads,
            "pdf_ocr_max_side_len": data.pdf_ocr_max_side_len,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RagConfig":
        return cls(
            collection_name=str((payload or {}).get("collection_name", cls.collection_name)).strip()
            or cls.collection_name,
            distance_metric=str((payload or {}).get("distance_metric", cls.distance_metric)).strip()
            or cls.distance_metric,
            chunk_size=int((payload or {}).get("chunk_size", cls.chunk_size)),
            chunk_overlap=int((payload or {}).get("chunk_overlap", cls.chunk_overlap)),
            top_k=int((payload or {}).get("top_k", cls.top_k)),
            retrieval_candidate_k=int((payload or {}).get("retrieval_candidate_k", cls.retrieval_candidate_k)),
            max_context_chars=int((payload or {}).get("max_context_chars", cls.max_context_chars)),
            vector_weight=float((payload or {}).get("vector_weight", cls.vector_weight)),
            keyword_weight=float((payload or {}).get("keyword_weight", cls.keyword_weight)),
            enable_hybrid_retrieval=_parse_bool(
                (payload or {}).get("enable_hybrid_retrieval", cls.enable_hybrid_retrieval),
                cls.enable_hybrid_retrieval,
            ),
            enable_exact_retrieval=_parse_bool(
                (payload or {}).get("enable_exact_retrieval", cls.enable_exact_retrieval),
                cls.enable_exact_retrieval,
            ),
            enable_structured_retrieval=_parse_bool(
                (payload or {}).get("enable_structured_retrieval", cls.enable_structured_retrieval),
                cls.enable_structured_retrieval,
            ),
            enable_query_planner=_parse_bool(
                (payload or {}).get("enable_query_planner", cls.enable_query_planner),
                cls.enable_query_planner,
            ),
            enable_cross_lingual_variants=_parse_bool(
                (payload or {}).get("enable_cross_lingual_variants", cls.enable_cross_lingual_variants),
                cls.enable_cross_lingual_variants,
            ),
            enable_query_rewrite=_parse_bool(
                (payload or {}).get("enable_query_rewrite", cls.enable_query_rewrite),
                cls.enable_query_rewrite,
            ),
            query_rewrite_count=int((payload or {}).get("query_rewrite_count", cls.query_rewrite_count)),
            default_answer_language=str((payload or {}).get("default_answer_language", cls.default_answer_language)),
            language_mode=str((payload or {}).get("language_mode", cls.language_mode)),
            rrf_k=int((payload or {}).get("rrf_k", cls.rrf_k)),
            parent_chunk_size=int((payload or {}).get("parent_chunk_size", cls.parent_chunk_size)),
            child_chunk_size=int((payload or {}).get("child_chunk_size", cls.child_chunk_size)),
            child_chunk_overlap=int((payload or {}).get("child_chunk_overlap", cls.child_chunk_overlap)),
            enable_reranker=_parse_bool((payload or {}).get("enable_reranker", cls.enable_reranker), cls.enable_reranker),
            reranker_model=str((payload or {}).get("reranker_model", cls.reranker_model)),
            reranker_device=str((payload or {}).get("reranker_device", cls.reranker_device)),
            reranker_batch_size=int((payload or {}).get("reranker_batch_size", cls.reranker_batch_size)),
            reranker_max_length=int((payload or {}).get("reranker_max_length", cls.reranker_max_length)),
            reranker_candidate_k=int((payload or {}).get("reranker_candidate_k", cls.reranker_candidate_k)),
            reranker_top_k=int((payload or {}).get("reranker_top_k", cls.reranker_top_k)),
            enable_evidence_judge=_parse_bool(
                (payload or {}).get("enable_evidence_judge", cls.enable_evidence_judge),
                cls.enable_evidence_judge,
            ),
            enable_pdf_ocr=_parse_bool((payload or {}).get("enable_pdf_ocr", cls.enable_pdf_ocr), cls.enable_pdf_ocr),
            pdf_ocr_language_hint=str((payload or {}).get("pdf_ocr_language_hint", cls.pdf_ocr_language_hint)),
            pdf_ocr_dpi=int((payload or {}).get("pdf_ocr_dpi", cls.pdf_ocr_dpi)),
            pdf_ocr_min_text_chars=int(
                (payload or {}).get("pdf_ocr_min_text_chars", cls.pdf_ocr_min_text_chars)
            ),
            pdf_ocr_device=str((payload or {}).get("pdf_ocr_device", cls.pdf_ocr_device)).strip().lower(),
            pdf_ocr_threads=int((payload or {}).get("pdf_ocr_threads", cls.pdf_ocr_threads)),
            pdf_ocr_max_side_len=int((payload or {}).get("pdf_ocr_max_side_len", cls.pdf_ocr_max_side_len)),
        ).normalized()


@dataclass
class UIConfig:
    """界面偏好配置。"""

    theme_mode: str = settings.default_theme_mode

    def to_dict(self) -> Dict[str, Any]:
        return {"theme_mode": str(self.theme_mode or settings.default_theme_mode).strip().lower()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UIConfig":
        return cls(theme_mode=str((payload or {}).get("theme_mode", settings.default_theme_mode)).strip().lower())


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _config_defaults(factory: Callable[[], TConfig]) -> TConfig:
    return factory()


def _is_versioned_config_store(payload: Dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and int(payload.get("schema_version") or 0) >= 2
        and isinstance(payload.get("scopes"), dict)
    )


def _scope_key(scope_id: str | None = None) -> str:
    return str(scope_id or DEFAULT_WORKSPACE_SCOPE_ID or "default").strip() or "default"


def _extract_scoped_payload(raw: Dict[str, Any], scope_id: str | None = None) -> Dict[str, Any]:
    if not raw:
        return {}
    if _is_versioned_config_store(raw):
        scopes = raw.get("scopes") or {}
        scoped = scopes.get(_scope_key(scope_id)) or scopes.get("default") or {}
        if isinstance(scoped, dict):
            payload = scoped.get(CONFIG_PAYLOAD_KEY, scoped)
            return dict(payload) if isinstance(payload, dict) else {}
        return {}
    return dict(raw)


def _build_scoped_store(path: Path, payload: Dict[str, Any], scope_id: str | None = None) -> Dict[str, Any]:
    scope = _scope_key(scope_id)
    try:
        existing = _read_json_file(path)
    except Exception:
        existing = {}
    if not _is_versioned_config_store(existing):
        existing = {
            "schema_version": CONFIG_STORE_VERSION,
            "default_scope_id": scope,
            "scopes": {},
        }
    scopes = dict(existing.get("scopes") or {})
    scopes[scope] = {
        "schema_version": CONFIG_STORE_VERSION,
        "scope_id": scope,
        CONFIG_PAYLOAD_KEY: payload,
    }
    existing["schema_version"] = CONFIG_STORE_VERSION
    existing["default_scope_id"] = str(existing.get("default_scope_id") or scope)
    existing["scopes"] = scopes
    return existing


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(text)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(tmp_path, path)


def _load_config(
    path: Path,
    factory: Callable[[], TConfig],
    parser: Callable[[Dict[str, Any]], TConfig],
    scope_id: str | None = None,
) -> TConfig:
    try:
        raw = _read_json_file(path)
        return parser(_extract_scoped_payload(raw, scope_id=scope_id))
    except Exception:
        return _config_defaults(factory)


def _save_config(
    config: Any,
    path: Path,
    scope_id: str | None = None,
) -> None:
    payload = config.to_dict()
    store = _build_scoped_store(path, payload=payload, scope_id=scope_id)
    _atomic_write_json(path, store)
    persisted = _extract_scoped_payload(_read_json_file(path), scope_id=scope_id)
    if persisted != payload:
        raise RuntimeError(f"Configuration persistence verification failed for {path}")


def load_api_config(path: Path = API_CONFIG_FILE, scope_id: str | None = None) -> ApiConfig:
    return _load_config(path, ApiConfig, ApiConfig.from_dict, scope_id=scope_id)


def save_api_config(config: ApiConfig, path: Path = API_CONFIG_FILE, scope_id: str | None = None) -> None:
    _save_config(config, path, scope_id=scope_id)


def load_rag_config(path: Path = RAG_CONFIG_FILE, scope_id: str | None = None) -> RagConfig:
    return _load_config(path, lambda: RagConfig().normalized(), RagConfig.from_dict, scope_id=scope_id)


def save_rag_config(config: RagConfig, path: Path = RAG_CONFIG_FILE, scope_id: str | None = None) -> None:
    _save_config(config, path, scope_id=scope_id)


def load_ui_config(path: Path = UI_CONFIG_FILE, scope_id: str | None = None) -> UIConfig:
    return _load_config(path, UIConfig, UIConfig.from_dict, scope_id=scope_id)


def save_ui_config(config: UIConfig, path: Path = UI_CONFIG_FILE, scope_id: str | None = None) -> None:
    _save_config(config, path, scope_id=scope_id)


def ensure_directories() -> None:
    for folder in (
        DATA_DIR,
        Path(settings.chroma_persist_dir),
        Path(settings.upload_dir),
        Path(settings.temp_dir),
        Path(settings.log_dir),
    ):
        folder.mkdir(parents=True, exist_ok=True)


ensure_directories()
