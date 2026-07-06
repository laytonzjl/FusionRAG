from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

import streamlit as st

from config import (
    ApiConfig,
    CONVERSATION_STORE_FILE,
    RagConfig,
    UIConfig,
    load_api_config,
    load_rag_config,
    load_ui_config,
    provider_api_key_hint,
    provider_default_chat_fields,
    provider_default_embedding_fields,
    provider_label,
    recommended_embedding_provider,
    save_api_config,
    save_rag_config,
    save_ui_config,
)
from rag_core import build_engine


WELCOME_MESSAGE = "你好，我是你的知识库助手。请先导入文件，我会基于检索到的来源证据回答问题。"
CONVERSATION_STORE_VERSION = 1
MAX_SAVED_CONVERSATIONS = 80
MAX_MESSAGES_PER_CONVERSATION = 120

CHAT_PROVIDER_KEYS = ["openai", "claude", "deepseek", "qwen", "custom"]
EMBEDDING_PROVIDER_KEYS = ["local", "openai", "qwen", "custom"]
API_FORM_SCHEMA_VERSION = "settings_entry_hydration_v10_structured_planner"
RAG_FORM_SCHEMA_VERSION = "settings_entry_hydration_v5_structured_planner"
CONFIG_HYDRATION_VERSION = "config_store_v4_settings_entry"


def _stable_signature(payload: Dict[str, Any]) -> str:
    try:
        return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"


@st.cache_resource(show_spinner=False)
def get_engine(config_payload: str):
    """基于当前配置缓存知识库引擎，避免频繁重复初始化。"""

    payload = json.loads(config_payload)
    api_config = ApiConfig.from_dict(payload.get("api", {}))
    rag_config = RagConfig.from_dict(payload.get("rag", {}))
    return build_engine(api_config=api_config, rag_config=rag_config)


def _load_saved_runtime_configs() -> tuple[ApiConfig, RagConfig, UIConfig]:
    """从持久化存储读取已保存的运行配置。"""

    return load_api_config(), load_rag_config(), load_ui_config()


def _runtime_config_signature(
    api_config: ApiConfig,
    rag_config: RagConfig,
    ui_config: UIConfig,
) -> str:
    """生成持久配置版本签名，用于阻止旧表单覆盖新配置。"""

    return _stable_signature(
        {
            "api": api_config.to_dict(),
            "rag": rag_config.to_dict(),
            "ui": ui_config.to_dict(),
        }
    )


def _refresh_saved_config_cache(
    api_config: ApiConfig | None = None,
    rag_config: RagConfig | None = None,
    ui_config: UIConfig | None = None,
) -> tuple[ApiConfig, RagConfig, UIConfig]:
    """刷新非 widget 的配置缓存，不自行覆盖当前编辑中的设置表单。"""

    if api_config is None or rag_config is None or ui_config is None:
        api_config, rag_config, ui_config = _load_saved_runtime_configs()

    st.session_state.api_config_data = api_config.to_dict()
    st.session_state.rag_config_data = rag_config.to_dict()
    st.session_state.ui_config_data = ui_config.to_dict()
    st.session_state.config_hydration_version = CONFIG_HYDRATION_VERSION
    st.session_state.config_hydrated = True
    return api_config, rag_config, ui_config


def hydrate_runtime_forms_from_saved_config() -> None:
    """以磁盘中的已保存配置重建设置表单。

    Streamlit 会清理未在当前页面渲染的 widget key。该函数只允许在
    设置页控件创建之前调用，或作为按钮 callback 调用，避免覆盖用户
    正在编辑的字段。
    """

    api_config, rag_config, ui_config = _load_saved_runtime_configs()
    _refresh_saved_config_cache(api_config, rag_config, ui_config)

    _apply_api_config_to_form(api_config, force=True)
    st.session_state.api_form_embedding_same_as_chat = bool(
        api_config.embedding_provider != "local"
        and api_config.embedding_api_key
        and api_config.embedding_api_key == api_config.chat_api_key
    )
    _normalize_api_embedding_form_state()

    _apply_rag_config_to_form(rag_config, force=True)
    st.session_state.ui_form_theme_mode = _normalize_theme_mode(ui_config.theme_mode)

    st.session_state.api_form_schema_version = API_FORM_SCHEMA_VERSION
    st.session_state.rag_form_schema_version = RAG_FORM_SCHEMA_VERSION
    st.session_state.api_form_loaded_signature = _stable_signature(api_config.to_dict())
    st.session_state.rag_form_loaded_signature = _stable_signature(rag_config.to_dict())
    st.session_state.ui_form_loaded_signature = _stable_signature(ui_config.to_dict())
    st.session_state.runtime_form_hydration_signature = _runtime_config_signature(
        api_config,
        rag_config,
        ui_config,
    )


def init_session_state() -> None:
    """初始化页面会话状态。"""

    _seed_conversations()
    if st.session_state.get("config_hydration_version") != CONFIG_HYDRATION_VERSION:
        _refresh_saved_config_cache()
    else:
        st.session_state.setdefault("api_config_data", load_api_config().to_dict())
        st.session_state.setdefault("rag_config_data", load_rag_config().to_dict())
        st.session_state.setdefault("ui_config_data", load_ui_config().to_dict())
        st.session_state.config_hydrated = True
    defaults = {
        "last_upload_result": None,
        "last_upload_errors": [],
        "last_retrieval": [],
        "last_answer_meta": None,
        "keyword_filter": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    _seed_api_form_state()
    _seed_rag_form_state()
    _seed_ui_form_state()


def _seed_conversations() -> None:
    if "conversations" not in st.session_state:
        store = load_conversation_store()
        st.session_state.conversations = store["conversations"]
        st.session_state.current_conversation_id = store["current_conversation_id"]
        current_id = st.session_state.current_conversation_id
        st.session_state.messages = st.session_state.conversations[current_id]["messages"]
        return

    current_id = st.session_state.get("current_conversation_id")
    if current_id not in st.session_state.conversations:
        current_id = next(iter(st.session_state.conversations))
        st.session_state.current_conversation_id = current_id
    st.session_state.messages = st.session_state.conversations[current_id]["messages"]


def clear_chat_history() -> None:
    create_new_conversation()


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_conversation_payload() -> Dict[str, Any]:
    conversation_id = uuid4().hex
    return {
        "id": conversation_id,
        "title": "新对话",
        "created_at": _now_text(),
        "updated_at": _now_text(),
        "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}],
    }


def _sanitize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    role = str(message.get("role") or "assistant")
    if role not in {"assistant", "user", "system"}:
        role = "assistant"
    sanitized: Dict[str, Any] = {
        "role": role,
        "content": str(message.get("content") or ""),
    }
    meta = message.get("meta")
    if isinstance(meta, dict):
        sanitized["meta"] = meta
    return sanitized


def _sanitize_conversation(conversation: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(conversation or {})
    conversation_id = str(payload.get("id") or uuid4().hex)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    clean_messages = [_sanitize_message(item) for item in messages[-MAX_MESSAGES_PER_CONVERSATION:] if isinstance(item, dict)]
    if not clean_messages:
        clean_messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    return {
        "id": conversation_id,
        "title": str(payload.get("title") or "新对话")[:60],
        "created_at": str(payload.get("created_at") or _now_text()),
        "updated_at": str(payload.get("updated_at") or payload.get("created_at") or _now_text()),
        "messages": clean_messages,
    }


def load_conversation_store() -> Dict[str, Any]:
    try:
        raw = json.loads(CONVERSATION_STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        conversation = _new_conversation_payload()
        return {
            "conversations": {conversation["id"]: conversation},
            "current_conversation_id": conversation["id"],
        }

    raw_conversations = raw.get("conversations") if isinstance(raw, dict) else None
    if not isinstance(raw_conversations, dict) or not raw_conversations:
        conversation = _new_conversation_payload()
        return {
            "conversations": {conversation["id"]: conversation},
            "current_conversation_id": conversation["id"],
        }

    conversations: Dict[str, Dict[str, Any]] = {}
    sorted_items = sorted(
        raw_conversations.values(),
        key=lambda item: str((item or {}).get("updated_at") or (item or {}).get("created_at") or ""),
        reverse=True,
    )
    for item in sorted_items[:MAX_SAVED_CONVERSATIONS]:
        if isinstance(item, dict):
            conversation = _sanitize_conversation(item)
            conversations[conversation["id"]] = conversation

    if not conversations:
        conversation = _new_conversation_payload()
        conversations = {conversation["id"]: conversation}

    current_id = str(raw.get("current_conversation_id") or "")
    if current_id not in conversations:
        current_id = next(iter(conversations))
    return {"conversations": conversations, "current_conversation_id": current_id}


def save_conversation_store() -> None:
    conversations = st.session_state.get("conversations", {})
    if not isinstance(conversations, dict) or not conversations:
        return
    sanitized: Dict[str, Dict[str, Any]] = {}
    for item in conversations.values():
        if isinstance(item, dict):
            conversation = _sanitize_conversation(item)
            sanitized[conversation["id"]] = conversation
    payload = {
        "version": CONVERSATION_STORE_VERSION,
        "current_conversation_id": st.session_state.get("current_conversation_id"),
        "conversations": sanitized,
    }
    CONVERSATION_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONVERSATION_STORE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_new_conversation() -> None:
    conversation = _new_conversation_payload()
    conversation_id = conversation["id"]
    st.session_state.conversations[conversation_id] = conversation
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = st.session_state.conversations[conversation_id]["messages"]
    st.session_state.last_retrieval = []
    st.session_state.last_answer_meta = None
    save_conversation_store()


def switch_conversation(conversation_id: str) -> None:
    if conversation_id not in st.session_state.get("conversations", {}):
        return
    st.session_state.current_conversation_id = conversation_id
    st.session_state.messages = st.session_state.conversations[conversation_id]["messages"]
    st.session_state.last_retrieval = []
    st.session_state.last_answer_meta = None
    save_conversation_store()


def delete_conversation(conversation_id: str) -> None:
    conversations = st.session_state.get("conversations", {})
    if conversation_id not in conversations:
        return
    conversations.pop(conversation_id, None)
    if not conversations:
        conversation = _new_conversation_payload()
        conversations[conversation["id"]] = conversation
    if st.session_state.get("current_conversation_id") == conversation_id:
        next_id = next(iter(conversations))
        st.session_state.current_conversation_id = next_id
        st.session_state.messages = conversations[next_id]["messages"]
    save_conversation_store()


def persist_current_conversation() -> None:
    conversation_id = st.session_state.get("current_conversation_id")
    if conversation_id in st.session_state.get("conversations", {}):
        st.session_state.conversations[conversation_id]["messages"] = st.session_state.messages
        st.session_state.conversations[conversation_id]["updated_at"] = _now_text()
        save_conversation_store()


def maybe_update_conversation_title(user_query: str) -> None:
    conversation_id = st.session_state.get("current_conversation_id")
    if conversation_id not in st.session_state.get("conversations", {}):
        return
    conversation = st.session_state.conversations[conversation_id]
    if conversation.get("title") == "新对话":
        title = (user_query or "").strip().replace("\n", " ")
        conversation["title"] = title[:18] or "新对话"
        conversation["updated_at"] = _now_text()
        save_conversation_store()


def _seed_api_form_state() -> None:
    """初始化表单状态，但不把局部编辑或临时空值视为“损坏配置”。

    设置页重新进入时的强制同步由 hydrate_runtime_forms_from_saved_config()
    负责；这里仅处理全新 session 或 schema / 已保存配置变更。
    """

    config = ApiConfig.from_dict(st.session_state.get("api_config_data") or {})
    saved_signature = _stable_signature(config.to_dict())
    force_refresh = (
        st.session_state.get("api_form_schema_version") != API_FORM_SCHEMA_VERSION
        or st.session_state.get("api_form_loaded_signature") != saved_signature
    )
    _apply_api_config_to_form(config, force=force_refresh)
    st.session_state.api_form_schema_version = API_FORM_SCHEMA_VERSION
    st.session_state.api_form_loaded_signature = saved_signature

    if "api_form_embedding_same_as_chat" not in st.session_state:
        st.session_state.api_form_embedding_same_as_chat = bool(
            config.embedding_provider != "local"
            and config.embedding_api_key
            and config.embedding_api_key == config.chat_api_key
        )
    _normalize_api_embedding_form_state()


def _normalize_api_embedding_form_state() -> None:
    """保持本地向量 UI 约束，不将其当作表单 hydration 的触发条件。"""

    if st.session_state.get("api_form_embedding_provider") != "local":
        return

    st.session_state.api_form_embedding_same_as_chat = False
    st.session_state.api_form_embedding_api_key = ""
    st.session_state.api_form_embedding_api_base = "local"
    current_embedding_model = str(st.session_state.get("api_form_embedding_model", ""))
    if current_embedding_model.startswith("text-embedding") or current_embedding_model.startswith("local-hashing"):
        st.session_state.api_form_embedding_model = "intfloat/multilingual-e5-small"


def _seed_rag_form_state() -> None:
    config = RagConfig.from_dict(st.session_state.get("rag_config_data") or {})
    saved_signature = _stable_signature(config.to_dict())
    force_refresh = (
        st.session_state.get("rag_form_schema_version") != RAG_FORM_SCHEMA_VERSION
        or st.session_state.get("rag_form_loaded_signature") != saved_signature
    )
    _apply_rag_config_to_form(config, force=force_refresh)
    st.session_state.rag_form_schema_version = RAG_FORM_SCHEMA_VERSION
    st.session_state.rag_form_loaded_signature = saved_signature


def _apply_rag_config_to_form(config: RagConfig, force: bool = True) -> None:
    mapping = {
        "rag_form_collection_name": config.collection_name,
        "rag_form_distance_metric": config.distance_metric,
        "rag_form_chunk_size": config.chunk_size,
        "rag_form_chunk_overlap": config.chunk_overlap,
        "rag_form_top_k": config.top_k,
        "rag_form_retrieval_candidate_k": config.retrieval_candidate_k,
        "rag_form_max_context_chars": config.max_context_chars,
        "rag_form_vector_weight": config.vector_weight,
        "rag_form_keyword_weight": config.keyword_weight,
        "rag_form_enable_hybrid_retrieval": config.enable_hybrid_retrieval,
        "rag_form_enable_exact_retrieval": config.enable_exact_retrieval,
        "rag_form_enable_structured_retrieval": config.enable_structured_retrieval,
        "rag_form_enable_query_planner": config.enable_query_planner,
        "rag_form_enable_cross_lingual_variants": config.enable_cross_lingual_variants,
        "rag_form_enable_query_rewrite": config.enable_query_rewrite,
        "rag_form_query_rewrite_count": config.query_rewrite_count,
        "rag_form_default_answer_language": config.default_answer_language,
        "rag_form_language_mode": config.language_mode,
        "rag_form_rrf_k": config.rrf_k,
        "rag_form_parent_chunk_size": config.parent_chunk_size,
        "rag_form_child_chunk_size": config.child_chunk_size,
        "rag_form_child_chunk_overlap": config.child_chunk_overlap,
        "rag_form_enable_reranker": config.enable_reranker,
        "rag_form_reranker_model": config.reranker_model,
        "rag_form_reranker_device": config.reranker_device,
        "rag_form_reranker_batch_size": config.reranker_batch_size,
        "rag_form_reranker_max_length": config.reranker_max_length,
        "rag_form_reranker_candidate_k": config.reranker_candidate_k,
        "rag_form_reranker_top_k": config.reranker_top_k,
        "rag_form_enable_evidence_judge": config.enable_evidence_judge,
        "rag_form_enable_pdf_ocr": config.enable_pdf_ocr,
        "rag_form_pdf_ocr_language_hint": config.pdf_ocr_language_hint,
        "rag_form_pdf_ocr_dpi": config.pdf_ocr_dpi,
        "rag_form_pdf_ocr_min_text_chars": config.pdf_ocr_min_text_chars,
        "rag_form_pdf_ocr_device": config.pdf_ocr_device,
        "rag_form_pdf_ocr_threads": config.pdf_ocr_threads,
        "rag_form_pdf_ocr_max_side_len": config.pdf_ocr_max_side_len,
    }
    for key, value in mapping.items():
        if force or key not in st.session_state:
            st.session_state[key] = value


def _seed_ui_form_state() -> None:
    config = UIConfig.from_dict(st.session_state.get("ui_config_data") or {})
    saved_signature = _stable_signature(config.to_dict())
    if (
        "ui_form_theme_mode" not in st.session_state
        or st.session_state.get("ui_form_loaded_signature") != saved_signature
    ):
        st.session_state.ui_form_theme_mode = _normalize_theme_mode(config.theme_mode)
    st.session_state.ui_form_loaded_signature = saved_signature


def _normalize_theme_mode(theme_mode: str) -> str:
    value = str(theme_mode or "light").strip().lower()
    return value if value in {"light", "dark", "graphite"} else "light"


def _apply_api_config_to_form(config: ApiConfig, force: bool = True) -> None:
    mapping = {
        "api_form_chat_provider": config.chat_provider,
        "api_form_chat_api_key": config.chat_api_key,
        "api_form_chat_api_base": config.chat_api_base,
        "api_form_chat_model": config.chat_model,
        "api_form_embedding_provider": config.embedding_provider,
        "api_form_embedding_api_key": config.embedding_api_key,
        "api_form_embedding_api_base": config.embedding_api_base,
        "api_form_embedding_model": config.embedding_model,
        "api_form_temperature": config.temperature,
        "api_form_max_output_tokens": config.max_output_tokens,
    }
    for key, value in mapping.items():
        if force or key not in st.session_state:
            st.session_state[key] = value
    default_model = provider_default_chat_fields(config.chat_provider)["chat_model"]
    if force or "api_form_chat_model_manual" not in st.session_state:
        st.session_state.api_form_chat_model_manual = bool(config.chat_model and config.chat_model != default_model)
    if force or "api_form_chat_model_manual_value" not in st.session_state:
        st.session_state.api_form_chat_model_manual_value = config.chat_model


def _apply_chat_provider_defaults(
    provider: str,
    sync_embedding: bool = True,
    update_provider_key: bool = True,
) -> None:
    chat_provider = str(provider or "openai").strip().lower()
    chat_defaults = provider_default_chat_fields(chat_provider)
    if update_provider_key:
        st.session_state.api_form_chat_provider = chat_provider
    st.session_state.api_form_chat_api_base = chat_defaults["chat_api_base"]
    st.session_state.api_form_chat_model = chat_defaults["chat_model"]
    st.session_state.api_form_chat_model_manual = False
    st.session_state.api_form_chat_model_manual_value = chat_defaults["chat_model"]

    if sync_embedding:
        embedding_provider = recommended_embedding_provider(chat_provider)
        st.session_state.api_form_embedding_provider = embedding_provider
        _apply_embedding_provider_defaults(embedding_provider)


def _apply_embedding_provider_defaults(provider: str, update_provider_key: bool = True) -> None:
    embedding_provider = str(provider or "openai").strip().lower()
    embedding_defaults = provider_default_embedding_fields(embedding_provider)
    if update_provider_key:
        st.session_state.api_form_embedding_provider = embedding_provider
    st.session_state.api_form_embedding_api_base = embedding_defaults["embedding_api_base"]
    st.session_state.api_form_embedding_model = embedding_defaults["embedding_model"]
    if embedding_provider == "local":
        st.session_state.api_form_embedding_same_as_chat = False
        st.session_state.api_form_embedding_api_key = ""


def on_chat_provider_change() -> None:
    _apply_chat_provider_defaults(
        st.session_state.get("api_form_chat_provider", "openai"),
        sync_embedding=True,
        update_provider_key=False,
    )


def on_chat_model_change() -> None:
    model_name = str(st.session_state.get("api_form_chat_model", "")).strip()
    st.session_state.api_form_chat_model_manual = True
    st.session_state.api_form_chat_model_manual_value = model_name


def on_embedding_provider_change() -> None:
    _apply_embedding_provider_defaults(
        st.session_state.get("api_form_embedding_provider", "openai"),
        update_provider_key=False,
    )


def on_chat_api_key_change() -> None:
    if (
        st.session_state.get("api_form_embedding_same_as_chat", True)
        and st.session_state.get("api_form_embedding_provider") != "local"
    ):
        st.session_state.api_form_embedding_api_key = st.session_state.get("api_form_chat_api_key", "")


def on_embedding_same_as_chat_change() -> None:
    if st.session_state.get("api_form_embedding_provider") == "local":
        st.session_state.api_form_embedding_same_as_chat = False
        st.session_state.api_form_embedding_api_key = ""
    elif st.session_state.get("api_form_embedding_same_as_chat", True):
        st.session_state.api_form_embedding_api_key = st.session_state.get("api_form_chat_api_key", "")


def apply_current_chat_provider_defaults() -> None:
    _apply_chat_provider_defaults(
        st.session_state.get("api_form_chat_provider", "openai"),
        sync_embedding=False,
        update_provider_key=False,
    )


def apply_current_embedding_provider_defaults() -> None:
    _apply_embedding_provider_defaults(
        st.session_state.get("api_form_embedding_provider", "openai"),
        update_provider_key=False,
    )
    on_embedding_same_as_chat_change()


def apply_deepseek_local_defaults() -> None:
    """一键设置为 DeepSeek 聊天 + 本地 Embedding，同时保留用户已输入的聊天 Key。"""

    existing_chat_key = st.session_state.get("api_form_chat_api_key", "")
    _apply_chat_provider_defaults("deepseek", sync_embedding=False)
    st.session_state.api_form_chat_api_key = existing_chat_key
    _apply_embedding_provider_defaults("local")


def apply_recommended_rag_defaults() -> None:
    """恢复适合多语言混合检索的推荐参数，不修改 API Key。"""

    recommended = RagConfig(
        collection_name=str(st.session_state.get("rag_form_collection_name", "enterprise_knowledge_base")).strip()
        or "enterprise_knowledge_base",
        distance_metric=str(st.session_state.get("rag_form_distance_metric", "cosine")).strip().lower() or "cosine",
        chunk_size=1000,
        chunk_overlap=120,
        top_k=6,
        retrieval_candidate_k=120,
        max_context_chars=7000,
        vector_weight=1.0,
        keyword_weight=0.0,
        enable_hybrid_retrieval=True,
        enable_exact_retrieval=True,
        enable_structured_retrieval=True,
        enable_query_planner=True,
        enable_cross_lingual_variants=True,
        enable_query_rewrite=True,
        query_rewrite_count=2,
        default_answer_language="auto",
        language_mode="auto",
        rrf_k=60,
        parent_chunk_size=1200,
        child_chunk_size=380,
        child_chunk_overlap=80,
        enable_reranker=True,
        reranker_model=str(st.session_state.get("rag_form_reranker_model", "BAAI/bge-reranker-v2-m3"))
        or "BAAI/bge-reranker-v2-m3",
        reranker_device=str(st.session_state.get("rag_form_reranker_device", "auto")) or "auto",
        reranker_batch_size=8,
        reranker_max_length=512,
        reranker_candidate_k=60,
        reranker_top_k=12,
        enable_evidence_judge=True,
        enable_pdf_ocr=True,
        pdf_ocr_language_hint=str(st.session_state.get("rag_form_pdf_ocr_language_hint", "auto")) or "auto",
        pdf_ocr_dpi=150,
        pdf_ocr_min_text_chars=80,
        pdf_ocr_device=str(st.session_state.get("rag_form_pdf_ocr_device", "cpu")) or "cpu",
        pdf_ocr_threads=-1,
        pdf_ocr_max_side_len=1400,
    ).normalized()
    st.session_state.rag_config_data = recommended.to_dict()
    for key, value in {
        "rag_form_collection_name": recommended.collection_name,
        "rag_form_distance_metric": recommended.distance_metric,
        "rag_form_chunk_size": recommended.chunk_size,
        "rag_form_chunk_overlap": recommended.chunk_overlap,
        "rag_form_top_k": recommended.top_k,
        "rag_form_retrieval_candidate_k": recommended.retrieval_candidate_k,
        "rag_form_max_context_chars": recommended.max_context_chars,
        "rag_form_vector_weight": recommended.vector_weight,
        "rag_form_keyword_weight": recommended.keyword_weight,
        "rag_form_enable_hybrid_retrieval": recommended.enable_hybrid_retrieval,
        "rag_form_enable_exact_retrieval": recommended.enable_exact_retrieval,
        "rag_form_enable_structured_retrieval": recommended.enable_structured_retrieval,
        "rag_form_enable_query_planner": recommended.enable_query_planner,
        "rag_form_enable_cross_lingual_variants": recommended.enable_cross_lingual_variants,
        "rag_form_enable_query_rewrite": recommended.enable_query_rewrite,
        "rag_form_query_rewrite_count": recommended.query_rewrite_count,
        "rag_form_default_answer_language": recommended.default_answer_language,
        "rag_form_language_mode": recommended.language_mode,
        "rag_form_rrf_k": recommended.rrf_k,
        "rag_form_parent_chunk_size": recommended.parent_chunk_size,
        "rag_form_child_chunk_size": recommended.child_chunk_size,
        "rag_form_child_chunk_overlap": recommended.child_chunk_overlap,
        "rag_form_enable_reranker": recommended.enable_reranker,
        "rag_form_reranker_model": recommended.reranker_model,
        "rag_form_reranker_device": recommended.reranker_device,
        "rag_form_reranker_batch_size": recommended.reranker_batch_size,
        "rag_form_reranker_max_length": recommended.reranker_max_length,
        "rag_form_reranker_candidate_k": recommended.reranker_candidate_k,
        "rag_form_reranker_top_k": recommended.reranker_top_k,
        "rag_form_enable_evidence_judge": recommended.enable_evidence_judge,
        "rag_form_enable_pdf_ocr": recommended.enable_pdf_ocr,
        "rag_form_pdf_ocr_language_hint": recommended.pdf_ocr_language_hint,
        "rag_form_pdf_ocr_dpi": recommended.pdf_ocr_dpi,
        "rag_form_pdf_ocr_min_text_chars": recommended.pdf_ocr_min_text_chars,
        "rag_form_pdf_ocr_device": recommended.pdf_ocr_device,
        "rag_form_pdf_ocr_threads": recommended.pdf_ocr_threads,
        "rag_form_pdf_ocr_max_side_len": recommended.pdf_ocr_max_side_len,
    }.items():
        st.session_state[key] = value


def _build_api_config_from_state() -> ApiConfig:
    chat_provider = str(st.session_state.get("api_form_chat_provider", "openai")).strip().lower()
    embedding_provider = str(st.session_state.get("api_form_embedding_provider", "openai")).strip().lower()

    chat_defaults = provider_default_chat_fields(chat_provider)
    embedding_defaults = provider_default_embedding_fields(embedding_provider)

    chat_api_key = str(st.session_state.get("api_form_chat_api_key", "")).strip()
    embedding_api_key = str(st.session_state.get("api_form_embedding_api_key", "")).strip()
    if embedding_provider == "local":
        embedding_api_key = ""
    elif st.session_state.get("api_form_embedding_same_as_chat", True):
        embedding_api_key = chat_api_key
    elif not embedding_api_key and embedding_provider == chat_provider:
        embedding_api_key = chat_api_key

    embedding_api_base = str(st.session_state.get("api_form_embedding_api_base", "")).strip()
    embedding_model = str(st.session_state.get("api_form_embedding_model", "")).strip()
    if embedding_provider == "local":
        embedding_api_base = embedding_defaults["embedding_api_base"]

    # 保存时必须以当前输入框为准。旧的 manual_model 缓存只用于兼容历史会话，
    # 不能覆盖用户刚刚输入的模型名。
    chat_model = str(st.session_state.get("api_form_chat_model", "")).strip()
    if not chat_model:
        chat_model = str(st.session_state.get("api_form_chat_model_manual_value", "")).strip()

    return ApiConfig(
        chat_provider=chat_provider,
        chat_api_key=chat_api_key,
        chat_api_base=str(st.session_state.get("api_form_chat_api_base", "")).strip()
        or chat_defaults["chat_api_base"],
        chat_model=chat_model or chat_defaults["chat_model"],
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        embedding_api_base=embedding_api_base or embedding_defaults["embedding_api_base"],
        embedding_model=embedding_model or embedding_defaults["embedding_model"],
        temperature=float(st.session_state.get("api_form_temperature", 0.2)),
        max_output_tokens=int(st.session_state.get("api_form_max_output_tokens", 1024)),
    )


def _build_rag_config_from_state() -> RagConfig:
    return RagConfig(
        collection_name=str(st.session_state.get("rag_form_collection_name", "enterprise_knowledge_base")).strip(),
        distance_metric=str(st.session_state.get("rag_form_distance_metric", "cosine")).strip().lower(),
        chunk_size=int(st.session_state.get("rag_form_chunk_size", 600)),
        chunk_overlap=int(st.session_state.get("rag_form_chunk_overlap", 120)),
        top_k=int(st.session_state.get("rag_form_top_k", 6)),
        retrieval_candidate_k=int(st.session_state.get("rag_form_retrieval_candidate_k", 120)),
        max_context_chars=int(st.session_state.get("rag_form_max_context_chars", 7000)),
        vector_weight=float(st.session_state.get("rag_form_vector_weight", 1.0)),
        keyword_weight=float(st.session_state.get("rag_form_keyword_weight", 0.0)),
        enable_hybrid_retrieval=bool(st.session_state.get("rag_form_enable_hybrid_retrieval", True)),
        enable_exact_retrieval=bool(st.session_state.get("rag_form_enable_exact_retrieval", True)),
        enable_structured_retrieval=bool(st.session_state.get("rag_form_enable_structured_retrieval", True)),
        enable_query_planner=bool(st.session_state.get("rag_form_enable_query_planner", True)),
        enable_cross_lingual_variants=bool(st.session_state.get("rag_form_enable_cross_lingual_variants", True)),
        enable_query_rewrite=bool(st.session_state.get("rag_form_enable_query_rewrite", True)),
        query_rewrite_count=int(st.session_state.get("rag_form_query_rewrite_count", 2)),
        default_answer_language=str(st.session_state.get("rag_form_default_answer_language", "auto")),
        language_mode=str(st.session_state.get("rag_form_language_mode", "auto")),
        rrf_k=int(st.session_state.get("rag_form_rrf_k", 60)),
        parent_chunk_size=int(st.session_state.get("rag_form_parent_chunk_size", 1200)),
        child_chunk_size=int(st.session_state.get("rag_form_child_chunk_size", 380)),
        child_chunk_overlap=int(st.session_state.get("rag_form_child_chunk_overlap", 80)),
        enable_reranker=bool(st.session_state.get("rag_form_enable_reranker", False)),
        reranker_model=str(st.session_state.get("rag_form_reranker_model", "BAAI/bge-reranker-v2-m3")),
        reranker_device=str(st.session_state.get("rag_form_reranker_device", "auto")),
        reranker_batch_size=int(st.session_state.get("rag_form_reranker_batch_size", 8)),
        reranker_max_length=int(st.session_state.get("rag_form_reranker_max_length", 512)),
        reranker_candidate_k=int(st.session_state.get("rag_form_reranker_candidate_k", 60)),
        reranker_top_k=int(st.session_state.get("rag_form_reranker_top_k", 12)),
        enable_evidence_judge=bool(st.session_state.get("rag_form_enable_evidence_judge", True)),
        enable_pdf_ocr=bool(st.session_state.get("rag_form_enable_pdf_ocr", True)),
        pdf_ocr_language_hint=str(st.session_state.get("rag_form_pdf_ocr_language_hint", "auto")),
        pdf_ocr_dpi=int(st.session_state.get("rag_form_pdf_ocr_dpi", 150)),
        pdf_ocr_min_text_chars=int(st.session_state.get("rag_form_pdf_ocr_min_text_chars", 80)),
        pdf_ocr_device=str(st.session_state.get("rag_form_pdf_ocr_device", "cpu")).strip().lower(),
        pdf_ocr_threads=int(st.session_state.get("rag_form_pdf_ocr_threads", -1)),
        pdf_ocr_max_side_len=int(st.session_state.get("rag_form_pdf_ocr_max_side_len", 1400)),
    ).normalized()


def _build_ui_config_from_state() -> UIConfig:
    return UIConfig(theme_mode=_normalize_theme_mode(st.session_state.get("ui_form_theme_mode", "light")))


def build_engine_payload() -> str:
    payload = {
        "api": _build_api_config_from_state().to_dict(),
        "rag": _build_rag_config_from_state().to_dict(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def get_saved_api_config() -> ApiConfig:
    return ApiConfig.from_dict(st.session_state.get("api_config_data") or {})


def get_saved_rag_config() -> RagConfig:
    return RagConfig.from_dict(st.session_state.get("rag_config_data") or {})


def get_saved_ui_config() -> UIConfig:
    return UIConfig.from_dict(st.session_state.get("ui_config_data") or {})


def is_runtime_config_ready(api_config: ApiConfig) -> bool:
    embedding_ready = bool(api_config.embedding_model)
    if api_config.embedding_provider != "local":
        embedding_ready = bool(api_config.embedding_api_key and api_config.embedding_model)

    return bool(
        api_config.chat_api_key
        and api_config.chat_model
        and embedding_ready
    )


def save_runtime_configs() -> None:
    if not st.session_state.get("config_hydrated"):
        raise RuntimeError("配置尚未完成初始化，已阻止保存以避免默认值覆盖持久配置。")

    # 设置表单必须来自本次进入设置页时的持久化快照，防止 Streamlit 清理
    # widget state 后的空值，或旧页面 state，回写覆盖 data/*.json。
    source_signature = st.session_state.get("runtime_form_hydration_signature")
    persisted_api, persisted_rag, persisted_ui = _load_saved_runtime_configs()
    persisted_signature = _runtime_config_signature(persisted_api, persisted_rag, persisted_ui)
    if not source_signature:
        raise RuntimeError("设置表单尚未从已保存配置同步；请重新进入“系统设置”后再保存。")
    if source_signature != persisted_signature:
        raise RuntimeError("已保存配置在本次编辑期间发生变化；请重新加载设置后再保存，避免覆盖最新配置。")

    api_config = _build_api_config_from_state()
    rag_config = _build_rag_config_from_state()
    ui_config = _build_ui_config_from_state()

    save_api_config(api_config)
    save_rag_config(rag_config)
    save_ui_config(ui_config)

    _refresh_saved_config_cache(api_config, rag_config, ui_config)
    st.session_state.api_form_loaded_signature = _stable_signature(st.session_state.api_config_data)
    st.session_state.rag_form_loaded_signature = _stable_signature(st.session_state.rag_config_data)
    st.session_state.ui_form_loaded_signature = _stable_signature(st.session_state.ui_config_data)
    st.session_state.runtime_form_hydration_signature = _runtime_config_signature(
        api_config,
        rag_config,
        ui_config,
    )
    st.session_state.api_form_chat_model_manual = True
    st.session_state.api_form_chat_model_manual_value = api_config.chat_model


def save_ui_theme_from_state() -> None:
    if not st.session_state.get("config_hydrated"):
        raise RuntimeError("配置尚未完成初始化，已阻止保存以避免默认值覆盖持久配置。")
    ui_config = _build_ui_config_from_state()
    save_ui_config(ui_config)
    st.session_state.ui_config_data = ui_config.to_dict()
    st.session_state.ui_form_loaded_signature = _stable_signature(st.session_state.ui_config_data)


def api_key_placeholder(provider: str) -> str:
    return provider_api_key_hint(provider)


def provider_display_name(provider: str) -> str:
    return provider_label(provider)


def chat_provider_options() -> Dict[str, str]:
    return {key: provider_label(key) for key in CHAT_PROVIDER_KEYS}


def embedding_provider_options() -> Dict[str, str]:
    return {key: provider_label(key) for key in EMBEDDING_PROVIDER_KEYS}


def provider_options() -> Dict[str, str]:
    return chat_provider_options()
