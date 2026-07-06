from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import (
    ApiConfig,
    RagConfig,
    UIConfig,
    load_api_config,
    load_rag_config,
    save_api_config,
    save_rag_config,
    save_ui_config,
)
import rag_ui.api as ui_api


class FakeSessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value


def test_current_chat_model_input_wins_over_stale_manual_cache(monkeypatch):
    state = FakeSessionState(
        {
            "api_form_chat_provider": "deepseek",
            "api_form_chat_api_key": "test-key",
            "api_form_chat_api_base": "https://api.deepseek.com/v1",
            "api_form_chat_model": "V4-flash",
            "api_form_chat_model_manual": True,
            "api_form_chat_model_manual_value": "deepseek-chat",
            "api_form_embedding_provider": "local",
            "api_form_embedding_api_key": "",
            "api_form_embedding_api_base": "local",
            "api_form_embedding_model": "intfloat/multilingual-e5-small",
            "api_form_embedding_same_as_chat": False,
            "api_form_temperature": 0.2,
            "api_form_max_output_tokens": 2048,
        }
    )
    monkeypatch.setattr(ui_api.st, "session_state", state)

    config = ui_api._build_api_config_from_state()

    assert config.chat_model == "V4-flash"


def test_blank_api_form_state_is_detected_as_broken(monkeypatch):
    state = FakeSessionState(
        {
            "api_form_chat_provider": "openai",
            "api_form_chat_api_base": "",
            "api_form_chat_model": "",
            "api_form_embedding_provider": "local",
            "api_form_embedding_model": "intfloat/multilingual-e5-small",
        }
    )
    monkeypatch.setattr(ui_api.st, "session_state", state)

    assert ui_api._api_form_state_is_broken() is True


def test_minimum_rag_form_state_is_detected_as_broken(monkeypatch):
    state = FakeSessionState(
        {
            "rag_form_collection_name": "enterprise_knowledge_base",
            "rag_form_top_k": 1,
            "rag_form_retrieval_candidate_k": 1,
            "rag_form_max_context_chars": 1000,
            "rag_form_rrf_k": 1,
            "rag_form_parent_chunk_size": 400,
            "rag_form_child_chunk_size": 120,
            "rag_form_enable_hybrid_retrieval": False,
            "rag_form_enable_exact_retrieval": False,
            "rag_form_enable_structured_retrieval": False,
            "rag_form_enable_query_planner": False,
        }
    )
    monkeypatch.setattr(ui_api.st, "session_state", state)

    assert ui_api._rag_form_state_is_broken() is True


def test_api_form_rehydrates_from_saved_signature_when_browser_state_is_stale(monkeypatch):
    state = FakeSessionState(
        {
            "api_config_data": {
                "chat_provider": "deepseek",
                "chat_api_key": "saved-key",
                "chat_api_base": "https://api.deepseek.com/v1",
                "chat_model": "deepseek-v4-flash",
                "embedding_provider": "local",
                "embedding_api_key": "",
                "embedding_api_base": "local",
                "embedding_model": "intfloat/multilingual-e5-small",
                "temperature": 0.25,
                "max_output_tokens": 1028,
            },
            "api_form_schema_version": ui_api.API_FORM_SCHEMA_VERSION,
            "api_form_loaded_signature": "stale-signature",
            "api_form_chat_provider": "openai",
            "api_form_chat_api_key": "",
            "api_form_chat_api_base": "https://api.openai.com/v1",
            "api_form_chat_model": "gpt-4o-mini",
            "api_form_embedding_provider": "local",
            "api_form_embedding_api_key": "",
            "api_form_embedding_api_base": "local",
            "api_form_embedding_model": "intfloat/multilingual-e5-small",
            "api_form_temperature": 0.0,
            "api_form_max_output_tokens": 128,
        }
    )
    monkeypatch.setattr(ui_api.st, "session_state", state)

    ui_api._seed_api_form_state()

    assert state["api_form_chat_provider"] == "deepseek"
    assert state["api_form_chat_api_base"] == "https://api.deepseek.com/v1"
    assert state["api_form_chat_model"] == "deepseek-v4-flash"


def test_rag_form_rehydrates_from_saved_signature_when_browser_state_is_stale(monkeypatch):
    state = FakeSessionState(
        {
            "rag_config_data": {
                "collection_name": "enterprise_knowledge_base",
                "distance_metric": "cosine",
                "chunk_size": 1000,
                "chunk_overlap": 120,
                "top_k": 6,
                "retrieval_candidate_k": 120,
                "max_context_chars": 7000,
                "vector_weight": 1.0,
                "keyword_weight": 0.0,
                "enable_hybrid_retrieval": True,
                "enable_exact_retrieval": True,
                "enable_structured_retrieval": True,
                "enable_query_planner": True,
                "enable_cross_lingual_variants": True,
                "enable_query_rewrite": False,
                "query_rewrite_count": 1,
                "default_answer_language": "auto",
                "language_mode": "auto",
                "rrf_k": 60,
                "parent_chunk_size": 1200,
                "child_chunk_size": 380,
                "child_chunk_overlap": 80,
                "enable_reranker": True,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_device": "auto",
                "reranker_batch_size": 8,
                "reranker_max_length": 512,
                "reranker_candidate_k": 60,
                "reranker_top_k": 12,
                "enable_evidence_judge": False,
                "enable_pdf_ocr": True,
                "pdf_ocr_language_hint": "auto",
                "pdf_ocr_dpi": 150,
                "pdf_ocr_min_text_chars": 80,
                "pdf_ocr_device": "directml",
                "pdf_ocr_threads": -1,
                "pdf_ocr_max_side_len": 1400,
            },
            "rag_form_schema_version": ui_api.RAG_FORM_SCHEMA_VERSION,
            "rag_form_loaded_signature": "stale-signature",
            "rag_form_collection_name": "",
            "rag_form_top_k": 1,
            "rag_form_retrieval_candidate_k": 1,
            "rag_form_max_context_chars": 1000,
            "rag_form_rrf_k": 1,
            "rag_form_parent_chunk_size": 400,
            "rag_form_child_chunk_size": 120,
            "rag_form_enable_hybrid_retrieval": False,
            "rag_form_enable_exact_retrieval": False,
            "rag_form_enable_structured_retrieval": False,
            "rag_form_enable_query_planner": False,
        }
    )
    monkeypatch.setattr(ui_api.st, "session_state", state)

    ui_api._seed_rag_form_state()

    assert state["rag_form_top_k"] == 6
    assert state["rag_form_retrieval_candidate_k"] == 120
    assert state["rag_form_enable_query_planner"] is True


def test_scoped_config_persists_across_reload_and_isolated_by_workspace(tmp_path):
    path = tmp_path / "api_config.json"
    first = ApiConfig(chat_provider="deepseek", chat_api_key="scope-a", chat_api_base="https://api.deepseek.com/v1", chat_model="deepseek-chat")
    second = ApiConfig(chat_provider="openai", chat_api_key="scope-b", chat_model="gpt-test")

    save_api_config(first, path=path, scope_id="workspace-a")
    save_api_config(second, path=path, scope_id="workspace-b")

    assert load_api_config(path=path, scope_id="workspace-a").chat_api_key == "scope-a"
    assert load_api_config(path=path, scope_id="workspace-b").chat_api_key == "scope-b"


def test_legacy_config_payload_is_migrated_on_next_save(tmp_path):
    path = tmp_path / "rag_config.json"
    path.write_text('{"top_k": 9, "collection_name": "legacy_space"}', encoding="utf-8")

    loaded = load_rag_config(path=path, scope_id="workspace-a")
    assert loaded.top_k == 9
    assert loaded.collection_name == "legacy_space"

    save_rag_config(loaded, path=path, scope_id="workspace-a")
    raw = path.read_text(encoding="utf-8")
    assert '"schema_version": 2' in raw
    assert '"workspace-a"' in raw


def test_save_failure_does_not_update_session_state_or_fake_success(monkeypatch):
    state = FakeSessionState(
        {
            "config_hydrated": True,
            "api_config_data": {"chat_model": "old-model"},
            "rag_config_data": RagConfig().to_dict(),
            "ui_config_data": UIConfig().to_dict(),
            "api_form_chat_provider": "openai",
            "api_form_chat_api_key": "key",
            "api_form_chat_api_base": "https://api.openai.com/v1",
            "api_form_chat_model": "new-model",
            "api_form_embedding_provider": "local",
            "api_form_embedding_api_key": "",
            "api_form_embedding_api_base": "local",
            "api_form_embedding_model": "intfloat/multilingual-e5-small",
            "api_form_embedding_same_as_chat": False,
            "api_form_temperature": 0.2,
            "api_form_max_output_tokens": 1024,
            "rag_form_collection_name": "enterprise_knowledge_base",
            "rag_form_distance_metric": "cosine",
            "rag_form_chunk_size": 1000,
            "rag_form_chunk_overlap": 120,
            "rag_form_top_k": 6,
            "rag_form_retrieval_candidate_k": 120,
            "rag_form_max_context_chars": 7000,
            "rag_form_vector_weight": 1.0,
            "rag_form_keyword_weight": 0.0,
            "rag_form_enable_hybrid_retrieval": True,
            "rag_form_enable_exact_retrieval": True,
            "rag_form_enable_structured_retrieval": True,
            "rag_form_enable_query_planner": True,
            "rag_form_enable_cross_lingual_variants": True,
            "rag_form_enable_query_rewrite": False,
            "rag_form_query_rewrite_count": 1,
            "rag_form_default_answer_language": "auto",
            "rag_form_language_mode": "auto",
            "rag_form_rrf_k": 60,
            "rag_form_parent_chunk_size": 1200,
            "rag_form_child_chunk_size": 380,
            "rag_form_child_chunk_overlap": 80,
            "rag_form_enable_reranker": False,
            "rag_form_reranker_model": "BAAI/bge-reranker-v2-m3",
            "rag_form_reranker_device": "auto",
            "rag_form_reranker_batch_size": 8,
            "rag_form_reranker_max_length": 512,
            "rag_form_reranker_candidate_k": 60,
            "rag_form_reranker_top_k": 12,
            "rag_form_enable_evidence_judge": True,
            "rag_form_enable_pdf_ocr": True,
            "rag_form_pdf_ocr_language_hint": "auto",
            "rag_form_pdf_ocr_dpi": 150,
            "rag_form_pdf_ocr_min_text_chars": 80,
            "rag_form_pdf_ocr_device": "cpu",
            "rag_form_pdf_ocr_threads": -1,
            "rag_form_pdf_ocr_max_side_len": 1400,
            "ui_form_theme_mode": "dark",
        }
    )
    monkeypatch.setattr(ui_api.st, "session_state", state)

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ui_api, "save_api_config", fail_save)

    with pytest.raises(OSError):
        ui_api.save_runtime_configs()

    assert state["api_config_data"]["chat_model"] == "old-model"


def test_save_is_blocked_before_hydration(monkeypatch):
    monkeypatch.setattr(ui_api.st, "session_state", FakeSessionState({"config_hydrated": False}))

    with pytest.raises(RuntimeError):
        ui_api.save_runtime_configs()
