from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from typing import Optional

import streamlit as st

from config import ApiConfig, RagConfig, UIConfig
from rag_core.parsers import get_ocr_runtime_info
from rag_ui.api import (
    api_key_placeholder,
    apply_current_chat_provider_defaults,
    apply_current_embedding_provider_defaults,
    apply_deepseek_local_defaults,
    apply_recommended_rag_defaults,
    chat_provider_options,
    clear_chat_history,
    delete_conversation,
    embedding_provider_options,
    get_engine,
    get_saved_api_config,
    get_saved_rag_config,
    get_saved_ui_config,
    hydrate_runtime_forms_from_saved_config,
    init_session_state,
    is_runtime_config_ready,
    maybe_update_conversation_title,
    on_chat_api_key_change,
    on_chat_model_change,
    on_chat_provider_change,
    on_embedding_provider_change,
    on_embedding_same_as_chat_change,
    persist_current_conversation,
    provider_display_name,
    save_runtime_configs,
    switch_conversation,
)
from rag_ui.theme import inject_custom_styles


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "高"
    if score >= 0.5:
        return "中"
    return "低"


def _confidence_class(score: float) -> str:
    if score >= 0.75:
        return "confidence-high"
    if score >= 0.5:
        return "confidence-medium"
    return "confidence-low"


def _build_answer_meta(results) -> dict:
    if not results:
        return {
            "confidence_score": 0.12,
            "confidence_label": "低",
            "source_count": 0,
            "source_type_label": "未命中来源",
            "source_cards": [],
            "source_tags": [],
        }

    confidence_score = _calibrated_confidence(results)
    if confidence_score >= 0.75:
        source_type_label = "直接证据"
    elif confidence_score >= 0.5:
        source_type_label = "多证据归纳"
    else:
        source_type_label = "间接证据"

    source_cards = []
    source_tags = []
    for item in results[:3]:
        metadata = item.metadata
        file_name = metadata.get("file_name") or metadata.get("source_ref") or "unknown"
        source_type = str(metadata.get("source_type") or "unknown").upper()
        page = metadata.get("page_start") or metadata.get("page")
        row_index = metadata.get("row_index")

        location_parts = []
        if page is not None:
            location_parts.append(f"第 {page} 页")
        if row_index is not None:
            location_parts.append(f"第 {row_index} 行")

        location_text = " / ".join(location_parts) if location_parts else "未提供定位"
        escaped_file_name = html.escape(str(file_name))
        escaped_source_type = html.escape(source_type)

        source_cards.append(
            {
                "file_name": escaped_file_name,
                "source_type": escaped_source_type,
                "location": html.escape(location_text),
                "score": f"{float(item.final_score or 0.0):.4f}",
                "preview": html.escape(item.content[:220].replace("\n", " ").strip()),
            }
        )
        source_tags.append(f"{escaped_file_name} | {escaped_source_type}")

    return {
        "confidence_score": confidence_score,
        "confidence_label": _confidence_label(confidence_score),
        "source_count": len(results),
        "source_type_label": source_type_label,
        "source_cards": source_cards,
        "source_tags": source_tags,
    }


def _calibrated_confidence(results) -> float:
    if not results:
        return 0.1
    top = results[0]
    entity_coverage_failed = any(
        (item.diagnostics or {}).get("entity_coverage_failed") for item in results[:3]
    )
    contributions = getattr(top, "contributions", []) or []
    channels = {item.channel for item in contributions}
    exact_bonus = 0.18 if any(channel.startswith("exact") for channel in channels) else 0.0
    structured_bonus = 0.12 if "structured" in channels else 0.0
    rerank_score = getattr(top, "rerank_score", None)
    rerank_bonus = 0.0
    if rerank_score is not None:
        rerank_bonus = max(0.0, min(0.22, (float(rerank_score) + 5.0) / 10.0 * 0.22))
    source_bonus = min(len(results), 4) * 0.05
    quality_values = []
    for item in results[:4]:
        try:
            quality_values.append(float((item.metadata or {}).get("text_quality_score", 0.75)))
        except Exception:
            quality_values.append(0.75)
    quality = sum(quality_values) / max(len(quality_values), 1)
    rrf_signal = min(float(getattr(top, "rrf_score", 0.0) or 0.0) * 18.0, 0.22)
    confidence = 0.18 + exact_bonus + structured_bonus + rerank_bonus + source_bonus + rrf_signal + quality * 0.18
    source_files = {
        str((item.metadata or {}).get("file_name") or (item.metadata or {}).get("source_ref") or "")
        for item in results[:4]
    }
    source_files.discard("")
    if len(source_files) > 1 and exact_bonus <= 0:
        confidence = min(confidence, 0.48)
    # 实体链接失败是审计风险，不是“答案只能有 28% 可信”的硬判决。
    # 检索器会保留该诊断，供 UI 展示和用户复核。
    if entity_coverage_failed:
        confidence *= 0.78
    return max(0.08, min(0.96, confidence))


def _adjust_answer_meta_by_answer(meta: dict, answer: str) -> dict:
    """模型拒答或证据不足时，置信度应代表“答案可信度”，而不是单纯检索相似度。"""

    if not meta:
        return meta
    answer_text = answer or ""
    has_sources = int(meta.get("source_count") or 0) > 0
    clear_uncertain_patterns = (
        "无法确定",
        "无法直接确定",
        "无法直接给出",
        "无法直接回答",
        "不能确定",
        "没有足够",
        "未提及",
        "未说明",
        "缺乏",
        "不含年龄",
        "无年龄信息",
    )
    if any((item.diagnostics or {}).get("entity_coverage_failed") for item in st.session_state.get("last_retrieval", [])[:3]):
        meta = dict(meta)
        meta["entity_coverage_failed"] = True
        meta["coverage_note"] = "主体实体链接不完整；请核验检索对象与来源。"
    clear_partial_patterns = (
        "现有证据可确认",
        "仅能确认",
        "可以确认",
        "可确认",
        "部分",
        "完整名单",
    )
    if any(pattern in answer_text for pattern in clear_uncertain_patterns) and any(
        pattern in answer_text for pattern in clear_partial_patterns
    ):
        adjusted = dict(meta)
        adjusted["confidence_score"] = min(max(float(adjusted.get("confidence_score", 0.0)), 0.45), 0.62)
        adjusted["confidence_label"] = _confidence_label(float(adjusted["confidence_score"]))
        adjusted["source_type_label"] = "部分证据"
        return adjusted
    if any(pattern in answer_text for pattern in clear_uncertain_patterns):
        adjusted = dict(meta)
        adjusted["source_type_label"] = "直接证据不足"
        adjusted["answer_mode"] = "insufficient_direct_evidence"
        return adjusted
    useful_answer_patterns = (
        "结论",
        "证据",
        "现有证据可确认",
        "可以确认",
        "可确认",
        "来源：",
    )
    partial_answer_patterns = (
        "无法直接给出",
        "完整名单",
        "所有",
        "仅能确认",
        "未列出",
    )
    uncertain_patterns = (
        "无法确定",
        "没有足够",
        "未提及",
        "无法直接回答",
        "不能确定",
        "缺少",
        "未系统列出",
    )
    has_useful_answer = has_sources and any(pattern in answer_text for pattern in useful_answer_patterns)
    has_partial_answer = has_sources and any(pattern in answer_text for pattern in partial_answer_patterns)
    if any(pattern in answer_text for pattern in uncertain_patterns) and not has_useful_answer:
        adjusted = dict(meta)
        adjusted["source_type_label"] = "直接证据不足"
        adjusted["answer_mode"] = "insufficient_direct_evidence"
        return adjusted
    if has_partial_answer:
        adjusted = dict(meta)
        adjusted["confidence_score"] = min(max(float(adjusted.get("confidence_score", 0.0)), 0.45), 0.68)
        adjusted["confidence_label"] = _confidence_label(float(adjusted["confidence_score"]))
        adjusted["source_type_label"] = "部分证据"
        return adjusted
    return meta


def render_header(
    api_config: ApiConfig,
    rag_config: RagConfig,
    ui_config: UIConfig,
    engine: Optional[object],
) -> None:
    file_count = 0
    chunk_count = 0
    try:
        if engine:
            file_count = len(engine.list_files())
            chunk_count = engine.collection_count()
    except Exception:
        file_count = 0
        chunk_count = 0

    title_col, stat_col = st.columns([0.62, 0.38], vertical_alignment="center")
    with title_col:
        st.markdown(
            f"""
            <div class="chat-shell-title-block">
                <div class="chat-title">知识库助手</div>
                <div class="chat-subtitle">
                    {provider_display_name(api_config.chat_provider)} | {rag_config.collection_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col:
        st.markdown(
            f"""
            <div class="status-row">
                <span class="status-pill">{file_count} 个文件</span>
                <span class="status-pill">{chunk_count} 个切片</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(engine: Optional[object]) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-row">
                <div class="sidebar-brand">知识库 <span>Plus</span></div>
                <div class="sidebar-collapse-icon">▣</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("↗ 新聊天", use_container_width=True, key="sidebar_new_chat"):
            clear_chat_history()
            st.session_state.current_view = "chat"
            st.rerun()
        if st.button("▣ 知识库管理", use_container_width=True, key="nav_knowledge"):
            st.session_state.current_view = "knowledge"
            st.rerun()
        if st.button("⚙ 系统设置", use_container_width=True, key="nav_settings"):
            st.session_state.current_view = "settings"
            # 显式标记一次性 hydration；避免聊天页未渲染设置控件后 widget key 被清理。
            st.session_state._settings_form_hydration_requested = True
            st.rerun()
        st.markdown('<div class="sidebar-section-title">最近</div>', unsafe_allow_html=True)
        conversations = st.session_state.get("conversations", {})
        current_id = st.session_state.get("current_conversation_id")
        sorted_conversations = sorted(
            conversations.values(),
            key=lambda item: str(item.get("updated_at") or item.get("created_at", "")),
            reverse=True,
        )
        for conversation in sorted_conversations[:12]:
            title = str(conversation.get("title") or "新对话")
            is_current = conversation.get("id") == current_id
            label = f"• {title}" if is_current else title
            st.markdown('<div class="conversation-row">', unsafe_allow_html=True)
            conv_col, delete_col = st.columns([0.84, 0.16], gap="small", vertical_alignment="center")
            with conv_col:
                if st.button(
                    label,
                    key=f"conversation_{conversation.get('id')}",
                    use_container_width=True,
                    help=title,
                ):
                    switch_conversation(str(conversation.get("id")))
                    st.session_state.current_view = "chat"
                    st.rerun()
            with delete_col:
                if st.button(
                    "×",
                    key=f"delete_conversation_{conversation.get('id')}",
                    use_container_width=True,
                    help="删除对话",
                ):
                    delete_conversation(str(conversation.get("id")))
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_knowledge_import_panel(engine: Optional[object]) -> None:
        with st.form("knowledge_ingest_form", clear_on_submit=False):
            uploaded_files = st.file_uploader(
                "上传文件",
                type=["txt", "pdf", "csv"],
                accept_multiple_files=True,
                key="uploaded_files",
                label_visibility="collapsed",
            )
            selected_count = len(uploaded_files or [])
            if selected_count:
                st.caption(f"已选择 {selected_count} 个文件，点击下方按钮后才会真正解析并写入知识库。")
            submitted = st.form_submit_button(
                "创建/更新索引",
                use_container_width=True,
                type="primary",
                disabled=engine is None,
            )

        if submitted:
            if engine is None:
                st.warning("请先完成聊天模型和向量配置。")
            elif not uploaded_files:
                st.warning("请至少选择一个文件。")
            else:
                results = []
                errors = []
                progress = st.progress(0, text="准备创建索引...")
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    progress.progress(
                        index / max(len(uploaded_files), 1),
                        text=f"正在处理：{uploaded_file.name}",
                    )
                    try:
                        results.append(engine.ingest_uploaded_file(uploaded_file))
                    except Exception as exc:
                        errors.append({"file_name": uploaded_file.name, "error": str(exc)})
                progress.empty()
                st.session_state.last_upload_result = results
                st.session_state.last_upload_errors = errors
                if results:
                    st.success(f"已成功入库 {len(results)} 个文件。")
                if errors:
                    st.error(f"{len(errors)} 个文件入库失败，请展开下方错误详情。")
                if results:
                    st.rerun()

        st.text_input(
            "关键词过滤",
            key="keyword_filter",
            placeholder="例如：创新点、方法、实验",
        )

        if st.session_state.get("last_upload_result"):
            with st.expander("最近导入", expanded=False):
                for item in st.session_state.last_upload_result:
                    if item.get("skipped"):
                        st.caption(f"{item['file_name']} | 已跳过：{item.get('skip_reason', '重复文件')}")
                        continue
                    timing = item.get("timing") or {}
                    total_seconds = timing.get("total_seconds")
                    timing_text = f" | 耗时 {total_seconds}s" if total_seconds is not None else ""
                    st.caption(f"{item['file_name']} | {item['chunk_count']} 个切片{timing_text}")
        if st.session_state.get("last_upload_errors"):
            with st.expander("最近失败", expanded=True):
                for item in st.session_state.last_upload_errors:
                    st.error(f"{item['file_name']}：{item['error']}")


def render_file_management_panel(engine: Optional[object]) -> None:
        if engine is None:
            st.caption("知识库尚未加载。")
            return

        try:
            file_items = engine.list_files()
        except Exception as exc:
            st.error(f"文件列表加载失败：{exc}")
            return

        try:
            compatibility = engine.index_compatibility()
            if not compatibility.get("compatible"):
                st.warning("当前索引与多语言/结构化检索配置不兼容，建议从已保存上传文件重建索引。")
                with st.expander("索引兼容性详情", expanded=False):
                    st.json(compatibility.get("mismatches") or {})
        except Exception:
            pass

        if not file_items:
            st.caption("暂无已入库文件。")
            if st.button("从已保存上传文件重建索引", key="rebuild_uploads_empty", use_container_width=True):
                try:
                    with st.spinner("正在从已保存文件重建索引..."):
                        result = engine.rebuild_from_uploads()
                    indexed = result.get("indexed", [])
                    errors = result.get("errors", [])
                    if indexed:
                        st.session_state.last_upload_result = indexed
                        st.success(f"已索引 {len(indexed)} 个已保存文件。")
                    else:
                        st.warning("没有可索引的已保存上传文件。")
                    if errors:
                        st.error(f"{len(errors)} 个文件索引失败，请查看日志。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"重建失败：{exc}")

        for item in file_items:
            with st.container(border=True):
                st.write(f"**{item['file_name']}**")
                st.caption(f"{item['source_type']} | {item['chunk_count']} 个切片")
                if st.button("删除", key=f"delete_{item['file_name']}", use_container_width=True):
                    try:
                        deleted = engine.delete_file(item["file_name"])
                        st.success(f"已删除 {deleted} 条索引记录，并清理对应上传源文件。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

        if file_items and st.button("从已保存上传文件重建索引", key="rebuild_uploads_existing", use_container_width=True):
            try:
                with st.spinner("正在从已保存文件重建索引..."):
                    result = engine.rebuild_from_uploads()
                indexed = result.get("indexed", [])
                errors = result.get("errors", [])
                if indexed:
                    st.session_state.last_upload_result = indexed
                    st.success(f"已索引 {len(indexed)} 个已保存文件。")
                else:
                    st.warning("没有可索引的已保存上传文件。")
                if errors:
                    st.error(f"{len(errors)} 个文件索引失败，请查看日志。")
                st.rerun()
            except Exception as exc:
                st.error(f"重建失败：{exc}")

        if st.button("清空当前知识库和上传文件", use_container_width=True):
            try:
                result = engine.reset_collection(delete_uploads=True)
                st.session_state.last_upload_result = []
                st.session_state.last_upload_errors = []
                deleted_uploads = int((result or {}).get("deleted_upload_files", 0))
                st.success(f"当前知识库已清空，并删除 {deleted_uploads} 个已上传源文件。")
                st.rerun()
            except Exception as exc:
                st.error(f"清空失败：{exc}")


def render_model_settings(api_config: ApiConfig, rag_config: RagConfig) -> None:
    with st.expander("运行配置与检索参数", expanded=False):
        _render_model_settings_content(api_config, rag_config)


def _render_model_settings_content(api_config: ApiConfig, rag_config: RagConfig) -> None:
    chat_provider_keys = list(chat_provider_options().keys())
    embedding_provider_keys = list(embedding_provider_options().keys())
    local_embedding_selected = st.session_state.get("api_form_embedding_provider", "local") == "local"

    if local_embedding_selected:
        st.session_state.api_form_embedding_same_as_chat = False
        st.session_state.api_form_embedding_api_key = ""
        st.session_state.api_form_embedding_api_base = "local"
        current_embedding_model = str(st.session_state.get("api_form_embedding_model", ""))
        if current_embedding_model.startswith("text-embedding") or current_embedding_model.startswith("local-hashing"):
            st.session_state.api_form_embedding_model = "intfloat/multilingual-e5-small"

    if (
        api_config.embedding_provider != "local"
        and api_config.embedding_provider == "openai"
        and "openai.com" in (api_config.embedding_api_base or "")
        and api_config.embedding_api_key
        and api_config.embedding_api_key == api_config.chat_api_key
        and api_config.chat_provider != "openai"
    ):
        st.warning(
            "当前向量配置为 OpenAI，但使用了非 OpenAI 聊天模型的同一个 Key。"
            "建议切换为本地向量，或填写真实可用的 OpenAI 向量 Key。"
        )

    with st.container(border=True):
        reload_col, quick_col, reset_col, save_col = st.columns([0.24, 0.27, 0.23, 0.26])
        with reload_col:
            st.button(
                "重新加载已保存配置",
                key="reload_saved_runtime_config",
                use_container_width=True,
                on_click=hydrate_runtime_forms_from_saved_config,
                help="放弃当前未保存的修改，重新读取 data/*.json 中的配置。",
            )
        with quick_col:
            if st.button(
                "一键切换为 DeepSeek + 本地向量",
                key="quick_deepseek_local",
                use_container_width=True,
            ):
                try:
                    apply_deepseek_local_defaults()
                    save_runtime_configs()
                    st.success("已应用！")
                    st.rerun()
                except Exception as exc:
                    st.error(f"保存失败：{exc}")
        with reset_col:
            if st.button(
                "恢复默认",
                key="reset_recommended_rag_defaults",
                use_container_width=True,
            ):
                try:
                    apply_recommended_rag_defaults()
                    save_runtime_configs()
                    st.success("已恢复推荐检索参数。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"恢复失败：{exc}")
        with save_col:
            if st.button("保存设置", key="save_settings_top", use_container_width=True):
                try:
                    save_runtime_configs()
                    st.success("设置已保存。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"保存失败：{exc}")

        tab_chat, tab_embedding, tab_retrieval = st.tabs(["聊天模型", "本地向量", "检索参数"])

        with tab_chat:
            st.selectbox(
                "聊天服务商",
                options=chat_provider_keys,
                key="api_form_chat_provider",
                format_func=provider_display_name,
                on_change=on_chat_provider_change,
            )
            st.button(
                "加载聊天默认配置",
                key="apply_chat_provider_defaults",
                use_container_width=True,
                on_click=apply_current_chat_provider_defaults,
            )
            st.text_input(
                "聊天 API Key",
                key="api_form_chat_api_key",
                type="password",
                placeholder=api_key_placeholder(st.session_state.get("api_form_chat_provider", "openai")),
                on_change=on_chat_api_key_change,
            )
            st.text_input("聊天接口地址 Base URL", key="api_form_chat_api_base")
            st.text_input("聊天模型名称", key="api_form_chat_model", on_change=on_chat_model_change)

            col_a, col_b = st.columns(2)
            with col_a:
                st.slider("生成随机性", 0.0, 1.0, key="api_form_temperature", step=0.05)
            with col_b:
                st.number_input(
                    "最大输出长度 Token",
                    min_value=128,
                    max_value=8192,
                    key="api_form_max_output_tokens",
                    step=64,
                )

        with tab_embedding:
            st.info(
                "推荐使用本地向量：文档切片和查询向量都在本机生成，不会请求 OpenAI 向量 API。"
            )
            st.selectbox(
                "向量服务",
                options=embedding_provider_keys,
                key="api_form_embedding_provider",
                format_func=provider_display_name,
                on_change=on_embedding_provider_change,
            )
            st.button(
                "加载向量默认配置",
                key="apply_embedding_provider_defaults",
                use_container_width=True,
                on_click=apply_current_embedding_provider_defaults,
            )

            local_embedding_selected = st.session_state.get("api_form_embedding_provider", "local") == "local"
            st.checkbox(
                "向量服务复用聊天 API Key",
                key="api_form_embedding_same_as_chat",
                on_change=on_embedding_same_as_chat_change,
                disabled=local_embedding_selected,
                help="本地向量不需要任何 API Key。只有当你的网关同时支持聊天和向量接口时，才建议开启。",
            )
            st.text_input(
                "向量 API Key",
                key="api_form_embedding_api_key",
                type="password",
                placeholder=api_key_placeholder(st.session_state.get("api_form_embedding_provider", "local")),
                disabled=local_embedding_selected or st.session_state.get("api_form_embedding_same_as_chat", False),
            )
            st.text_input("向量接口地址 Base URL", key="api_form_embedding_api_base", disabled=local_embedding_selected)
            st.text_input(
                "向量模型名或本地模型路径",
                key="api_form_embedding_model",
                help="默认使用真实本地语义模型 intfloat/multilingual-e5-small；也可填写其他 Hugging Face 模型名或本地模型目录。",
            )
            if local_embedding_selected:
                st.caption("当前模式：本地向量。API Key 和接口地址会被忽略。")

        with tab_retrieval:
            st.text_input("知识库集合名", key="rag_form_collection_name")
            st.selectbox("距离度量", options=["cosine", "l2", "ip"], key="rag_form_distance_metric")
            st.caption(
                "当前版本使用 Parent/Child 结构化切块 + Exact/Lexical/Dense/Structured 多路召回 + RRF/rerank。"
                "旧切片和线性权重参数仅用于兼容旧索引，不建议继续调。"
            )

            col_c, col_d = st.columns(2)
            with col_c:
                st.number_input(
                    "切片大小（旧版兼容）",
                    min_value=200,
                    max_value=4000,
                    key="rag_form_chunk_size",
                    step=50,
                    disabled=True,
                    help="旧版平铺 Chunk 参数。新版建库请调高级检索里的 Parent/Child Chunk。",
                )
                st.number_input("返回片段数 Top-K", min_value=1, max_value=20, key="rag_form_top_k", step=1)
                st.slider(
                    "向量得分权重（旧版兼容）",
                    0.0,
                    1.0,
                    key="rag_form_vector_weight",
                    step=0.05,
                    disabled=True,
                    help="旧版向量/关键词线性融合权重。新版使用 Weighted RRF，不再直接使用该权重。",
                )
            with col_d:
                st.number_input(
                    "切片重叠（旧版兼容）",
                    min_value=0,
                    max_value=2000,
                    key="rag_form_chunk_overlap",
                    step=20,
                    disabled=True,
                    help="旧版平铺 Chunk overlap。新版建库请调 Child overlap。",
                )
                st.number_input(
                    "候选片段数",
                    min_value=1,
                    max_value=200,
                    key="rag_form_retrieval_candidate_k",
                    step=1,
                    help="进入多路召回和 RRF 的基础候选池。推荐 120；开启 rerank 时可用 120-180。",
                )
                st.slider(
                    "关键词得分权重（旧版兼容）",
                    0.0,
                    1.0,
                    key="rag_form_keyword_weight",
                    step=0.05,
                    disabled=True,
                    help="旧版线性融合权重。新版词法检索通过独立通道参与 RRF。",
                )

            st.number_input(
                "最大上下文字符数",
                min_value=1000,
                max_value=20000,
                key="rag_form_max_context_chars",
                step=500,
            )
            st.divider()
            with st.expander("高级检索 / 准确模式 / 多语言模式", expanded=False):
                col_fast, col_accurate = st.columns(2)
                with col_fast:
                    st.checkbox("启用混合检索", key="rag_form_enable_hybrid_retrieval")
                    st.checkbox("启用精精确实体检索", key="rag_form_enable_exact_retrieval")
                    st.checkbox("启用结构化检索", key="rag_form_enable_structured_retrieval")
                    st.checkbox("启用 Query Planner", key="rag_form_enable_query_planner")
                    st.checkbox("启用跨语言检索变体", key="rag_form_enable_cross_lingual_variants")
                    st.selectbox("默认回答语言", options=["auto", "zh", "en", "ja", "ko"], key="rag_form_default_answer_language")
                    st.selectbox("语言识别模式", options=["auto", "force"], key="rag_form_language_mode")
                with col_accurate:
                    st.number_input("RRF k", min_value=1, max_value=200, key="rag_form_rrf_k", step=1)
                    st.number_input("Parent Chunk 大小", min_value=400, max_value=5000, key="rag_form_parent_chunk_size", step=100)
                    st.number_input("Child Chunk 大小", min_value=120, max_value=1600, key="rag_form_child_chunk_size", step=20)
                    st.number_input("Child overlap", min_value=0, max_value=800, key="rag_form_child_chunk_overlap", step=10)
                st.checkbox(
                    "启用 LLM 结构化问题规划（含检索改写）",
                    key="rag_form_enable_query_rewrite",
                    help="主路径将问题解析为受约束 JSON 语义计划（实体、目标属性、操作、直接事实/受控推断），再驱动实体链接、检索与证据判断；失败时才回退本地 Planner。",
                )
                st.number_input("LLM 检索改写数量", min_value=0, max_value=4, key="rag_form_query_rewrite_count", step=1)
                st.checkbox("启用 Cross-Encoder Rerank", key="rag_form_enable_reranker")
                st.text_input("Reranker 模型名或本地路径", key="rag_form_reranker_model")
                st.selectbox("Reranker 设备", options=["auto", "cpu", "cuda"], key="rag_form_reranker_device")
                rerank_col_a, rerank_col_b = st.columns(2)
                with rerank_col_a:
                    st.number_input("Reranker 候选数", min_value=1, max_value=200, key="rag_form_reranker_candidate_k", step=1)
                    st.number_input("Reranker batch size", min_value=1, max_value=64, key="rag_form_reranker_batch_size", step=1)
                with rerank_col_b:
                    st.number_input("Reranker 保留数", min_value=1, max_value=50, key="rag_form_reranker_top_k", step=1)
                    st.number_input("Reranker max length", min_value=128, max_value=2048, key="rag_form_reranker_max_length", step=64)
                st.checkbox("启用 LLM Evidence Judge", key="rag_form_enable_evidence_judge")
            st.divider()
            st.checkbox(
                "启用 PDF OCR",
                key="rag_form_enable_pdf_ocr",
                help="上传扫描版 PDF 或图片型 PDF 时，系统会自动识别页面图片中的文字并入库。",
            )
            ocr_col_a, ocr_col_b = st.columns(2)
            with ocr_col_a:
                st.number_input(
                    "OCR 渲染 DPI",
                    min_value=100,
                    max_value=300,
                    key="rag_form_pdf_ocr_dpi",
                    step=10,
                    help="DPI 越高识别可能越准，但速度越慢。推荐 160-220。",
                )
            with ocr_col_b:
                st.number_input(
                    "低文本页 OCR 阈值",
                    min_value=0,
                    max_value=1000,
                    key="rag_form_pdf_ocr_min_text_chars",
                    step=20,
                    help="某页原生可复制文本少于该字符数时，会尝试 OCR 兜底。",
                )
            st.selectbox(
                "OCR 语言提示",
                options=["auto", "zh_en", "latin", "ja", "ko"],
                key="rag_form_pdf_ocr_language_hint",
                help="当前 OCR 引擎语言覆盖受 RapidOCR 模型限制；该字段用于诊断和未来可插拔 OCR。",
            )

            runtime_info = get_ocr_runtime_info()
            provider_text = ", ".join(runtime_info.get("available_providers") or ["未检测到 ONNX Runtime Provider"])
            st.caption(f"OCR 运行时：{runtime_info.get('platform')} | {runtime_info.get('onnx_device')} | {provider_text}")
            if not runtime_info.get("cuda_ready") and not runtime_info.get("directml_ready"):
                st.info(
                    "当前 OCR 仍是 CPU 推理。Windows 机器可安装 DirectML 运行时后选择 DirectML；"
                    "NVIDIA CUDA 环境可安装 onnxruntime-gpu 后选择 CUDA。"
                )
            selected_ocr_device = st.session_state.get("rag_form_pdf_ocr_device", "cpu")
            if selected_ocr_device == "directml" and not runtime_info.get("directml_ready"):
                st.warning("你选择了 DirectML，但当前环境没有 DmlExecutionProvider，实际会回落到 CPU。")
            if selected_ocr_device == "cuda" and not runtime_info.get("cuda_ready"):
                st.warning("你选择了 CUDA，但当前环境没有 CUDAExecutionProvider，实际会回落到 CPU。")

            ocr_runtime_col_a, ocr_runtime_col_b = st.columns(2)
            with ocr_runtime_col_a:
                st.selectbox(
                    "OCR 推理设备",
                    options=["cpu", "auto", "directml", "cuda"],
                    key="rag_form_pdf_ocr_device",
                    format_func={
                        "cpu": "CPU（最稳）",
                        "auto": "自动选择 GPU/CPU",
                        "directml": "DirectML GPU（Windows 通用）",
                        "cuda": "CUDA GPU（NVIDIA）",
                    }.get,
                    help="需要安装对应 ONNX Runtime GPU 包才会真正启用 GPU；否则会自动回落 CPU。",
                )
                st.number_input(
                    "OCR CPU 线程数",
                    min_value=-1,
                    max_value=32,
                    key="rag_form_pdf_ocr_threads",
                    step=1,
                    help="-1 表示由 ONNX Runtime 自动决定；CPU 模式可尝试 4、6、8。",
                )
            with ocr_runtime_col_b:
                st.number_input(
                    "OCR 最大图像边长",
                    min_value=960,
                    max_value=3000,
                    key="rag_form_pdf_ocr_max_side_len",
                    step=100,
                    help="边长越小越快但可能丢细小文字。扫描小说/合同推荐 1400-1800，复杂表格可提高。",
                )

        if st.button("保存", key="save_settings_bottom", use_container_width=True):
            try:
                save_runtime_configs()
                st.success("配置已保存。")
                st.rerun()
            except Exception as exc:
                st.error(f"保存失败：{exc}")


def render_welcome_state() -> None:
    st.markdown(
        """
        <div class="welcome-panel">
            <h1>想从知识库里了解什么？</h1>
            <p>导入文档后，可以直接提问。我会检索相关片段，并基于来源证据回答。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_meta(meta: dict) -> None:
    if not meta:
        return

    confidence_score = float(meta.get("confidence_score", 0.0))
    confidence_pct = round(confidence_score * 100)
    confidence_label = str(meta.get("confidence_label", "低"))
    confidence_class = _confidence_class(confidence_score)
    source_count = int(meta.get("source_count", 0))
    source_type_label = html.escape(str(meta.get("source_type_label", "未命中来源")))
    source_tags = meta.get("source_tags", [])
    source_cards = meta.get("source_cards", [])
    coverage_note = html.escape(str(meta.get("coverage_note", "")))
    tags_html = "".join(f'<span class="source-chip">{tag}</span>' for tag in source_tags[:3])

    st.markdown(
        f"""
        <div class="answer-meta">
            <div><strong>证据支持度：</strong> <span class="{confidence_class}">{confidence_pct}% | {confidence_label}</span></div>
            <div><strong>结论性质：</strong> {source_type_label}</div>
            {f'<div><strong>检索提示：</strong> {coverage_note}</div>' if coverage_note else ''}
            <div class="source-chip-row">
                <span class="source-chip">引用片段：{source_count}</span>
                {tags_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_cards:
        with st.expander("引用来源", expanded=False):
            for card in source_cards:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-card-title">{card['file_name']}</div>
                        <div class="source-card-subtitle">
                            {card['source_type']} | {card['location']} | 得分 {card['score']}
                        </div>
                        <div class="source-card-subtitle">{card['preview']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_retrieval_diagnostics(diagnostics: dict, results) -> None:
    if not diagnostics:
        return
    with st.expander("检索诊断", expanded=False):
        plan = diagnostics.get("query_plan") or {}
        st.markdown("##### Query Plan")
        st.json(
            {
                "intent": plan.get("intent"),
                "query_language": plan.get("query_language"),
                "language_confidence": plan.get("language_confidence"),
                "script_distribution": plan.get("script_distribution"),
                "entities": plan.get("entities"),
                "retrieval_queries": plan.get("retrieval_queries"),
                "preferred_chunk_kinds": plan.get("preferred_chunk_kinds"),
                "required_evidence": plan.get("required_evidence"),
                "planner_source": plan.get("planner_source"),
                "semantics": plan.get("semantics"),
                "entity_linking_confidence": plan.get("entity_linking_confidence"),
                "entity_coverage_failed": plan.get("entity_coverage_failed"),
                "warnings": plan.get("warnings"),
            }
        )
        st.markdown("##### 检索通道候选")
        st.json(diagnostics.get("candidates_by_channel") or {})
        st.markdown("##### Reranker / Evidence Judge")
        st.json(
            {
                "reranker_enabled": diagnostics.get("reranker_enabled"),
                "reranker_status": diagnostics.get("reranker_status"),
                "evidence_judge_enabled": diagnostics.get("evidence_judge_enabled"),
                "entity_coverage_failed": diagnostics.get("entity_coverage_failed"),
                "warnings": diagnostics.get("warnings"),
            }
        )
        st.markdown("##### 最终证据块")
        evidence_rows = []
        for item in results or []:
            evidence_rows.append(
                {
                    "chunk_id": getattr(item, "chunk_id", ""),
                    "chunk_kind": getattr(item, "chunk_kind", ""),
                    "file": (item.metadata or {}).get("file_name"),
                    "page": (item.metadata or {}).get("page_start") or (item.metadata or {}).get("page"),
                    "language": (item.metadata or {}).get("chunk_language"),
                    "rrf_score": round(float(getattr(item, "rrf_score", 0.0) or 0.0), 6),
                    "rerank_score": getattr(item, "rerank_score", None),
                    "contributions": [
                        {
                            "channel": contrib.channel,
                            "rank": contrib.rank,
                            "weight": contrib.weight,
                            "contribution": round(contrib.contribution, 6),
                        }
                        for contrib in getattr(item, "contributions", [])
                    ],
                }
            )
        st.json(evidence_rows)


def thinking_state_html(title: str, detail: str) -> str:
    return f"""
    <div class="thinking-wave" aria-label="{html.escape(title)}">
        <span></span><span></span><span></span>
        <div class="thinking-caption">{html.escape(detail)}</div>
    </div>
    """


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("meta"):
                render_answer_meta(message["meta"])


def run_app() -> None:
    st.set_page_config(
        page_title="RAG 知识库",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"

    current_view = st.session_state.current_view
    previous_view = st.session_state.get("_last_rendered_view")
    settings_entry_requested = bool(st.session_state.pop("_settings_form_hydration_requested", False))
    if current_view == "settings" and (previous_view != "settings" or settings_entry_requested):
        # 在设置控件实例化前重建 widget state。停留在设置页编辑时绝不自动回填。
        hydrate_runtime_forms_from_saved_config()

    ui_config = get_saved_ui_config()
    inject_custom_styles(ui_config.theme_mode)

    st.markdown(
        """
        <style>
        div.stButton > button {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }
        div.stButton > button p {
            text-align: center !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            margin-bottom: -14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    api_config = get_saved_api_config()
    rag_config = get_saved_rag_config()

    engine = None
    if is_runtime_config_ready(api_config):
        try:
            payload = {"api": api_config.to_dict(), "rag": rag_config.to_dict()}
            engine = get_engine(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            st.error(f"知识库引擎初始化失败：{exc}")
    else:
        st.warning("请先完成聊天模型和向量配置，然后再导入文件或提问。")

    render_sidebar(engine)
    render_header(api_config, rag_config, ui_config, engine)

    if st.session_state.current_view == "knowledge":
        st.markdown("### ▣ 知识库管理中心")
        tab_import, tab_manage = st.tabs(["导入文件并创建索引", "已入库文件管理"])
        with tab_import:
            render_knowledge_import_panel(engine)
        with tab_manage:
            render_file_management_panel(engine)
    elif st.session_state.current_view == "settings":
        st.markdown("### ⚙ 系统运行配置")
        _render_model_settings_content(api_config, rag_config)
    else:
        if len(st.session_state.messages) <= 1:
            render_welcome_state()
        render_chat_history()

        query = st.chat_input("向知识库提问", disabled=engine is None)
        if query:
            maybe_update_conversation_title(query)
            st.session_state.messages.append({"role": "user", "content": query})
            persist_current_conversation()
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                placeholder = st.empty()
                accumulated = []

                try:
                    thinking_placeholder.markdown(
                        thinking_state_html(
                            "正在理解问题并检索知识库",
                            "正在检索知识库",
                        ),
                        unsafe_allow_html=True,
                    )
                    results, stream = engine.answer_stream(
                        query=query,
                        history_messages=st.session_state.messages[:-1],
                        top_k=rag_config.top_k,
                        keyword_filter=st.session_state.get("keyword_filter", ""),
                    )
                    thinking_placeholder.markdown(
                        thinking_state_html(
                            "已找到候选证据，正在生成回答",
                            "正在组织答案",
                        ),
                        unsafe_allow_html=True,
                    )
                    answer_meta = _build_answer_meta(results)
                    st.session_state.last_retrieval = results
                    st.session_state.last_answer_meta = answer_meta

                    for delta in stream:
                        if not accumulated:
                            thinking_placeholder.empty()
                        accumulated.append(delta)
                        placeholder.markdown("".join(accumulated) + "|")

                    final_answer = "".join(accumulated).strip()
                    if not final_answer:
                        final_answer = "当前知识库没有检索到足够证据来回答这个问题。"
                    answer_meta = _adjust_answer_meta_by_answer(answer_meta, final_answer)
                    st.session_state.last_answer_meta = answer_meta
                    thinking_placeholder.empty()
                    placeholder.markdown(final_answer)
                    render_answer_meta(answer_meta)
                    diagnostics = engine.retrieval_diagnostics() if engine else {}
                    st.session_state.last_retrieval_diagnostics = diagnostics
                    render_retrieval_diagnostics(diagnostics, results)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": final_answer,
                            "meta": answer_meta,
                        }
                    )
                    persist_current_conversation()
                except Exception as exc:
                    thinking_placeholder.empty()
                    error_message = f"回答生成失败：{exc}"
                    placeholder.error(error_message)
                    st.session_state.last_answer_meta = None
                    st.session_state.last_retrieval = []
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    persist_current_conversation()

    st.session_state._last_rendered_view = current_view
    st.caption(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
