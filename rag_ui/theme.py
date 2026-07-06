from __future__ import annotations

import streamlit as st


def inject_custom_styles(theme_mode: str = "native") -> None:
    """Inject a ChatGPT-like shell that follows Streamlit's native theme."""

    st.markdown(
        """
        <style>
            :root {
                --app-bg: var(--background-color);
                --sidebar-bg: var(--secondary-background-color);
                --panel-bg: var(--background-color);
                --panel-soft: var(--secondary-background-color);
                --control-bg: var(--background-color);
                --text: var(--text-color);
                --muted: color-mix(in srgb, var(--text-color) 58%, transparent);
                --border: color-mix(in srgb, var(--text-color) 14%, transparent);
                --accent: var(--primary-color);
                --accent-text: #ffffff;
                --user-bubble: var(--secondary-background-color);
                --assistant-bubble: var(--background-color);
                --surface-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
            }

            @media (prefers-color-scheme: light) {
                :root {
                    --app-bg: #ffffff;
                    --sidebar-bg: #f7f7f7;
                    --panel-bg: #ffffff;
                    --panel-soft: #f3f3f3;
                    --control-bg: #ffffff;
                    --text: #111111;
                    --muted: #6b6b6b;
                    --border: rgba(0, 0, 0, 0.08);
                    --accent: #10a37f;
                    --user-bubble: #f4f4f4;
                    --assistant-bubble: #ffffff;
                    --surface-shadow: 0 10px 34px rgba(0, 0, 0, 0.08);
                }
            }

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"] {
                background: var(--app-bg) !important;
                color: var(--text) !important;
            }

            .stApp {
                font-family: "Sohne", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            }

            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {
                background: var(--sidebar-bg) !important;
                color: var(--text) !important;
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] {
                min-width: 300px !important;
            }

            [data-testid="stSidebarContent"] {
                padding: 0.85rem 0.85rem 1rem;
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
            }

            section.main > div.block-container {
                max-width: 1180px;
                padding: 0.75rem 1.25rem 7.5rem;
            }

            .chat-shell-header {
                position: sticky;
                top: 0;
                z-index: 10;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.3rem 0 0.65rem;
                background: var(--app-bg);
                border-bottom: 0;
                margin-bottom: 0.25rem;
            }

            .chat-title {
                font-size: 1rem;
                font-weight: 650;
                color: var(--text);
            }

            .chat-subtitle {
                margin-top: 0.15rem;
                font-size: 0.82rem;
                color: var(--muted);
            }

            .status-row {
                display: flex;
                flex-wrap: wrap;
                justify-content: flex-end;
                gap: 0.4rem;
            }

            .status-pill {
                border: 1px solid var(--border);
                background: var(--panel-soft);
                color: var(--muted);
                border-radius: 999px;
                padding: 0.28rem 0.62rem;
                font-size: 0.75rem;
                white-space: nowrap;
            }

            .runtime-setup-banner {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                border: 1px solid var(--border);
                background: var(--panel-soft);
                color: var(--text);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                margin: 0.75rem 0;
            }

            .runtime-eyebrow {
                color: var(--muted);
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .runtime-title {
                color: var(--text);
                font-size: 1rem;
                font-weight: 700;
                margin-top: 0.15rem;
            }

            .runtime-note {
                color: var(--muted);
                font-size: 0.82rem;
                text-align: right;
            }

            .welcome-panel {
                max-width: 720px;
                margin: min(24vh, 12rem) auto 2rem;
                text-align: center;
                color: var(--text);
            }

            .welcome-panel h1 {
                margin: 0 0 0.6rem;
                font-size: clamp(1.9rem, 3vw, 2.45rem);
                font-weight: 500;
                letter-spacing: 0;
            }

            .welcome-panel p {
                margin: 0 auto;
                max-width: 520px;
                color: var(--muted);
                line-height: 1.7;
                font-size: 0.95rem;
            }

            div[data-testid="stChatMessage"] {
                background: transparent !important;
                color: var(--text) !important;
                border: 0 !important;
                margin: 0 auto 1.15rem;
                max-width: 860px;
            }

            div[data-testid="stChatMessageContent"] {
                background: transparent !important;
            }

            div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
                color: var(--text);
                line-height: 1.8;
            }

            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                background: var(--user-bubble) !important;
                border-radius: 20px;
                padding: 0.8rem 1rem;
                max-width: min(76%, 760px);
                margin-left: auto;
                box-shadow: none;
            }

            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
                background: var(--assistant-bubble) !important;
                padding: 0.3rem 0;
            }

            .stChatInputContainer {
                background: linear-gradient(to top, var(--app-bg) 70%, rgba(255, 255, 255, 0));
                padding: 0.85rem 0 1.1rem;
            }

            .stChatInput textarea {
                min-height: 64px !important;
                border-radius: 999px !important;
                border: 1px solid var(--border) !important;
                background: var(--control-bg) !important;
                color: var(--text) !important;
                box-shadow: 0 10px 34px rgba(0, 0, 0, 0.09) !important;
                padding: 1.15rem 4.8rem 1.15rem 1.35rem !important;
                font-size: 1rem !important;
                line-height: 1.35 !important;
                caret-color: var(--text) !important;
            }

            div[data-testid="stChatInput"] {
                max-width: 760px;
                margin: 0 auto;
                position: relative;
            }

            div[data-testid="stChatInput"]::before {
                content: none;
            }

            div[data-testid="stChatInput"]::after {
                content: none;
            }

            .stChatInput button,
            div[data-testid="stChatInputSubmitButton"] button {
                border-radius: 999px !important;
                background: #111111 !important;
                color: #ffffff !important;
                border: 0 !important;
                width: 2.7rem !important;
                height: 2.7rem !important;
            }

            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea,
            .stSelectbox div[data-baseweb="select"] > div,
            .stMultiSelect div[data-baseweb="select"] > div {
                background: var(--control-bg) !important;
                color: var(--text) !important;
                border-color: var(--border) !important;
                border-radius: 8px !important;
            }

            .stTextInput input::placeholder,
            .stTextArea textarea::placeholder {
                color: var(--muted) !important;
            }

            .stButton > button {
                border-radius: 12px;
                border: 1px solid transparent;
                background: transparent;
                color: var(--text);
                min-height: 2.35rem;
                box-shadow: none;
                justify-content: flex-start;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                background: #ececec !important;
                border-color: transparent !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                padding-left: 0.7rem;
            }

            .stButton > button[kind="primary"] {
                background: var(--accent) !important;
                color: var(--accent-text) !important;
                border-color: var(--accent) !important;
            }

            div[data-testid="stFileUploaderDropzone"] {
                background: var(--panel-soft) !important;
                border: 1px dashed var(--border) !important;
                border-radius: 8px !important;
            }

            details,
            .stExpander {
                background: var(--panel-bg) !important;
                border: 1px solid var(--border) !important;
                border-radius: 8px !important;
                box-shadow: none !important;
            }

            .settings-panel {
                border-bottom: 1px solid var(--border);
                padding-bottom: 0.75rem;
                margin-bottom: 0.75rem;
            }

            .sidebar-brand-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin: 0.25rem 0 1rem;
            }

            .sidebar-brand {
                font-size: 1.08rem;
                font-weight: 750;
                letter-spacing: -0.02em;
            }

            .sidebar-brand span {
                color: var(--muted);
                font-size: 0.92rem;
                font-weight: 650;
            }

            .sidebar-collapse-icon {
                width: 1.75rem;
                height: 1.75rem;
                border-radius: 0.55rem;
                display: grid;
                place-items: center;
                color: var(--muted);
                border: 1px solid var(--border);
            }

            .sidebar-nav {
                margin: 0.55rem 0 1.1rem;
            }

            .sidebar-nav-item {
                border-radius: 12px;
                padding: 0.52rem 0.68rem;
                font-size: 0.82rem;
                line-height: 1.25;
                color: var(--muted);
                background: rgba(0, 0, 0, 0.025);
                margin-bottom: 0.35rem;
            }

            .sidebar-section-title {
                margin: 1.15rem 0 0.45rem;
                font-size: 0.82rem;
                font-weight: 700;
                color: #3b3b3b;
            }

            [data-testid="stSidebar"] details,
            [data-testid="stSidebar"] .stExpander {
                background: transparent !important;
                border-color: rgba(0, 0, 0, 0.07) !important;
                border-radius: 12px !important;
                margin-bottom: 0.5rem;
            }

            [data-testid="stSidebar"] details summary {
                font-size: 0.9rem;
                font-weight: 650;
            }

            [data-testid="stSidebar"] .conversation-row + div[data-testid="stHorizontalBlock"] {
                gap: 0.25rem;
                align-items: center;
            }

            [data-testid="stSidebar"] .conversation-row + div[data-testid="stHorizontalBlock"] .stButton > button {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                font-size: 0.9rem;
                min-height: 2.05rem;
                color: var(--text) !important;
                text-align: left;
                display: block;
                width: 100%;
            }

            [data-testid="stSidebar"] .conversation-row + div[data-testid="stHorizontalBlock"] div:last-child .stButton > button {
                justify-content: center;
                padding: 0;
                color: var(--muted);
                opacity: 0;
                text-align: center;
            }

            [data-testid="stSidebar"] .conversation-row + div[data-testid="stHorizontalBlock"]:hover div:last-child .stButton > button {
                opacity: 1;
            }

            .answer-meta,
            .source-card {
                border: 1px solid var(--border);
                background: var(--panel-soft);
                border-radius: 8px;
                padding: 0.75rem 0.85rem;
                margin-top: 0.7rem;
                color: var(--muted);
                font-size: 0.86rem;
            }

            .answer-meta strong,
            .source-card-title {
                color: var(--text);
            }

            .source-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                margin-top: 0.45rem;
            }

            .source-chip {
                border: 1px solid var(--border);
                background: var(--panel-bg);
                color: var(--muted);
                border-radius: 999px;
                padding: 0.2rem 0.5rem;
                font-size: 0.74rem;
            }

            .source-card-subtitle {
                color: var(--muted);
                font-size: 0.8rem;
                line-height: 1.55;
                word-break: break-word;
            }

            .thinking-wave {
                display: inline-flex;
                align-items: center;
                gap: 0.25rem;
                min-height: 2.1rem;
                padding: 0.2rem 0.1rem;
                margin: 0.25rem 0 0.6rem;
            }

            .thinking-wave span {
                width: 0.48rem;
                height: 0.48rem;
                border-radius: 999px;
                background: #111111;
                opacity: 0.24;
                animation: thinking-wave 1.05s infinite ease-in-out;
            }

            .thinking-wave span:nth-child(2) {
                animation-delay: 0.14s;
            }

            .thinking-wave span:nth-child(3) {
                animation-delay: 0.28s;
            }

            .thinking-caption {
                margin-left: 0.45rem;
                font-size: 0.86rem;
                color: var(--muted);
            }

            @keyframes thinking-wave {
                0%, 70%, 100% {
                    transform: translateY(0);
                    opacity: 0.24;
                }
                35% {
                    transform: translateY(-0.38rem);
                    opacity: 1;
                }
            }

            .confidence-high { color: #16a34a; font-weight: 700; }
            .confidence-medium { color: #d97706; font-weight: 700; }
            .confidence-low { color: #dc2626; font-weight: 700; }

            @media (max-width: 760px) {
                section.main > div.block-container {
                    padding-left: 0.8rem;
                    padding-right: 0.8rem;
                }

                .chat-shell-header {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .runtime-setup-banner {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .runtime-note {
                    text-align: left;
                }

                div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
                    max-width: 100%;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
