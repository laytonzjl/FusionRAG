from __future__ import annotations

"""
兼容层：
保留旧的 `rag_engine.py` 导入路径，避免外部代码或历史脚本失效。
新的核心实现已经拆分到 `rag_core/` 包中。
"""

from rag_core import MultiProviderClient, ParsedChunk, RAGEngine, SearchResult, build_engine

__all__ = [
    "MultiProviderClient",
    "ParsedChunk",
    "RAGEngine",
    "SearchResult",
    "build_engine",
]

