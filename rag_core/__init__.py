"""Public exports for the RAG core package.

Heavy optional dependencies (Chroma, embedding backends) are imported lazily so
utility modules such as query planning and multilingual normalization can be
used without paying the full engine import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import ParsedChunk, SearchResult

if TYPE_CHECKING:  # pragma: no cover
    from .engine import RAGEngine, build_engine
    from .providers import MultiProviderClient

__all__ = [
    "MultiProviderClient",
    "ParsedChunk",
    "RAGEngine",
    "SearchResult",
    "build_engine",
]


def __getattr__(name: str):
    if name in {"RAGEngine", "build_engine"}:
        from .engine import RAGEngine, build_engine
        return {"RAGEngine": RAGEngine, "build_engine": build_engine}[name]
    if name == "MultiProviderClient":
        from .providers import MultiProviderClient
        return MultiProviderClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
