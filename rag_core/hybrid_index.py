from __future__ import annotations

import json
import logging
import sqlite3
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from config import settings
from rag_core.multilingual import normalize_entity_name, normalize_for_lexical_search, tokenize_for_search


logger = logging.getLogger(__name__)
LEXICAL_INDEX_VERSION = "lexical_index_v1"
# The v3 metadata contains source positions and navigation regions; old records
# must be rebuilt rather than silently treated as equivalent.
INDEX_SCHEMA_VERSION = "hybrid_schema_v5_segment_topology"


@dataclass
class HybridRecord:
    chunk_id: str
    document_id: str
    parent_chunk_id: str
    content: str
    metadata: Dict[str, object]
    chunk_kind: str = "child"
    language: str = "unknown"
    title: str = ""
    section_path: str = ""
    aliases: List[str] = field(default_factory=list)
    exact_terms: List[str] = field(default_factory=list)


@dataclass
class HybridHit:
    chunk_id: str
    content: str
    metadata: Dict[str, object]
    rank: int
    score: float
    channel: str
    matched_terms: List[str] = field(default_factory=list)


class HybridIndex:
    """本地 SQLite 混合索引：Exact / Lexical / Structured。"""

    def __init__(self, path: str | Path | None = None, collection_name: str = "default") -> None:
        self.path = Path(path or settings.hybrid_index_path)
        self.collection_name = collection_name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    collection_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT,
                    parent_chunk_id TEXT,
                    chunk_kind TEXT,
                    language TEXT,
                    title TEXT,
                    section_path TEXT,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    lexical_text TEXT NOT NULL,
                    PRIMARY KEY(collection_name, chunk_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS aliases (
                    collection_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    entity_type TEXT DEFAULT 'unknown',
                    source TEXT DEFAULT 'document',
                    PRIMARY KEY(collection_name, normalized_name, chunk_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_meta (
                    collection_name TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(collection_name, key)
                )
                """
            )
            # The primary key already makes exact lookup efficient.  This index
            # speeds navigation scans over an anchored document.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_collection_document_kind "
                "ON chunks(collection_name, document_id, chunk_kind)"
            )
            # Navigation lookups must filter and sort by source position in SQL.
            # Without this expression index, a terminal-region read can degrade
            # into an arbitrary full-table scan on large PDFs.
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_navigation_region_position "
                    "ON chunks("
                    "collection_name, document_id, chunk_kind, "
                    "json_extract(metadata_json, '$.navigation_region'), "
                    "CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL)"
                    ")"
                )
            except sqlite3.Error as exc:
                logger.warning("无法创建导航表达式索引，将继续使用正确但较慢的查询: %s", exc)
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        collection_name UNINDEXED,
                        chunk_id UNINDEXED,
                        lexical_text,
                        title,
                        section_path,
                        content
                    )
                    """
                )
            except sqlite3.Error as exc:
                logger.warning("SQLite FTS5 不可用，将使用 LIKE 词法回退: %s", exc)

    def reset_collection(self) -> None:
        with self._connect() as conn:
            for table in ("chunks", "aliases", "index_meta"):
                conn.execute(f"DELETE FROM {table} WHERE collection_name = ?", (self.collection_name,))
            try:
                conn.execute("DELETE FROM chunks_fts WHERE collection_name = ?", (self.collection_name,))
            except sqlite3.Error:
                pass

    def delete_file(self, file_name: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE collection_name = ? AND json_extract(metadata_json, '$.file_name') = ?",
                (self.collection_name, file_name),
            ).fetchall()
            chunk_ids = [str(row["chunk_id"]) for row in rows]
            if not chunk_ids:
                return
            placeholders = ",".join("?" for _ in chunk_ids)
            conn.execute(
                f"DELETE FROM chunks WHERE collection_name = ? AND chunk_id IN ({placeholders})",
                (self.collection_name, *chunk_ids),
            )
            conn.execute(
                f"DELETE FROM aliases WHERE collection_name = ? AND chunk_id IN ({placeholders})",
                (self.collection_name, *chunk_ids),
            )
            try:
                conn.execute(
                    f"DELETE FROM chunks_fts WHERE collection_name = ? AND chunk_id IN ({placeholders})",
                    (self.collection_name, *chunk_ids),
                )
            except sqlite3.Error:
                pass

    def upsert_records(self, records: Iterable[HybridRecord]) -> None:
        with self._connect() as conn:
            for record in records:
                metadata_json = json.dumps(record.metadata or {}, ensure_ascii=False, sort_keys=True)
                lexical_text = self._lexical_text(record)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (collection_name, chunk_id, document_id, parent_chunk_id, chunk_kind, language,
                     title, section_path, content, metadata_json, lexical_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.collection_name,
                        record.chunk_id,
                        record.document_id,
                        record.parent_chunk_id,
                        record.chunk_kind,
                        record.language,
                        record.title,
                        record.section_path,
                        record.content,
                        metadata_json,
                        lexical_text,
                    ),
                )
                try:
                    conn.execute("DELETE FROM chunks_fts WHERE collection_name = ? AND chunk_id = ?", (self.collection_name, record.chunk_id))
                    conn.execute(
                        """
                        INSERT INTO chunks_fts(collection_name, chunk_id, lexical_text, title, section_path, content)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (self.collection_name, record.chunk_id, lexical_text, record.title, record.section_path, record.content),
                    )
                except sqlite3.Error:
                    pass
                for alias in list(record.aliases) + list(record.exact_terms):
                    normalized = normalize_entity_name(alias)
                    if not normalized:
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO aliases(collection_name, normalized_name, alias, chunk_id, entity_type, source)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (self.collection_name, normalized, alias, record.chunk_id, str(record.metadata.get("entity_type", "unknown")), str(record.metadata.get("alias_source", "document"))),
                    )

    def resolve_entity_candidates(self, mention: str, limit: int = 4) -> List[Dict[str, object]]:
        """将 LLM 抽取的实体提及链接到本地 aliases 索引。

        不依赖领域词表：先做标准化精确匹配，再做 SQLite 子串候选与通用字符相似度排序。
        返回的是索引内真实存在的别名，供检索阶段扩展，而不是由 LLM 编造的别名。
        """

        raw_mention = str(mention or "").strip()
        normalized = normalize_entity_name(raw_mention)
        if len(normalized) < 2:
            return []

        def rows_for(where_sql: str, params: Sequence[object]):
            query = f"""
                SELECT normalized_name, MIN(alias) AS alias, MAX(entity_type) AS entity_type,
                       MAX(source) AS source, COUNT(DISTINCT chunk_id) AS support_count
                FROM aliases
                WHERE collection_name = ? AND ({where_sql})
                GROUP BY normalized_name
                LIMIT ?
            """
            with self._connect() as conn:
                return conn.execute(query, (self.collection_name, *params, max(limit * 12, 24))).fetchall()

        try:
            rows = rows_for("normalized_name = ?", [normalized])
            exact = bool(rows)
            if not rows:
                # 通用的 substring candidate generation，不是针对某个问法或实体的规则。
                rows = rows_for("normalized_name LIKE ? OR ? LIKE '%' || normalized_name || '%'", [f"%{normalized}%", normalized])
                exact = False
        except sqlite3.Error as exc:
            logger.warning("实体链接查询失败: %s", exc)
            return []

        candidates: List[Dict[str, object]] = []
        for row in rows:
            candidate_name = str(row["normalized_name"] or "")
            alias = str(row["alias"] or candidate_name)
            if not candidate_name:
                continue
            ratio = SequenceMatcher(a=normalized, b=candidate_name).ratio()
            substring = normalized in candidate_name or candidate_name in normalized
            score = 1.0 if exact and candidate_name == normalized else max(ratio, 0.82 if substring else 0.0)
            if score < 0.62:
                continue
            candidates.append(
                {
                    "canonical": alias,
                    "normalized_name": candidate_name,
                    "alias": alias,
                    "entity_type": str(row["entity_type"] or "unknown"),
                    "source": str(row["source"] or "document"),
                    "support_count": int(row["support_count"] or 0),
                    "score": round(min(score + min(int(row["support_count"] or 0), 8) * 0.005, 1.0), 4),
                }
            )
        candidates.sort(key=lambda item: (float(item["score"]), int(item["support_count"])), reverse=True)
        return candidates[:limit]

    def exact_search(self, terms: Sequence[str], limit: int = 20) -> List[HybridHit]:
        normalized_terms = [normalize_entity_name(term) for term in terms if normalize_entity_name(term)]
        if not normalized_terms:
            return []
        placeholders = ",".join("?" for _ in normalized_terms)
        params = [self.collection_name, *normalized_terms]
        sql = f"""
            SELECT c.*, a.alias
            FROM aliases a
            JOIN chunks c ON c.collection_name = a.collection_name AND c.chunk_id = a.chunk_id
            WHERE a.collection_name = ? AND a.normalized_name IN ({placeholders})
            LIMIT ?
        """
        hits: List[HybridHit] = []
        with self._connect() as conn:
            rows = conn.execute(sql, (*params, limit)).fetchall()
        for index, row in enumerate(rows, start=1):
            hits.append(self._row_to_hit(row, index, "exact_entity", matched_terms=[str(row["alias"])]))
        return hits

    def lexical_search(self, query: str, language: str = "unknown", limit: int = 50) -> List[HybridHit]:
        tokens = tokenize_for_search(query, language=language)
        if not tokens:
            return []
        hits = self._fts_search(tokens, limit=limit)
        if hits:
            return hits
        return self._like_search(tokens, limit=limit)

    def structured_search(
        self,
        query: str,
        preferred_chunk_kinds: Sequence[str],
        language: str = "unknown",
        limit: int = 30,
    ) -> List[HybridHit]:
        if not preferred_chunk_kinds:
            preferred_chunk_kinds = ["document_card", "entity_card", "section_card", "metadata"]
        tokens = tokenize_for_search(query, language=language)
        placeholders = ",".join("?" for _ in preferred_chunk_kinds)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM chunks
                WHERE collection_name = ?
                  AND chunk_kind IN ({placeholders})
                ORDER BY CASE chunk_kind
                    WHEN 'entity_card' THEN 0
                    WHEN 'document_card' THEN 1
                    WHEN 'section_card' THEN 2
                    WHEN 'table_card' THEN 3
                    WHEN 'code_card' THEN 4
                    ELSE 9 END
                LIMIT ?
                """,
                (self.collection_name, *preferred_chunk_kinds, limit * 3),
            ).fetchall()
        scored = []
        for row in rows:
            text = " ".join([str(row["title"] or ""), str(row["section_path"] or ""), str(row["content"] or "")])
            match_count = sum(1 for token in tokens if token in normalize_for_lexical_search(text, language=language))
            if match_count > 0 or not tokens:
                scored.append((match_count, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_hit(row, index, "structured", matched_terms=tokens[:8]) for index, (_, row) in enumerate(scored[:limit], start=1)]

    def navigation_search(
        self,
        document_ids: Sequence[str],
        regions: Sequence[str],
        query: str = "",
        language: str = "unknown",
        limit: int = 24,
        position_bias: str = "none",
        segment_ids: Sequence[str] | None = None,
        coverage: bool = False,
    ) -> List[HybridHit]:
        """Return raw source windows from an anchored document/segment.

        ``coverage`` is a semantic-plan execution flag.  It does not inspect
        question wording: it asks for position-diverse source windows before
        later semantic ranking and atomic-evidence judging.
        """

        document_ids = [str(item) for item in document_ids if str(item)]
        if not document_ids:
            return []
        requested_regions = [str(item).strip().lower() for item in regions if str(item).strip()]
        broad_read = not requested_regions or "all" in requested_regions
        wanted = ["front", "middle", "terminal"] if broad_read else requested_regions
        wanted = [item for item in wanted if item in {"front", "middle", "terminal"}]
        if not wanted:
            wanted = ["front", "middle", "terminal"]
        segment_ids = [str(item) for item in (segment_ids or []) if str(item)]

        doc_marks = ",".join("?" for _ in document_ids)
        region_marks = ",".join("?" for _ in wanted)
        segment_clause = ""
        segment_params: List[object] = []
        if segment_ids:
            segment_marks = ",".join("?" for _ in segment_ids)
            segment_clause = (
                " AND COALESCE(json_extract(metadata_json, '$.document_segment_id'), 'segment:0') "
                f"IN ({segment_marks})"
            )
            segment_params.extend(segment_ids)

        if position_bias == "front":
            position_order = "position ASC"
        elif position_bias == "terminal":
            position_order = "position DESC"
        else:
            position_order = "position ASC"

        # A broad read must not privilege the terminal window in SQL.  Do not
        # emit ``ORDER BY 0`` here: SQLite interprets a bare integer as a
        # 1-based output-column reference, so zero is an invalid ORDER BY term.
        kind_order = (
            ""
            if broad_read
            else (
                "CASE WHEN chunk_kind = 'terminal_window' THEN 0 "
                "WHEN chunk_kind = 'navigation_window' THEN 1 ELSE 9 END, "
            )
        )
        row_limit = max(min(int(limit), 160), 1)
        if coverage:
            # Coverage reads are intentionally rarer and may inspect a larger
            # source span; this avoids a position-ascending SQL limit that only
            # exposes the beginning of a long document.
            row_limit = max(min(int(limit) * 16, 800), row_limit)

        sql = f"""
            SELECT *,
                   CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL) AS position
            FROM chunks
            WHERE collection_name = ?
              AND document_id IN ({doc_marks})
              AND chunk_kind IN ('terminal_window', 'navigation_window')
              AND LOWER(COALESCE(json_extract(metadata_json, '$.navigation_region'), 'middle'))
                  IN ({region_marks})
              {segment_clause}
            ORDER BY
                {kind_order}{position_order},
                chunk_id ASC
            LIMIT ?
        """
        params: List[object] = [
            self.collection_name,
            *document_ids,
            *wanted,
            *segment_params,
            row_limit,
        ]
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("文档导航检索失败: %s", exc)
            return []

        tokens = tokenize_for_search(query, language=language)
        scored: List[tuple[float, float, sqlite3.Row, List[str]]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            region = str(metadata.get("navigation_region") or "middle").lower()
            if region not in wanted:
                continue
            position = _metadata_float(metadata.get("document_position_ratio"), 0.5)
            text = " ".join([str(row["title"] or ""), str(row["section_path"] or ""), str(row["content"] or "")])
            normalized = normalize_for_lexical_search(text, language=language)
            matched = [token for token in tokens if token and token in normalized]
            lexical = len(matched) / max(len(set(tokens)), 1) if tokens else 0.0
            kind = str(row["chunk_kind"] or "")
            if position_bias == "terminal":
                position_bonus = position
            elif position_bias == "front":
                position_bonus = 1.0 - position
            elif position_bias == "chronological":
                position_bonus = 0.15
            else:
                position_bonus = 0.0
            window_bonus = 0.10 if (not broad_read and kind == "terminal_window") else 0.0
            score = lexical * 0.9 + position_bonus * 0.55 + window_bonus
            scored.append((score, position, row, matched))

        if coverage:
            selected = self._position_diverse_navigation_hits(scored, limit=limit)
        else:
            selected = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[:limit]
        return [
            self._row_to_hit(row, rank, "navigation", matched_terms=matched)
            for rank, (_, _, row, matched) in enumerate(selected, start=1)
        ]

    def _position_diverse_navigation_hits(
        self,
        scored: Sequence[tuple[float, float, sqlite3.Row, List[str]]],
        limit: int,
    ) -> List[tuple[float, float, sqlite3.Row, List[str]]]:
        """Seed a source read with windows from multiple document positions."""

        wanted = max(1, int(limit))
        if len(scored) <= wanted:
            return sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)

        bin_count = min(8, max(3, wanted // 4))
        buckets: List[List[tuple[float, float, sqlite3.Row, List[str]]]] = [
            [] for _ in range(bin_count)
        ]
        for item in scored:
            position = max(0.0, min(1.0, float(item[1])))
            bucket = min(bin_count - 1, int(position * bin_count))
            buckets[bucket].append(item)
        for bucket in buckets:
            bucket.sort(key=lambda item: (item[0], item[1]), reverse=True)

        selected: List[tuple[float, float, sqlite3.Row, List[str]]] = []
        selected_ids: set[str] = set()
        # One best source window from every populated position band gives the
        # evidence judge a chance to reject/accept events across the document.
        for bucket in buckets:
            if not bucket:
                continue
            item = bucket[0]
            chunk_id = str(item[2]["chunk_id"] or "")
            if chunk_id and chunk_id in selected_ids:
                continue
            if chunk_id:
                selected_ids.add(chunk_id)
            selected.append(item)
            if len(selected) >= wanted:
                return selected

        for item in sorted(scored, key=lambda value: (value[0], value[1]), reverse=True):
            chunk_id = str(item[2]["chunk_id"] or "")
            if chunk_id and chunk_id in selected_ids:
                continue
            if chunk_id:
                selected_ids.add(chunk_id)
            selected.append(item)
            if len(selected) >= wanted:
                break
        return selected

    def entity_window_search(
        self,
        terms: Sequence[str],
        document_ids: Sequence[str],
        segment_ids: Sequence[str] | None = None,
        limit: int = 72,
    ) -> List[HybridHit]:
        """Return every source window that explicitly mentions a linked entity.

        This is a generic entity-coverage primitive for enumerative plans.  It
        does not infer relationship type or parse question wording; it merely
        prevents a source-window read from overlooking pages where the target
        entity is explicitly present.  Relation/event qualification remains the
        responsibility of the semantic evidence contract and judge.
        """

        normalized_terms = list(
            dict.fromkeys(
                normalize_entity_name(term)
                for term in terms
                if len(normalize_entity_name(term)) >= 2
            )
        )
        document_ids = [str(item) for item in document_ids if str(item)]
        segment_ids = [str(item) for item in (segment_ids or []) if str(item)]
        if not normalized_terms or not document_ids:
            return []

        doc_marks = ",".join("?" for _ in document_ids)
        segment_clause = ""
        params: List[object] = [self.collection_name, *document_ids]
        if segment_ids:
            segment_marks = ",".join("?" for _ in segment_ids)
            segment_clause = (
                " AND COALESCE(json_extract(metadata_json, '$.document_segment_id'), 'segment:0') "
                f"IN ({segment_marks})"
            )
            params.extend(segment_ids)
        params.append(max(200, min(int(limit) * 12, 1600)))

        sql = f"""
            SELECT *,
                   CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL) AS position
            FROM chunks
            WHERE collection_name = ?
              AND document_id IN ({doc_marks})
              AND chunk_kind IN ('navigation_window', 'terminal_window')
              {segment_clause}
            ORDER BY position ASC, chunk_id ASC
            LIMIT ?
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("实体覆盖窗口查询失败: %s", exc)
            return []

        matched_rows: List[tuple[float, sqlite3.Row, List[str]]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            searchable = normalize_entity_name(
                " ".join(
                    [
                        str(row["content"] or ""),
                        " ".join(str(value) for value in (metadata.get("aliases") or [])),
                    ]
                )
            )
            matched = [term for term in normalized_terms if term and term in searchable]
            if not matched:
                continue
            position = _metadata_float(metadata.get("document_position_ratio"), 0.5)
            matched_rows.append((position, row, matched))

        # Preserve source order.  Later retrieval code supplies semantic ranking
        # and position diversity; this method is intentionally evidence-only.
        matched_rows.sort(key=lambda item: (item[0], str(item[1]["chunk_id"] or "")))
        return [
            self._row_to_hit(row, rank, "entity_coverage", matched_terms=matched)
            for rank, (_, row, matched) in enumerate(matched_rows[: max(1, int(limit))], start=1)
        ]

    def entity_terminal_neighborhood_search(
        self,
        terms: Sequence[str],
        document_ids: Sequence[str],
        limit: int = 16,
    ) -> List[HybridHit]:
        """Read windows around the latest source-backed entity evidence.

        Source-position ordering is performed in SQL, avoiding a random LIMIT
        that can hide the latest entity occurrence in a large document.
        """

        normalized_terms = [normalize_entity_name(term) for term in terms if len(normalize_entity_name(term)) >= 2]
        document_ids = [str(item) for item in document_ids if str(item)]
        if not normalized_terms or not document_ids:
            return []
        doc_marks = ",".join("?" for _ in document_ids)
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *,
                           CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL) AS position
                    FROM chunks
                    WHERE collection_name = ? AND document_id IN ({doc_marks})
                      AND chunk_kind IN ('child', 'parent')
                    ORDER BY position DESC, chunk_id ASC
                    LIMIT 3000
                    """,
                    (self.collection_name, *document_ids),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("实体时间线检索失败: %s", exc)
            return []

        entity_rows: List[tuple[float, sqlite3.Row, Dict[str, object], List[str]]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            source_index = metadata.get("source_unit_index")
            if source_index is None:
                continue
            source = normalize_entity_name(
                " ".join([str(row["content"] or ""), " ".join(str(item) for item in metadata.get("aliases") or [])])
            )
            matched = [term for term in normalized_terms if term in source]
            if not matched:
                continue
            position = _metadata_float(metadata.get("document_position_ratio"), 0.0)
            entity_rows.append((position, row, metadata, matched))
        entity_rows.sort(key=lambda item: item[0], reverse=True)
        anchors = entity_rows[: min(5, len(entity_rows))]
        if not anchors:
            return []

        output_rows: Dict[str, tuple[float, sqlite3.Row, List[str]]] = {}
        for position, row, metadata, matched_terms in anchors:
            source_index = int(metadata.get("source_unit_index") or 0)
            document_id = str(row["document_id"] or "")
            try:
                with self._connect() as conn:
                    window_rows = conn.execute(
                        """
                        SELECT *,
                               CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL) AS position
                        FROM chunks
                        WHERE collection_name = ? AND document_id = ?
                          AND chunk_kind IN ('terminal_window', 'navigation_window')
                          AND CAST(json_extract(metadata_json, '$.source_unit_start') AS INTEGER) <= ?
                          AND CAST(json_extract(metadata_json, '$.source_unit_end') AS INTEGER) >= ?
                        ORDER BY
                            CASE WHEN chunk_kind = 'terminal_window' THEN 0 ELSE 1 END,
                            position DESC,
                            chunk_id ASC
                        LIMIT 12
                        """,
                        (self.collection_name, document_id, source_index, source_index),
                    ).fetchall()
            except sqlite3.Error as exc:
                logger.warning("实体邻域窗口查询失败: %s", exc)
                continue
            for window in window_rows:
                key = str(window["chunk_id"] or "")
                score = 2.2 + position * 0.4 + (0.35 if str(window["chunk_kind"]) == "terminal_window" else 0.0)
                existing = output_rows.get(key)
                if existing is None or score > existing[0]:
                    output_rows[key] = (score, window, matched_terms)

        ordered = sorted(output_rows.values(), key=lambda item: item[0], reverse=True)[:limit]
        return [
            self._row_to_hit(row, rank, "entity_terminal_neighborhood", matched_terms=matched)
            for rank, (_, row, matched) in enumerate(ordered, start=1)
        ]

    def navigation_inventory(self, document_ids: Sequence[str] | None = None) -> List[Dict[str, object]]:
        """Return a read-only inventory of source navigation windows."""

        document_ids = [str(item) for item in (document_ids or []) if str(item)]
        where = ["collection_name = ?", "chunk_kind IN ('terminal_window', 'navigation_window')"]
        params: List[object] = [self.collection_name]
        if document_ids:
            marks = ",".join("?" for _ in document_ids)
            where.append(f"document_id IN ({marks})")
            params.extend(document_ids)
        sql = f"""
            SELECT
                document_id,
                COALESCE(json_extract(metadata_json, '$.file_name'), '') AS file_name,
                chunk_kind,
                LOWER(COALESCE(json_extract(metadata_json, '$.navigation_region'), 'unknown')) AS navigation_region,
                COUNT(*) AS window_count,
                MIN(CAST(json_extract(metadata_json, '$.page_start') AS INTEGER)) AS page_start,
                MAX(CAST(json_extract(metadata_json, '$.page_end') AS INTEGER)) AS page_end,
                MIN(CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL)) AS min_position,
                MAX(CAST(json_extract(metadata_json, '$.document_position_ratio') AS REAL)) AS max_position
            FROM chunks
            WHERE {' AND '.join(where)}
            GROUP BY document_id, file_name, chunk_kind, navigation_region
            ORDER BY document_id, navigation_region, chunk_kind
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("读取导航索引清单失败: %s", exc)
            return []
        return [dict(row) for row in rows]

    def document_outline(self, document_ids: Sequence[str], limit: int = 80) -> List[Dict[str, object]]:
        """Return locally indexed section metadata for diagnostics or a second-pass planner."""

        document_ids = [str(item) for item in document_ids if str(item)]
        if not document_ids:
            return []
        marks = ",".join("?" for _ in document_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_id, document_id, title, section_path, metadata_json
                FROM chunks
                WHERE collection_name = ? AND document_id IN ({marks})
                  AND chunk_kind = 'section_card'
                LIMIT ?
                """,
                (self.collection_name, *document_ids, limit),
            ).fetchall()
        outline: List[Dict[str, object]] = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = {}
            outline.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "section_id": str(metadata.get("section_id") or ""),
                    "title": str(row["section_path"] or row["title"] or ""),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "position_start": metadata.get("document_position_start"),
                    "position_end": metadata.get("document_position_end"),
                }
            )
        return outline

    def get_records(self, chunk_ids: Sequence[str]) -> Dict[str, HybridRecord]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE collection_name = ? AND chunk_id IN ({placeholders})",
                (self.collection_name, *chunk_ids),
            ).fetchall()
        records = {}
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            records[str(row["chunk_id"])] = HybridRecord(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"] or ""),
                parent_chunk_id=str(row["parent_chunk_id"] or ""),
                chunk_kind=str(row["chunk_kind"] or "child"),
                language=str(row["language"] or "unknown"),
                title=str(row["title"] or ""),
                section_path=str(row["section_path"] or ""),
                content=str(row["content"] or ""),
                metadata=metadata,
            )
        return records

    def set_meta(self, values: Dict[str, object]) -> None:
        with self._connect() as conn:
            for key, value in values.items():
                conn.execute(
                    "INSERT OR REPLACE INTO index_meta(collection_name, key, value) VALUES (?, ?, ?)",
                    (self.collection_name, key, json.dumps(value, ensure_ascii=False)),
                )

    def get_meta(self) -> Dict[str, object]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM index_meta WHERE collection_name = ?", (self.collection_name,)).fetchall()
        output: Dict[str, object] = {}
        for row in rows:
            try:
                output[str(row["key"])] = json.loads(row["value"])
            except Exception:
                output[str(row["key"])] = row["value"]
        return output

    def _fts_search(self, tokens: Sequence[str], limit: int) -> List[HybridHit]:
        query = " OR ".join(self._escape_fts_token(token) for token in tokens[:20])
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT c.*, bm25(chunks_fts) AS fts_score
                    FROM chunks_fts
                    JOIN chunks c ON c.collection_name = chunks_fts.collection_name AND c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts.collection_name = ? AND chunks_fts MATCH ?
                    ORDER BY fts_score
                    LIMIT ?
                    """,
                    (self.collection_name, query, limit),
                ).fetchall()
            return [self._row_to_hit(row, index, "lexical", matched_terms=list(tokens[:8])) for index, row in enumerate(rows, start=1)]
        except sqlite3.Error:
            return []

    def _like_search(self, tokens: Sequence[str], limit: int) -> List[HybridHit]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE collection_name = ?", (self.collection_name,)).fetchall()
        scored = []
        for row in rows:
            lexical_text = str(row["lexical_text"] or "")
            score = sum(1 for token in tokens if token in lexical_text)
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._row_to_hit(row, index, "lexical", matched_terms=list(tokens[:8])) for index, (_, row) in enumerate(scored[:limit], start=1)]

    def _row_to_hit(self, row: sqlite3.Row, rank: int, channel: str, matched_terms: Sequence[str]) -> HybridHit:
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata.setdefault("chunk_id", row["chunk_id"])
        metadata.setdefault("parent_chunk_id", row["parent_chunk_id"])
        metadata.setdefault("chunk_kind", row["chunk_kind"])
        metadata.setdefault("chunk_language", row["language"])
        return HybridHit(
            chunk_id=str(row["chunk_id"]),
            content=str(row["content"] or ""),
            metadata=metadata,
            rank=rank,
            score=1.0 / rank,
            channel=channel,
            matched_terms=list(matched_terms),
        )

    def _lexical_text(self, record: HybridRecord) -> str:
        raw = " ".join([record.title, record.section_path, record.content, " ".join(record.aliases), " ".join(record.exact_terms)])
        tokens = tokenize_for_search(raw, language=record.language)
        return " ".join(tokens + [normalize_for_lexical_search(raw, record.language)])

    def _escape_fts_token(self, token: str) -> str:
        escaped = token.replace('"', '""')
        return f'"{escaped}"'


def _metadata_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))
