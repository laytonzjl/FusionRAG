from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from rag_core.models import ParsedChunk
from rag_core.multilingual import detect_language, normalize_entity_name, surface_variants


# v3 turns a flat chunk collection into a navigable document map.  Existing
# collections must be rebuilt because the metadata and card kinds change.
DOCUMENT_STRUCTURE_VERSION = "document_structure_v5_segment_topology"


@dataclass
class DocumentSignals:
    document_type: str = "unknown"
    layout_type: str = "unknown"
    has_tables: bool = False
    has_code_blocks: bool = False
    has_sections: bool = False
    is_scanned: bool = False
    document_title: str = ""
    author_or_source: str = ""
    document_date: str = ""
    document_version: str = ""
    topics: List[str] = field(default_factory=list)


@dataclass
class DocumentSection:
    section_id: str
    title: str
    start_unit: int
    end_unit: int
    page_start: Optional[int]
    page_end: Optional[int]


@dataclass
class DocumentSegment:
    """A source-order span bounded by a structural restart.

    Segments are inferred only from already detected section titles and repeated
    structural markers. They do not encode document-topic words or question
    templates. This separates bundled works and repeated manuals without
    generating any summary of their content.
    """

    segment_id: str
    start_unit: int
    end_unit: int
    page_start: Optional[int]
    page_end: Optional[int]
    label: str = ""


@dataclass
class DocumentMap:
    """Offline navigation map used by the retrieval executor.

    The map is deliberately evidence-only: it stores source order, explicit
    headings, structural segments, entity mentions, and windows. It does not
    generate story summaries or inferred facts during indexing.
    """

    unit_metadata: Dict[int, Dict[str, object]] = field(default_factory=dict)
    sections: List[DocumentSection] = field(default_factory=list)
    segments: List[DocumentSegment] = field(default_factory=list)
    entities: List[Dict[str, object]] = field(default_factory=list)
    total_units: int = 0
    total_pages: int = 0

    def metadata_for_unit(self, index: int) -> Dict[str, object]:
        return dict(self.unit_metadata.get(index) or {})

    def section_for_unit(self, index: int) -> Optional[DocumentSection]:
        for section in self.sections:
            if section.start_unit <= index <= section.end_unit:
                return section
        return self.sections[0] if self.sections else None

    def segment_for_unit(self, index: int) -> Optional[DocumentSegment]:
        for segment in self.segments:
            if segment.start_unit <= index <= segment.end_unit:
                return segment
        return self.segments[0] if self.segments else None


def analyze_document(original_name: str, documents: Sequence[Dict[str, object]]) -> DocumentSignals:
    first_text = "\n".join(str(document.get("text", "")) for document in documents[:8])
    all_text_sample = "\n".join(str(document.get("text", "")) for document in documents[:80])
    lower_name = original_name.casefold()
    lower_text = all_text_sample.casefold()

    document_type = "unknown"
    if re.search(r"(合同|协议|甲方|乙方|违约|contract|agreement)", lower_text):
        document_type = "contract"
    elif re.search(r"(制度|政策|办法|条例|policy|procedure)", lower_text):
        document_type = "policy"
    elif re.search(r"(def |class |function|api|README|```|参数|接口)", all_text_sample):
        document_type = "technical_doc"
    elif re.search(r"(报告|年度|季度|分析|report)", lower_text):
        document_type = "report"
    elif re.search(r"(小说|历险记|chapter|第[一二三四五六七八九十百\d]+章|文学)", lower_text + lower_name):
        document_type = "book"

    has_tables = "|" in all_text_sample or "\t" in all_text_sample or re.search(r"(列名|表格|table)", lower_text) is not None
    has_code = "```" in all_text_sample or re.search(r"\b(def|class|import|function|const|let|var)\b", all_text_sample) is not None
    has_sections = re.search(r"(第[一二三四五六七八九十百\d]+[章节条]|^\s*\d+(?:\.\d+)+)", all_text_sample, flags=re.M) is not None
    is_scanned = any((document.get("metadata") or {}).get("ocr") for document in documents)
    title = infer_title(original_name, first_text)
    return DocumentSignals(
        document_type=document_type,
        layout_type="table_heavy" if has_tables else "mixed" if has_code else "text",
        has_tables=has_tables,
        has_code_blocks=has_code,
        has_sections=has_sections,
        is_scanned=bool(is_scanned),
        document_title=title,
        author_or_source=infer_author(first_text),
        document_date=infer_date(first_text),
        document_version=infer_version(first_text),
        topics=extract_topics(first_text, title),
    )


def build_document_map(
    original_name: str,
    documents: Sequence[Dict[str, object]],
    signals: Optional[DocumentSignals] = None,
) -> DocumentMap:
    """Create source-order, section, and entity-mention metadata for all units.

    This function is intentionally generic.  It detects explicit document
    structure (headings / title-like source lines) and links entities based on
    entities actually observed in the indexed document.  It has no task- or
    title-specific vocabulary such as "ending", "friend", or "age".
    """

    signals = signals or analyze_document(original_name, documents)
    clean_documents = [document for document in documents if str(document.get("text", "")).strip()]
    total_units = len(clean_documents)
    page_values = [
        _as_int((document.get("metadata") or {}).get("page"))
        for document in clean_documents
    ]
    page_values = [page for page in page_values if page is not None]
    page_count_values = [
        _as_int((document.get("metadata") or {}).get("page_count"))
        for document in clean_documents
    ]
    page_count_values = [count for count in page_count_values if count is not None]
    total_pages = max(page_count_values) if page_count_values else (max(page_values) if page_values else 0)

    entities = extract_entities(signals, clean_documents)
    sections = _build_sections(clean_documents, total_units)
    if not sections:
        sections = [
            DocumentSection(
                section_id="section:root",
                title="正文",
                start_unit=0,
                end_unit=max(total_units - 1, 0),
                page_start=min(page_values) if page_values else None,
                page_end=max(page_values) if page_values else None,
            )
        ]
    segments = _build_segments(sections, clean_documents, total_units)

    entity_patterns: List[tuple[str, Dict[str, object]]] = []
    for entity in entities:
        canonical = normalize_entity_name(str(entity.get("name") or ""))
        if len(canonical) >= 2:
            entity_patterns.append((canonical, entity))

    unit_metadata: Dict[int, Dict[str, object]] = {}
    for index, document in enumerate(clean_documents):
        text = str(document.get("text") or "")
        metadata = dict(document.get("metadata") or {})
        position = index / max(total_units - 1, 1)
        section = next((item for item in sections if item.start_unit <= index <= item.end_unit), sections[0])
        segment = next((item for item in segments if item.start_unit <= index <= item.end_unit), segments[0])
        segment_position = (index - segment.start_unit) / max(segment.end_unit - segment.start_unit, 1)
        normalized_text = normalize_entity_name(text)
        aliases: List[str] = []
        mentioned_entities: List[str] = []
        for canonical, entity in entity_patterns:
            if canonical and canonical in normalized_text:
                name = str(entity.get("name") or "")
                if name:
                    mentioned_entities.append(name)
                aliases.extend([str(alias) for alias in entity.get("aliases") or []])
        page = _as_int(metadata.get("page"))
        unit_metadata[index] = {
            "source_unit_index": index,
            "source_unit_count": total_units,
            "document_position_ratio": round(position, 6),
            "document_position_start": round(position, 6),
            "document_position_end": round(position, 6),
            "document_segment_id": segment.segment_id,
            "document_segment_index": segments.index(segment),
            "document_segment_start_unit": segment.start_unit,
            "document_segment_end_unit": segment.end_unit,
            "segment_position_ratio": round(segment_position, 6),
            "navigation_region": _region_for_position(segment_position),
            "section_id": section.section_id,
            "section_path": section.title or "正文",
            "section_index": sections.index(section),
            "section_unit_start": section.start_unit,
            "section_unit_end": section.end_unit,
            "source_page": page,
            "document_page_count": total_pages,
            "aliases": list(dict.fromkeys(alias for alias in aliases if alias))[:16],
            "entity_mentions": list(dict.fromkeys(mentioned_entities))[:16],
        }

    return DocumentMap(
        unit_metadata=unit_metadata,
        sections=sections,
        segments=segments,
        entities=entities,
        total_units=total_units,
        total_pages=total_pages,
    )


def build_cards(
    original_name: str,
    stored_name: str,
    document_id: str,
    upload_time: str,
    file_sha256: str,
    documents: Sequence[Dict[str, object]],
    document_map: Optional[DocumentMap] = None,
) -> List[ParsedChunk]:
    """Build document, section, entity, and source-backed navigation cards.

    Navigation cards are raw source windows with page/order metadata.  They are
    not generated summaries, so an online answer can always cite their source
    spans and remain grounded in the uploaded material.
    """

    signals = analyze_document(original_name, documents)
    document_map = document_map or build_document_map(original_name, documents, signals=signals)
    profile = detect_language("\n".join(str(document.get("text", "")) for document in documents[:5]))
    source_pages = sorted(
        {
            int((document.get("metadata") or {}).get("page"))
            for document in documents
            if _as_int((document.get("metadata") or {}).get("page")) is not None
        }
    )
    evidence_pages = source_pages[:5]
    base_metadata = {
        "document_id": document_id,
        "document_title": signals.document_title,
        "document_type": signals.document_type,
        "layout_type": signals.layout_type,
        "has_tables": signals.has_tables,
        "has_code_blocks": signals.has_code_blocks,
        "has_sections": signals.has_sections,
        "is_scanned": signals.is_scanned,
        "document_language": profile.language,
        "language_confidence": profile.confidence,
        "script_distribution": profile.script_distribution,
        "file_name": original_name,
        "stored_file_name": stored_name,
        "upload_time": upload_time,
        "file_sha256": file_sha256,
        "text_source": "structured_card",
        "text_quality_score": 1.0,
        "document_unit_count": document_map.total_units,
        "document_page_count": document_map.total_pages,
    }

    cards: List[ParsedChunk] = [
        ParsedChunk(
            content=(
                f"文档标题：{signals.document_title}\n"
                f"原始文件名：{original_name}\n"
                f"作者/发布方：{signals.author_or_source or 'unknown'}\n"
                f"文档类型：{signals.document_type}\n"
                f"版本/日期：{signals.document_version or 'unknown'} {signals.document_date or ''}\n"
                f"主题：{'、'.join(signals.topics) if signals.topics else signals.document_title}\n"
                f"页数：{document_map.total_pages or 'unknown'}\n"
                f"证据页码：{', '.join(map(str, evidence_pages)) if evidence_pages else 'unknown'}"
            ),
            metadata={
                **base_metadata,
                "chunk_kind": "document_card",
                "section_id": "document_card",
                "section_path": "Document Card",
                "source_chunk_ids": "",
                "page_start": evidence_pages[0] if evidence_pages else None,
                "page_end": evidence_pages[-1] if evidence_pages else None,
                "document_position_ratio": 0.0,
                "navigation_region": "front",
                "aliases": list_aliases(signals.document_title, original_name),
                "entity_type": "文档",
            },
        )
    ]

    cards.extend(_build_section_cards(base_metadata, document_map, documents))
    cards.extend(_build_navigation_window_cards(base_metadata, document_map, documents))

    for entity in document_map.entities:
        cards.append(
            ParsedChunk(
                content=(
                    f"实体：{entity['name']}\n"
                    f"实体类型：{entity['type']}\n"
                    f"所属文档：{signals.document_title}\n"
                    f"别名：{'、'.join(entity['aliases'])}\n"
                    f"证据页码：{', '.join(map(str, entity['pages'])) if entity['pages'] else 'unknown'}"
                ),
                metadata={
                    **base_metadata,
                    "chunk_kind": "entity_card",
                    "section_id": f"entity:{normalize_entity_name(entity['name'])}",
                    "section_path": f"Entity Card / {entity['name']}",
                    "page_start": entity["pages"][0] if entity["pages"] else None,
                    "page_end": entity["pages"][-1] if entity["pages"] else None,
                    "navigation_region": "all",
                    "aliases": entity["aliases"],
                    "entity_type": entity["type"],
                    "confidence": entity["confidence"],
                },
            )
        )
    return cards


def _build_section_cards(
    base_metadata: Dict[str, object],
    document_map: DocumentMap,
    documents: Sequence[Dict[str, object]],
) -> List[ParsedChunk]:
    cards: List[ParsedChunk] = []
    for section in document_map.sections:
        texts = [
            str(documents[index].get("text") or "").strip()
            for index in range(section.start_unit, min(section.end_unit + 1, len(documents)))
            if str(documents[index].get("text") or "").strip()
        ]
        if not texts:
            continue
        excerpt = _bounded_source_excerpt(texts, head_chars=900, tail_chars=700)
        midpoint = (section.start_unit + section.end_unit) / max(2 * max(document_map.total_units - 1, 1), 1)
        segment = document_map.segment_for_unit((section.start_unit + section.end_unit) // 2)
        segment_position = (
            ((section.start_unit + section.end_unit) // 2 - segment.start_unit) / max(segment.end_unit - segment.start_unit, 1)
            if segment is not None
            else midpoint
        )
        cards.append(
            ParsedChunk(
                content=(
                    f"文档区段：{section.title}\n"
                    f"来源位置：单元 {section.start_unit + 1}-{section.end_unit + 1}/{document_map.total_units}\n"
                    f"页码：{_page_label(section.page_start, section.page_end)}\n"
                    f"区段原文摘录：\n{excerpt}"
                ),
                metadata={
                    **base_metadata,
                    "chunk_kind": "section_card",
                    "section_id": section.section_id,
                    "section_path": section.title,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "source_unit_start": section.start_unit,
                    "source_unit_end": section.end_unit,
                    "document_position_start": round(section.start_unit / max(document_map.total_units - 1, 1), 6),
                    "document_position_end": round(section.end_unit / max(document_map.total_units - 1, 1), 6),
                    "document_position_ratio": round(midpoint, 6),
                    "document_segment_id": segment.segment_id if segment else "segment:0",
                    "document_segment_start_unit": segment.start_unit if segment else section.start_unit,
                    "document_segment_end_unit": segment.end_unit if segment else section.end_unit,
                    "segment_position_ratio": round(segment_position, 6),
                    "navigation_region": _region_for_position(segment_position),
                    "aliases": list_aliases(section.title, section.title),
                    "entity_type": "文档区段",
                },
            )
        )
    return cards


def _build_navigation_window_cards(
    base_metadata: Dict[str, object],
    document_map: DocumentMap,
    documents: Sequence[Dict[str, object]],
) -> List[ParsedChunk]:
    """Build source windows inside structural segments.

    Windows never cross a detected segment boundary. A terminal window therefore
    denotes the endpoint of the relevant structural span, not necessarily the
    last page of a bundled file.
    """

    if document_map.total_units <= 0:
        return []

    cards: List[ParsedChunk] = []
    for segment in document_map.segments:
        span = segment.end_unit - segment.start_unit + 1
        if span <= 0:
            continue
        window_size = 4 if span >= 8 else max(1, span)
        step = max(1, window_size // 2)
        starts = list(range(segment.start_unit, max(segment.end_unit - window_size + 2, segment.start_unit + 1), step))
        terminal_start = max(segment.end_unit - window_size + 1, segment.start_unit)
        if terminal_start not in starts:
            starts.append(terminal_start)

        for start in sorted(set(starts)):
            end = min(segment.end_unit, start + window_size - 1)
            source_texts = [
                str(documents[index].get("text") or "").strip()
                for index in range(start, end + 1)
                if str(documents[index].get("text") or "").strip()
            ]
            if not source_texts:
                continue
            first_meta = document_map.metadata_for_unit(start)
            last_meta = document_map.metadata_for_unit(end)
            doc_ratio = (start + end) / max(2 * max(document_map.total_units - 1, 1), 1)
            segment_ratio = ((start + end) / 2 - segment.start_unit) / max(segment.end_unit - segment.start_unit, 1)
            region = _region_for_position(segment_ratio)
            page_start = _as_int(first_meta.get("source_page"))
            page_end = _as_int(last_meta.get("source_page"))
            source_text = _bounded_source_excerpt(source_texts, head_chars=4200, tail_chars=2600)
            chunk_kind = "terminal_window" if region == "terminal" else "navigation_window"
            section = document_map.section_for_unit((start + end) // 2)
            cards.append(
                ParsedChunk(
                    content=(
                        f"文档导航原文窗口\n"
                        f"结构片段：{segment.segment_id}\n"
                        f"区域：{region}\n"
                        f"来源位置：单元 {start + 1}-{end + 1}/{document_map.total_units}\n"
                        f"页码：{_page_label(page_start, page_end)}\n"
                        f"所属区段：{section.title if section else '正文'}\n"
                        f"原文：\n{source_text}"
                    ),
                    metadata={
                        **base_metadata,
                        "chunk_kind": chunk_kind,
                        "section_id": section.section_id if section else "section:root",
                        "section_path": section.title if section else "正文",
                        "page_start": page_start,
                        "page_end": page_end,
                        "source_unit_start": start,
                        "source_unit_end": end,
                        "document_position_start": round(start / max(document_map.total_units - 1, 1), 6),
                        "document_position_end": round(end / max(document_map.total_units - 1, 1), 6),
                        "document_position_ratio": round(doc_ratio, 6),
                        "document_segment_id": segment.segment_id,
                        "document_segment_start_unit": segment.start_unit,
                        "document_segment_end_unit": segment.end_unit,
                        "segment_position_ratio": round(segment_ratio, 6),
                        "navigation_region": region,
                        "aliases": list(dict.fromkeys([*first_meta.get("aliases", []), *last_meta.get("aliases", [])]))[:16],
                        "entity_type": "文档导航窗口",
                    },
                )
            )
    return cards


def enrich_chunk_metadata(
    chunk: ParsedChunk,
    document_id: str,
    document_title: str,
    document_type: str,
    parent_chunk_id: str,
    chunk_id: str,
    chunk_kind: str,
    sequence_no: int,
    content_hash: str,
) -> ParsedChunk:
    profile = detect_language(chunk.content)
    metadata = dict(chunk.metadata)
    page = _as_int(metadata.get("page"))
    page_start = _as_int(metadata.get("page_start"))
    page_end = _as_int(metadata.get("page_end"))
    if page_start is None:
        page_start = page
    if page_end is None:
        page_end = page
    position = _safe_float(metadata.get("document_position_ratio"), default=0.0)
    metadata.update(
        {
            "document_id": document_id,
            "document_title": document_title,
            "document_type": document_type,
            "section_id": metadata.get("section_id") or infer_section_id(chunk.content, sequence_no),
            "section_path": metadata.get("section_path") or infer_section_path(chunk.content),
            "chunk_id": chunk_id,
            "parent_chunk_id": parent_chunk_id,
            "chunk_kind": chunk_kind,
            "page_start": page_start,
            "page_end": page_end,
            "sequence_no": sequence_no,
            "content_hash": content_hash,
            "chunk_language": profile.language,
            "language_confidence": profile.confidence,
            "script_distribution": profile.script_distribution,
            "text_source": "ocr" if metadata.get("ocr") else metadata.get("text_source", "native"),
            "text_quality_score": score_text_quality(chunk.content, metadata),
            "document_position_ratio": round(max(0.0, min(1.0, position)), 6),
            "navigation_region": metadata.get("navigation_region") or _region_for_position(position),
        }
    )
    return ParsedChunk(content=chunk.content, metadata=metadata)


def infer_title(original_name: str, first_text: str) -> str:
    stem = Path(original_name).stem
    lines = [line.strip() for line in (first_text or "").splitlines() if 2 <= len(line.strip()) <= 80]
    bad_patterns = ("ISBN", "CIP", "责任编辑", "出版社")
    for line in lines[:8]:
        if any(pattern in line for pattern in bad_patterns):
            continue
        if re.search(r"第\s*\d+\s*页", line):
            continue
        return line
    return stem


def infer_author(text: str) -> str:
    patterns = [
        r"(?:作者|著|发布方|编者|Author|By)[:：\s]+([^\n]{2,80})",
        r"([^\n]{1,40})(?:著|编著|译)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.I)
        if match:
            return match.group(1).strip()
    return ""


def infer_date(text: str) -> str:
    match = re.search(r"(20\d{2}|19\d{2})[-年./](\d{1,2})?[-月./]?(\d{1,2})?", text or "")
    return match.group(0) if match else ""


def infer_version(text: str) -> str:
    match = re.search(r"\b(v?\d+(?:\.\d+){1,3})\b|版本[:：\s]*([A-Za-z0-9_.-]+)", text or "", flags=re.I)
    if not match:
        return ""
    return next(group for group in match.groups() if group)


def extract_topics(text: str, title: str) -> List[str]:
    topics = [title]
    for match in re.findall(r"《([^》]{2,40})》", text or ""):
        topics.append(match)
    return list(dict.fromkeys(topic.strip() for topic in topics if topic.strip()))[:8]


def list_aliases(title: str, original_name: str) -> List[str]:
    aliases = []
    for item in [title, Path(original_name).stem, original_name]:
        aliases.extend(surface_variants(item))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def extract_entities(signals: DocumentSignals, documents: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Extract document-observed entities across the full document.

    This is not a relation extractor.  It only makes source-observed aliases
    available for local entity linking and exact retrieval.  The all-document
    pass replaces the previous first-20-unit sampling, which omitted entities
    that occur later in long documents.
    """

    entities: Dict[str, Dict[str, object]] = {}
    seed_names = list_aliases(signals.document_title, signals.document_title)
    for name in seed_names:
        add_entity(entities, name, "作品" if signals.document_type == "book" else "文档", [], 0.9)

    for document in documents:
        text = str(document.get("text", ""))
        metadata = document.get("metadata") or {}
        page = _as_int(metadata.get("page"))
        for name in re.findall(r"《([^》]{2,40})》", text):
            add_entity(entities, name, "作品", [page] if page else [], 0.84)
        # Includes transliterated names such as 汤姆·索亚.  Surface variants add
        # dotless forms for query/entity linking without encoding any title list.
        for name in re.findall(r"[\u4e00-\u9fff]{2,10}[·・•][\u4e00-\u9fff]{1,10}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", text):
            add_entity(entities, name, "人物", [page] if page else [], 0.66)
        for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\(\)|\b[A-Za-z_][A-Za-z0-9_]*\b", text):
            if len(name) >= 3 and re.search(r"(api|class|function|timeout|token|config|client|server)", name, re.I):
                add_entity(entities, name, "代码模块", [page] if page else [], 0.6)

    ordered = sorted(
        entities.values(),
        key=lambda item: (float(item.get("confidence") or 0.0), len(item.get("pages") or [])),
        reverse=True,
    )
    return ordered[:160]


def add_entity(
    entities: Dict[str, Dict[str, object]],
    name: str,
    entity_type: str,
    pages: Sequence[object],
    confidence: float,
) -> None:
    key = normalize_entity_name(name)
    if not key or len(key) < 2:
        return
    item = entities.setdefault(
        key,
        {
            "name": name,
            "type": entity_type,
            "aliases": surface_variants(name),
            "pages": [],
            "confidence": confidence,
        },
    )
    item["confidence"] = max(float(item["confidence"]), confidence)
    for page in pages:
        if page is not None and page not in item["pages"]:
            item["pages"].append(page)
    item["pages"].sort()


def infer_section_id(text: str, sequence_no: int) -> str:
    title = infer_section_path(text)
    return normalize_entity_name(title) or f"section-{sequence_no}"


def infer_section_path(text: str) -> str:
    for line in (text or "").splitlines()[:5]:
        line = line.strip()
        if re.match(r"^(第[一二三四五六七八九十百\d]+[章节条]|#{1,6}\s+|\d+(?:\.\d+)+)", line):
            return line[:120]
    return "正文"


def score_text_quality(text: str, metadata: Dict[str, object]) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    valid = re.findall(r"[A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]", compact)
    valid_ratio = len(valid) / max(len(compact), 1)
    repeated_ratio = len(re.findall(r"(.)\1{4,}", compact)) / max(len(compact), 1)
    quality = max(0.0, min(1.0, valid_ratio - repeated_ratio * 2.0))
    if metadata.get("ocr"):
        quality *= 0.88
    return round(quality, 4)


def _build_sections(documents: Sequence[Dict[str, object]], total_units: int) -> List[DocumentSection]:
    headings: List[tuple[int, str]] = []
    for index, document in enumerate(documents):
        metadata = document.get("metadata") or {}
        text = str(document.get("text") or "")
        candidate = _heading_candidate(text, metadata)
        if candidate:
            headings.append((index, candidate))

    # De-duplicate repeated page headers while retaining source order.  A title
    # must be separated from the previous identical heading by a meaningful
    # source span to begin a new section.
    accepted: List[tuple[int, str]] = []
    previous_key = ""
    for index, title in headings:
        key = normalize_entity_name(title)
        if key == previous_key and accepted and index - accepted[-1][0] < 3:
            continue
        accepted.append((index, title))
        previous_key = key

    if not accepted:
        return []
    if accepted[0][0] > 0:
        accepted.insert(0, (0, "正文"))

    sections: List[DocumentSection] = []
    for section_index, (start, title) in enumerate(accepted):
        end = (accepted[section_index + 1][0] - 1) if section_index + 1 < len(accepted) else total_units - 1
        if end < start:
            continue
        pages = [
            _as_int((documents[item].get("metadata") or {}).get("page"))
            for item in range(start, min(end + 1, len(documents)))
        ]
        pages = [page for page in pages if page is not None]
        sections.append(
            DocumentSection(
                section_id=f"section:{section_index}:{normalize_entity_name(title) or 'untitled'}",
                title=title[:120] or "正文",
                start_unit=start,
                end_unit=end,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
        )
    return sections



def _build_segments(
    sections: Sequence[DocumentSection],
    documents: Sequence[Dict[str, object]],
    total_units: int,
) -> List[DocumentSegment]:
    """Infer source spans from repeated structural headings.

    A repeated normalized heading after a meaningful source span indicates a
    structural restart. This is a topology signal only; it does not depend on
    topic vocabulary, language-specific question wording, or generated text.
    """

    if total_units <= 0:
        return []
    boundaries = [0]
    seen_titles: Dict[str, int] = {}
    for section in sections:
        key = normalize_entity_name(section.title)
        if not key:
            continue
        previous = seen_titles.get(key)
        if previous is not None and section.start_unit - previous >= 3:
            boundaries.append(section.start_unit)
        seen_titles[key] = section.start_unit

    starts = sorted(set(boundary for boundary in boundaries if 0 <= boundary < total_units))
    segments: List[DocumentSegment] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else total_units - 1
        pages = [
            _as_int((documents[position].get("metadata") or {}).get("page"))
            for position in range(start, min(end + 1, len(documents)))
        ]
        pages = [page for page in pages if page is not None]
        section = next((item for item in sections if item.start_unit <= start <= item.end_unit), None)
        segments.append(
            DocumentSegment(
                segment_id=f"segment:{index}",
                start_unit=start,
                end_unit=end,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                label=section.title if section else "正文",
            )
        )
    return segments or [
        DocumentSegment(
            segment_id="segment:0",
            start_unit=0,
            end_unit=max(total_units - 1, 0),
            page_start=None,
            page_end=None,
            label="正文",
        )
    ]

def _heading_candidate(text: str, metadata: Dict[str, object]) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    # Parser prefixes PDF blocks with "第 N 页".  Skip that technical marker.
    lines = [line for line in lines if not re.match(r"^第\s*\d+\s*页(?:（OCR 识别）)?$", line)]
    if not lines:
        return ""
    first = lines[0]
    if re.match(r"^(第[一二三四五六七八九十百\d]+[章节条]|CHAPTER\s+\w+|PART\s+\w+|#{1,6}\s+|\d+(?:\.\d+)+)", first, re.I):
        return first[:120]
    # A title-like source line is used only when it is a short standalone line
    # at the beginning of a source unit.  This is generic document layout
    # detection, not query-specific text matching.
    compact = re.sub(r"\s+", "", first)
    if (
        2 <= len(compact) <= 52
        and not re.search(r"[。！？!?；;，,：:]", compact)
        and re.fullmatch(r"[A-Za-z0-9\u4e00-\u9fff·・•《》()（）\-—_ ]+", first)
        and (_as_int(metadata.get("page_block_index")) in {None, 0})
    ):
        return first[:120]
    return ""


def _bounded_source_excerpt(texts: Sequence[str], head_chars: int, tail_chars: int) -> str:
    combined = "\n\n".join(text for text in texts if text)
    if len(combined) <= head_chars + tail_chars + 120:
        return combined
    return f"{combined[:head_chars]}\n\n[...中间原文窗口省略...]\n\n{combined[-tail_chars:]}"


def _region_for_position(position: float) -> str:
    if position <= 0.16:
        return "front"
    if position >= 0.78:
        return "terminal"
    return "middle"


def _page_label(start: Optional[int], end: Optional[int]) -> str:
    if start is None and end is None:
        return "unknown"
    if start == end or end is None:
        return str(start)
    return f"{start}-{end}"


def _as_int(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
