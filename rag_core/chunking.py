from __future__ import annotations

import re
from typing import Iterable, List


def normalize_text(text: str) -> str:
    """尽量保留段落分隔，同时清理多余空白。"""

    text = remove_control_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = fix_pdf_line_breaks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def remove_control_chars(text: str) -> str:
    return "".join(ch for ch in text or "" if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def fix_pdf_line_breaks(text: str) -> str:
    """修复 PDF 常见换行问题，同时保留真实段落。"""

    lines = text.split("\n")
    merged: List[str] = []
    buffer = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append("")
            continue

        if not buffer:
            buffer = line
            continue

        if should_join_lines(buffer, line):
            if buffer.endswith("-") and line and line[0].islower():
                buffer = buffer[:-1] + line
            elif is_cjk_text(buffer + line):
                buffer += line
            else:
                buffer += " " + line
        else:
            merged.append(buffer.strip())
            buffer = line

    if buffer:
        merged.append(buffer.strip())
    return "\n".join(merged)


def should_join_lines(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.endswith((".", "。", "！", "?", "？", "；", ";", "：", ":")):
        return False
    if re.match(r"^\s*(第?\d+[章节条\.、)]|[A-Z][A-Z\s]{4,})", current):
        return False
    return True


def is_cjk_text(text: str) -> bool:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin_count = len(re.findall(r"[A-Za-z]", text or ""))
    return cjk_count > latin_count


def split_text_with_overlap(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """
    段落感知切片：
    - 优先保留自然段落
    - 长段落按句子继续切
    - 只在必要时使用重叠，避免把语义块切碎
    """

    normalized = normalize_text(text)
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in iter_semantic_units(normalized):
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > chunk_size:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(split_long_unit(unit, chunk_size=chunk_size, overlap=overlap))
            continue

        projected_len = current_len + len(unit) + (2 if current else 0)
        if current and projected_len > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = build_overlap_tail(chunks[-1], overlap)
            current_len = sum(len(part) for part in current)

        current.append(unit)
        current_len += len(unit) + (2 if current_len else 0)

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def iter_semantic_units(text: str) -> Iterable[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        yield paragraph


def split_long_unit(text: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sliding_window_split(text, chunk_size=chunk_size, overlap=overlap)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(sliding_window_split(sentence, chunk_size=chunk_size, overlap=overlap))
            continue
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current).strip())
            current = build_overlap_tail(chunks[-1], overlap)
            current_len = sum(len(part) for part in current)
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [part.strip() for part in parts if part and part.strip()]


def sliding_window_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_overlap_tail(text: str, overlap: int) -> List[str]:
    if overlap <= 0 or not text:
        return []
    tail = text[-overlap:].strip()
    return [tail] if tail else []
