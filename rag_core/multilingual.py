from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Sequence

from rag_core.models import LanguageProfile


NORMALIZATION_VERSION = "multilingual_norm_v2"
LANGUAGE_PROCESSING_VERSION = "language_processing_v2"


SCRIPT_RANGES = {
    "Han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "Hiragana": [(0x3040, 0x309F)],
    "Katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "Hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "Cyrillic": [(0x0400, 0x04FF)],
    "Latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "Digit": [(0x0030, 0x0039), (0xFF10, 0xFF19)],
}


LANGUAGE_ALIASES = {
    "CHINESE": "zh",
    "ENGLISH": "en",
    "JAPANESE": "ja",
    "KOREAN": "ko",
    "SPANISH": "es",
    "FRENCH": "fr",
    "GERMAN": "de",
    "PORTUGUESE": "pt",
    "RUSSIAN": "ru",
    "ARABIC": "ar",
}


QUESTION_NOISE = {
    "请问",
    "介绍一下",
    "说明一下",
    "解释一下",
    "告诉我",
    "这个",
    "那个",
    "一下",
    "是谁",
    "是什么",
    "什么是",
    "为什么",
    "怎么",
    "如何",
    "多少",
    "多少岁",
    "几岁",
    "多大",
    "有哪些",
    "哪一些",
    "哪些",
    "哪里",
    "哪儿",
    "什么地方",
    "什么地点",
    "最好的朋友",
    "朋友",
    "关系",
    "的",
    "了",
    "吗",
    "呢",
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "about",
    "tell",
    "me",
}

RELATION_QUESTION_NOISE = {
    "认识哪些人",
    "认识谁",
    "认识的人",
    "和谁来往",
    "跟谁来往",
    "和谁互动",
    "跟谁互动",
    "有哪些朋友",
    "朋友有哪些",
    "有哪些家人",
    "家人有哪些",
    "有哪些亲人",
    "亲人有哪些",
    "和谁发生过冲突",
    "跟谁发生过冲突",
    "冲突对象",
    "人物关系",
    "主要人物关系",
    "认识",
    "来往",
    "互动",
    "朋友",
    "家人",
    "亲人",
    "冲突",
    "同学",
    "伙伴",
    "哪些人",
    "谁",
    "人",
    "whom",
    "who",
    "interacts",
    "interact",
    "associate",
    "associates",
    "with",
    "friends",
    "family",
    "relatives",
    "companions",
    "conflicts",
    "conflict",
    "relationships",
}


def normalize_for_exact_match(text: str, language: str | None = None) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = _normalize_punctuation(value)
    value = _strip_accents(value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_for_lexical_search(text: str, language: str | None = None) -> str:
    value = normalize_for_exact_match(text, language=language)
    # Keep code/path/version symbols, but normalize human punctuation to spaces.
    value = re.sub(r"[，。！？；：、,!?;:()\[\]{}<>《》「」『』“”\"']", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_entity_name(text: str) -> str:
    value = normalize_for_exact_match(text)
    # Entity keys remove separators but keep code symbols such as C++, C#, .NET and v1.2.0.
    value = re.sub(r"[\s·・•\-—–]+", "", value)
    value = re.sub(r"[，。！？；：、,!?;:()\[\]{}<>《》「」『』“”\"']", "", value)
    return value


def detect_script_distribution(text: str) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for char in text or "":
        if char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        script = script_of_char(char)
        if script == "Other":
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if total <= 0:
        return {}
    return {key: round(value / total, 4) for key, value in sorted(counts.items())}


def script_of_char(char: str) -> str:
    codepoint = ord(char)
    for script, ranges in SCRIPT_RANGES.items():
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return "Other"


def detect_language(text: str) -> LanguageProfile:
    distribution = detect_script_distribution(text)
    if not distribution:
        return LanguageProfile(language="unknown", confidence=0.0, script_distribution={})

    major_script, major_ratio = max(distribution.items(), key=lambda item: item[1])
    non_digit_scripts = {key: value for key, value in distribution.items() if key != "Digit"}
    mixed = len([value for value in non_digit_scripts.values() if value >= 0.18]) >= 2

    if "Han" in distribution and ("Hiragana" in distribution or "Katakana" in distribution):
        confidence = distribution.get("Han", 0) + distribution.get("Hiragana", 0) + distribution.get("Katakana", 0)
        return LanguageProfile("ja", min(0.95, confidence), distribution, mixed)
    if "Hangul" in distribution:
        return LanguageProfile("ko", distribution["Hangul"], distribution, mixed)
    if "Han" in distribution and not mixed:
        return LanguageProfile("zh", distribution["Han"], distribution, mixed)
    if "Arabic" in distribution and not mixed:
        return LanguageProfile("ar", distribution["Arabic"], distribution, mixed)
    if "Cyrillic" in distribution and not mixed:
        return LanguageProfile("ru", distribution["Cyrillic"], distribution, mixed)
    if mixed:
        return LanguageProfile("mixed", major_ratio, distribution, True)
    if major_script == "Latin":
        language, confidence = detect_latin_language(text)
        return LanguageProfile(language, confidence, distribution, False)
    return LanguageProfile("unknown", major_ratio, distribution, mixed)


@lru_cache(maxsize=1)
def _get_lingua_detector():
    try:
        from lingua import LanguageDetectorBuilder

        return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    except Exception:
        return None


def detect_latin_language(text: str) -> tuple[str, float]:
    detector = _get_lingua_detector()
    if detector is None:
        return "en", 0.45
    try:
        confidence_values = detector.compute_language_confidence_values(text or "")
        if not confidence_values:
            return "unknown", 0.0
        best = confidence_values[0]
        language_name = getattr(best.language, "name", str(best.language)).upper()
        return LANGUAGE_ALIASES.get(language_name, language_name.lower()[:2]), float(best.value)
    except Exception:
        return "unknown", 0.0


def tokenize_for_search(text: str, language: str | None = None) -> List[str]:
    normalized = normalize_for_lexical_search(text, language=language)
    tokens: List[str] = []
    tokens.extend(re.findall(r"[a-z0-9_+#./-]{2,}", normalized))

    cjk_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", normalized)
    for size in (2, 3):
        tokens.extend("".join(cjk_chars[index : index + size]) for index in range(len(cjk_chars) - size + 1))

    symbols = re.findall(
        r"\.[A-Za-z][A-Za-z0-9_+#-]*|[A-Za-z_][A-Za-z0-9_]*\(\)|[A-Za-z]:\\[^\s]+|/[^\s]+|v\d+(?:\.\d+)+|C\+\+|C#",
        text or "",
    )
    tokens.extend(item.casefold() for item in symbols)
    return [token for token in dict.fromkeys(tokens) if token and token not in _stopwords()]


def query_focus_terms(text: str) -> List[str]:
    raw = text or ""
    terms: List[str] = []

    relation_subject = extract_relation_subject(raw)
    if relation_subject:
        normalized_relation_subject = (
            normalize_for_lexical_search(relation_subject)
            if re.search(r"[A-Za-z]\s+[A-Za-z]", relation_subject)
            else normalize_entity_name(relation_subject)
        )
        if len(normalized_relation_subject) >= 2:
            terms.append(normalized_relation_subject)

    frame_patterns = [
        r"^\s*(?:请问)?(.+?)(?:认识哪些人|认识谁|认识的人|和谁来往|跟谁来往|和谁互动|跟谁互动|有哪些朋友|朋友有哪些|有哪些家人|家人有哪些|有哪些亲人|亲人有哪些|和谁发生过冲突|跟谁发生过冲突|人物关系)[?？。！!]*\s*$",
        r"(?i)^\s*(?:who\s+does\s+)?(.+?)\s+(?:interact|interacts|associate|associates)\s+with(?:\s+whom|\s+who)?[?!.]*\s*$",
        r"(?i)^\s*(?:who|what|where)\s+(?:is|are|was|were)\s+(.+?)[?？。！!]*\s*$",
        r"^\s*(?:请问)?(.+?)(?:的?最好的朋友|的?朋友|和谁|与谁|关系)(?:是谁|是什么|有哪些)?[?？。！!]*\s*$",
        r"^\s*(?:请问|介绍一下|说明一下|解释一下)?(.+?)(?:是谁|是什么|是什么角色|是什么人物|多少岁|几岁|多大|有哪些|去过哪里|去过哪些地方|在哪里|在哪儿)[?？。！!]*\s*$",
    ]
    for pattern in frame_patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        captured = match.group(1).strip(" \t\r\n\"'“”‘’")
        normalized = normalize_for_lexical_search(captured) if re.search(r"[A-Za-z]\s+[A-Za-z]", captured) else normalize_entity_name(captured)
        if len(normalized) >= 2 and normalized not in _stopwords():
            terms.append(normalized)

    compact = normalize_entity_name(raw)
    candidate = compact
    for item in QUESTION_NOISE:
        candidate = candidate.replace(normalize_entity_name(item), "")
    if len(candidate) >= 2 and re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", candidate):
        terms.append(candidate)

    latin_tokens: List[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_+#.-]{1,}", raw):
        normalized = normalize_entity_name(token)
        for item in QUESTION_NOISE:
            normalized = normalized.replace(normalize_entity_name(item), "")
        if not normalized or normalized in _stopwords():
            continue
        if re.fullmatch(r"[a-z][a-z0-9_+#.-]*", normalized):
            latin_tokens.append(normalized)
        if len(normalized) >= 2:
            terms.append(normalized)

    if len(latin_tokens) >= 2:
        phrase = " ".join(latin_tokens[:4])
        terms.insert(0, phrase)

    unique_terms = list(dict.fromkeys(item for item in terms if item))
    latin_phrases = [item for item in unique_terms if " " in item and re.fullmatch(r"[a-z0-9_+#./ -]+", item)]
    if latin_phrases:
        phrase_parts = {part for phrase in latin_phrases for part in phrase.split()}
        unique_terms = [
            item
            for item in unique_terms
            if not (item in phrase_parts and re.fullmatch(r"[a-z][a-z0-9_+#.-]*", item))
        ]
    return unique_terms[:5]


def extract_relation_subject(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"[?？。！!]+$", "", raw).strip()
    patterns = [
        r"^\s*(?:请问)?(.+?)(?:认识哪些人|认识谁|认识的人|和谁来往|跟谁来往|和谁互动|跟谁互动|有哪些朋友|朋友有哪些|有哪些家人|家人有哪些|有哪些亲人|亲人有哪些|和谁发生过冲突|跟谁发生过冲突|的主要人物关系|人物关系)\s*$",
        r"^\s*(?:who\s+does\s+)?(.+?)\s+(?:interact|interacts|associate|associates)\s+with(?:\s+whom|\s+who)?\s*$",
        r"^\s*(.+?)(?:'s|’s)?\s+(?:friends|family|relatives|companions|conflicts|relationships)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.I)
        if match:
            return _strip_relation_noise(match.group(1))
    return ""


def _strip_relation_noise(text: str) -> str:
    candidate = (text or "").strip(" \t\r\n\"'“”‘’的")
    for item in QUESTION_NOISE | RELATION_QUESTION_NOISE:
        normalized = normalize_entity_name(item)
        if normalized:
            candidate = candidate.replace(item, "")
    return candidate.strip(" \t\r\n\"'“”‘’的")


def surface_variants(text: str) -> List[str]:
    variants = [text]
    compact = normalize_entity_name(text)
    if compact and compact != text:
        variants.append(compact)
    for term in re.findall(r"[\u4e00-\u9fffA-Za-z]+[·・•\s]+[\u4e00-\u9fffA-Za-z·・•\s]+", text or ""):
        compact_term = normalize_entity_name(term)
        if compact_term:
            variants.append(compact_term)
    return list(dict.fromkeys(item for item in variants if item.strip()))


def language_name(language: str) -> str:
    return {
        "zh": "中文",
        "en": "English",
        "ja": "日本語",
        "ko": "한국어",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
        "pt": "Português",
        "ru": "Русский",
        "ar": "العربية",
        "mixed": "混合语言",
        "unknown": "未知语言",
    }.get(language, language)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _normalize_punctuation(text: str) -> str:
    return (
        (text or "")
        .replace("·", " ")
        .replace("・", " ")
        .replace("•", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("－", "-")
    )


def _stopwords() -> set[str]:
    return QUESTION_NOISE | {
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "a",
        "an",
    }


def best_language(profiles: Sequence[LanguageProfile]) -> LanguageProfile:
    if not profiles:
        return LanguageProfile()
    merged_scripts: Dict[str, float] = {}
    language_counts: Dict[str, float] = {}
    for profile in profiles:
        language_counts[profile.language] = language_counts.get(profile.language, 0.0) + max(profile.confidence, 0.1)
        for script, ratio in profile.script_distribution.items():
            merged_scripts[script] = merged_scripts.get(script, 0.0) + ratio
    language = max(language_counts.items(), key=lambda item: item[1])[0]
    total = sum(merged_scripts.values()) or 1.0
    distribution = {key: round(value / total, 4) for key, value in merged_scripts.items()}
    return LanguageProfile(
        language=language,
        confidence=min(1.0, language_counts[language] / max(len(profiles), 1)),
        script_distribution=distribution,
        is_mixed=language == "mixed",
    )
