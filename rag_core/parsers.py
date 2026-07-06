from __future__ import annotations

import json
import logging
import platform
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd

from rag_core.chunking import normalize_text
from rag_core.multilingual import detect_language


logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv"}
OCR_MIN_CONFIDENCE = 0.45
_OCR_ENGINE_CACHE: Dict[Tuple[str, int, int], Any] = {}
_OCR_IMPORT_ERROR: Optional[Exception] = None


def parse_file(
    path: Path,
    original_name: str,
    enable_pdf_ocr: bool = True,
    pdf_ocr_dpi: int = 180,
    pdf_ocr_min_text_chars: int = 80,
    pdf_ocr_device: str = "cpu",
    pdf_ocr_threads: int = -1,
    pdf_ocr_max_side_len: int = 1600,
) -> List[Dict[str, object]]:
    """按文件类型解析出结构化文本。"""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(path, original_name)
    if suffix == ".pdf":
        return parse_pdf(
            path,
            original_name,
            enable_ocr=enable_pdf_ocr,
            ocr_dpi=pdf_ocr_dpi,
            ocr_min_text_chars=pdf_ocr_min_text_chars,
            ocr_device=pdf_ocr_device,
            ocr_threads=pdf_ocr_threads,
            ocr_max_side_len=pdf_ocr_max_side_len,
        )
    if suffix == ".csv":
        return parse_csv(path, original_name)
    raise ValueError(f"不支持的文件类型：{suffix}")


def parse_txt(path: Path, original_name: str) -> List[Dict[str, object]]:
    """TXT 文件采用多编码兜底读取。"""

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            text = normalize_text(text.strip())
            if not text:
                return []
            return [
                {
                    "text": text,
                    "metadata": {"source_type": "txt", "source_ref": original_name},
                }
            ]
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"TXT 文件读取失败：{last_error}")


def parse_pdf(
    path: Path,
    original_name: str,
    enable_ocr: bool = True,
    ocr_dpi: int = 180,
    ocr_min_text_chars: int = 80,
    ocr_device: str = "cpu",
    ocr_threads: int = -1,
    ocr_max_side_len: int = 1600,
) -> List[Dict[str, object]]:
    """解析 PDF。

    策略：
    1. 优先使用 PyMuPDF 提取原生可复制文本，保留页码和块级结构。
    2. 如果某页原生文本过少，自动对该页渲染图片并执行 OCR。
    3. OCR 只作为兜底，不污染普通 PDF 的原生版面结构。
    """

    documents: List[Dict[str, object]] = []
    text_page_count = 0
    ocr_page_count = 0
    low_text_pages = 0
    ocr_errors: List[str] = []

    try:
        with fitz.open(path) as pdf:
            total_pages = len(pdf)
            for page_index in range(total_pages):
                page_number = page_index + 1
                page = pdf[page_index]
                page_documents = extract_pdf_page_documents(page, original_name, page_number, total_pages)
                page_text_chars = page_document_char_count(page_documents)

                if page_documents and page_text_chars >= ocr_min_text_chars:
                    text_page_count += 1
                    documents.extend(page_documents)
                    continue

                if page_documents:
                    low_text_pages += 1

                ocr_document = None
                if enable_ocr:
                    try:
                        ocr_document = extract_pdf_page_ocr_document(
                            page=page,
                            original_name=original_name,
                            page_number=page_number,
                            total_pages=total_pages,
                            dpi=ocr_dpi,
                            device=ocr_device,
                            threads=ocr_threads,
                            max_side_len=ocr_max_side_len,
                        )
                    except Exception as exc:
                        message = f"第 {page_number} 页 OCR 失败：{exc}"
                        ocr_errors.append(message)
                        logger.warning(message)

                if ocr_document:
                    ocr_text_chars = len(re.sub(r"\s+", "", str(ocr_document.get("text", ""))))
                    if ocr_text_chars > page_text_chars:
                        ocr_page_count += 1
                        documents.append(ocr_document)
                        continue

                if page_documents:
                    text_page_count += 1
                    documents.extend(page_documents)
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"PDF 打开或解析失败：{exc}") from exc

    if documents:
        logger.info(
            "PDF 解析完成：%s，原生文本页=%s，OCR页=%s，低文本页=%s，OCR错误=%s",
            original_name,
            text_page_count,
            ocr_page_count,
            low_text_pages,
            len(ocr_errors),
        )
        return documents

    if enable_ocr and ocr_errors:
        raise RuntimeError(
            "PDF 未提取到可入库文本，且 OCR 执行失败。"
            f"首个错误：{ocr_errors[0]}。"
            "请确认已安装 rapidocr-onnxruntime，并重新运行：pip install -r requirements.txt"
        )
    if enable_ocr:
        raise RuntimeError("PDF 已执行 OCR，但仍未识别到可入库文本。请确认文件不是空白页或严重模糊扫描件。")
    raise RuntimeError("PDF 未提取到可入库文本。该文件可能是扫描版 PDF，请在设置中启用 PDF OCR 后重新导入。")


def extract_pdf_page_documents(
    page: fitz.Page,
    original_name: str,
    page_number: int,
    total_pages: int,
) -> List[Dict[str, object]]:
    """提取单页原生文本块，并按常见阅读顺序排序。"""

    try:
        raw = page.get_text("dict", sort=False)
    except Exception as exc:
        logger.warning("提取 PDF 第 %s 页文本失败：%s", page_number, exc)
        return []

    blocks = []
    for block_index, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:
            continue
        text = normalize_text(extract_text_from_block(block))
        if not is_useful_pdf_text(text):
            continue
        x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
        if is_probable_header_footer(y0, y1, page.rect.height, page_number, total_pages, text):
            continue
        blocks.append(
            {
                "text": text,
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "block_index": block_index,
            }
        )

    ordered_blocks = order_pdf_blocks(blocks, page.rect.width)
    documents: List[Dict[str, object]] = []
    for local_index, block in enumerate(ordered_blocks):
        documents.append(
            {
                "text": f"第 {page_number} 页\n\n{block['text']}",
                "metadata": {
                    "source_type": "pdf",
                    "source_ref": original_name,
                    "page": page_number,
                    "page_count": total_pages,
                    "block_index": int(block["block_index"]),
                    "page_block_index": local_index,
                    "ocr": False,
                    "text_source": "native",
                    "page_language": detect_language(block["text"]).language,
                    "page_language_confidence": detect_language(block["text"]).confidence,
                    "script_distribution": detect_language(block["text"]).script_distribution,
                    "bbox": json.dumps(block["bbox"], ensure_ascii=False),
                },
            }
        )
    return documents


def extract_pdf_page_ocr_document(
    page: fitz.Page,
    original_name: str,
    page_number: int,
    total_pages: int,
    dpi: int,
    device: str = "cpu",
    threads: int = -1,
    max_side_len: int = 1600,
) -> Optional[Dict[str, object]]:
    """将 PDF 页面渲染为图片后执行 OCR，返回单页 OCR 文档。"""

    text = run_ocr_on_pdf_page(page, dpi=dpi, device=device, threads=threads, max_side_len=max_side_len)
    text = normalize_text(text)
    if not is_useful_pdf_text(text):
        return None

    return {
        "text": f"第 {page_number} 页（OCR 识别）\n\n{text}",
        "metadata": {
            "source_type": "pdf_ocr",
            "source_ref": original_name,
            "page": page_number,
            "page_count": total_pages,
            "block_index": -1,
            "page_block_index": 0,
            "ocr": True,
            "text_source": "ocr",
            "page_language": detect_language(text).language,
            "page_language_confidence": detect_language(text).confidence,
            "script_distribution": detect_language(text).script_distribution,
            "ocr_dpi": int(dpi),
            "ocr_device": normalize_ocr_device(device),
            "ocr_max_side_len": int(max_side_len),
        },
    }


def run_ocr_on_pdf_page(
    page: fitz.Page,
    dpi: int = 180,
    device: str = "cpu",
    threads: int = -1,
    max_side_len: int = 1600,
) -> str:
    """调用 RapidOCR 识别单页图片文字。"""

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("当前环境缺少 numpy，无法执行 OCR。请重新安装 requirements.txt。") from exc

    engine = get_ocr_engine(device=device, threads=threads, max_side_len=max_side_len)
    scale = max(100, min(int(dpi), 300)) / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        image = image[:, :, :3]

    raw_result = engine(image)
    lines = normalize_rapidocr_result(raw_result)
    return "\n".join(lines)


def get_ocr_engine(device: str = "cpu", threads: int = -1, max_side_len: int = 1600):
    """懒加载 OCR 引擎，避免普通 PDF 导入时增加启动成本。"""

    global _OCR_IMPORT_ERROR
    requested_device = normalize_ocr_device(device)
    resolved_device = resolve_ocr_device(device)
    normalized_threads = normalize_ocr_threads(threads)
    normalized_max_side_len = normalize_ocr_max_side_len(max_side_len)
    cache_key = (resolved_device, normalized_threads, normalized_max_side_len)
    if cache_key in _OCR_ENGINE_CACHE:
        return _OCR_ENGINE_CACHE[cache_key]
    if _OCR_IMPORT_ERROR is not None:
        raise RuntimeError(
            "OCR 依赖尚未安装或加载失败。请先运行：pip install -r requirements.txt"
        ) from _OCR_IMPORT_ERROR

    try:
        from rapidocr_onnxruntime import RapidOCR

        engine_kwargs = build_ocr_engine_kwargs(resolved_device, normalized_threads, normalized_max_side_len)
        engine = RapidOCR(**engine_kwargs)
        actual_providers = get_rapidocr_actual_providers(engine)
        expected_provider = expected_provider_for_ocr_device(resolved_device)
        if expected_provider and requested_device in {"cuda", "directml"} and expected_provider not in actual_providers:
            raise RuntimeError(
                f"你选择了 OCR {requested_device.upper()} 加速，但实际 ONNX Runtime 会话没有使用 {expected_provider}，"
                f"当前实际 Provider={actual_providers or ['unknown']}。"
                "为避免继续 CPU 满载，已停止 OCR。"
                "如果使用 CUDA，请安装匹配 onnxruntime-gpu 的 CUDA 12.x、cuDNN 9.x 和最新版 MSVC Runtime，并确保在 PATH 中；"
                "如果只是 Windows 本地加速，建议改用 DirectML。"
            )
        if expected_provider and expected_provider not in actual_providers:
            logger.warning(
                "OCR 请求设备 %s 但实际 Provider=%s，已自动回落。若不希望回落，请显式选择 CUDA 或 DirectML。",
                resolved_device,
                actual_providers,
            )
        _OCR_ENGINE_CACHE[cache_key] = engine
        logger.info(
            "OCR 引擎已加载：requested=%s, resolved=%s, actual_providers=%s, threads=%s, max_side_len=%s",
            requested_device,
            resolved_device,
            actual_providers,
            normalized_threads,
            normalized_max_side_len,
        )
        return _OCR_ENGINE_CACHE[cache_key]
    except RuntimeError:
        raise
    except Exception as exc:
        _OCR_IMPORT_ERROR = exc
        raise RuntimeError(
            "OCR 依赖尚未安装或加载失败。请先运行：pip install -r requirements.txt"
        ) from exc


def expected_provider_for_ocr_device(device: str) -> str:
    if device == "cuda":
        return "CUDAExecutionProvider"
    if device == "directml":
        return "DmlExecutionProvider"
    return ""


def get_rapidocr_actual_providers(engine: Any) -> List[str]:
    """读取 RapidOCR 三个 ONNX session 的真实 Provider，识别 CUDA/DirectML 是否实际生效。"""

    providers: List[str] = []
    session_holders = [
        getattr(getattr(engine, "text_det", None), "infer", None),
        getattr(getattr(engine, "text_cls", None), "infer", None),
        getattr(getattr(engine, "text_rec", None), "session", None),
    ]
    for holder in session_holders:
        session = getattr(holder, "session", None)
        if session is None or not hasattr(session, "get_providers"):
            continue
        try:
            for provider in session.get_providers():
                if provider not in providers:
                    providers.append(provider)
        except Exception:
            continue
    return providers


def build_ocr_engine_kwargs(device: str, threads: int, max_side_len: int) -> Dict[str, Any]:
    """构建 RapidOCR 参数，分别控制检测、方向分类和文字识别三个 ONNX 会话。"""

    use_cuda = device == "cuda"
    use_dml = device == "directml"
    kwargs: Dict[str, Any] = {
        "det_use_cuda": use_cuda,
        "cls_use_cuda": use_cuda,
        "rec_use_cuda": use_cuda,
        "det_use_dml": use_dml,
        "cls_use_dml": use_dml,
        "rec_use_dml": use_dml,
        "max_side_len": max_side_len,
    }
    if threads > 0:
        kwargs["intra_op_num_threads"] = threads
        kwargs["inter_op_num_threads"] = 1
    return kwargs


def normalize_ocr_device(device: str) -> str:
    value = str(device or "cpu").strip().lower()
    aliases = {
        "gpu": "auto",
        "dml": "directml",
        "direct-ml": "directml",
        "direct_ml": "directml",
    }
    return aliases.get(value, value if value in {"cpu", "auto", "cuda", "directml"} else "cpu")


def resolve_ocr_device(device: str) -> str:
    requested = normalize_ocr_device(device)
    providers = set(get_available_ocr_providers())
    if requested == "auto":
        if "CUDAExecutionProvider" in providers:
            return "cuda"
        if "DmlExecutionProvider" in providers:
            return "directml"
        return "cpu"
    if requested == "cuda" and "CUDAExecutionProvider" not in providers:
        logger.warning("CUDAExecutionProvider 不可用，OCR 自动回落到 CPU。当前 providers=%s", sorted(providers))
        return "cpu"
    if requested == "directml" and "DmlExecutionProvider" not in providers:
        logger.warning("DmlExecutionProvider 不可用，OCR 自动回落到 CPU。当前 providers=%s", sorted(providers))
        return "cpu"
    return requested


def normalize_ocr_threads(threads: int) -> int:
    try:
        value = int(threads)
    except Exception:
        return -1
    if value < 1:
        return -1
    return min(value, 32)


def normalize_ocr_max_side_len(max_side_len: int) -> int:
    try:
        value = int(max_side_len)
    except Exception:
        return 1600
    return max(960, min(value, 3000))


def get_available_ocr_providers() -> List[str]:
    try:
        from onnxruntime import get_available_providers

        return list(get_available_providers())
    except Exception:
        return []


def get_ocr_runtime_info() -> Dict[str, Any]:
    providers = get_available_ocr_providers()
    device = "unknown"
    try:
        from onnxruntime import get_device

        device = str(get_device())
    except Exception:
        pass
    return {
        "platform": platform.system(),
        "onnx_device": device,
        "available_providers": providers,
        "cuda_ready": "CUDAExecutionProvider" in providers,
        "directml_ready": "DmlExecutionProvider" in providers,
    }


def normalize_rapidocr_result(raw_result: Any) -> List[str]:
    """兼容 RapidOCR 常见返回格式，提取可信文本行。"""

    if raw_result is None:
        return []

    result = raw_result
    if isinstance(raw_result, tuple) and raw_result:
        result = raw_result[0]
    if not result:
        return []

    lines: List[str] = []
    for item in result:
        text = ""
        score = 1.0
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("rec_text") or "")
            score = safe_float(item.get("score") or item.get("confidence"), 1.0)
        elif isinstance(item, (list, tuple)):
            if len(item) >= 3:
                text = str(item[1] or "")
                score = safe_float(item[2], 1.0)
            elif len(item) >= 2:
                text = str(item[1] or "")
        if text.strip() and score >= OCR_MIN_CONFIDENCE:
            lines.append(text.strip())
    return lines


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def page_document_char_count(documents: List[Dict[str, object]]) -> int:
    return sum(len(re.sub(r"\s+", "", str(document.get("text", "")))) for document in documents)


def extract_text_from_block(block: Dict[str, object]) -> str:
    lines: List[Tuple[float, str]] = []
    for line in block.get("lines", []):
        spans = []
        for span in line.get("spans", []):
            span_text = str(span.get("text") or "")
            if span_text.strip():
                spans.append(span_text)
        if not spans:
            continue
        y0 = float((line.get("bbox") or [0, 0, 0, 0])[1])
        lines.append((y0, "".join(spans)))
    lines.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in lines)


def order_pdf_blocks(blocks: List[Dict[str, object]], page_width: float) -> List[Dict[str, object]]:
    """兼容常见单栏/双栏 PDF 的阅读顺序排序。"""

    if not blocks:
        return []

    centers = [((block["bbox"][0] + block["bbox"][2]) / 2.0) for block in blocks]
    left_count = sum(1 for center in centers if center < page_width * 0.48)
    right_count = sum(1 for center in centers if center > page_width * 0.52)
    looks_like_two_columns = left_count >= 2 and right_count >= 2

    if looks_like_two_columns:

        def column_key(block: Dict[str, object]) -> Tuple[int, float, float]:
            x0, y0, x1, _ = block["bbox"]
            center = (x0 + x1) / 2.0
            column = 0 if center < page_width / 2.0 else 1
            return column, y0, x0

        return sorted(blocks, key=column_key)

    return sorted(blocks, key=lambda block: (block["bbox"][1], block["bbox"][0]))


def is_probable_header_footer(
    y0: float,
    y1: float,
    page_height: float,
    page_number: int,
    total_pages: int,
    text: str,
) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if y1 < page_height * 0.06 or y0 > page_height * 0.94:
        page_marks = {str(page_number), f"{page_number}/{total_pages}", f"-{page_number}-"}
        if len(compact) <= 30 or compact in page_marks:
            return True
    return False


def is_useful_pdf_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 8:
        return False
    alpha_num_cjk = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", compact)
    return len(alpha_num_cjk) / max(len(compact), 1) >= 0.35


def parse_csv(path: Path, original_name: str) -> List[Dict[str, object]]:
    """把 CSV 逐行转成可检索文本，兼容 JSON 和 Markdown 风格。"""

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"]
    df = None
    last_error = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc
            continue
    if df is None:
        raise RuntimeError(f"CSV 文件读取失败：{last_error}")

    documents: List[Dict[str, object]] = []
    columns = [str(col) for col in df.columns.tolist()]
    header_text = " | ".join(columns)
    for idx, row in df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                value = ""
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                value = value
            else:
                value = str(value)
            row_dict[str(col)] = value

        row_json = json.dumps(row_dict, ensure_ascii=False)
        row_md = " | ".join(f"{key}: {value}" for key, value in row_dict.items())
        text = (
            f"CSV 文件：{original_name}\n"
            f"列名：{header_text}\n"
            f"第 {idx + 1} 行：{row_json}\n"
            f"Markdown 视图：{row_md}"
        )
        documents.append(
            {
                "text": text,
                "metadata": {
                    "source_type": "csv",
                    "source_ref": original_name,
                    "row_index": int(idx) + 1,
                },
            }
        )
    return documents
