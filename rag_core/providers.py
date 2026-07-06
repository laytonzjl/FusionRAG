from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Dict, Generator, List, Sequence, Tuple

from openai import OpenAI

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None

from config import ApiConfig


logger = logging.getLogger(__name__)

LOCAL_EMBEDDING_DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class MultiProviderClient:
    """统一封装聊天模型与向量模型调用。"""

    def __init__(self, api_config: ApiConfig) -> None:
        self.api_config = api_config
        self.chat_provider = (api_config.chat_provider or "openai").strip().lower()
        self.embedding_provider = (api_config.embedding_provider or "openai").strip().lower()
        self.embedding_client = None
        self.local_embedding_model = None
        self.local_embedding_backend = ""
        self.local_hashing_dimension = 1024

        if not api_config.chat_api_key:
            raise ValueError("未检测到聊天 API Key，请先在设置中填写并保存。")

        if self.embedding_provider == "local":
            self._init_local_embedding_model()
        else:
            if not api_config.embedding_api_key:
                raise ValueError("未检测到向量 API Key。若不想配置向量 API，请将向量服务设为本地向量。")
            self.embedding_client = OpenAI(
                api_key=api_config.embedding_api_key,
                base_url=api_config.embedding_api_base or "https://api.openai.com/v1",
            )

        if self.chat_provider == "claude":
            if Anthropic is None:
                raise RuntimeError("当前环境缺少 `anthropic` 依赖，无法调用 Claude。")
            self.chat_client = Anthropic(
                api_key=api_config.chat_api_key,
                base_url=api_config.chat_api_base or "https://api.anthropic.com",
            )
        else:
            self.chat_client = OpenAI(
                api_key=api_config.chat_api_key,
                base_url=api_config.chat_api_base or "https://api.openai.com/v1",
            )

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
        """批量生成向量，避免一次请求过大。"""

        if self.embedding_provider == "local":
            return self.embed_documents(texts=texts, batch_size=batch_size)

        return self._embed_texts_remote(texts=texts, batch_size=batch_size)

    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
        """为入库文档生成向量。"""

        if self.embedding_provider == "local":
            return self._embed_texts_local(texts=texts, batch_size=batch_size, mode="document")
        return self._embed_texts_remote(texts=texts, batch_size=batch_size)

    def embed_query(self, query: str) -> List[float]:
        """为用户查询生成向量。"""

        if self.embedding_provider == "local":
            vectors = self._embed_texts_local(texts=[query], batch_size=1, mode="query")
            return vectors[0] if vectors else []
        vectors = self._embed_texts_remote(texts=[query], batch_size=1)
        return vectors[0] if vectors else []

    def embed_queries(self, queries: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        """Batch query embeddings while preserving query-side model prefixes.

        ``embed_documents`` must not be reused here because E5-family models
        require a different query prefix.  This method is intentionally small
        and remains backward compatible with callers that still use
        :meth:`embed_query`.
        """

        cleaned = [str(query or "").strip() for query in queries if str(query or "").strip()]
        if not cleaned:
            return []
        if self.embedding_provider == "local":
            return self._embed_texts_local(texts=cleaned, batch_size=max(1, int(batch_size or 1)), mode="query")
        return self._embed_texts_remote(texts=cleaned, batch_size=max(1, int(batch_size or 1)))

    def _embed_texts_remote(self, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
        """调用兼容 OpenAI API 的远程向量服务。"""

        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [text for text in texts[start : start + batch_size] if text]
            if not batch:
                continue
            try:
                if self.embedding_client is None:
                    raise RuntimeError("Embedding client 未初始化。")
                response = self.embedding_client.embeddings.create(
                    model=self.api_config.embedding_model,
                    input=batch,
                )
            except Exception as exc:
                logger.exception("Embedding 请求失败: %s", exc)
                raise RuntimeError(
                    "向量请求失败。请检查向量服务商、接口地址、模型名称和 API Key 是否匹配。"
                    f"当前向量服务商：{self.embedding_provider}，"
                    f"接口地址：{self.api_config.embedding_api_base}，模型：{self.api_config.embedding_model}。"
                    f"原始错误：{exc}"
                ) from exc
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def _init_local_embedding_model(self) -> None:
        """初始化本地向量模型，可选择离线 Hashing 或 sentence-transformers。"""

        model_name = (self.api_config.embedding_model or LOCAL_EMBEDDING_DEFAULT_MODEL).strip()
        if self._is_hashing_embedding_model(model_name):
            self.local_embedding_backend = "hashing"
            self.local_hashing_dimension = self._parse_hashing_dimension(model_name)
            logger.info("使用离线 Hashing Embedding，维度: %s", self.local_hashing_dimension)
            return

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "当前环境缺少 `sentence-transformers`，无法使用真实本地语义向量。"
                "请先运行：pip install -r requirements.txt。"
                "如果只是临时离线演示，可手动把向量模型名改为 local-hashing-1024。"
            ) from exc

        try:
            self.local_embedding_model = self._load_sentence_transformer(SentenceTransformer, model_name)
            self.local_embedding_backend = "sentence_transformers"
        except Exception as exc:
            logger.exception("本地 Embedding 模型加载失败: %s", exc)
            raise RuntimeError(
                "本地 Embedding 模型加载失败。系统默认优先离线加载本机缓存，避免启动时卡在网络重试。"
                "请先联网运行一次模型下载，或把向量模型名改成本地模型目录。"
                "临时离线演示才建议使用 local-hashing-1024。"
                f"当前模型：{model_name}。原始错误：{exc}"
            ) from exc

    def _load_sentence_transformer(self, model_cls, model_name: str):
        if Path(model_name).exists():
            return model_cls(model_name)
        try:
            return model_cls(model_name, local_files_only=True)
        except TypeError:
            return model_cls(model_name)

    def _embed_texts_local(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        mode: str = "document",
    ) -> List[List[float]]:
        """使用本地模型生成向量，不请求任何 Embedding API。"""

        clean_texts = [text for text in texts if text]
        if self.local_embedding_backend == "hashing":
            return [self._hash_text_to_embedding(text) for text in clean_texts]

        if self.local_embedding_model is None:
            raise RuntimeError("本地 Embedding 模型未初始化。")

        embeddings: List[List[float]] = []
        model_name = (self.api_config.embedding_model or "").lower()
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            if not batch:
                continue
            encoded_batch = self._prepare_local_embedding_texts(batch, model_name=model_name, mode=mode)
            try:
                vectors = self.local_embedding_model.encode(
                    encoded_batch,
                    batch_size=min(batch_size, 64),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                logger.exception("本地 Embedding 生成失败: %s", exc)
                raise RuntimeError(f"本地 Embedding 生成失败：{exc}") from exc
            if hasattr(vectors, "tolist"):
                embeddings.extend(vectors.tolist())
            else:
                embeddings.extend([list(vector) for vector in vectors])
        return embeddings

    def _prepare_local_embedding_texts(self, texts: Sequence[str], model_name: str, mode: str) -> List[str]:
        """根据模型族补充推荐前缀，提升通用语义召回质量。"""

        if "e5" in model_name:
            prefix = "query: " if mode == "query" else "passage: "
            return [prefix + text for text in texts]
        if "bge" in model_name and mode == "query":
            return [text for text in texts]
        return list(texts)

    def _is_hashing_embedding_model(self, model_name: str) -> bool:
        normalized = (model_name or "").strip().lower()
        return normalized in {"hashing", "local-hashing"} or normalized.startswith("local-hashing-")

    def _parse_hashing_dimension(self, model_name: str) -> int:
        match = re.search(r"(\d+)$", model_name or "")
        if not match:
            return 1024
        return max(128, min(int(match.group(1)), 8192))

    def _hash_text_to_embedding(self, text: str) -> List[float]:
        """离线 Hashing Embedding：无需模型文件，适合本地演示和轻量部署。"""

        dimension = self.local_hashing_dimension
        vector = [0.0] * dimension
        tokens = self._tokenize_for_hashing(text)
        if not tokens:
            return self._fallback_hash_embedding(text, dimension)

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="little", signed=False)
            index = value % dimension
            sign = 1.0 if value & (1 << 63) else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return self._fallback_hash_embedding(text, dimension)
        return [value / norm for value in vector]

    def _fallback_hash_embedding(self, text: str, dimension: int) -> List[float]:
        """为纯符号/公式片段生成非零向量，避免 Chroma cosine 索引处理零向量。"""

        seed = (text or "__empty_text__").encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(seed, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        vector = [0.0] * dimension
        vector[value % dimension] = 1.0
        return vector

    def _tokenize_for_hashing(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        word_tokens = re.findall(r"[a-z0-9_]+", lowered)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", lowered)

        tokens = list(word_tokens)
        tokens.extend(chinese_chars)
        for size in (2, 3):
            tokens.extend("".join(chinese_chars[index : index + size]) for index in range(len(chinese_chars) - size + 1))

        if len(word_tokens) > 1:
            tokens.extend(f"{word_tokens[index]}_{word_tokens[index + 1]}" for index in range(len(word_tokens) - 1))
        return tokens

    def stream_chat(self, messages: Sequence[Dict[str, str]]) -> Generator[str, None, None]:
        """流式调用聊天模型。"""

        try:
            if self.chat_provider == "claude":
                system_prompt, chat_messages = self._split_anthropic_messages(messages)
                with self.chat_client.messages.stream(
                    model=self.api_config.chat_model,
                    max_tokens=self.api_config.max_output_tokens,
                    temperature=self.api_config.temperature,
                    system=system_prompt,
                    messages=chat_messages,
                ) as stream:
                    for text in stream.text_stream:
                        if text:
                            yield text
                return

            stream = self.chat_client.chat.completions.create(
                model=self.api_config.chat_model,
                messages=list(messages),
                temperature=self.api_config.temperature,
                max_tokens=self.api_config.max_output_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as exc:
            logger.exception("聊天模型调用失败: %s", exc)
            raise RuntimeError(f"聊天模型调用失败：{exc}") from exc

    def complete_chat(self, messages: Sequence[Dict[str, str]], max_tokens: int = 256) -> str:
        """非流式调用聊天模型，供查询改写等轻量内部任务使用。"""

        try:
            if self.chat_provider == "claude":
                system_prompt, chat_messages = self._split_anthropic_messages(messages)
                response = self.chat_client.messages.create(
                    model=self.api_config.chat_model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    system=system_prompt,
                    messages=chat_messages,
                )
                parts = []
                for block in getattr(response, "content", []) or []:
                    text = getattr(block, "text", "")
                    if text:
                        parts.append(text)
                return "\n".join(parts).strip()

            response = self.chat_client.chat.completions.create(
                model=self.api_config.chat_model,
                messages=list(messages),
                temperature=0.0,
                max_tokens=max_tokens,
                stream=False,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("内部查询改写调用失败，将回退到本地通用扩展: %s", exc)
            return ""

    def _split_anthropic_messages(
        self, messages: Sequence[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Anthropic SDK 需要单独的 system prompt。"""

        system_prompt = ""
        chat_messages: List[Dict[str, str]] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")
            if index == 0 and role == "system":
                system_prompt = content
                continue
            if role in {"user", "assistant"} and content:
                chat_messages.append({"role": role, "content": content})
        return system_prompt, chat_messages
