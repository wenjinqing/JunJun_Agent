"""Embedding 客户端：bge-m3 (SiliconFlow, 1024 维)。

对齐原 lpmm embedding 配置。SILICONFLOW_API_KEY 未设置时 available=False，
上层（长期记忆/知识库）自动降级关键词检索，不阻塞主循环。
"""

import asyncio
import os
from typing import List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.embedding")

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
_BASE_URL = "https://api.siliconflow.cn/v1"


class EmbeddingClient:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key if api_key is not None else os.environ.get("SILICONFLOW_API_KEY", "")
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=_BASE_URL, timeout=15)
        return self._client

    async def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量向量化。不可用/失败返回 None（上层降级）。"""
        if not self.available or not texts:
            return None
        try:
            resp = await self._get_client().embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            logger.warning(f"embedding 调用失败（降级关键词检索）: {e}")
            return None

    async def embed_one(self, text: str) -> Optional[List[float]]:
        out = await self.embed([text])
        return out[0] if out else None


_client: Optional[EmbeddingClient] = None


def get_embedding_client() -> EmbeddingClient:
    global _client
    if _client is None:
        _client = EmbeddingClient()
        if not _client.available:
            logger.warning("SILICONFLOW_API_KEY 未设置，向量检索禁用（关键词兜底）")
    return _client
