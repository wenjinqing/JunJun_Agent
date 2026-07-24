"""Embedding 客户端：bge-m3 (SiliconFlow, 1024 维)。

对齐原 lpmm embedding 配置。支持两套 env 配置：
1. EMBEDDING_BASE_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY（专用 embedding 服务）
2. 降级：VLM_BASE_URL / VLM_MODEL / VLM_API_KEY（复用多模态服务的 embedding 能力）
3. 再降级：SILICONFLOW_API_KEY（默认 bge-m3）

任一不可用则 available=False，上层自动降级关键词检索，不阻塞主循环。
"""

import os
import socket
from typing import List, Optional

from junjun_core.observability import get_logger

logger = get_logger("memory.embedding")

EMBED_DIM = 1024
# 兼容旧引用（long_term.py 索引头校验用）——实际模型名从 client._model 读
EMBED_MODEL = "BAAI/bge-m3"


class EmbeddingClient:
    def __init__(self):
        self._api_key = ""
        self._base_url = ""
        self._model = ""
        self._client = None
        self._load_config()

    def _load_config(self) -> None:
        """按优先级加载 embedding 配置。"""
        # 1. 专用 embedding 配置
        self._api_key = os.environ.get("EMBEDDING_API_KEY", "")
        self._base_url = os.environ.get("EMBEDDING_BASE_URL", "")
        self._model = os.environ.get("EMBEDDING_MODEL", "")
        if self._api_key and self._base_url and self._model:
            logger.info(f"embedding 配置: {self._base_url[:40]}... / {self._model}")
            return
        # 2. SiliconFlow bge-m3（专用 embedding 服务，支持 embeddings endpoint）
        sf_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if sf_key:
            self._api_key = sf_key
            self._base_url = "https://api.siliconflow.cn/v1"
            self._model = "BAAI/bge-m3"
            logger.info("embedding 配置: siliconflow / BAAI/bge-m3")
            return
        # 3. VLM 复用（仅当无专用 embedding 服务时；多数 VLM 服务不支持 embeddings endpoint）
        self._api_key = os.environ.get("VLM_API_KEY", "")
        self._base_url = os.environ.get("VLM_BASE_URL", "")
        self._model = os.environ.get("VLM_MODEL", "")
        if self._api_key and self._base_url and self._model:
            logger.info(f"embedding 复用 VLM 配置: {self._base_url[:40]}... / {self._model}")
            return
        logger.warning("embedding 未配置（EMBEDDING_*/SILICONFLOW_API_KEY/VLM_* 均无），向量检索禁用")

    @property
    def available(self) -> bool:
        return bool(self._api_key and self._base_url and self._model)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            import httpx
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=15,
                http_client=httpx.AsyncClient(
                    timeout=httpx.Timeout(15, connect=5),
                    transport=httpx.AsyncHTTPTransport(
                        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                        socket_options=[
                            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
                            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 120),
                        ],
                    ),
                ),
            )
        return self._client

    async def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量向量化。不可用/失败返回 None（上层降级）。"""
        if not self.available or not texts:
            return None
        try:
            resp = await self._get_client().embeddings.create(model=self._model, input=texts)
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
    return _client
