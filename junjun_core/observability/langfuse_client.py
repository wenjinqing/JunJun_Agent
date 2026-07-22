"""Langfuse 可观测客户端（未启用时空操作降级，不阻塞主循环）。"""

import os
from typing import Any, Optional

try:
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore


class _NoopSpan:
    """未启用时的占位 span，链式调用全部空操作。"""
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def span(self, *a, **k): return self
    def generation(self, *a, **k): return self
    def update(self, *a, **k): return self
    def end(self, *a, **k): return self
    def attribute(self, *a, **k): return self


class LangfuseClient:
    """统一封装，屏蔽未启用场景。"""
    def __init__(self) -> None:
        self._enabled = False
        self._client: Optional[Any] = None
        self._init()

    def _init(self) -> None:
        if not _LANGFUSE_AVAILABLE:
            return
        if os.environ.get("LANGFUSE_ENABLED", "false").lower() != "true":
            return
        host = os.environ.get("LANGFUSE_HOST")
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_SECRET_KEY")
        if not (host and pk and sk):
            return
        try:
            self._client = Langfuse(host=host, public_key=pk, secret_key=sk)
            self._enabled = True
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_span(self, name: str, **kwargs):
        """v3 统一入口：未启用时返回空操作 span。"""
        if not self._enabled:
            return _NoopSpan()
        try:
            return self._client.start_as_current_observation(name=name, as_type="span", **kwargs)
        except Exception:
            return _NoopSpan()

    def start_trace(self, *a, **k):
        if not self._enabled:
            return _NoopSpan()
        try:
            return self._client.start_trace(*a, **k)  # type: ignore
        except Exception:
            return _NoopSpan()

    def get_callback_handler(self, *a, **k):
        if not self._enabled:
            return None
        try:
            return self._client.get_langchain_handler(*a, **k)  # type: ignore
        except Exception:
            return None


lf = LangfuseClient()
