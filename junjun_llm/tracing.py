"""Langfuse v3 追踪：业务层只拿 callbacks 列表，不直接持有 Langfuse 句柄。"""

import os
from typing import List, Optional

from junjun_core.observability import get_logger

logger = get_logger("llm.tracing")

_handler: Optional[object] = None
_checked = False


def get_callbacks() -> List:
    """返回 LangChain callbacks。Langfuse 未启用/不可达时返回空列表（静默降级）。"""
    global _handler, _checked
    if _checked:
        return [_handler] if _handler else []
    _checked = True

    if os.environ.get("LANGFUSE_ENABLED", "false").lower() != "true":
        logger.info("Langfuse 未启用（LANGFUSE_ENABLED != true）")
        return []
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        client = get_client()
        if not client.auth_check():
            logger.warning("Langfuse auth_check 失败，降级为无追踪")
            return []
        _handler = CallbackHandler()
        logger.info(f"Langfuse v3 追踪已启用 -> {os.environ.get('LANGFUSE_HOST', '')}")
        return [_handler]
    except Exception as e:
        logger.warning(f"Langfuse 初始化失败，降级为无追踪: {e}")
        return []
