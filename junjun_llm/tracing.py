"""Langfuse v3 tracing: business layer only gets callbacks list, does not hold Langfuse handle directly."""

import os
from typing import List, Optional

from junjun_core.observability import get_logger

logger = get_logger("llm.tracing")

_handler: Optional[object] = None
_checked = False


def get_callbacks() -> List:
    """Return LangChain callbacks. Return empty list when Langfuse is disabled/unreachable (silent degrade)."""
    global _handler, _checked
    if _checked:
        return [_handler] if _handler else []
    _checked = True

    if os.environ.get("LANGFUSE_ENABLED", "false").lower() != "true":
        logger.info("Langfuse not enabled (LANGFUSE_ENABLED != true)")
        return []
    try:
        from langfuse.langchain import CallbackHandler

        _handler = CallbackHandler()
        logger.info(f"Langfuse v3 tracing enabled -> {os.environ.get('LANGFUSE_HOST', '')}")
        return [_handler]
    except Exception as e:
        logger.warning(f"Langfuse init failed, degraded to no tracing: {e}")
        return []
