"""junjun_llm: LLM 抽象层（任务槽模型 + Langfuse 追踪）。"""

from junjun_llm.models import get_chat_model
from junjun_llm.tracing import get_callbacks

__all__ = ["get_chat_model", "get_callbacks"]
