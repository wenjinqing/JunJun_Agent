"""junjun_llm: LLM 抽象层（任务槽模型 + Langfuse 追踪）。"""

from junjun_llm.models import get_chat_model, reset_slots
from junjun_llm.tracing import get_callbacks

__all__ = ["get_chat_model", "reset_slots", "get_callbacks"]
