"""情绪系统：对齐原 mood/mood_manager 语义。

- ChatMood: 按会话维护情绪文本描述
- 新消息进入 L3 时触发重评（跟随 gate，省 token）；超时衰退回平静
- enable_mood / emotion_style 配置对齐；情绪块进 persona prompt
"""

import time
from dataclasses import dataclass, field
from typing import Dict

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.mood")

_REGRESS_AFTER = 1800.0   # 30 分钟无互动衰退
_EVAL_COOLDOWN = 120.0    # 重评冷却
_DEFAULT_MOOD = "平静"

_EVAL_PROMPT = """你是 QQ 机器人"{nickname}"。根据最近对话判断你现在的情绪状态。
{emotion_style}
当前情绪：{current}
最近对话：
{conversation}

只输出新的情绪状态，一个短语（如：开心、有点无语、兴奋、被夸了很得意、平静），不要解释。"""


@dataclass
class ChatMood:
    state: str = _DEFAULT_MOOD
    updated_at: float = field(default_factory=time.time)
    last_eval: float = 0.0


class MoodManager:
    def __init__(self):
        self._moods: Dict[str, ChatMood] = {}

    def _enabled(self) -> bool:
        return bool(get_global_config().raw.get("mood", {}).get("enable_mood", True))

    def get_mood(self, chat_id: str) -> str:
        """读取情绪（自动衰退）。"""
        if not self._enabled():
            return ""
        mood = self._moods.get(chat_id)
        if mood is None:
            return _DEFAULT_MOOD
        if time.time() - mood.updated_at > _REGRESS_AFTER and mood.state != _DEFAULT_MOOD:
            mood.state = _DEFAULT_MOOD
            logger.debug(f"[{chat_id}] 情绪衰退回平静")
        return mood.state

    def build_mood_block(self, chat_id: str) -> str:
        mood = self.get_mood(chat_id)
        return f"你当前的情绪：{mood}（让回复语气自然反映这个情绪）" if mood else ""

    def should_evaluate(self, chat_id: str) -> bool:
        if not self._enabled():
            return False
        mood = self._moods.setdefault(chat_id, ChatMood())
        return (time.time() - mood.last_eval) >= _EVAL_COOLDOWN

    async def evaluate(self, chat_id: str, conversation: str, *, model=None, callbacks=None) -> None:
        """小模型重评情绪。失败静默保持原状态。"""
        mood = self._moods.setdefault(chat_id, ChatMood())
        mood.last_eval = time.time()
        cfg = get_global_config()
        emotion_style = cfg.raw.get("mood", {}).get("emotion_style", "")
        from langchain_core.messages import HumanMessage
        try:
            if model is None:
                from junjun_llm import get_chat_model
                model = get_chat_model("utils")
            resp = await model.ainvoke(
                [HumanMessage(content=_EVAL_PROMPT.format(
                    nickname=cfg.bot.nickname, emotion_style=emotion_style,
                    current=mood.state, conversation=conversation,
                ))],
                config={"callbacks": callbacks or []},
            )
            new_state = str(resp.content).strip().splitlines()[0][:20]
            if new_state and new_state != mood.state:
                logger.info(f"[{chat_id}] 情绪变化: {mood.state} -> {new_state}")
                mood.state = new_state
            mood.updated_at = time.time()
        except Exception as e:
            logger.warning(f"情绪评估失败（保持 {mood.state}）: {e}")

    def set_mood(self, chat_id: str, state: str) -> None:
        """skill 手动调整。"""
        mood = self._moods.setdefault(chat_id, ChatMood())
        mood.state = state[:20]
        mood.updated_at = time.time()


mood_manager = MoodManager()
