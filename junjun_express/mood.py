"""情绪系统：两层心境。

- ChatMood（会话层）：按会话维护情绪文本描述，新消息进 L3 时触发重评
  （跟随 gate，省 token）；30 分钟无互动衰退回平静
- 全局自我心境（SelfMood 表持久化）：跨场景持续的心境，由最近一次
  情绪变化与每晚的日记共同塑造，重启不丢；超过 self_mood_hours 视为回到平静
- enable_mood / emotion_style 配置对齐；情绪块进 persona prompt
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("express.mood")

_REGRESS_AFTER = 1800.0   # 30 分钟无互动衰退
_EVAL_COOLDOWN = 120.0    # 重评冷却
_DEFAULT_MOOD = "平静"

# 情绪 -> 行为倾向映射（情绪连贯性：不只影响措辞，也影响回复长度/工具使用意愿）
_MOOD_NEGATIVE = ("无语", "烦", "生气", "怒", "难过", "郁闷", "累", "疲", "丧", "低落", "委屈", "emo")
_MOOD_POSITIVE = ("开心", "兴奋", "得意", "高兴", "快乐", "期待", "满足", "甜", "惊喜")

_EVAL_PROMPT = """你是 QQ 群里的"{nickname}"。根据最近对话判断你现在的情绪状态。
{emotion_style}
当前情绪：{current}
最近对话：
{conversation}

判断规则：
- 评估的是「你」被怎么对待后的情绪，不是群聊氛围——群聊吵闹/复读/玩梗不等于你无语，你是见惯了的群老人
- 只有对话里出现明确冲着你来的情绪信号（夸你/怼你/求你/撩你/冷落你）才改变情绪；信号不明显就保持现状，或向「平静」回落
- 情绪会自然流动：同一件事不会让你停在一个情绪里太久，事情过去就往「平静」回落
只输出新的情绪状态，一个 2~8 字的短语，不要解释。"""


@dataclass
class ChatMood:
    state: str = _DEFAULT_MOOD
    updated_at: float = field(default_factory=time.time)
    last_eval: float = 0.0


class MoodManager:
    def __init__(self):
        self._moods: Dict[str, ChatMood] = {}
        self._self_mood: Optional[ChatMood] = None  # 懒加载自 SelfMood 表

    def _enabled(self) -> bool:
        return bool(get_global_config().raw.get("mood", {}).get("enable_mood", True))

    # ---------------------------------------------------------------- 全局自我心境

    def _self_fresh_hours(self) -> float:
        try:
            return float(get_global_config().raw.get("mood", {}).get("self_mood_hours", 12))
        except (TypeError, ValueError):
            return 12.0

    def _load_self(self) -> ChatMood:
        if self._self_mood is None:
            try:
                from junjun_core.database.models import SelfMood, _bot_id
                row = SelfMood.get_or_none(SelfMood.bot_id == _bot_id())
                if row:
                    self._self_mood = ChatMood(state=row.state, updated_at=row.updated_at)
            except Exception:
                pass
            if self._self_mood is None:
                self._self_mood = ChatMood()
        return self._self_mood

    def get_self_mood(self) -> str:
        """全局自我心境（超新鲜期视为回到平静，不落库）。"""
        if not self._enabled():
            return ""
        sm = self._load_self()
        if sm.state == _DEFAULT_MOOD:
            return _DEFAULT_MOOD
        if time.time() - sm.updated_at > self._self_fresh_hours() * 3600:
            return _DEFAULT_MOOD
        return sm.state

    def set_self_mood(self, state: str, reason: str = "") -> None:
        """更新全局自我心境并持久化（情绪重评变化 / 日记沉淀都会走这里）。"""
        state = (state or "").strip()[:20]
        if not state:
            return
        self._self_mood = ChatMood(state=state, updated_at=time.time())
        try:
            from junjun_core.database.models import SelfMood, _bot_id
            row = SelfMood.get_or_none(SelfMood.bot_id == _bot_id())
            if row is None:
                SelfMood.create(state=state, reason=reason[:100], updated_at=time.time())
            else:
                row.state, row.reason, row.updated_at = state, reason[:100], time.time()
                row.save()
        except Exception as e:
            logger.warning(f"自我心境持久化失败（仅内存保留）: {e}")

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
        if not mood:
            return ""
        base = f"你当前的情绪：{mood}"
        if any(w in mood for w in _MOOD_NEGATIVE):
            # 负面情绪只调语气，不压人格：话少一点淡一点，但别人需要时依然在
            # （旧版「不想说话/不想折腾工具」会把学姐的温柔和主动行为全压没，
            # 还抑制工具调用意愿——2026-08-04 情绪卡死在「无语」事件）
            block = base + "。让情绪体现在语气上：话少一点、淡一点，主动闹的心思低了；" \
                          "但有人找你、需要你时，你依然会认真回应。"
        elif any(w in mood for w in _MOOD_POSITIVE):
            block = base + "。让情绪体现在行为上：现在很愿意聊，可以活泼一点；" \
                          "合适时主动发表情包/语音，被夸就大方得意。"
        else:
            block = base + "（让回复语气自然反映这个情绪）"
        # 全局自我心境：与会话情绪不同时补充（它是更缓慢、跨场景的底色）
        self_mood = self.get_self_mood()
        if self_mood and self_mood != _DEFAULT_MOOD and self_mood != mood:
            block += f"\n你的整体心境：{self_mood}（最近经历沉淀下来的底色，跨场景持续，让它自然地影响你的状态）"
        return block

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
                # 会话情绪变化同时塑造全局自我心境（最近的经历定义现在的自己）；
                # 但「平静」不写入——别群评出的平静不该冲掉这边真实沉淀的情绪
                if new_state != _DEFAULT_MOOD:
                    self.set_self_mood(new_state, reason=chat_id)
            mood.updated_at = time.time()
        except Exception as e:
            logger.warning(f"情绪评估失败（保持 {mood.state}）: {e}")

    def set_mood(self, chat_id: str, state: str) -> None:
        """skill 手动调整。"""
        mood = self._moods.setdefault(chat_id, ChatMood())
        mood.state = state[:20]
        mood.updated_at = time.time()


mood_manager = MoodManager()
