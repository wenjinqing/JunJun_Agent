"""自我反思：定期复盘最近回复质量，私聊反馈给管理员。

每 N 轮回复触发一次（[reflection].every_n）：
- 从 Messages 表拉最近对话（含自己的发言）
- utils 槽 LLM 自评：复读/答非所问/情绪连贯/工具使用
- 反思摘要经 notify_admin 私聊发管理员

设计：counter 驱动（回复即计数），后台任务执行不阻塞会话；
失败静默（反思是锦上添花，绝不能影响聊天主链路）。
"""

import asyncio
import time

from junjun_core.config import get_global_config
from junjun_core.observability import get_logger

logger = get_logger("loop.reflection")

_PROMPT = """你是 QQ 机器人「{nickname}」，现在复盘自己最近的回复质量。
检查四点：1) 有没有复读/重复表达 2) 有没有答非所问或没接住上下文
3) 情绪语气是否连贯、像不像真人 4) 工具使用是否合理（有没有该用不用/乱用）。

最近对话记录（「{nickname}」是你自己）：
{transcript}

输出一段 150 字以内的反思：先说一句总体表现，再列 1-2 个具体要改进的点（各附一个例子）。
对管理员（你的主人）汇报，口吻诚恳简短，不要 markdown。"""


def _cfg() -> dict:
    return get_global_config().raw.get("reflection", {})


class ReflectionLoop:
    """回复计数驱动的自评触发器。"""

    def __init__(self) -> None:
        self._count = 0
        self._running = False

    def note_reply(self) -> None:
        """每发出一条回复计数；到阈值后台触发反思（幂等，执行中不重复触发）。"""
        if not _cfg().get("enable", True):
            return
        self._count += 1
        every = int(_cfg().get("every_n", 50))
        if self._count < every or self._running:
            return
        self._count = 0
        from junjun_core.bg_tasks import fire_and_forget
        fire_and_forget(self._run_safe(), name="reflection")

    async def _run_safe(self) -> None:
        self._running = True
        try:
            await self.reflect()
        except Exception as e:
            logger.warning(f"自我反思失败（静默）: {type(e).__name__}: {e}")
        finally:
            self._running = False

    async def reflect(self) -> str:
        """拉最近对话 -> LLM 自评 -> 私聊管理员。返回反思文本（失败空串）。"""
        transcript = self._recent_transcript(limit=60)
        if not transcript:
            return ""
        cfg = get_global_config()
        from langchain_core.messages import HumanMessage
        from junjun_llm import get_chat_model
        model = get_chat_model("utils")
        resp = await model.ainvoke([HumanMessage(content=_PROMPT.format(
            nickname=cfg.bot.nickname, transcript=transcript,
        ))])
        text = (resp.content or "").strip()
        if not text:
            return ""
        from junjun_core.security import notify_admin
        ok = await notify_admin(f"【君君自我反思】\n{text}")
        if ok:
            logger.info(f"自我反思已上报管理员（{len(text)} 字）")
        return text

    @staticmethod
    def _recent_transcript(limit: int) -> str:
        """Messages 表最近 N 条渲染为「昵称: 内容」行（截断防超长）。"""
        try:
            from junjun_core.database import Messages
            rows = list(Messages.select()
                        .where(Messages.processed_plain_text != "")
                        .order_by(Messages.time.desc()).limit(limit))
            rows.reverse()
            lines = []
            for r in rows:
                name = "君君" if r.is_bot else (r.user_nickname or "群友")
                text = (r.processed_plain_text or "").replace("\n", " ")[:120]
                if text:
                    lines.append(f"{name}: {text}")
            return "\n".join(lines)[-3000:]
        except Exception as e:
            logger.debug(f"拉取反思语料失败: {e}")
            return ""


reflection_loop = ReflectionLoop()
