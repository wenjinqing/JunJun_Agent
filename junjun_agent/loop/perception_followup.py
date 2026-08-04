"""感知后续推送：决策时没看完的图/语音/视频，看完后主动补一句。

「我看不到图片」的根治（2026-08-02 生产反馈）：
- 旧行为：3s 决策窗内感知没完成 -> Agent 看到占位 -> 只能说「看不到」，
  在途任务完成后结果躺缓存里没人说
- 新行为：决策时「还在看」的条目注册到这里 -> 后台等任务完成 ->
  utils 模型用君君口吻主动补一条观后感

防刷屏三原则：只对「已接话」的消息补（Agent 沉默就不补）；
同一条目的只补一次（chat+task 去重）；单条消息的多路感知合并成一条补发。
"""

import asyncio
import time
from typing import Dict, List

from junjun_core.observability import get_logger

logger = get_logger("loop.perception_followup")

_MAX_WAIT = 90.0       # 等在途任务的总上限（视频感知最重 10-30s）
_WATCHED: Dict[str, set] = {}   # chat_id -> {id(task)}（去重：同任务只补一次）

# 各感知类型的中文名（prompt/日志用）
_KIND_NAMES = {"image": "图片", "voice": "语音", "video": "视频"}


def schedule(session, entries: List[dict]) -> None:
    """注册感知后续：entries = [{"kind": "image"|"voice"|"video", "task": Task}]。

    仅在 bot 已接话（回复了）时调用。去重后无可补条目则静默返回。
    """
    chat_id = session.chat_id
    watched = _WATCHED.setdefault(chat_id, set())
    fresh = [e for e in entries
             if e.get("task") is not None and id(e["task"]) not in watched]
    if not fresh:
        return
    for e in fresh:
        watched.add(id(e["task"]))
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_followup(session, fresh),
                     name=f"perception-followup-{chat_id}")
    logger.info(f"[{chat_id}] 感知后续已登记: {len(fresh)} 项在途"
                f"（{','.join(_KIND_NAMES.get(e['kind'], e['kind']) for e in fresh)}）")


async def _followup(session, entries: List[dict]) -> None:
    """等在途感知完成 -> 组织观后感 -> 主动推送。任何失败静默。"""
    chat_id = session.chat_id
    try:
        done, still = await asyncio.wait(
            [e["task"] for e in entries], timeout=_MAX_WAIT)
        results = []
        for e in entries:
            t = e["task"]
            if t in done and not t.cancelled() and t.exception() is None:
                val = t.result()
                if val and val not in ("[图片]", "[语音]", "[视频]", "[表情]"):
                    results.append((_KIND_NAMES.get(e["kind"], e["kind"]), val))
        if still:
            logger.debug(f"[{chat_id}] 感知后续超时（{len(still)} 项没等到），放弃补发")
        if not results:
            return
        text = await _compose(session, results)
        if not text:
            return
        parts = chat_id.split(":")
        platform, target_id = parts[0], parts[1]
        kind = parts[2] if len(parts) > 2 else "private"
        from junjun_core.contracts import ReplySegment, ReplySet
        from junjun_core.gateway.router import get_gateway
        await get_gateway().send_reply(ReplySet(
            platform=platform,
            target_group_id=target_id if kind == "group" else None,
            target_user_id=target_id if kind != "group" else None,
            segments=[ReplySegment(type="text", data=text)],
            should_reply=True,
        ))
        # 补发也进会话记忆——后续对话里君君记得自己说过这句
        try:
            session.memory.add_bot(text)
        except Exception:
            pass
        logger.info(f"[{chat_id}] 感知后续已补发（{len(results)} 项）")
    except Exception as e:
        logger.debug(f"[{chat_id}] 感知后续失败（忽略）: {type(e).__name__}: {e}")


_COMPOSE_PROMPT = """你是"{nickname}"——{persona_brief}
对方刚发了{what}，你当时说「等我看/听一下」，现在看完了。
内容：{results}
用你的口吻告诉对方你看到/听到了什么（2-3 句口语，自然带出内容，不要念清单，不要前缀）。只输出要说的话。"""


async def _compose(session, results: List[tuple]) -> str:
    """utils 模型写观后感（人设口吻）；失败降级模板。"""
    what = "、".join(sorted({k for k, _ in results}))
    body = "；".join(f"{k}里是：{v}" for k, v in results)
    fallback = f"看完啦——{body}"
    try:
        from junjun_core.config import get_global_config
        from junjun_llm import get_callbacks, get_chat_model
        from langchain_core.messages import HumanMessage
        from junjun_agent.persona import persona_brief
        nickname = get_global_config().bot.nickname
        resp = await get_chat_model("utils").ainvoke(
            [HumanMessage(content=_COMPOSE_PROMPT.format(
                nickname=nickname, persona_brief=persona_brief(),
                what=what, results=body))],
            config={"callbacks": get_callbacks()})
        out = str(resp.content).strip()
        if out:
            return out
    except Exception:
        pass
    return fallback
