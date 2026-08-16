"""主循环工具结果压缩（2026-08-16，DSH pi-quiet-tools 思想移植）。

病灶：工具大产出（fetch_page 深读全文、run_code 长输出、query_chat_history
批量翻记录）整条 ToolMessage 进上下文——一条结果几千上万字，把背景对话
挤出有效注意力区，还按轮重复计费（前缀缓存只保头部稳定，长结果驻留
消息列每轮都烧输入溢价）。

移植 DeepSeek Harness 的「头尾预览 + 全文外置」：超过阈值的结果只留
头部+尾部+省略说明，全文落进本会话工作区 artifacts/——模型要全文
自己调 workspace_read 读回来（工具自给，不丢信息只延迟取）。

挂载点选 _build_agent 中间件而非注册表包装：TaskKernel 执行器直调注册表
工具，它的大产出走材料库（task_kernel/materials.py 全文落盘+摘要指针），
两条链路各管各的，互不截对方全文。

落盘失败一律降级纯截断（无指针），压缩是增强不是硬依赖。
"""

import itertools
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from junjun_core.observability import get_logger

logger = get_logger("agent.result_compress")

_INLINE_CHARS = 4000     # [agent] tool_result_inline_chars：超过才压缩（短结果原样）
_HEAD_CHARS = 1800       # 预览头部长度
_TAIL_CHARS = 900        # 预览尾部长度（尾部常带结论/错误行，比中间值钱）

_SEQ = itertools.count(1)   # 文件名防碰撞（同秒同工具多次产出）


def _cfg() -> tuple:
    """([agent] tool_result_compress, tool_result_inline_chars)。配置坏了按默认。"""
    try:
        from junjun_core.config import get_global_config
        agent = get_global_config().raw.get("agent", {}) or {}
        return (bool(agent.get("tool_result_compress", True)),
                int(agent.get("tool_result_inline_chars", _INLINE_CHARS)))
    except Exception:
        return True, _INLINE_CHARS


def _store_artifact(chat_id: str, tool_name: str, text: str) -> str:
    """全文存进会话工作区 artifacts/，返回工作区相对路径；失败返回 ""。"""
    try:
        from junjun_skills.plugins.workspace import tools as wt
        fname = f"{wt._safe_name(tool_name)}-{int(time.time())}-{next(_SEQ)}.txt"
        d = wt._ROOT / wt._safe_name(chat_id or "unknown") / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(text, encoding="utf-8")
        return f"artifacts/{fname}"
    except Exception as e:
        logger.warning(f"工具结果落盘失败（降级纯截断）: {type(e).__name__}: {e}")
        return ""


def compress_result(text: str, *, chat_id: str, tool_name: str) -> str:
    """超长结果 -> 头尾预览+指针；未超阈值/开关关闭时原样返回。"""
    enabled, inline = _cfg()
    if not enabled or len(text) <= inline:
        return text
    # 阈值调得比头+尾还小时按比例收缩，保证压缩后一定变短而不是变长
    head_n = min(_HEAD_CHARS, max(200, inline // 2))
    tail_n = min(_TAIL_CHARS, max(100, inline // 4))
    omitted = len(text) - head_n - tail_n
    rel = _store_artifact(chat_id, tool_name, text)
    if rel:
        pointer = (f"……（中间省略 {omitted} 字；全文 {len(text)} 字已存工作区 "
                   f"{rel}，要看全文调 workspace_read 读这个路径）……")
    else:
        pointer = f"……（中间省略 {omitted} 字，全文共 {len(text)} 字）……"
    return f"{text[:head_n]}\n{pointer}\n{text[-tail_n:]}"


class ToolResultCompressMiddleware(AgentMiddleware):
    """超长 ToolMessage 头尾预览 + 全文存工作区 artifacts/。

    实例随 _build_agent 每轮新建；只动 ToolMessage.content（str），
    tool_call_id/name 等字段原样保留。检索刹车/复读熔断的短路文本很短，
    天然不过阈值，不会被二次加工。
    """

    name = "tool_result_compress"

    def __init__(self, chat_id: str = ""):
        self._chat_id = chat_id

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        compressed = compress_result(
            result.content, chat_id=self._chat_id,
            tool_name=str(request.tool_call.get("name", "") or result.name or ""))
        if compressed == result.content:
            return result
        logger.info(f"[{self._chat_id or '?'}] 工具结果压缩："
                    f"{result.name} {len(result.content)} -> {len(compressed)} 字")
        return result.model_copy(update={"content": compressed})
