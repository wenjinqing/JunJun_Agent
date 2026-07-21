"""chat_frequency 插件：发言频率调节命令（迁移自 ChatFrequency，新架构重写）。

命令：
  /chat talk_frequency <倍率>（/chat t <倍率>）——管理员专用，直接设调节因子
  /chat show（/chat s）——查看当前 talk_value/倍率/基础值
底层接 junjun_agent.funnel.frequency.frequency_control（LLM 动态调节的同一个因子）。
"""

from junjun_agent.commands import register_command
from junjun_agent.funnel.frequency import frequency_control, resolve_talk_value
from junjun_core.observability import get_logger

logger = get_logger("plugin.chat_frequency")

_MIN, _MAX = 0.1, 3.0


def _show(chat_id: str) -> str:
    st = frequency_control.state(chat_id)
    base = resolve_talk_value(chat_id)
    eff = frequency_control.effective_talk_value(chat_id)
    return (f"当前会话发言频率：\n- 生效值 {eff:.2f}（基础 {base:.2f} × 倍率 {st.adjust_factor:.2f}）\n"
            f"- 倍率可调范围 {_MIN}~{_MAX}，/chat talk_frequency <倍率> 调整")


@register_command("chat", plugin="chat_frequency",
                  description="/chat talk_frequency <倍率>（管理员）| /chat show")
async def chat_cmd(ctx):
    parts = ctx.args.split()
    sub = parts[0].lower() if parts else "show"

    if sub in ("show", "s"):
        return _show(ctx.session.chat_id)

    if sub in ("talk_frequency", "t"):
        from junjun_core.security import is_admin, report_violation
        if not is_admin(ctx.meta.user_id):
            report_violation("调整发言频率", ctx.meta.user_id or "", ctx.meta.nickname,
                             ctx.session.chat_id, ctx.args[:60])
            return "调频率只有管理员能操作哦（已通知管理员）。"
        if len(parts) < 2:
            return "用法：/chat talk_frequency <倍率>（0.1~3.0）"
        try:
            value = float(parts[1])
        except ValueError:
            return f"「{parts[1]}」不是数字，用法：/chat talk_frequency <倍率>（0.1~3.0）"
        value = max(_MIN, min(_MAX, value))
        st = frequency_control.state(ctx.session.chat_id)
        st.adjust_factor = value
        logger.info(f"[{ctx.session.chat_id}] 管理员手动调频率 -> {value:.2f}")
        return _show(ctx.session.chat_id)

    return "用法：/chat talk_frequency <倍率>（管理员）| /chat show"


TOOLS = []
