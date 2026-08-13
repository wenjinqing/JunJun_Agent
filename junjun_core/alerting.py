"""系统告警挂钩（2026-08-13 审查 P1）。

告警通道（notify_admin 私聊 + watchdog SMTP）早就建好，却只发业务流程——
LLM 全线挂、号池见底、花费失控全靠管理员翻日志/群友吐槽才发现。
三条软告警（只喊人不硬熔断，单人项目保守方向）：

1. 号池空：junjun_llm.models 在 healthy_keys() 为空时调 note_pool_empty（4h 防抖）
2. 日 token 超阈：agent._record_usage 顺手 note_usage（每日一次，阈值
   [alerts] daily_token_threshold，默认 3_000_000，0=关）
3. 每日用量汇总：scheduler cron 推昨日 LLMUsage 聚合（[alerts] daily_report_time
   默认 "23:50"，空=关）

住 core 不住 agent：llm（下层）也要调，agent/scheduler（上层）同样可调，无环。
"""

import asyncio
import time

from junjun_core.observability import get_logger

logger = get_logger("core.alerting")

_pool_empty_last = 0.0
_POOL_DEBOUNCE = 4 * 3600     # 号池空告警防抖：号池重建每次启动都查，不防抖会刷屏

_day = ""
_day_tokens = 0
_day_alerted = False


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return dict(get_global_config().raw.get("alerts", {}) or {})
    except Exception:
        return {}


async def _safe_notify(text: str) -> None:
    try:
        from junjun_core.security import notify_admin
        await notify_admin(text)
    except Exception as e:
        logger.warning(f"告警发送失败（仅日志）: {e} | {text[:60]}")


def _fire(text: str) -> None:
    """同步上下文点火：有事件循环就派任务发，没有就降级日志（启动早期路径）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(f"告警（无事件循环，仅日志）: {text}")
        return
    loop.create_task(_safe_notify(text))


def note_pool_empty() -> None:
    """号池空告警（4h 防抖）。号池空 = 号池腿全撤靠静态条目兜底，是 LLM
    全线失败的前兆——必须让人知道。"""
    global _pool_empty_last
    now = time.time()
    if now - _pool_empty_last < _POOL_DEBOUNCE:
        return
    _pool_empty_last = now
    _fire("【告警】号池空了（sf_keys 全部欠费/失效或文件读不出），"
          "号池腿全撤，正在靠静态条目兜底。去看看 data/sf_keys.txt。")


def note_usage(tokens: int) -> None:
    """token 日累计 + 超阈告警（每日一次）。软告警不熔断——用来发现
    「不像今天的用法」（loop 死循环刷调用、评测忘关之类）。"""
    global _day, _day_tokens, _day_alerted
    today = time.strftime("%Y-%m-%d")
    if today != _day:
        _day, _day_tokens, _day_alerted = today, 0, False
    _day_tokens += max(0, int(tokens))
    threshold = int(_cfg().get("daily_token_threshold", 3_000_000) or 0)
    if threshold > 0 and _day_tokens >= threshold and not _day_alerted:
        _day_alerted = True
        _fire(f"【告警】今日 token 已 {_day_tokens:,}，超过阈值 {threshold:,}。"
              f"没有硬熔断，但如果这不像你今天的用法，查查是不是有 loop 在刷调用。")


async def daily_usage_report() -> None:
    """每日用量汇总推送：LLMUsage 最近 24h 按槽聚合。24h 零调用本身也是信号
    （= 一天没人理她或全线静默），照发。"""
    try:
        from peewee import fn
        from junjun_core.database import LLMUsage
        since = time.time() - 86400
        rows = list(LLMUsage
                    .select(LLMUsage.request_type,
                            fn.SUM(LLMUsage.prompt_tokens).alias("pt"),
                            fn.SUM(LLMUsage.completion_tokens).alias("ct"),
                            fn.COUNT(LLMUsage.id).alias("n"))
                    .where(LLMUsage.time >= since)
                    .group_by(LLMUsage.request_type))
    except Exception as e:
        logger.warning(f"用量日报查询失败: {e}")
        return
    if not rows:
        await _safe_notify("【日报】最近 24h 零 LLM 调用——如果今天没特意停用她，"
                           "这本身就是不对劲的信号。")
        return
    lines = [f"· {r.request_type or '?'}：{r.n} 次，in={int(r.pt or 0):,} out={int(r.ct or 0):,}"
             for r in sorted(rows, key=lambda r: -(r.pt or 0) - (r.ct or 0))]
    total = sum(int(r.pt or 0) + int(r.ct or 0) for r in rows)
    await _safe_notify(f"【日报】最近 24h token 用量（共 {total:,}）：\n" + "\n".join(lines))


def _reset_for_test() -> None:
    global _pool_empty_last, _day, _day_tokens, _day_alerted
    _pool_empty_last = 0.0
    _day, _day_tokens, _day_alerted = "", 0, False
