"""junjun_cli：本地终端直接对 TaskKernel 下单（TaskKernel Phase 4 唯一必做项）。

调内核不用开 QQ 全栈：真实 planner/执行器/工具（提醒落库、跑码进沙箱、
文件写工作区都是真副作用），只有消息出口从网关换成终端打印。

用法：
    uv run python scripts/junjun_cli.py "帮我把工作区 sales.csv 做个汇总图"
    uv run python scripts/junjun_cli.py          # 无参数进 REPL，/exit 退出
    uv run python scripts/junjun_cli.py --user 123456 "..."   # 模拟非管理员
    uv run python scripts/junjun_cli.py --timeout 600 "..."   # 整体看门狗（秒）

补丁面（照 eval_tasks.py 先例，只换出口与记录，不动决策链路）：
- outbound.send_proactive -> 终端打印（终态汇报/规划拒收交代都从这看）
- security.notify_admin   -> 终端打印（审批请求）
- task_manager._record_outcome -> no-op（cli:local 结局不进生产决策注入）
- executor._cfg 强制 enable=True（灰度开关是生产的，CLI 是调试工具）
- LangGraph 引擎换 MemorySaver：生产 aiosqlite 连接的 worker 是非守护线程，
  短命脚本退出时 join 它 = 永远挂死（2026-08-12 eval py-spy 实锤）

边界（如实反馈，不装）：
- 发送类工具（send_message/workspace_send 等）没有网关可发，会失败
- 审批交互在终端进行：挂起时提示「发/算了」，超时走生产默认（跳过）
- 一次跟一单；并发下单不是 CLI 的场景
- 日志写 logs/cli.log，不碰 bot.log/adapter.log
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 注意：load_dotenv 不许在 import 期跑——pytest 收编本模块时 override=True 会把
# 真 .env 注进测试进程，「无密钥降级」类测试全线假败 + 真网络调用把套件拖慢
# 7 倍（2026-08-15 实锤）。环境装配只走 _main。

CLI_CHAT_ID = "cli:local:private"


class CliSink:
    """内核出口的终端捕获：completed 存终态 plan（等终态的信号），
    rejected 记规划拒收（planner 没接单，等下去是白等）。"""

    def __init__(self) -> None:
        self.completed: dict = {}
        self.rejected = False

    async def on_send(self, chat_id, segments, **kw) -> bool:
        text = "\n".join(str(getattr(s, "data", "")) for s in segments
                         if getattr(s, "type", "") == "text")
        if chat_id == CLI_CHAT_ID and "拆不动" in text:
            self.rejected = True
        print(f"\n[君君] {text}\n")
        return True

    async def on_notify_admin(self, text, **kw) -> bool:
        print(f"\n[审批请求]\n{text}\n")
        return True


def force_enable(executor) -> None:
    """灰度开关是生产放量用的；CLI 是调试工具，永远接单。其余配置照生产。"""
    orig = executor._cfg
    executor._cfg = lambda: {**orig(), "enable": True}


def patch_surface(sink: CliSink) -> None:
    """换出口/记录，返回原 send_proactive 之前的 kernel._report 包装也已挂。"""
    import junjun_agent.outbound as outbound
    import junjun_core.security as sec
    from junjun_agent.tasks import task_manager

    outbound.send_proactive = sink.on_send
    sec.notify_admin = sink.on_notify_admin
    task_manager._record_outcome = lambda *a, **kw: None


def capture_terminal(kernel, sink: CliSink) -> None:
    """终态信号：_report 是双引擎（legacy/langgraph）唯一的共同终态必经点。"""
    orig_report = kernel._report

    async def _capture(plan):
        if plan.chat_id == CLI_CHAT_ID:
            sink.completed[plan.plan_id] = plan
        await orig_report(plan)

    kernel._report = _capture


async def _ask_approval(plan_id: str, info: dict, runner) -> None:
    """终端交互审批：与生产「发/算了」同语义，超时由内核看门狗兜底。"""
    desc = info.get("desc", "")
    try:
        ans = await asyncio.to_thread(
            input, f"[审批] 步骤「{desc[:40]}」——回「发」放行，「算了」跳过: ")
    except (EOFError, KeyboardInterrupt):
        ans = "算了"
    await runner.resume(plan_id, ans.strip() == "发")


async def run_order(text: str, *, user_id: str, timeout: float) -> int:
    """下一单并等到终态。返回 0=done 1=failed/拒收/超时。"""
    from junjun_agent.task_kernel import executor
    from junjun_agent.task_kernel.graph import runner

    sink = CliSink()
    patch_surface(sink)
    capture_terminal(executor.kernel, sink)

    ack = await executor.kernel.try_submit(
        text, chat_id=CLI_CHAT_ID, user_id=user_id)
    if ack is None:
        print("[CLI] 内核未接单（enable 被关了？force_enable 应该已顶住）")
        return 1
    print(f"[君君] {ack}")

    deadline = time.time() + timeout
    seen_status: dict = {}
    while time.time() < deadline:
        if sink.completed:
            plan = next(iter(sink.completed.values()))
            return 0 if plan.state == "done" else 1
        if sink.rejected and not any(
                p.chat_id == CLI_CHAT_ID for p in executor.kernel._plans.values()):
            return 1   # 交代话术已打印，等下去是白等
        # 人审挂起：终端问管理员
        for plan_id, info in list(runner.pending_approvals.items()):
            await _ask_approval(plan_id, info, runner)
        # 步骤进度（状态迁移才打印，不刷屏）
        for p in executor.kernel._plans.values():
            if p.chat_id != CLI_CHAT_ID:
                continue
            for s in p.steps:
                key = (p.plan_id, s.id)
                if seen_status.get(key) != s.status:
                    seen_status[key] = s.status
                    if s.status in ("done", "failed"):
                        print(f"  [步骤 {s.id} {s.status}] {s.desc[:50]}"
                              + (f"  -- {s.error[:60]}" if s.error else ""))
        await asyncio.sleep(0.5)
    print(f"[CLI] 超时（{timeout:.0f}s 未到终态）")
    return 1


async def _main(args) -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from junjun_core import get_global_config, initialize_logging
    initialize_logging("INFO", log_name="cli")   # 独立文件，不碰 bot/adapter.log
    nickname = get_global_config().bot.nickname

    from junjun_skills.registry import load_builtin
    from junjun_skills.plugin_loader import load_plugins
    load_builtin()
    load_plugins()

    from junjun_agent.task_kernel import executor
    force_enable(executor)

    # LangGraph 引擎：短命脚本必须换 MemorySaver（aiosqlite worker 非守护线程
    # 会卡死解释器退出）；不配 persistence——CLI 不做崩溃续跑，也不许碰
    # 生产的 data/task_kernel 注册表。
    if executor.engine() == "langgraph":
        from langgraph.checkpoint.memory import MemorySaver
        from junjun_agent.task_kernel import graph as tk_graph
        tk_graph.runner._graph = tk_graph.build_graph(MemorySaver())

    # 提醒/记忆类工具的副作用是真的（和生产同库，WAL 多进程安全）
    try:
        from junjun_core.database import init_database, db_writer
        init_database()
        db_writer.start()
    except Exception as e:
        print(f"[CLI] 数据库初始化失败（依赖 DB 的工具会失败）: {e}")

    from junjun_core.security import get_admin_id
    user_id = args.user or get_admin_id()
    if not user_id:
        print("[CLI] 未配置 ADMIN_QQ 且没给 --user，按非管理员跑（run_code 会走人审门）")

    try:
        if args.text:
            return await run_order(" ".join(args.text), user_id=user_id,
                                   timeout=args.timeout)
        # REPL
        print(f"{nickname} CLI（TaskKernel 直连，/exit 退出）")
        while True:
            try:
                line = await asyncio.to_thread(input, "下单> ")
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line or line in ("/exit", "/quit"):
                break
            await run_order(line, user_id=user_id, timeout=args.timeout)
        return 0
    finally:
        try:
            from junjun_core.database import db_writer
            await db_writer.stop()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="本地终端直接对 TaskKernel 下单")
    ap.add_argument("text", nargs="*", help="任务描述（空则进 REPL）")
    ap.add_argument("--user", default="", help="发起者 QQ（默认 ADMIN_QQ）")
    ap.add_argument("--timeout", type=float, default=1800,
                    help="整体看门狗秒数（默认 1800，对齐生产 deadline_minutes=30）")
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(_main(args)))
    except KeyboardInterrupt:
        sys.exit(130)
