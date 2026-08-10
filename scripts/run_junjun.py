"""君君 AGENT 统一入口。

职责：
1. 固定工作目录到仓库根。
2. 加载 .env 环境变量。
3. 初始化日志与配置。
4. 启动消息网关。
5. 优雅关闭。
"""

import asyncio
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_env() -> None:
    from dotenv import load_dotenv

    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(str(env_path), override=True)
    else:
        print(f"[warn] 未找到 .env，可参考 .env.example 创建：{env_path}")


async def _run() -> int:
    from junjun_core import get_logger, initialize_logging, get_global_config, get_router

    initialize_logging()
    logger = get_logger("main")

    try:
        cfg = get_global_config()
        logger.info("=" * 60)
        logger.info("启动君君 AGENT (JunJun_Agent)")
        logger.info(f"昵称: {cfg.bot.nickname}  平台: {cfg.bot.platform}")
        logger.info(f"工作目录: {ROOT}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1

    router = get_router()

    # 后台任务结局落盘 + 恢复（重启前的「在画了」不再是无头承诺）
    try:
        from junjun_agent.tasks import enable_persistence
        enable_persistence(ROOT / "data" / "task_outcomes.jsonl")
    except Exception as e:
        logger.warning(f"任务结局持久化挂接失败（忽略）: {e}")

    # 复杂任务（TaskKernel）计划落盘 + 中断标记（重启不续跑，照实登记结局）
    try:
        from junjun_agent.task_kernel import enable_persistence as tk_persistence
        tk_persistence(ROOT / "data" / "task_kernel")
    except Exception as e:
        logger.warning(f"任务内核持久化挂接失败（忽略）: {e}")

    # LangGraph 引擎（[task_kernel] engine=langgraph 时）：崩溃任务断点续跑
    try:
        from junjun_agent.task_kernel.graph import runner as tk_runner
        asyncio.create_task(tk_runner.recover(), name="task-kernel-recover")
    except Exception as e:
        logger.warning(f"任务内核断点恢复挂接失败（忽略）: {e}")

    # 深度研究 LangGraph 引擎（[deep_research] engine=langgraph 时）：
    # AsyncJob 是内存任务，进程重启整单丢——研究图把中间态落 sqlite 断点续跑
    try:
        from junjun_skills.plugins.async_task import research_graph
        research_graph.configure(ROOT / "data" / "task_kernel")
        asyncio.create_task(research_graph.recover(), name="research-recover")
    except Exception as e:
        logger.warning(f"深度研究断点恢复挂接失败（忽略）: {e}")

    # 数据库建表 + 写队列
    try:
        from junjun_core.database import init_database, db_writer
        init_database()
        db_writer.start()
        logger.info("数据库已就绪 (data/junjun.db, WAL)")
    except Exception as e:
        logger.error(f"数据库初始化失败（继续运行，不落库）: {e}")

    # 注入决策漏斗 processor（失败则保持 echo 占位，便于排障）
    try:
        from junjun_agent import junjun_processor
        router.set_processor(junjun_processor)
    except Exception as e:
        logger.error(f"Agent processor 注入失败，回退 echo 模式: {e}")

    # skill 插件（vrchat/tts 等静态扫描）+ MCP 客户端工具
    try:
        from junjun_skills.registry import load_builtin
        from junjun_skills.plugin_loader import load_plugins
        load_builtin()
        load_plugins()
    except Exception as e:
        logger.error(f"插件加载失败（内置 skill 不受影响）: {e}")
    # MCP 客户端：后台并发连接，不阻塞 Agent 启动（2026-07-29 用户反馈启动等待）
    # 连上后工具自动注册进 registry——注册前创建的会话用内置工具，
    # 之后新建的会话自动带 MCP 工具（会话 Agent 构建时快照工具列表）。
    async def _mcp_bootstrap():
        try:
            from junjun_mcp_client.client import mcp_manager
            n = await mcp_manager.start()
            if n:
                mcp_manager.register_all()
                logger.info(f"MCP 工具已注入 registry: {n} 个")
        except Exception as e:
            logger.error(f"MCP 客户端启动失败（降级无 MCP）: {e}")
    asyncio.create_task(_mcp_bootstrap())

    # 定时任务：记忆遗忘 + 摘要兜底
    try:
        from junjun_agent.loop import scheduler, register_default_tasks
        register_default_tasks()
        scheduler.start()
    except Exception as e:
        logger.error(f"调度器启动失败（继续运行）: {e}")

    # WebUI（WEBUI_ENABLED=true 时同进程启动）
    webui_task = None
    try:
        from junjun_webui.server import start_webui
        webui_task = await start_webui()
    except Exception as e:
        logger.error(f"WebUI 启动失败（继续运行）: {e}")

    await router.start()
    logger.info("君君网关运行中，等待 Adapter 消息（Ctrl+C 退出）")

    stop_event = asyncio.Event()

    def _on_signal(*_):
        logger.info("收到退出信号，开始优雅关闭...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    await stop_event.wait()

    # 优雅关闭：WebUI -> 调度器 -> 会话队列 -> DB 写队列 -> 网关
    try:
        if webui_task is not None:
            webui_task.cancel()
        from junjun_agent.loop import scheduler
        await scheduler.stop()
        from junjun_agent.tasks import task_manager
        await task_manager.shutdown()
        # 长期记忆批量落盘后的收尾：关停前把脏数据写出去
        try:
            from junjun_memory.long_term import get_long_term_memory
            get_long_term_memory().flush()
        except Exception:
            pass
        from junjun_agent.funnel.session_queue import session_queues
        await session_queues.stop_all()
        from junjun_core.database import db_writer
        await db_writer.stop()
    except Exception:
        pass
    await router.stop()
    logger.info("君君已关闭")
    return 0


def main() -> None:
    _setup_env()
    try:
        exit_code = asyncio.run(_run())
    except KeyboardInterrupt:
        exit_code = 0
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
