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
        logger.info("启动君君 AGENT (JunJun_Agent) [阶段 2 Agent 最小可用]")
        logger.info(f"昵称: {cfg.bot.nickname}  平台: {cfg.bot.platform}")
        logger.info(f"工作目录: {ROOT}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1

    router = get_router()

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

    # 优雅关闭：会话队列 -> DB 写队列 -> 网关
    try:
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
