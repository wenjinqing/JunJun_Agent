"""生产库每日备份（2026-08-13 审查 P0）。

为什么存在：junjun.db 曾零自动备份，唯一快照和生产库在同一块盘上；
且 WAL 模式下只复制 .db 文件会丢未 checkpoint 的最近写入（审查时
junjun.db-wal 已 4.2MB 超过主库）——必须用 sqlite3 的 .backup API。
画像/好感度/订阅/提醒/屏蔽名单/自我叙事全在这一个文件里，丢了不可重建。

用法：
    uv run python scripts/backup_db.py                  # 备份到默认目录
    set BACKUP_DIR=D:\\junjun_backups && uv run ...     # 异盘更好（推荐）

默认目录：仓库旁 ../JunJun_backups（至少和仓库不同目录；异盘/网盘同步目录更稳）。
保留最近 7 代。同时打包 data/memory/（长期记忆 json，同一批不可重建资产）。

Windows 每日定时（用户自行注册，一条命令）：
    schtasks /create /tn "junjun-backup" /tr "\"C:\\路径\\到\\uv.exe\" run python scripts\\backup_db.py" /sc daily /st 04:30 /f
（在仓库目录下运行；或用完整 python 路径）
"""

import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SRC = REPO / "data" / "junjun.db"
MEMORY_DIR = REPO / "data" / "memory"
DEFAULT_DEST = (REPO.parent / "JunJun_backups").resolve()
KEEP = 7

from junjun_core.observability import get_logger  # noqa: E402  GBK 安全输出

logger = get_logger("scripts.backup_db")


def _backup_db(src: Path, dest_dir: Path, keep: int = KEEP) -> Path:
    """sqlite3 .backup API 复制（WAL 安全，含未 checkpoint 数据）+ 完整性校验。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"junjun_{ts}.db"
    src_conn = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    try:
        dest_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dest_conn)
            ok = dest_conn.execute("PRAGMA integrity_check").fetchone()[0]
            if ok != "ok":
                raise RuntimeError(f"备份完整性校验失败: {ok}")
        finally:
            dest_conn.close()
    finally:
        src_conn.close()
    _rotate(dest_dir, "junjun_*.db", keep)
    return dest


def _backup_memory(dest_dir: Path, keep: int = KEEP) -> Path | None:
    """data/memory/ 整树打包 zip（长期记忆/日记等 json 资产）。"""
    if not MEMORY_DIR.is_dir():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = shutil.make_archive(str(dest_dir / f"memory_{ts}"), "zip", MEMORY_DIR)
    _rotate(dest_dir, "memory_*.zip", keep)
    return Path(out)


def _rotate(dest_dir: Path, pattern: str, keep: int) -> None:
    """按文件名时间序保留最新 keep 份，其余删。"""
    files = sorted(dest_dir.glob(pattern))
    for old in files[:-keep] if len(files) > keep else []:
        old.unlink()
        logger.info(f"备份轮换删除: {old.name}")


def backup(src: Path = SRC, dest_dir: Path | None = None, keep: int = KEEP) -> Path:
    dest_dir = dest_dir or Path(os.environ.get("BACKUP_DIR") or DEFAULT_DEST)
    if not src.is_file():
        raise FileNotFoundError(f"生产库不存在: {src}")
    dest = _backup_db(src, dest_dir, keep)
    logger.info(f"库备份完成: {dest}（{dest.stat().st_size // 1024}KB）")
    mem = _backup_memory(dest_dir, keep)
    if mem:
        logger.info(f"记忆打包完成: {mem}（{mem.stat().st_size // 1024}KB）")
    if dest_dir.resolve().drive == src.resolve().drive:
        logger.warning("备份与生产库在同一磁盘——盘坏两者俱失，"
                       "建议 BACKUP_DIR 指到异盘或网盘同步目录")
    return dest


if __name__ == "__main__":
    try:
        backup()
    except Exception as e:
        logger.error(f"备份失败: {type(e).__name__}: {e}")
        sys.exit(1)
