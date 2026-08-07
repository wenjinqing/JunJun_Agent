"""从迁移日备份恢复用户数据（2026-08-06 DB 体检发现迁移漏迁）。

旧机备份 data/backup_20260805_full/junjun.db 里有 2 条启用中的订阅
（pixiv_author 16689973、bili_up 1482451696，30 分钟轮询推送到群里），
迁移后新库从空开始——订阅者两天没收到更新了。remindertasks 备份里
全是已完成的历史行，无需恢复。

用法（建议先停 bot）：
    uv run python scripts/restore_from_backup.py            # 预演
    uv run python scripts/restore_from_backup.py --apply    # 恢复订阅
    uv run python scripts/restore_from_backup.py --apply --images   # 连带识图缓存
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "data" / "backup_20260805_full" / "junjun.db"
DB = ROOT / "data" / "junjun.db"

_SUB_COLS = ["bot_id", "kind", "target_id", "target_name", "chat_id", "user_id",
             "user_nickname", "last_seen", "interval_minutes", "enabled",
             "created_at", "last_checked"]
_IMG_COLS = ["bot_id", "image_hash", "description", "timestamp"]


def main() -> int:
    apply = "--apply" in sys.argv
    with_images = "--images" in sys.argv
    print(f"[{'APPLY' if apply else 'DRY-RUN'}] {BACKUP.name} -> {DB.name}")
    if not BACKUP.exists():
        print(f"备份不存在: {BACKUP}")
        return 1

    src = sqlite3.connect(f"file:{BACKUP.as_posix()}?mode=ro", uri=True)
    uri = f"file:{DB.as_posix()}" if apply else f"file:{DB.as_posix()}?mode=ro"
    dst = sqlite3.connect(uri, uri=True)

    subs = src.execute(
        f"SELECT {','.join(_SUB_COLS)} FROM subscription WHERE enabled=1").fetchall()
    print(f"启用中的订阅: {len(subs)}")
    for s in subs:
        print(f"   {s[1]} {s[2]}({s[3]}) -> {s[4]} 每{s[8]}min")
    if apply:
        for s in subs:
            exists = dst.execute(
                "SELECT 1 FROM subscription WHERE kind=? AND target_id=? AND chat_id=?",
                (s[1], s[2], s[4])).fetchone()
            if exists:
                print(f"   跳过（已存在）: {s[1]} {s[2]}")
                continue
            dst.execute(
                f"INSERT INTO subscription ({','.join(_SUB_COLS)}) "
                f"VALUES ({','.join('?' * len(_SUB_COLS))})", s)
        dst.commit()
        print("订阅已恢复（last_seen 保留，不会补推旧内容）。")

    if with_images:
        rows = src.execute(
            f"SELECT {','.join(_IMG_COLS)} FROM images").fetchall()
        # 备份本身带着旧库的测试污染（2026-08-04 Images 事故：675 行
        # 「一只猫」假识图缓存骑备份回流，2026-08-07 实锤）——按
        # cleanup_test_pollution.py 的同一定义在源头滤掉
        rows = [r for r in rows
                if not (r[2] == "一只猫" and r[3] <= 1786031506.0)]
        have = {r[0] for r in dst.execute("SELECT image_hash FROM images")}
        new = [r for r in rows if r[1] not in have]  # r[1] = image_hash
        print(f"识图缓存: 备份 {len(rows)} 行，可新增 {len(new)} 行")
        if apply and new:
            dst.executemany(
                f"INSERT INTO images ({','.join(_IMG_COLS)}) "
                f"VALUES ({','.join('?' * len(_IMG_COLS))})", new)
            dst.commit()
            print(f"已恢复 {len(new)} 行识图缓存（省 VLM 重识别）。")

    src.close()
    dst.close()
    if not apply:
        print("预演完成——加 --apply 实际恢复（建议先停 bot）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
