"""清理第三次测试污染事故的存量垃圾（2026-08-06 DB 体检实锤）。

污染源（已在 b76cf87 修堵，本脚本只清存量）：
1. messages：test_processor/test_queue_predecision 写入的测试会话行
   （qq:999:group / qq:12345:private / qq:3:group，共 167 行）
2. intimacy：测试带入的 user_id=111 行
3. images：test_vision_prewarm 写入的「一只猫」假识图缓存（90 行，
   全部时间戳落在 pytest 运行簇内，已逐簇核对无真实图片）
4. data/tool_health.json：flaky_async/flaky_sync/noisy 等测试工具的
   降级状态（bot 启动会读到不存在的工具）
5. data/tool_failures.jsonl：全部 189 行均为测试工具名

用法（建议先停 bot）：
    uv run python scripts/cleanup_test_pollution.py          # 预演，只打印
    uv run python scripts/cleanup_test_pollution.py --apply  # 实际清理
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "junjun.db"
HEALTH = ROOT / "data" / "tool_health.json"
FAILURES = ROOT / "data" / "tool_failures.jsonl"

TEST_CHATS = ("qq:999:group", "qq:12345:private", "qq:3:group")
JUNK_TOOLS = ("exploding_tool", "flaky_tool", "flaky_async", "flaky_sync", "noisy")
# 截止纪元：脚本生成时刻（2026-08-06 23:58）。此后到达的真实猫图不受影响。
CAT_CUTOFF_TS = 1786031506.0


def main() -> int:
    apply = "--apply" in sys.argv
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] DB: {DB}")

    uri = f"file:{DB.as_posix()}" if apply else f"file:{DB.as_posix()}?mode=ro"
    # 不带 mode=ro 时路径不存在会静默创建空库文件——先查存在性，防路径笔误
    if apply and not DB.exists():
        print(f"错误：DB 不存在（--apply 拒绝创建新库）: {DB}")
        return 1
    con = sqlite3.connect(uri, uri=True)

    # 1. messages 测试会话行
    n = con.execute(
        f"SELECT COUNT(*) FROM messages WHERE chat_id IN "
        f"({','.join('?' * len(TEST_CHATS))})", TEST_CHATS).fetchone()[0]
    print(f"messages 测试会话行: {n}")
    if apply and n:
        con.execute(
            f"DELETE FROM messages WHERE chat_id IN "
            f"({','.join('?' * len(TEST_CHATS))})", TEST_CHATS)

    # 2. intimacy user_id=111
    n = con.execute("SELECT COUNT(*) FROM intimacy WHERE user_id='111'").fetchone()[0]
    print(f"intimacy user_id=111 行: {n}")
    if apply and n:
        con.execute("DELETE FROM intimacy WHERE user_id='111'")

    # 3. images 一只猫测试缓存（时间簇核对过，cutoff 前的全部是测试行）
    n = con.execute(
        "SELECT COUNT(*) FROM images WHERE description='一只猫' AND timestamp <= ?",
        (CAT_CUTOFF_TS,)).fetchone()[0]
    print(f"images「一只猫」测试行: {n}")
    if apply and n:
        con.execute(
            "DELETE FROM images WHERE description='一只猫' AND timestamp <= ?",
            (CAT_CUTOFF_TS,))

    if apply:
        con.commit()
    con.close()

    # 4. tool_health.json 测试工具降级状态
    if HEALTH.exists():
        data = json.loads(HEALTH.read_text(encoding="utf-8"))
        junk = [k for k in data if k in JUNK_TOOLS]
        print(f"tool_health.json 测试工具键: {len(junk)} {junk}")
        if apply and junk:
            for k in junk:
                data.pop(k, None)
            HEALTH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # 5. tool_failures.jsonl 测试工具行
    if FAILURES.exists():
        lines = FAILURES.read_text(encoding="utf-8").splitlines()
        keep, drop = [], 0
        for l in lines:
            try:
                if json.loads(l).get("tool") in JUNK_TOOLS:
                    drop += 1
                    continue
            except Exception:
                pass
            keep.append(l)
        print(f"tool_failures.jsonl 测试行: {drop}（保留 {len(keep)}）")
        if apply and drop:
            FAILURES.write_text("\n".join(keep) + ("\n" if keep else ""),
                                encoding="utf-8")

    print("完成。" if apply else "预演完成——加 --apply 实际清理（建议先停 bot）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
