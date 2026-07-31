"""订阅系统验收脚本：列出全部订阅 + 干跑检查器（不推送、不改状态）。

用法：
    .venv\\Scripts\\python.exe scripts\\check_subscriptions.py          # 列出订阅
    .venv\\Scripts\\python.exe scripts\\check_subscriptions.py --dry    # 干跑：以 last_seen=0 试拉，验证抓取链路
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    from junjun_core.database.models import init_database, Subscription
    init_database()

    subs = list(Subscription.select())
    if not subs:
        print("还没有任何订阅。在群里/私聊对君君说「帮我盯着 P 站作者 xxx」试试。")
        return

    dry = "--dry" in sys.argv
    print(f"共 {len(subs)} 条订阅：")
    for s in subs:
        state = "启用" if s.enabled else "已取消"
        print(f"  #{s.id} [{state}] {s.kind}:{s.target_id}"
              f"（{s.target_name or '名字待回填'}）-> {s.chat_id}"
              f"  last_seen={s.last_seen or '(空)'}")

    if not dry:
        print("\n加 --dry 参数可干跑检查器，验证抓取链路是否通。")
        return

    from junjun_skills.plugins.subscription.tools import _CHECKERS

    async def _dry():
        for s in subs:
            if not s.enabled:
                continue
            checker = _CHECKERS.get(s.kind)
            if not checker:
                print(f"  #{s.id} 未知 kind {s.kind}")
                continue
            # last_seen=0 干跑：能拉出内容就说明抓取链路通（不推送、不落库）
            fake = type("Sub", (), {"target_id": s.target_id, "last_seen": "0"})()
            try:
                items, name = await checker(fake)
                head = items[-1]["title"] if items else "(无内容)"
                print(f"  #{s.id} 抓取 OK：显示名「{name or '?'}」，"
                      f"可拉取 {len(items)} 条，最新《{head}》")
            except Exception as e:
                print(f"  #{s.id} 抓取失败: {type(e).__name__}: {e}")

    asyncio.run(_dry())


if __name__ == "__main__":
    main()
