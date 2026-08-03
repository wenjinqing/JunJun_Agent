"""实测：P 站图代下 base64 -> NapCat 发给君君自己（2026-08-03）。

背景：NapCat 直连图床（i.pixiv.re/i.pximg.net）Connect Timeout（被墙无代理），
发图必须本侧（有代理）下载转 base64 交给 NapCat。本脚本端到端验证：
  1) 官方 API 拿今日榜第一张图的地址
  2) fetch_image_b64 本侧代下（代理 + Referer）
  3) NapCat send_private_msg 发给 bot 自己（base64:// 载荷）
  4) 对照组（--url）：直接发 URL 给 NapCat，复现超时（约 10s）

用法：uv run scripts/probe_pixiv_image_send.py [--url]
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _target_qq() -> str:
    """发送目标：实测 NapCat 不能给 bot 自己发私信（NTQQ 拒绝），发管理员。

    管理员 = .env ADMIN_QQ（君君的主人，实测接收方）。
    """
    return os.environ.get("ADMIN_QQ", "").strip()


async def main() -> None:
    _load_env()
    from junjun_core import napcat_client
    from junjun_skills.plugins.pixiv import client, illust

    if not napcat_client.available():
        print("❌ NAPCAT_HTTP_BASE 未配置")
        return
    target = _target_qq()
    if not target:
        print("❌ .env 里没有 ADMIN_QQ")
        return
    print(f"发送目标（管理员）: {target}  NapCat: {napcat_client._base()}")

    print("\n[1/3] 官方 API 取今日榜榜首图...")
    items = await illust._ranking("daily", "illust")
    if not items:
        print("❌ 排行榜抓取失败")
        return
    iid = items[0]["id"]
    urls = await illust._illust_page_urls(iid, 1)
    print(f"  「{items[0]['title']}」id={iid}  url={urls[0][:80]}...")

    print("\n[2/3] 本侧代下（代理+Referer）转 base64...")
    b64 = await client.fetch_image_b64(urls[0])
    if not b64:
        print("❌ 代下失败")
        return
    size_kb = (len(b64) - 9) * 3 // 4 // 1024
    print(f"  ✅ {size_kb} KB")

    print("\n[3/3] NapCat 发给管理员（base64 载荷）...")
    ret = await napcat_client.call("send_private_msg", {
        "user_id": int(target),
        "message": [{"type": "text", "data": {"text": "【自检】P站图代下 base64 发送测试"}},
                    {"type": "image", "data": {"file": b64}}],
    }, timeout=60.0)
    print(f"  {'✅ 发送成功 message_id=' + str(ret.get('message_id')) if ret else '❌ 发送失败'}")

    if "--url" in sys.argv:
        print("\n[对照] 直接发 URL 给 NapCat（预期超时复现）...")
        ret2 = await napcat_client.call("send_private_msg", {
            "user_id": int(target),
            "message": [{"type": "image", "data": {"file": urls[0]}}],
        }, timeout=30.0)
        print(f"  {'意外成功？' if ret2 else '✅ 如预期失败（NapCat 拉不到图床）'}")


if __name__ == "__main__":
    asyncio.run(main())
