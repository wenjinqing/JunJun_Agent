# -*- coding: utf-8 -*-
"""ranking.php 原始 JSON 形状确认（顶层无 body 包装）。"""
import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for line in open(".env", encoding="utf-8"):
    if line.startswith("PIXIV_COOKIE="):
        os.environ["PIXIV_COOKIE"] = line.split("=", 1)[1].strip().strip('"').strip("'")

from junjun_skills.plugins.pixiv.client import _headers  # noqa: E402


async def main():
    from curl_cffi.requests import AsyncSession
    from junjun_skills.plugins.pixiv.client import _proxy
    async with AsyncSession(impersonate="chrome", proxy=_proxy() or None) as s:
        for content in ("illust", "manga", "novel"):
            url = f"https://www.pixiv.net/ranking.php?mode=daily&content={content}&p=1&format=json"
            resp = await s.get(url, headers=_headers("https://www.pixiv.net/ranking.php"), timeout=30)
            print(f"[{content}] HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"  顶层 keys={list(data.keys())[:10]}")
                contents = data.get("contents") or []
                print(f"  条数={len(contents)} date={data.get('date')}")
                if contents:
                    c = contents[0]
                    print(f"  首条字段={list(c.keys())[:16]}")
                    print(f"  rank={c.get('rank')} id={c.get('illust_id') or c.get('id')} "
                          f"title={c.get('title')!r} user={c.get('user_name')!r} "
                          f"url={str(c.get('url'))[:55]}")


asyncio.run(main())
