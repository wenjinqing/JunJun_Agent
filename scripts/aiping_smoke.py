"""AI Ping 四条新腿连通性冒烟（2026-08-12）。

直连 httpx 原始 POST（不过 LangChain），看最原始的返回结构：
1. ***REMOVED*** 文本（utils/utils_small 主力、gate/agent_light 备腿）
2. ***REMOVED*** 带双关思考字段（agent 备2）——验证两个 extra_body 字段都被网关接受，
   并检查是否真的没烧 reasoning tokens
3. ***REMOVED*** 纯文本（vlm 主力）
4. ***REMOVED*** 带 data-url 图片——验证视觉消息格式在该网关可用
结果写 JSON 文件，避免控制台 GBK 中文乱码。
"""

import base64
import io
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values  # noqa: E402

cfg = dotenv_values(ROOT / ".env")
BASE = cfg["AIPING_BASE_URL"].rstrip("/")
KEY = cfg["AIPING_API_KEY"]
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
OUT = Path(os.environ.get("AIPING_SMOKE_OUT", ROOT / "data" / "aiping_smoke.json"))


def _png_b64(rgb: tuple) -> str:
    """生成 8x8 纯色 PNG data-url（不引第三方库，手写最小 PNG）。"""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    w = h = 8
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def post(payload: dict) -> dict:
    try:
        r = httpx.post(f"{BASE}/chat/completions", headers=HEADERS,
                       json=payload, timeout=60.0)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300]
        return {"http": r.status_code, "body": body if r.status_code != 200 else {
            "content": (body["choices"][0]["message"].get("content") or "")[:120],
            "reasoning_present": bool(body["choices"][0]["message"].get("reasoning_content")),
            "finish": body["choices"][0].get("finish_reason"),
            "usage": body.get("usage"),
            "model_returned": body.get("model"),
        }}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    results = {}

    results["flash_text"] = post({
        "model": cfg["AIPING_FLASH_MODEL"],
        "messages": [{"role": "user", "content": "用三个中文词形容晴天"}],
        "max_tokens": 32, "temperature": 0.1,
    })

    results["glm52_agent_leg"] = post({
        "model": cfg["AIPING_GLM_MODEL"],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32, "temperature": 0.1,
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    })

    results["vlm_text"] = post({
        "model": cfg["AIPING_VLM_MODEL"],
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 16, "temperature": 0.1,
    })

    results["vlm_vision"] = post({
        "model": cfg["AIPING_VLM_MODEL"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _png_b64((255, 0, 0))}},
            {"type": "text", "text": "这张图是什么颜色？用一个词回答"},
        ]}],
        "max_tokens": 32, "temperature": 0.1,
    })

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = all(v.get("http") == 200 for v in results.values())
    print("ALL_OK" if ok else "HAS_FAILURE", "->", OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
