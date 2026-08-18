"""小涩猫咖啡厅站点管理插件（2026-08-18 站主接入）。

通过站主开放的管理接口（X-API-Key 认证）让君君兼任网站管理员：
读站点内容、看运营数据、发公告、改 slogan / 作者状态。

安全边界：
- key 只放 .env（CATCAFE_API_KEY），永不入库、永不打印（日志只给错误类别）
- 写操作工具体内硬校验管理员特权——公开发布动作不靠 prompt 自觉
- 更新接口是全量替换：一律先 GET 再改再 PUT（read-modify-write），
  PUT 前校验 tag 白名单与 1MB 上限，返回非 saved 如实说明
"""

import json
import os
import time

import httpx
from langchain_core.tools import tool

_DEFAULT_BASE = "https://alicefans.asia"
_TIMEOUT = 15.0
_MAX_PUT_BYTES = 1024 * 1024  # 接口上限 1MB
_VALID_TAGS = ("公告", "新坑", "更新", "活动")


def _base() -> str:
    return (os.environ.get("CATCAFE_BASE_URL", "").strip() or _DEFAULT_BASE).rstrip("/")


def _key() -> str:
    return os.environ.get("CATCAFE_API_KEY", "").strip()


async def _request(method: str, path: str, raw_body: bytes | None = None):
    """统一请求入口。返回 (dict|None, 错误文本|None)——错误文本可直接回给模型。
    请求体由调用方序列化好（ensure_ascii=False 的 UTF-8），这样 1MB 上限
    校验与实际发送字节一致。"""
    if not _key():
        return None, "站点管理 key 未配置（CATCAFE_API_KEY），跟我说也没法开工。"
    headers = {"X-API-Key": _key()}
    if raw_body is not None:
        headers["Content-Type"] = "application/json"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.request(method, f"{_base()}{path}",
                                        headers=headers, content=raw_body)
    except Exception as e:
        return None, f"连不上站点接口（{type(e).__name__}），稍后再试。"
    if resp.status_code == 401:
        return None, "站点接口说 key 无效（401），得让管理员去核对 key。"
    if resp.status_code != 200:
        return None, f"站点接口返回 HTTP {resp.status_code}，稍后再试。"
    try:
        return resp.json(), None
    except Exception:
        return None, "站点接口返回的不是 JSON，稍后再试。"


def _admin_refusal() -> str | None:
    """写操作管理员门（体内硬校验，与 junzone 同模式）。是管理员返回 None。"""
    from junjun_core.security import is_admin_privileged
    if is_admin_privileged():
        return None
    return "网站发布是公开动作，只有管理员本人能用——这事我可以帮你转达给管理员。"


async def _mutate_content(mutate, action_desc: str) -> str:
    """read-modify-write 骨架：GET -> mutate(data) -> 校验 -> PUT -> 确认 saved。
    mutate 返回错误文本则中止（不发 PUT）。"""
    data, err = await _request("GET", "/api/agent/content")
    if err:
        return err
    err = mutate(data)
    if err:
        return err
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if len(body) > _MAX_PUT_BYTES:
        return (f"改动后整体内容超接口 1MB 上限（{len(body)} 字节），"
                f"本次{action_desc}没做成。")
    result, err = await _request("PUT", "/api/agent/content", body)
    if err:
        return err
    if isinstance(result, dict) and result.get("status") == "saved":
        return f"{action_desc}完成，访客刷新前台就能看到。"
    return (f"{action_desc}的返回和预期不一样（{json.dumps(result, ensure_ascii=False)[:120]}），"
            f"建议让我重新读一遍站点确认状态。")


def _fmt_content(data: dict) -> str:
    lines = [f"站点：{data.get('title', '?')}（作者：{data.get('author', '?')}）"]
    if data.get("slogan"):
        lines.append(f"slogan：{data['slogan']}")
    if data.get("authorStatus"):
        lines.append(f"作者状态：{data['authorStatus']}")
    notices = data.get("notices") or []
    lines.append(f"公告栏 {len(notices)} 条（最新 {min(5, len(notices))} 条）：")
    for n in notices[:5]:
        lines.append(f"- [{n.get('date', '?')}] {n.get('tag', '?')}：{str(n.get('text', ''))[:60]}")
    novels = data.get("novels") or []
    if novels:
        lines.append(f"小说 {len(novels)} 部：" + "、".join(str(x.get("title", "?")) for x in novels[:10]))
    gallery = data.get("gallery") or []
    if gallery:
        lines.append(f"画廊 {len(gallery)} 张图")
    fc = data.get("fanClub") or {}
    if fc.get("name"):
        lines.append(f"后援会：{fc['name']}")
    return "\n".join(lines)


def _fmt_stats(data: dict) -> str:
    lines = ["累计数据："]
    for k, label in (("visits", "访问"), ("pets", "撸猫"), ("feeds", "投喂"),
                     ("urges", "催更"), ("messages", "留言"), ("comments", "评论"),
                     ("postcards", "明信片"), ("likes", "点赞"), ("pigmis", "猪咪")):
        if k in data:
            lines.append(f"- {label}：{data[k]}")
    day = data.get("day") or {}
    if day:
        parts = [f"{k}={v}" for k, v in day.items() if k != "date"]
        lines.append(f"今日（{day.get('date', '?')}）：" + "、".join(parts))
    if data.get("topPost"):
        tp = data["topPost"]
        lines.append(f"最热作品：{tp.get('nick', '?')}（{tp.get('likes', 0)} 赞）")
    if data.get("topPigmi"):
        tm = data["topPigmi"]
        lines.append(f"最活跃猪咪：{tm.get('nick', '?')}（{tm.get('points', 0)} 分）")
    return "\n".join(lines)


@tool
async def catcafe_get_content() -> str:
    """查看「小涩猫咖啡厅」网站当前内容。对方问网站上现在有什么、最新公告、
    小说列表、站点长什么样、咖啡厅怎么样时使用。返回摘要（公告列最新 5 条）。
    区别于 catcafe_get_stats（那是访问/互动运营数据）。"""
    data, err = await _request("GET", "/api/agent/content")
    return err if err else _fmt_content(data)


@tool
async def catcafe_get_stats() -> str:
    """查看「小涩猫咖啡厅」网站的运营数据。对方问网站访问量、撸猫数、留言数、
    今日数据、哪部作品最热、咖啡厅人气怎么样时使用。
    区别于 catcafe_get_content（那是站点的文案内容）。"""
    data, err = await _request("GET", "/api/agent/stats")
    return err if err else _fmt_stats(data)


@tool
async def catcafe_post_notice(text: str, tag: str = "公告") -> str:
    """在「小涩猫咖啡厅」网站的公告栏发一条新公告（挂在最前）。
    管理员（站主）说「发个公告」「网站上通知一下」「挂个更新」「发个新坑预告」时使用。
    tag 只能是：公告 / 新坑 / 更新 / 活动。
    发布成功访客刷新即见；非管理员想发会被拒，照实转达即可。

    Args:
        text: 公告正文（可爱温和的语气，一两句话）
        tag: 公告类型：公告 / 新坑 / 更新 / 活动
    """
    refusal = _admin_refusal()
    if refusal:
        return refusal
    text = (text or "").strip()
    if not text:
        return "公告正文是空的，没法发。"
    if tag not in _VALID_TAGS:
        return f"tag 只能是 {'/'.join(_VALID_TAGS)} 之一，「{tag}」不行。"

    def _prepend(data: dict):
        notices = data.setdefault("notices", [])
        notices.insert(0, {"date": time.strftime("%Y-%m-%d"),
                           "tag": tag, "text": text})
        return None

    return await _mutate_content(_prepend, f"发公告（{tag}）")


@tool
async def catcafe_set_slogan(text: str) -> str:
    """修改「小涩猫咖啡厅」网站顶部的 slogan。管理员（站主）说「 slogan 换成 xxx 」
    「改下网站标语」时使用。只改这一个字段，其余内容原样保留（内部先读后写）；
    非管理员想改会被拒，照实转达即可。

    Args:
        text: 新 slogan
    """
    refusal = _admin_refusal()
    if refusal:
        return refusal
    text = (text or "").strip()
    if not text:
        return "slogan 是空的，没法改。"

    def _set(data: dict):
        data["slogan"] = text
        return None

    return await _mutate_content(_set, "改 slogan")


@tool
async def catcafe_set_status(text: str) -> str:
    """修改「小涩猫咖啡厅」网站的作者状态（如「赶稿中」「休假中」）。
    管理员（站主）说「把作者状态改成 xxx」「状态更新一下」时使用。
    只改这一个字段，其余内容原样保留（内部先读后写）；
    非管理员想改会被拒，照实转达即可。

    Args:
        text: 新状态文字
    """
    refusal = _admin_refusal()
    if refusal:
        return refusal
    text = (text or "").strip()
    if not text:
        return "状态文字是空的，没法改。"

    def _set(data: dict):
        data["authorStatus"] = text
        return None

    return await _mutate_content(_set, "改作者状态")


TOOLS = [catcafe_get_content, catcafe_get_stats, catcafe_post_notice,
         catcafe_set_slogan, catcafe_set_status]
