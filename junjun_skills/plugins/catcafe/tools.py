"""小涩猫咖啡厅站点管理插件（2026-08-18 站主接入）。

通过站主开放的管理接口（X-API-Key 认证）让君君兼任网站管理员：
读站点内容、看运营数据、发公告、改 slogan / 作者状态，
并以「猪咪君君」身份照看留言板（列表 / 回复 / 删除不当留言）。

安全边界：
- key 只放 .env（CATCAFE_API_KEY），永不入库、永不打印（日志只给错误类别）
- 写操作工具体内硬校验管理员特权——公开发布动作不靠 prompt 自觉
- 更新接口是全量替换：一律先 GET 再改再 PUT（read-modify-write），
  PUT 前校验 tag 白名单与 1MB 上限，返回非 saved 如实说明
- 留言 reply/delete 无 id 字段，靠 nick+time+content 精确三元组定位：
  工具只收列表编号，内部重新拉列表取原文再 POST，不让模型转抄长文本；
  空回复 = 清除已有回复（探测事故实锤），回复文本硬校验非空
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
        hint = ""
        try:
            hint = str(resp.json().get("error", "")).strip()
        except Exception:
            pass
        tail = "，稍后再试。" if resp.status_code >= 500 else "。"
        return None, f"站点接口返回 HTTP {resp.status_code}{f'（{hint}）' if hint else ''}{tail}"
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


# ---------------- 留言板（猪咪君君看店职责） ----------------
# 接口无 id 字段：reply/delete 靠 nick+time+content 精确三元组定位留言。
# 因此工具只收「列表编号」，内部重新 GET 列表取回原文再 POST——
# 不让模型转手长文本（转抄错一个字符就是 404，截断更是直接没救）。
# 已实测的接口语义：reply 字段缺省/为空 = 清除该条已有回复（2026-08-18 探测事故），
# 所以回复文本在工具内硬校验非空，空串一律拒。

async def _fetch_messages():
    """拉留言列表（最新在前）。返回 (list|None, 错误文本|None)。"""
    data, err = await _request("GET", "/api/messages")
    if err:
        return None, err
    if not isinstance(data, list):
        return None, "留言列表格式和预期不一样，稍后再试。"
    return data, None


def _pick_message(msgs: list, index: int, expect_nick: str = ""):
    """按编号取留言，可选复核昵称。返回 (msg|None, 错误文本|None)。"""
    if not msgs:
        return None, "留言板是空的，没有什么可操作的。"
    if index < 0 or index >= len(msgs):
        return (None, f"编号 #{index} 不存在（当前共 {len(msgs)} 条，编号 0~{len(msgs) - 1}）。"
                      f"先用 catcafe_list_messages 看最新列表再操作。")
    msg = msgs[index]
    if expect_nick and msg.get("nick") != expect_nick:
        return (None, f"编号 #{index} 现在的留言者是「{msg.get('nick')}」，"
                      f"和预期的「{expect_nick}」对不上——列表可能变了，没动手。"
                      f"先重新 list 再操作。")
    return msg, None


def _msg_ref(msg: dict) -> dict:
    """接口定位三元组。"""
    return {"nick": msg.get("nick", ""), "time": msg.get("time", ""),
            "content": msg.get("content", "")}


@tool
async def catcafe_list_messages() -> str:
    """查看「小涩猫咖啡厅」网站的访客留言板。对方问网站上有没有新留言、
    留言板近况、谁留了什么话、有没有要处理的留言时使用。
    返回带编号的列表（#0 为最新）；之后用 catcafe_reply_message 回复、
    catcafe_delete_message 删除，都按这里的编号来。"""
    msgs, err = await _fetch_messages()
    if err:
        return err
    if not msgs:
        return "留言板暂时是空的。"
    lines = [f"留言板共 {len(msgs)} 条（#0 最新）："]
    for i, m in enumerate(msgs):
        content = str(m.get("content", "")).replace("\n", " ")
        preview = content[:80] + ("…" if len(content) > 80 else "")
        line = f"#{i} [{m.get('time', '?')}] {m.get('nick', '?')}：{preview}"
        if m.get("reply"):
            r = str(m["reply"]).replace("\n", " ")
            line += f"\n   ↳ 已回复（{m.get('replyBy', '?')}）：{r[:60]}{'…' if len(r) > 60 else ''}"
        lines.append(line)
    return "\n".join(lines)


@tool
async def catcafe_reply_message(index: int, text: str) -> str:
    """以「猪咪君君」的身份回复「小涩猫咖啡厅」网站上的一条留言，回复公开显示在留言下方。
    站主让你回复留言、照看留言板时使用；index 是 catcafe_list_messages
    给的编号（#0 最新）。回复语气可爱温和，自称「猪咪君君」；若有留言问店长的
    更新计划、私生活，那条就回「店长在赶稿，帮你转达~」，不要代答。

    Args:
        index: 留言编号（来自 catcafe_list_messages，#0 为最新一条）
        text: 回复内容（不能为空——接口上空内容等于清掉该条已有回复）
    """
    refusal = _admin_refusal()
    if refusal:
        return refusal
    text = (text or "").strip()
    if not text:
        return "回复内容不能为空（接口语义上空内容 = 清除已有回复），重写一条再来。"
    msgs, err = await _fetch_messages()
    if err:
        return err
    msg, err = _pick_message(msgs, index)
    if err:
        return err
    payload = _msg_ref(msg)
    payload["reply"] = text
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    result, err = await _request("POST", "/api/agent/messages/reply", body)
    if err:
        return err
    if isinstance(result, dict) and result.get("status") == "ok":
        return (f"已回复 #{index}（{msg.get('nick')}）的留言，访客刷新即见。"
                f"回复内容：{text[:80]}")
    return (f"回复接口的返回和预期不一样（{json.dumps(result, ensure_ascii=False)[:120]}），"
            f"建议重新读一遍留言板确认状态。")


@tool
async def catcafe_delete_message(index: int, expect_nick: str = "", reason: str = "") -> str:
    """删除「小涩猫咖啡厅」留言板上的一条不当留言（广告、骚扰等），删除动作站点会记日志。
    站主让你清理留言板、删广告时使用；index 来自 catcafe_list_messages。
    删错不可恢复：建议带上 expect_nick 复核，昵称对不上会中止不动手。
    拿不准该不该删的（只是不顺眼、不算违规），先问站主，别自作主张。

    Args:
        index: 留言编号（来自 catcafe_list_messages，#0 为最新一条）
        expect_nick: 可选，期望的留言者昵称；和实际对不上就中止（防列表变动删错人）
        reason: 可选，删除原因（随请求提交）
    """
    refusal = _admin_refusal()
    if refusal:
        return refusal
    msgs, err = await _fetch_messages()
    if err:
        return err
    msg, err = _pick_message(msgs, index, expect_nick.strip())
    if err:
        return err
    payload = _msg_ref(msg)
    if reason.strip():
        payload["reason"] = reason.strip()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    result, err = await _request("POST", "/api/agent/messages/delete", body)
    if err:
        return err
    if isinstance(result, dict) and result.get("status") == "ok":
        preview = str(msg.get("content", "")).replace("\n", " ")[:60]
        return (f"已删除 #{index}（{msg.get('nick')}）的留言：「{preview}」。"
                f"删除动作站点会记日志。")
    return (f"删除接口的返回和预期不一样（{json.dumps(result, ensure_ascii=False)[:120]}），"
            f"建议重新读一遍留言板确认状态。")


TOOLS = [catcafe_get_content, catcafe_get_stats, catcafe_post_notice,
         catcafe_set_slogan, catcafe_set_status,
         catcafe_list_messages, catcafe_reply_message, catcafe_delete_message]
