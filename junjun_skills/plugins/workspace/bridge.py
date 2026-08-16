"""沙箱工具桥（PTC 试点，2026-08-15）：让沙箱里的代码能调宿主的白名单只读工具。

思想来源 DeepSeek Harness 的 PTC（程序化工具调用）：模型写代码一次性编排
多次工具调用，中间数据留在沙箱、只有最终结果回模型上下文。

关键约束：沙箱容器 --network=none（隔离底线，出入全断），任何 HTTP/socket
通道都不成立。本桥走**唯一合法通道——/workspace 挂载目录的文件**，协议是
「挂起-回放」（经典无网络 PTC 沙箱方案）：

1. run_code 执行前把 jjtools.py 注入会话工作区根目录（沙箱 cwd=/workspace，
   `import jjtools` 直接可用；sys 在静态预检黑名单，不能玩 sys.path 花招）
2. 沙箱代码 jjtools.call("web_search", query=...)：查 .jj_cache.json，
   第 i 次调用已有缓存则直接返回（回放）；没有则把请求写进
   .jj_request.json 并 sys.exit(42) 挂起
3. 宿主侧（本模块 drive()）看到 returncode==42 + 请求文件：校验白名单、
   按发起者身份执行工具、结果追加进 .jj_cache.json，原样重跑代码
4. 重跑时前 i 次调用从缓存回放，代码推进到第 i+1 个调用……直到正常结束

代价与规矩：每个工具调用代码整体重跑一次——编排代码必须纯粹可重入
（工具调用前别做不可逆副作用，print 放最后）；单轮工具调用数上限
（[sandbox] tool_bridge_max_calls，默认 8）防无限挂起。

安全模型：
- 白名单宿主侧强制（[sandbox] tool_bridge_tools，默认 get_time/web_search/
  fetch_page/query_chat_history——全只读；副作用工具永远不该进名单）
- 身份/会话边界：工具按 run_code 发起者的 chat_id/user_id 执行，
  query_chat_history 等按原会话取数——桥不是提权/跨会话通道
- 文件通道天然不出沙箱挂载：无 token 概念，无端口暴露，隔离等级不变
- 默认关：[sandbox] tool_bridge = true 才启用（试点期灰度）
"""

import json
from pathlib import Path
from typing import Awaitable, Callable, Optional

from junjun_core.observability import get_logger

logger = get_logger("plugin.workspace.bridge")

_DEFAULT_TOOLS = ["get_time", "web_search", "fetch_page", "query_chat_history"]
_CALL_TIMEOUT = 45.0
_TEXT_CAP = 16000        # 单次回传上限：中间数据该留沙箱，回传只是出口
_REQUEST_FILE = ".jj_request.json"
_CACHE_FILE = ".jj_cache.json"
EXIT_SUSPENDED = 42      # jjtools 挂起退出码（与宿主约定）


def _cfg() -> dict:
    try:
        from junjun_core.config import get_global_config
        return get_global_config().raw.get("sandbox", {})
    except Exception:
        return {}


def enabled() -> bool:
    return bool(_cfg().get("tool_bridge", False))


def _whitelist() -> list:
    tools = _cfg().get("tool_bridge_tools")
    if isinstance(tools, list) and tools:
        return [str(t) for t in tools]
    return list(_DEFAULT_TOOLS)


def _max_calls() -> int:
    try:
        return max(1, int(_cfg().get("tool_bridge_max_calls", 8)))
    except Exception:
        return 8


async def execute(tool_name: str, args: dict, *, chat_id: str,
                  user_id: str) -> dict:
    """白名单校验 + 按发起者身份执行工具。返回 {"ok", "text"/"error"}。"""
    if tool_name not in _whitelist():
        return {"ok": False, "error": f"工具 {tool_name or '(空)'} 不在沙箱桥白名单"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "args 必须是对象"}
    from junjun_skills.registry import get_tools
    tool = next((t for t in get_tools() if t.name == tool_name), None)
    if tool is None:
        return {"ok": False, "error": f"工具 {tool_name} 未注册/已降级"}

    from junjun_core.security import set_caller
    from junjun_skills.builtin.memory_skills import current_chat_id
    set_caller(user_id, at_bot=True, is_group=chat_id.endswith(":group"))
    token = current_chat_id.set(chat_id)
    try:
        import asyncio
        out = await asyncio.wait_for(tool.ainvoke(args), timeout=_CALL_TIMEOUT)
        text = out if isinstance(out, str) else str(out)
        return {"ok": True, "text": text[:_TEXT_CAP]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        current_chat_id.reset(token)


async def drive(post_run: Callable[[], Awaitable[dict]], workdir: Path, *,
                chat_id: str, user_id: str) -> dict:
    """挂起-回放驱动循环。post_run：执行一次沙箱 /run 并返回响应 dict。

    返回最终一轮的响应（中途轮次的 stdout 丢弃——代码重放会再打印，
    只留最后一轮不重复）。超过调用上限：给代码注入错误结果再跑一次
    （让它自己收场），仍挂起则按最后一轮响应返回并追加说明。
    """
    req_p = Path(workdir) / _REQUEST_FILE
    cache_p = Path(workdir) / _CACHE_FILE
    for p in (req_p, cache_p):          # 清上一轮残留，调用链从 0 开始
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    cache: list = []
    max_calls = _max_calls()
    resp: Optional[dict] = None
    for _round in range(max_calls + 2):
        resp = await post_run()
        if resp.get("returncode") != EXIT_SUSPENDED or not req_p.is_file():
            return resp
        try:
            payload = json.loads(req_p.read_text(encoding="utf-8"))
            req_p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"桥请求文件损坏（按普通失败返回）: {e}")
            return resp
        tool_name = str(payload.get("tool") or "")
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        if len(cache) < max_calls:
            result = await execute(tool_name, args,
                                   chat_id=chat_id, user_id=user_id)
            logger.info(f"沙箱桥调用 {tool_name}（第 {len(cache) + 1} 次）"
                        f"-> {'ok' if result.get('ok') else result.get('error', '')[:60]}")
        else:
            result = {"ok": False,
                      "error": f"超过单次运行工具调用上限（{max_calls}）"}
        cache.append(result)
        try:
            cache_p.write_text(json.dumps(cache, ensure_ascii=False),
                               encoding="utf-8")
        except Exception as e:
            logger.warning(f"桥缓存写入失败（中止回放）: {e}")
            return resp
    # 到上限后仍挂起：返回最后一轮并说明
    if resp is not None:
        resp = dict(resp)
        resp["stderr"] = (str(resp.get("stderr") or "")
                          + f"\n[bridge] 工具调用超过上限 {max_calls}，运行中止").strip()
    return resp


_SDK_SOURCE = '''"""君君沙箱工具桥 SDK（run_code 自动注入，每次运行刷新，请勿手改）。

可用（白名单只读工具，宿主侧强制）：
    import jjtools
    jjtools.web_search("关键词")                 # 联网搜索
    jjtools.fetch_page("https://...")            # 深读网页正文
    jjtools.get_time()                           # 当前时间
    jjtools.call("query_chat_history", keyword="...", user="")  # 翻聊天记录

工作机制（重要）：每次工具调用会把请求落盘并暂停整个程序，宿主执行完
工具后【从头重跑你的代码】，已完成的调用从缓存直接回放。所以：
- 工具调用前的代码必须纯粹可重入（别做不可逆副作用，别依赖随机/计时）
- print 结果放到所有工具调用之后，中间数据存变量或写工作区文件
- 别把大段原始数据 print 出来——中间结果留沙箱，只输出结论
"""

import json as _json
import sys as _sys
from pathlib import Path as _Path

_IDX = 0
_CACHE_PATH = _Path(".jj_cache.json")
_REQUEST_PATH = _Path(".jj_request.json")


def _load_cache():
    try:
        return _json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def call(tool, **args):
    """调宿主白名单工具，返回文本结果；工具失败抛 RuntimeError。"""
    global _IDX
    idx = _IDX
    _IDX += 1
    cache = _load_cache()
    if idx < len(cache):
        item = cache[idx]
        if item.get("ok"):
            return item.get("text", "")
        raise RuntimeError("工具桥调用失败: " + str(item.get("error", "")))
    _REQUEST_PATH.write_text(
        _json.dumps({"idx": idx, "tool": tool, "args": args},
                    ensure_ascii=False),
        encoding="utf-8")
    _sys.exit(42)  # 与宿主的挂起约定：宿主执行工具后从头重放本程序


def web_search(query, num_results=10):
    return call("web_search", query=query, num_results=num_results)


def fetch_page(url, save_as=""):
    return call("fetch_page", url=url, save_as=save_as)


def get_time():
    return call("get_time")
'''


def install_sdk(workdir: Path) -> bool:
    """把 jjtools.py 写进会话工作区根目录（沙箱 cwd），返回是否写入。

    文件会在 workspace_list 里可见、内容无秘密（文件通道无 token），
    每次 run_code 刷新覆盖。
    """
    try:
        d = Path(workdir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "jjtools.py").write_text(_SDK_SOURCE, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"jjtools SDK 注入失败（忽略，本次 run_code 无桥）: {e}")
        return False
