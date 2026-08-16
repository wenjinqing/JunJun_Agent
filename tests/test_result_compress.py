"""主循环工具结果压缩测试（2026-08-16，DSH pi-quiet-tools 移植）。

核心断言：短结果原样不动；长结果头尾预览+省略计数+工作区材料指针
（全文落盘可 workspace_read 读回）；落盘失败降级纯截断；配置关闭直透；
阈值调小时头尾按比例收缩（保证压缩后一定变短）；中间件只动 ToolMessage
的 str 内容、字段原样保留。工作区根一律指 tmp——绝不写真 data/。
"""

import pytest
from langchain_core.messages import ToolMessage

import junjun_agent.loop.result_compress as rc


def _set_cfg(monkeypatch, raw):
    import junjun_core.config.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw=raw))


@pytest.fixture
def ws_root(monkeypatch, tmp_path):
    from junjun_skills.plugins.workspace import tools as wt
    monkeypatch.setattr(wt, "_ROOT", tmp_path / "ws")
    return tmp_path / "ws"


class TestCompressResult:
    def test_short_result_untouched(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        text = "短结果" * 30
        assert rc.compress_result(text, chat_id="qq:1:group",
                                  tool_name="web_search") == text
        assert not ws_root.exists()          # 短结果不落盘

    def test_long_result_head_tail_and_artifact(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        text = "头" * 2000 + "中" * 4000 + "尾" * 2000     # 8000 字
        out = rc.compress_result(text, chat_id="qq:1:group", tool_name="fetch_page")
        assert len(out) < len(text)
        assert out.startswith("头" * 100)
        assert out.endswith("尾" * 100)
        assert "中间省略 5300 字" in out                  # 8000-1800-900
        assert "workspace_read" in out and "artifacts/fetch_page-" in out
        # 全文落盘：工作区 artifacts/ 里能完整读回
        files = list((ws_root / "qq_1_group" / "artifacts").glob("fetch_page-*.txt"))
        assert len(files) == 1
        assert files[0].read_text(encoding="utf-8") == text

    def test_store_failure_degrades_to_plain_cut(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        monkeypatch.setattr(rc, "_store_artifact", lambda *a, **kw: "")
        text = "头" * 2000 + "中" * 4000 + "尾" * 2000
        out = rc.compress_result(text, chat_id="qq:1:group", tool_name="fetch_page")
        assert "全文共 8000 字" in out
        assert "workspace_read" not in out               # 没落盘就不给假指针

    def test_disabled_passthrough(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {"agent": {"tool_result_compress": False}})
        text = "长" * 9000
        assert rc.compress_result(text, chat_id="c", tool_name="t") == text

    def test_tiny_threshold_scales_head_tail(self, monkeypatch, ws_root):
        """阈值调到比头+尾还小：按比例收缩，保证压缩后一定更短。"""
        _set_cfg(monkeypatch, {"agent": {"tool_result_inline_chars": 500}})
        text = "长" * 2000
        out = rc.compress_result(text, chat_id="qq:1:group", tool_name="t")
        assert len(out) < len(text) and "中间省略" in out

    def test_config_broken_falls_back_to_defaults(self, monkeypatch, ws_root):
        """配置源炸了：_cfg 静默回默认（开+4000），压缩照常工作。"""
        import junjun_core.config as cfg_pkg

        def _boom():
            raise RuntimeError("配置加载炸了")
        monkeypatch.setattr(cfg_pkg, "get_global_config", _boom)
        text = "长" * 9000
        out = rc.compress_result(text, chat_id="qq:1:group", tool_name="t")
        assert "中间省略" in out and len(out) < len(text)


class _Req:
    def __init__(self, name):
        self.tool_call = {"name": name, "id": "t1"}


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_tool_message_compressed_fields_kept(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        long_text = "结" * 9000

        async def handler(request):
            return ToolMessage(content=long_text, tool_call_id="t1",
                               name="fetch_page")

        mw = rc.ToolResultCompressMiddleware("qq:1:group")
        out = await mw.awrap_tool_call(_Req("fetch_page"), handler)
        assert isinstance(out, ToolMessage)
        assert out.tool_call_id == "t1" and out.name == "fetch_page"
        assert len(out.content) < len(long_text)
        assert "workspace_read" in out.content

    @pytest.mark.asyncio
    async def test_short_result_same_object(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        msg = ToolMessage(content="短", tool_call_id="t1", name="get_time")

        async def handler(request):
            return msg

        mw = rc.ToolResultCompressMiddleware("qq:1:group")
        out = await mw.awrap_tool_call(_Req("get_time"), handler)
        assert out is msg                                # 未过阈值原样返回

    @pytest.mark.asyncio
    async def test_non_tool_message_passthrough(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        payload = {"not": "a ToolMessage"}

        async def handler(request):
            return payload

        mw = rc.ToolResultCompressMiddleware("qq:1:group")
        out = await mw.awrap_tool_call(_Req("x"), handler)
        assert out is payload

    @pytest.mark.asyncio
    async def test_non_str_content_passthrough(self, monkeypatch, ws_root):
        _set_cfg(monkeypatch, {})
        msg = ToolMessage(content=[{"type": "text", "text": "长" * 9000}],
                          tool_call_id="t1", name="vlm")

        async def handler(request):
            return msg

        mw = rc.ToolResultCompressMiddleware("qq:1:group")
        out = await mw.awrap_tool_call(_Req("vlm"), handler)
        assert out is msg                                # 多模态内容不动
