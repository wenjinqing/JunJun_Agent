"""B站视频内容理解测试：三个数据源解析/材料构建/缓存共享/摘要/最近视频注入/工具。

HTTP 与 LLM 全部打桩；模块级缓存每个用例清空。
"""

import asyncio

import pytest

import junjun_core.config.config as cfg_mod
import junjun_skills.plugins.bilibili.tools as bili_tools
from junjun_skills.plugins.bilibili import content

CHAT = "qq:999:group"

_VIEW = {
    "bvid": "BV1xx411c7mD", "aid": 17001, "cid": 280001,
    "title": "猫猫弹琴", "desc": "一只会弹琴的猫",
    "duration": 100, "pic": "http://i0.hdslb.com/cover.jpg", "owner": "UP主甲",
}


@pytest.fixture
def env(monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={"bilibili": {"enable_understand": True, "subtitle_max_chars": 100}})
    content._MATERIAL_CACHE.clear()
    content._PENDING.clear()
    content._RECENT.clear()

    async def _extract(url):
        return "BV1xx411c7mD"

    async def _view(bvid):
        return dict(_VIEW)

    monkeypatch.setattr(bili_tools, "extract_bvid", _extract)
    monkeypatch.setattr(bili_tools, "_fetch_view", _view)
    builds = []

    async def _subtitle(aid, cid):
        return ""

    async def _danmaku(cid):
        return []

    async def _replies(aid):
        return []

    monkeypatch.setattr(content, "_fetch_subtitle_text", _subtitle)
    monkeypatch.setattr(content, "_fetch_danmaku_sample", _danmaku)
    monkeypatch.setattr(content, "_fetch_top_replies", _replies)
    yield monkeypatch
    cfg_mod.global_config = old


class TestParsers:
    def test_pick_subtitle_url(self):
        payload = {"data": {"subtitle": {"subtitles": [
            {"lan": "ai-zh", "subtitle_url": "//aisubtitle.hdslb.com/x.json"}]}}}
        assert content._pick_subtitle_url(payload) == "https://aisubtitle.hdslb.com/x.json"
        assert content._pick_subtitle_url({}) == ""
        assert content._pick_subtitle_url({"data": {"subtitle": {"subtitles": []}}}) == ""

    def test_parse_subtitle_body(self):
        data = {"body": [{"from": 0, "to": 1, "content": "大家好"},
                         {"from": 1, "to": 2, "content": "今天讲配队"},
                         {"from": 2, "to": 3}]}
        assert content._parse_subtitle_body(data, 100) == "大家好\n今天讲配队"
        assert len(content._parse_subtitle_body(data, 5)) == 5

    def test_parse_danmaku_sampling(self):
        xml = "".join(f'<d p="{i},1,25,16777215,0,0,0,0">弹幕{i}</d>' for i in range(100))
        out = content._parse_danmaku(xml, sample=10)
        assert len(out) == 10
        assert out[0] == "弹幕0" and out[-1] == "弹幕90"  # 均匀采样不扎堆头部
        assert content._parse_danmaku("no xml") == []

    def test_parse_replies(self):
        payload = {"data": {"replies": [
            {"content": {"message": "第一！\n沙发"}},
            {"content": {"message": "  "}},
            {"content": {"message": "学到了"}},
        ]}}
        assert content._parse_replies(payload) == ["第一！ 沙发", "学到了"]
        assert content._parse_replies({}) == []


class TestMaterial:
    @pytest.mark.asyncio
    async def test_subtitle_preferred(self, env, monkeypatch):
        async def _subtitle(aid, cid):
            return "字幕第一行\n字幕第二行"
        monkeypatch.setattr(content, "_fetch_subtitle_text", _subtitle)
        m = await content.get_material("https://www.bilibili.com/video/BV1xx411c7mD")
        assert m["source"] == "字幕"
        assert "字幕全文" in m["material"] and "字幕第一行" in m["material"]
        assert "弹幕采样" not in m["material"]  # 有字幕就不掺弹幕

    @pytest.mark.asyncio
    async def test_fallback_danmaku_replies(self, env, monkeypatch):
        async def _danmaku(cid):
            return ["awsl", "名场面"]
        async def _replies(aid):
            return ["课代表总结到位"]
        monkeypatch.setattr(content, "_fetch_danmaku_sample", _danmaku)
        monkeypatch.setattr(content, "_fetch_top_replies", _replies)
        m = await content.get_material("x")
        assert m["source"] == "弹幕热评"
        assert "awsl" in m["material"] and "课代表总结到位" in m["material"]

    @pytest.mark.asyncio
    async def test_all_empty_uses_desc(self, env):
        m = await content.get_material("x")
        assert m["source"] == "简介"
        assert "猫猫弹琴" in m["material"] and "一只会弹琴的猫" in m["material"]

    @pytest.mark.asyncio
    async def test_view_failure_returns_none(self, env, monkeypatch):
        async def _view(bvid):
            return None
        monkeypatch.setattr(bili_tools, "_fetch_view", _view)
        assert await content.get_material("x") is None

    @pytest.mark.asyncio
    async def test_cache_and_inflight_sharing(self, env, monkeypatch):
        calls = []

        async def _view(bvid):
            calls.append(bvid)
            return dict(_VIEW)
        monkeypatch.setattr(bili_tools, "_fetch_view", _view)
        # 并发两次：in-flight 共享，只构建一次
        m1, m2 = await asyncio.gather(content.get_material("x"), content.get_material("x"))
        assert m1 is m2 and len(calls) == 1
        # 串行第三次：命中缓存
        m3 = await content.get_material("x")
        assert m3 is m1 and len(calls) == 1


class TestWatchAndRender:
    @pytest.mark.asyncio
    async def test_watch_fills_recent_and_render(self, env, monkeypatch):
        class _FakeModel:
            async def ainvoke(self, messages, config=None):
                return type("R", (), {"content": "猫猫弹琴很治愈"})()
        # summarize_material 走注入的 model；_watch 内部不传 model——桩掉 get_chat_model
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot: _FakeModel())
        await content._watch(CHAT, "https://www.bilibili.com/video/BV1xx411c7mD")
        block = content.render_recent_block(CHAT)
        assert "猫猫弹琴" in block and "猫猫弹琴很治愈" in block and "UP主甲" in block

    @pytest.mark.asyncio
    async def test_render_empty_and_disabled(self, env, monkeypatch):
        assert content.render_recent_block(CHAT) == ""
        cfg_mod.global_config.raw["bilibili"]["enable_understand"] = False
        from collections import deque
        content._RECENT.setdefault(CHAT, deque()).append(
            (0, {"title": "t", "owner": "o", "summary": "s", "page_url": "u"}))
        # ts=0 已过期 + 开关关闭，双保险
        assert content.render_recent_block(CHAT) == ""


class TestTool:
    @pytest.mark.asyncio
    async def test_bilibili_summary_tool(self, env, monkeypatch):
        async def _subtitle(aid, cid):
            return "这期讲绝区零丹的配队"
        monkeypatch.setattr(content, "_fetch_subtitle_text", _subtitle)

        class _FakeModel:
            async def ainvoke(self, messages, config=None):
                return type("R", (), {"content": "讲丹的配队攻略"})()
        import junjun_llm
        monkeypatch.setattr(junjun_llm, "get_chat_model", lambda slot: _FakeModel())

        out = await bili_tools.bilibili_summary.ainvoke(
            {"url": "https://www.bilibili.com/video/BV1xx411c7mD"})
        assert "猫猫弹琴" in out and "讲丹的配队攻略" in out and "依据字幕" in out

    @pytest.mark.asyncio
    async def test_tool_no_material(self, env, monkeypatch):
        async def _view(bvid):
            return None
        monkeypatch.setattr(bili_tools, "_fetch_view", _view)
        out = await bili_tools.bilibili_summary.ainvoke({"url": "x"})
        assert "没拿到" in out
