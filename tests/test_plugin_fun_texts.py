"""fun_texts 插件测试：答案之书/毒鸡汤/抽签/二维码/历史上的今天/每日60s。"""

import json

import pytest

from junjun_skills.plugins.fun_texts import tools as ft


def _payload(data):
    return {"code": 200, "data": data}


class TestExtractText:
    def test_plain_string(self):
        assert ft._extract_text(_payload("一切都是最好的安排")) == "一切都是最好的安排"

    def test_dict_preferred_keys(self):
        assert ft._extract_text(_payload({"content": "签文内容", "other": 1})) == "签文内容"

    def test_list_of_events(self):
        data = [{"year": "1949", "title": "开国大典"}, {"year": "2000", "title": "千禧年"}]
        out = ft._extract_text(_payload(data))
        assert "· 1949: 开国大典" in out and "· 2000: 千禧年" in out

    def test_garbage(self):
        assert ft._extract_text(None) == ""
        assert ft._extract_text({}) == ""

    def test_answer_book_zh_preferred(self):
        """答案之书真实结构：优先中文字段，不输出英文。"""
        data = {"description_en": "Stay strong", "description_zh": "坚定不移",
                "title_en": "Stay", "title_zh": "坚定"}
        out = ft._extract_text(_payload(data))
        assert out == "坚定不移"

    def test_format_lot(self):
        """灵签真实结构：签级+签名+签诗+解签。"""
        data = {"fortune": "上签", "name": "金精试窦儿", "palace": "卯宫",
                "poem_version_1": "一条金秤等君情", "explanation": "心平正直",
                "meaning": "凡事平稳无凶也", "image": "https://x/67.gif"}
        out = ft._format_lot(_payload(data))
        assert "「上签」金精试窦儿" in out
        assert "一条金秤等君情" in out
        assert "解签：心平正直" in out
        assert "gif" not in out  # 图片字段不入文案


def _mock_json(monkeypatch, payload):
    async def _get(path, params=None):
        return payload
    monkeypatch.setattr(ft, "_get_json", _get)


class TestTools:
    @pytest.mark.asyncio
    async def test_answer_book(self, monkeypatch):
        _mock_json(monkeypatch, _payload("别再犹豫了"))
        out = await ft.answer_book.ainvoke({"question": "该不该表白"})
        assert "别再犹豫了" in out and "该不该表白" in out

    @pytest.mark.asyncio
    async def test_answer_book_failure(self, monkeypatch):
        _mock_json(monkeypatch, None)
        out = await ft.answer_book.ainvoke({"question": "x"})
        assert "没翻开" in out

    @pytest.mark.asyncio
    async def test_fun_quote(self, monkeypatch):
        _mock_json(monkeypatch, _payload("你全力以赴了，才知道自己真的不行"))
        out = await ft.fun_quote.ainvoke({})
        assert "全力以赴" in out

    @pytest.mark.asyncio
    async def test_draw_lot_paths(self, monkeypatch):
        seen = []

        async def _get(path, params=None):
            seen.append(path)
            return _payload("上上签")

        monkeypatch.setattr(ft, "_get_json", _get)
        out = await ft.draw_lot.ainvoke({"kind": "wenchang"})
        assert seen == ["wenchangdijunrandom"] and "文昌" in out
        out = await ft.draw_lot.ainvoke({"kind": "garbage"})  # 非法值回落观音
        assert seen[-1] == "guanyinrandom" and "观音" in out

    @pytest.mark.asyncio
    async def test_make_qrcode_sends_image(self, monkeypatch):
        from junjun_agent.tasks import task_manager
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:1:group")
        sent = []

        async def _redirect(path, params=None):
            return "https://v2.xxapi.cn/qr/abc.png"

        monkeypatch.setattr(ft, "_get_redirect_url", _redirect)

        async def _send(chat_id, segments):
            sent.append((chat_id, segments))

        monkeypatch.setattr(task_manager, "_send", _send)
        ack = await ft.make_qrcode.ainvoke({"text": "https://example.com"})
        assert "在生成" in ack
        import asyncio
        tasks = list(task_manager._running.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        assert sent
        images = [s for s in sent[0][1] if s.type == "image"]
        assert images and images[0].data == "https://v2.xxapi.cn/qr/abc.png"

    @pytest.mark.asyncio
    async def test_decode_qrcode_explicit_url(self, monkeypatch):
        _mock_json(monkeypatch, _payload("https://evil.example.com/prize"))
        out = await ft.decode_qrcode.ainvoke({"url": "http://x/qr.png"})
        assert "evil.example.com" in out
        assert "别乱点" in out  # 链接类内容带风险提示

    @pytest.mark.asyncio
    async def test_decode_qrcode_uses_recent_image(self, monkeypatch):
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:1:group")
        seen = []

        async def _get(path, params=None):
            seen.append(params.get("url"))
            return _payload("HELLO")

        monkeypatch.setattr(ft, "_get_json", _get)
        monkeypatch.setattr("junjun_memory.vision.recent_image_urls",
                            lambda chat: [("image", "http://x/recent.png")])
        out = await ft.decode_qrcode.ainvoke({"url": ""})
        assert seen == ["http://x/recent.png"]
        assert "HELLO" in out and "别乱点" not in out  # 非链接不带风险提示

    @pytest.mark.asyncio
    async def test_decode_qrcode_no_image(self, monkeypatch):
        from junjun_skills.builtin.memory_skills import current_chat_id
        current_chat_id.set("qq:1:group")
        monkeypatch.setattr("junjun_memory.vision.recent_image_urls", lambda chat: [])
        out = await ft.decode_qrcode.ainvoke({"url": ""})
        assert "没找到" in out

    @pytest.mark.asyncio
    async def test_today_in_history(self, monkeypatch):
        _mock_json(monkeypatch, _payload([{"year": "1969", "title": "阿波罗登月"}]))
        out = await ft.today_in_history.ainvoke({})
        assert "阿波罗登月" in out


class TestDaily60s:
    @pytest.mark.asyncio
    async def test_disabled_skips(self, monkeypatch):
        monkeypatch.setattr(ft, "_daily60s_cfg", lambda: {"daily60s_enable": False})
        called = []

        async def _redirect(path, params=None):
            called.append(path)
            return "u"

        monkeypatch.setattr(ft, "_get_redirect_url", _redirect)
        await ft.daily60s_push()
        assert called == []

    @pytest.mark.asyncio
    async def test_push_to_active_groups(self, monkeypatch):
        monkeypatch.setattr(ft, "_daily60s_cfg", lambda: {"daily60s_enable": True})
        monkeypatch.setattr(ft, "_active_groups", lambda: ["qq:100:group", "qq:200:group"])

        async def _redirect(path, params=None):
            return "https://img/60s.png"

        monkeypatch.setattr(ft, "_get_redirect_url", _redirect)
        sent = []

        class _GW:
            async def send_reply(self, reply):
                sent.append(reply)

        import junjun_core.gateway.router as router_mod
        monkeypatch.setattr(router_mod, "get_gateway", lambda: _GW())
        await ft.daily60s_push()
        assert len(sent) == 2
        assert sent[0].segments[0].data == "https://img/60s.png"

    @pytest.mark.asyncio
    async def test_no_active_groups_skips(self, monkeypatch):
        monkeypatch.setattr(ft, "_daily60s_cfg", lambda: {"daily60s_enable": True})
        monkeypatch.setattr(ft, "_active_groups", lambda: [])

        async def _redirect(path, params=None):
            return "https://img/60s.png"

        monkeypatch.setattr(ft, "_get_redirect_url", _redirect)
        # gateway 不可用也应正常返回不抛异常
        await ft.daily60s_push()
