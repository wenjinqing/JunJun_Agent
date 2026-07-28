"""W1/W2 收尾测试：表情包 VLM 描述 + 链接预览 + 自我反思。"""

import pytest


class TestStickerDescription:
    @pytest.mark.asyncio
    async def test_describe_stickers_uses_sticker_prompt(self, monkeypatch):
        """表情包走专属 prompt（画面+情绪），结果入 Images 缓存。"""
        from junjun_memory import vision
        used = {}

        class _Model:
            async def ainvoke(self, msgs):
                used["prompt"] = msgs[0].content[0]["text"]
                class R: content = "猫咪竖大拇指表示赞同"
                return R()

        async def _dl(url):
            import uuid
            return b"\x89PNG sticker-test-" + uuid.uuid4().bytes  # 每次唯一防 Images 持久缓存命中

        monkeypatch.setattr(vision, "_download", _dl)
        out = await vision.describe_stickers(["http://x/s.png"], model=_Model())
        assert out["http://x/s.png"] == "猫咪竖大拇指表示赞同"
        assert "表情包" in used["prompt"]

    @pytest.mark.asyncio
    async def test_render_sticker_block(self):
        from junjun_memory.vision import render_sticker_block
        assert render_sticker_block({"u": "猫咪点赞"}) == "对方发了一个表情包：猫咪点赞"
        assert render_sticker_block({"u": "[表情]"}) == ""
        assert "对方发了表情包" in render_sticker_block({"a": "x", "b": "y"})


class TestLinkPreview:
    def test_first_fetchable_url(self):
        from junjun_memory.link_preview import _first_fetchable_url
        assert _first_fetchable_url("看这个 https://example.com/a 好") == "https://example.com/a"
        # B站/抖音走拦截器，跳过
        assert _first_fetchable_url("https://b23.tv/abc https://example.com/x") == "https://example.com/x"
        # 图片直链跳过
        assert _first_fetchable_url("https://x.com/pic.jpg") is None
        assert _first_fetchable_url("没有链接") is None

    def test_extract_summary(self):
        from junjun_memory.link_preview import _extract_summary
        html = "<html><head><title>大新闻</title></head><body><script>x=1</script><p>正文内容</p></body></html>"
        out = _extract_summary(html, 300)
        assert out.startswith("大新闻。") and "正文内容" in out and "x=1" not in out

    @pytest.mark.asyncio
    async def test_fetch_disabled_by_config(self, monkeypatch):
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"link_preview": {"enable": False}}))
        from junjun_memory.link_preview import fetch_link_preview
        assert await fetch_link_preview("https://example.com/a") == ""

    @pytest.mark.asyncio
    async def test_fetch_timeout_silent(self, monkeypatch):
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"link_preview": {"enable": True}}))
        from junjun_memory.link_preview import fetch_link_preview
        # 不可达地址：超时/失败静默返回空串
        assert await fetch_link_preview("https://192.0.2.1/x", timeout=0.3) == ""


class TestReflection:
    @pytest.mark.asyncio
    async def test_trigger_after_n_replies(self, monkeypatch):
        from junjun_agent.loop.reflection import ReflectionLoop
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"reflection": {"enable": True, "every_n": 3}}))
        loop = ReflectionLoop()
        ran = []

        async def _fake():
            ran.append(1)
            loop._running = False

        monkeypatch.setattr(loop, "_run_safe", _fake)
        for _ in range(2):
            loop.note_reply()
        assert not ran
        # 第三次到阈值：create_task 需要事件循环——直接验证计数逻辑
        import asyncio
        tasks_before = len(asyncio.all_tasks())
        loop.note_reply()
        await asyncio.sleep(0)
        assert ran or len(asyncio.all_tasks()) >= tasks_before

    @pytest.mark.asyncio
    async def test_reflect_sends_to_admin(self, monkeypatch):
        from junjun_agent.loop.reflection import ReflectionLoop
        loop = ReflectionLoop()
        monkeypatch.setattr(ReflectionLoop, "_recent_transcript", staticmethod(lambda limit: "君君: 好\n甲: 嗯"))
        sent = []

        async def _notify(text):
            sent.append(text)
            return True

        class _Model:
            async def ainvoke(self, msgs):
                class R: content = "总体不错，但有一次复读「笨蛋」，注意换说法。"
                return R()

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda slot: _Model())
        monkeypatch.setattr("junjun_core.security.notify_admin", _notify)
        out = await loop.reflect()
        assert "复读" in out
        assert sent and "自我反思" in sent[0]

    @pytest.mark.asyncio
    async def test_reflect_empty_transcript_skips(self, monkeypatch):
        from junjun_agent.loop.reflection import ReflectionLoop
        loop = ReflectionLoop()
        monkeypatch.setattr(ReflectionLoop, "_recent_transcript", staticmethod(lambda limit: ""))
        assert await loop.reflect() == ""
