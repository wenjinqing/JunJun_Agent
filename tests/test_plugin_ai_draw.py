"""ai_draw 插件测试：命令 /draw、红线拒绝、扩写降级、限流、tool、人设注入。"""

from types import SimpleNamespace

import pytest

from junjun_agent import commands


@pytest.fixture(autouse=True)
def _clean_buses():
    commands.clear_commands()
    yield
    commands.clear_commands()


def _session(is_group=True):
    return SimpleNamespace(platform="qq", group_id="999" if is_group else None,
                           is_group=is_group, chat_id="qq:999:group" if is_group else "qq:1:private")


def _meta(text):
    return SimpleNamespace(text=text, user_id="12345", nickname="甲", at_bot=False, message_id="m1")


@pytest.fixture
def _fake_gateway(monkeypatch):
    sent = []

    class _FakeGW:
        async def send_reply(self, reply_set):
            sent.append(reply_set)

    import junjun_core.gateway.router as router_mod
    monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
    return sent


def _ctx(text, is_group=True):
    return commands.CommandContext(session=_session(is_group), meta=_meta(text),
                                   args=text.split(" ", 1)[1] if " " in text else "")


@pytest.fixture
def _plugin(monkeypatch):
    """导入插件并清空限流、配好密钥与假生成/扩写 helper。"""
    import junjun_skills.plugins.ai_draw.tools as ad
    ad._last_use.clear()
    monkeypatch.setenv("MODELSCOPE_API_KEY", "ms-test")

    async def _gen(prompt, model):
        return "http://x/draw.png"

    async def _expand(p):
        return p

    monkeypatch.setattr(ad, "generate", _gen)
    monkeypatch.setattr(ad, "expand_prompt", _expand)
    yield ad
    ad._last_use.clear()


class TestDrawCommand:
    @pytest.mark.asyncio
    async def test_success_sends_image(self, _fake_gateway, _plugin):
        result = await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        assert result is None
        assert len(_fake_gateway) == 1
        segs = _fake_gateway[0].segments
        assert segs[0].type == "text" and "画好啦" in segs[0].data
        assert segs[1].type == "image" and segs[1].data == "http://x/draw.png"

    @pytest.mark.asyncio
    async def test_empty_args_usage(self, _plugin):
        assert "用法" in await _plugin.draw_cmd(_ctx("/draw"))

    @pytest.mark.asyncio
    async def test_minor_nsfw_rejected(self, _fake_gateway, _plugin, monkeypatch):
        called = []

        async def _gen(prompt, model):
            called.append(prompt)
            return "http://x/bad.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        result = await _plugin.draw_cmd(_ctx("/draw 萝莉 裸体"))
        assert "不画" in result
        assert not called           # 绝不调用生成
        assert not _fake_gateway    # 不发任何段

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, _plugin):
        await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        result = await _plugin.draw_cmd(_ctx("/draw 狗娘"))
        assert "秒后" in result
        assert len(_fake_gateway) == 1  # 第二次没发图

    @pytest.mark.asyncio
    async def test_no_api_key_degrades(self, _plugin, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_KEY", raising=False)
        result = await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        assert "MODELSCOPE_API_KEY" in result

    @pytest.mark.asyncio
    async def test_generate_failure_degrades(self, _plugin, monkeypatch):
        async def _none(prompt, model):
            return None

        monkeypatch.setattr(_plugin, "generate", _none)
        result = await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        assert "失败" in result

    @pytest.mark.asyncio
    async def test_anime_model_routing(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model):
            captured["model"] = model
            return "http://x/a.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        await _plugin.draw_cmd(_ctx("/draw 二次元少女"))
        assert captured["model"] == _plugin._DEFAULT_ANIME_MODEL


class TestExpandPrompt:
    @pytest.mark.asyncio
    async def test_expand_failure_falls_back_to_raw(self, _plugin, monkeypatch):
        def _boom(task):
            raise RuntimeError("模型槽未配置")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        assert await _plugin.expand_prompt("猫") == "猫"  # 降级用原文

    @pytest.mark.asyncio
    async def test_long_prompt_skips_expand(self, _plugin, monkeypatch):
        def _boom(task):
            raise RuntimeError("不应被调用")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        long_prompt = "一只站在樱花树下的白毛猫娘少女，日系插画风格"
        assert await _plugin.expand_prompt(long_prompt) == long_prompt


class TestSelfPrompt:
    @pytest.mark.asyncio
    async def test_persona_injected(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model):
            captured["prompt"] = prompt
            return "http://x/me.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"personality": {"personality": "猫娘 白发 红瞳 可爱 萌"}}))

        await _plugin.draw_cmd(_ctx("/draw 画一张你自己"))
        assert "猫娘 白发" in captured["prompt"]
        assert "你自己" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_no_self_word_no_persona(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model):
            captured["prompt"] = prompt
            return "http://x/n.png"

        monkeypatch.setattr(_plugin, "generate", _gen)

        def _boom():
            raise AssertionError("不应读取人设配置")

        monkeypatch.setattr("junjun_core.config.get_global_config", _boom)
        await _plugin.draw_cmd(_ctx("/draw 星空"))
        assert captured["prompt"] == "星空"


class TestTool:
    @pytest.mark.asyncio
    async def test_tool_returns_url(self, _plugin):
        out = await _plugin.ai_draw.ainvoke({"prompt": "星空下的城市"})
        assert out == "http://x/draw.png"

    @pytest.mark.asyncio
    async def test_tool_rejects_minor_nsfw(self, _plugin):
        out = await _plugin.ai_draw.ainvoke({"prompt": "小学生 sex"})
        assert "拒绝" in out

    @pytest.mark.asyncio
    async def test_tool_no_key_degrades(self, _plugin, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_KEY", raising=False)
        out = await _plugin.ai_draw.ainvoke({"prompt": "猫娘"})
        assert "MODELSCOPE_API_KEY" in out
