"""ai_draw 插件测试：命令 /draw、红线拒绝、扩写降级、限流、tool、人设注入、异步直发。"""

import asyncio
from types import SimpleNamespace

import pytest

from junjun_agent import commands
from junjun_agent.tasks import task_manager


async def _drain():
    """等待全部后台任务完成。"""
    tasks = list(task_manager._running.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


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

    async def _gen(prompt, model, negative=""):
        return "http://x/draw.png"

    async def _expand(p, *, anime=False):
        return p

    monkeypatch.setattr(ad, "generate", _gen)
    monkeypatch.setattr(ad, "expand_prompt", _expand)
    yield ad
    ad._last_use.clear()


class TestDrawCommand:
    @pytest.mark.asyncio
    async def test_success_sends_image(self, _fake_gateway, _plugin):
        result = await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        assert "在弄了" in result       # 提交即返回 ack
        await _drain()                   # 后台画完直发
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

        async def _gen(prompt, model, negative=""):
            called.append(prompt)
            return "http://x/bad.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        result = await _plugin.draw_cmd(_ctx("/draw 萝莉 裸体"))
        assert "不画" in result
        await _drain()
        assert not called           # 绝不调用生成
        assert not _fake_gateway    # 不发任何段

    @pytest.mark.asyncio
    async def test_rate_limit(self, _fake_gateway, _plugin):
        await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        await _drain()
        result = await _plugin.draw_cmd(_ctx("/draw 狗娘"))
        assert "秒后" in result
        await _drain()
        assert len(_fake_gateway) == 1  # 第二次没发图

    @pytest.mark.asyncio
    async def test_no_api_key_degrades(self, _plugin, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_KEY", raising=False)
        result = await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        assert "MODELSCOPE_API_KEY" in result

    @pytest.mark.asyncio
    async def test_generate_failure_degrades(self, _fake_gateway, _plugin, monkeypatch):
        async def _none(prompt, model, negative=""):
            return None

        monkeypatch.setattr(_plugin, "generate", _none)
        result = await _plugin.draw_cmd(_ctx("/draw 猫娘"))
        assert "在弄了" in result          # 先回 ack
        await _drain()                      # 后台失败发降级文案
        assert len(_fake_gateway) == 1
        assert "失败" in _fake_gateway[0].segments[0].data

    @pytest.mark.asyncio
    async def test_anime_model_routing(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model, negative=""):
            captured["model"] = model
            captured["negative"] = negative
            return "http://x/a.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        await _plugin.draw_cmd(_ctx("/draw 二次元少女"))
        await _drain()
        assert captured["model"] == _plugin._DEFAULT_ANIME_MODEL
        assert captured["negative"] == _plugin._ANIME_NEGATIVE  # 二次元走专属负面词


class TestExpandPrompt:
    """真实 expand_prompt（不走 fixture 的 mock）：按模型家族转写 + 降级。"""

    @staticmethod
    def _fake_llm(monkeypatch, content="catgirl, cat ears, white hair"):
        class _Resp:
            pass

        class _Model:
            async def ainvoke(self, msgs):
                r = _Resp()
                r.content = content
                return r

        monkeypatch.setattr("junjun_llm.get_chat_model", lambda task: _Model())

    @pytest.mark.asyncio
    async def test_anime_tags_with_quality_suffix(self, monkeypatch):
        import junjun_skills.plugins.ai_draw.tools as ad
        self._fake_llm(monkeypatch)
        out = await ad.expand_prompt("猫娘少女", anime=True)
        assert out.startswith("catgirl, cat ears, white hair")
        assert ad._ANIME_QUALITY_SUFFIX in out
        assert "猫娘" not in out  # Danbooru 标签串不混中文原文

    @pytest.mark.asyncio
    async def test_default_natural_language_keeps_origin(self, monkeypatch):
        import junjun_skills.plugins.ai_draw.tools as ad
        self._fake_llm(monkeypatch, content="A cat girl under cherry blossoms, soft light")
        out = await ad.expand_prompt("樱花下的猫娘", anime=False)
        assert out.startswith("樱花下的猫娘，")  # 原文前置保主体
        assert "soft light" in out

    @pytest.mark.asyncio
    async def test_expand_failure_anime_keeps_suffix(self, monkeypatch):
        import junjun_skills.plugins.ai_draw.tools as ad

        def _boom(task):
            raise RuntimeError("模型槽未配置")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        out = await ad.expand_prompt("猫", anime=True)
        assert out.startswith("猫") and ad._ANIME_QUALITY_SUFFIX in out

    @pytest.mark.asyncio
    async def test_expand_failure_default_falls_back_to_raw(self, monkeypatch):
        import junjun_skills.plugins.ai_draw.tools as ad

        def _boom(task):
            raise RuntimeError("模型槽未配置")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        assert await ad.expand_prompt("猫") == "猫"

    @pytest.mark.asyncio
    async def test_long_prompt_skips_expand(self, monkeypatch):
        import junjun_skills.plugins.ai_draw.tools as ad

        def _boom(task):
            raise RuntimeError("不应被调用")

        monkeypatch.setattr("junjun_llm.get_chat_model", _boom)
        long_prompt = "一只站在樱花树下的白毛猫娘少女" * 20  # >200 字
        assert await ad.expand_prompt(long_prompt) == long_prompt


class TestSubmitTask:
    @pytest.mark.asyncio
    async def test_negative_prompt_400_retry(self, monkeypatch):
        """模型拒绝 negative_prompt（HTTP 400）时自动去掉重试。"""
        import junjun_skills.plugins.ai_draw.tools as ad
        monkeypatch.setenv("MODELSCOPE_API_KEY", "ms-test")
        calls = []

        class _Resp:
            def __init__(self, code, data=None):
                self.status_code = code
                self._data = data or {}

            def json(self):
                return self._data

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                calls.append(dict(json))
                if "negative_prompt" in json:
                    return _Resp(400)
                return _Resp(200, {"task_id": "t1"})

        monkeypatch.setattr(ad.httpx, "AsyncClient", _Client)
        task_id = await ad.submit_task("p", "m", "neg")
        assert task_id == "t1"
        assert len(calls) == 2
        assert "negative_prompt" in calls[0] and "negative_prompt" not in calls[1]


class TestSelfPrompt:
    @pytest.mark.asyncio
    async def test_persona_injected(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model, negative=""):
            captured["prompt"] = prompt
            return "http://x/me.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        import junjun_core.config.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"personality": {"personality": "猫娘 白发 红瞳 可爱 萌"}}))

        await _plugin.draw_cmd(_ctx("/draw 画一张你自己"))
        await _drain()
        assert "猫娘 白发" in captured["prompt"]
        assert "你自己" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_no_self_word_no_persona(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model, negative=""):
            captured["prompt"] = prompt
            return "http://x/n.png"

        monkeypatch.setattr(_plugin, "generate", _gen)

        def _boom():
            raise AssertionError("不应读取人设配置")

        monkeypatch.setattr("junjun_core.config.get_global_config", _boom)
        await _plugin.draw_cmd(_ctx("/draw 星空"))
        await _drain()
        assert captured["prompt"] == "星空"


class TestTool:
    @pytest.mark.asyncio
    async def test_tool_returns_url(self, _plugin):
        # 无会话路由（contextvar 显式清空）：同步降级，返回 [IMAGE:] 标记
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("")
        try:
            out = await _plugin.ai_draw.ainvoke({"prompt": "星空下的城市"})
        finally:
            current_chat_id.reset(token)
        assert out == "[IMAGE:http://x/draw.png]"

    @pytest.mark.asyncio
    async def test_tool_async_direct_send(self, _fake_gateway, _plugin):
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:999:group")
        try:
            out = await _plugin.ai_draw.ainvoke({"prompt": "星空下的城市"})
        finally:
            current_chat_id.reset(token)
        assert "在弄了" in out           # 立即返回 ack，不等轮询
        await _drain()                    # 后台画完直发图片
        assert len(_fake_gateway) == 1
        segs = _fake_gateway[0].segments
        assert segs[-1].type == "image" and segs[-1].data == "http://x/draw.png"

    @pytest.mark.asyncio
    async def test_tool_rejects_minor_nsfw(self, _plugin):
        out = await _plugin.ai_draw.ainvoke({"prompt": "小学生 sex"})
        assert "拒绝" in out

    @pytest.mark.asyncio
    async def test_tool_no_key_degrades(self, _plugin, monkeypatch):
        monkeypatch.delenv("MODELSCOPE_API_KEY", raising=False)
        out = await _plugin.ai_draw.ainvoke({"prompt": "猫娘"})
        assert "MODELSCOPE_API_KEY" in out


class TestQwenModel:
    """Qwen-Image-2512：写实/文字域自动路由 + 显式别名。"""

    def test_route_model_matrix(self, _plugin):
        qwen = _plugin._DEFAULT_QWEN_MODEL
        anime = _plugin._DEFAULT_ANIME_MODEL
        default = _plugin._DEFAULT_MODEL
        assert _plugin.route_model("写实照片风的猫咪") == qwen      # 写实域
        assert _plugin.route_model("一张带字的海报") == qwen        # 文字渲染域
        assert _plugin.route_model("二次元少女") == anime
        assert _plugin.route_model("画一个涩图") == anime   # R18 词路由到唯一能出的模型
        assert _plugin.route_model("星空下的城市") == default
        assert _plugin.route_model("二次元少女", explicit="qwen") == qwen  # 显式优先
        assert _plugin.route_model("猫", explicit="anime") == anime
        assert _plugin.route_model("猫", explicit="不存在") == default    # 未知别名忽略

    def test_model_style(self, _plugin):
        reg = _plugin._model_registry()
        assert _plugin.model_style(reg["anime"]) == "anime"
        assert _plugin.model_style(reg["qwen"]) == "default"  # qwen 吃自然语言细描
        assert _plugin.model_style(reg["zimage"]) == "default"

    def test_parse_model_alias(self, _plugin):
        assert _plugin._parse_model_alias("猫娘少女 qwen") == ("猫娘少女", "qwen")
        assert _plugin._parse_model_alias("猫娘少女") == ("猫娘少女", "")
        assert _plugin._parse_model_alias("带字海报 anime") == ("带字海报", "anime")

    @pytest.mark.asyncio
    async def test_qwen_routing_via_cmd(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model, negative=""):
            captured["model"] = model
            captured["negative"] = negative
            return "http://x/a.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        await _plugin.draw_cmd(_ctx("/draw 写实照片风的猫咪"))
        await _drain()
        assert captured["model"] == _plugin._DEFAULT_QWEN_MODEL
        assert captured["negative"] == _plugin._DEFAULT_NEGATIVE  # 非 anime 家族

    @pytest.mark.asyncio
    async def test_explicit_alias_via_cmd(self, _fake_gateway, _plugin, monkeypatch):
        captured = {}

        async def _gen(prompt, model, negative=""):
            captured["model"] = model
            captured["prompt"] = prompt
            return "http://x/a.png"

        monkeypatch.setattr(_plugin, "generate", _gen)
        _plugin._last_use.clear()
        await _plugin.draw_cmd(_ctx("/draw 猫娘少女 qwen"))
        await _drain()
        assert captured["model"] == _plugin._DEFAULT_QWEN_MODEL
        assert "qwen" not in captured["prompt"]  # 别名被剥离

    @pytest.mark.asyncio
    async def test_tool_bad_alias_rejected(self, _plugin):
        out = await _plugin.ai_draw.ainvoke({"prompt": "猫娘", "model": "gpt"})
        assert "不认识模型" in out
