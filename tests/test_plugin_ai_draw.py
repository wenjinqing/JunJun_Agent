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
        default = _plugin._DEFAULT_KOLORS_MODEL   # 2026-08-12 起默认 SFW = AI Ping Kolors
        assert _plugin.route_model("写实照片风的猫咪") == qwen      # 写实域
        assert _plugin.route_model("一张带字的海报") == qwen        # 文字渲染域
        assert _plugin.route_model("二次元少女") == anime
        assert _plugin.route_model("画一个涩图") == anime   # R18 词路由到唯一能出的模型
        assert _plugin.route_model("星空下的城市") == default
        assert _plugin.route_model("二次元少女", explicit="qwen") == qwen  # 显式优先
        assert _plugin.route_model("猫", explicit="anime") == anime
        assert _plugin.route_model("猫", explicit="不存在") == default    # 未知别名忽略
        assert _plugin.route_model("猫", explicit="zimage") == _plugin._DEFAULT_MODEL  # 旧默认仍可显式指定

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


class TestNsfwDrawInterceptor:
    """私聊涩图直通道（2026-08-04）：agent 模型对 NSFW 请求空内容自我审查，
    无歧义私聊请求绕过 LLM 直接走 /draw 同链路。"""

    def _ctx(self, _plugin, text, is_group=False):
        from junjun_agent.commands import CommandContext
        session = SimpleNamespace(
            platform="qq", is_group=is_group,
            group_id="999" if is_group else None,
            chat_id="qq:999:group" if is_group else "qq:12345:private")
        meta = SimpleNamespace(text=text, user_id="12345", nickname="甲",
                               at_bot=False, message_id="m1")
        return CommandContext(session=session, meta=meta, args=text)

    @pytest.fixture
    def _env(self, _plugin, monkeypatch):
        import junjun_core.gateway.router as router_mod
        sent = []

        class _FakeGW:
            async def send_reply(self, rs):
                sent.append(rs)
        monkeypatch.setattr(router_mod, "get_gateway", lambda: _FakeGW())
        monkeypatch.setenv("MODELSCOPE_API_KEY", "sk-test")

        async def _fake_submit(**kw):
            return "在弄了，好了直接发出来。"
        monkeypatch.setattr(_plugin.task_manager, "submit", _fake_submit)
        yield sent
        _plugin._PENDING.clear()
        _plugin._last_use.clear()

    @pytest.mark.asyncio
    async def test_private_consumed_and_submits(self, _plugin, _env):
        out = await _plugin.nsfw_draw_hit(self._ctx(_plugin, "画一个涩图"))
        assert out is True
        assert "在弄了" in _env[0].segments[0].data

    @pytest.mark.asyncio
    async def test_group_not_consumed(self, _plugin, _env):
        """群聊不拦截，交给 LLM 按手册婉拒。"""
        out = await _plugin.nsfw_draw_hit(self._ctx(_plugin, "画一个涩图", is_group=True))
        assert out is False
        assert not _env

    @pytest.mark.asyncio
    async def test_negation_not_consumed(self, _plugin, _env):
        """「别画涩图了」是制止不是请求。"""
        out = await _plugin.nsfw_draw_hit(self._ctx(_plugin, "别画涩图了，难看"))
        assert out is False
        assert not _env

    @pytest.mark.asyncio
    async def test_minor_red_line_still_refuses(self, _plugin, _env):
        """未成年红线在直通道同样生效（draw_cmd 的 is_minor_nsfw）。"""
        out = await _plugin.nsfw_draw_hit(self._ctx(_plugin, "画一个萝莉涩图"))
        assert out is True                       # 消费掉 + 文本拒绝
        assert "不行" in _env[0].segments[0].data


class TestGroupR18HardGate:
    """群聊 R18 硬门（2026-08-06 实锤「分不清群聊私聊」：群场景此前只靠模型
    自觉，命令/工具层无兜底）。私聊放开、未成年红线不受此门影响。"""

    def _ctx(self, _plugin, text, is_group=False):
        from junjun_agent.commands import CommandContext
        session = SimpleNamespace(
            platform="qq", is_group=is_group,
            group_id="999" if is_group else None,
            chat_id="qq:999:group" if is_group else "qq:12345:private")
        meta = SimpleNamespace(text=text, user_id="12345", nickname="甲",
                               at_bot=False, message_id="m1")
        return CommandContext(session=session, meta=meta, args=text)

    @pytest.fixture
    def _env(self, _plugin, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_API_KEY", "sk-test")
        submitted = []

        async def _fake_submit(**kw):
            submitted.append(kw)
            return "在弄了，好了直接发出来。"
        monkeypatch.setattr(_plugin.task_manager, "submit", _fake_submit)
        yield submitted
        _plugin._PENDING.clear()
        _plugin._last_use.clear()

    @pytest.mark.asyncio
    async def test_draw_cmd_group_r18_blocked(self, _plugin, _env):
        """/draw 涩图 xxx 在群里：拒绝文案，绝不派单。"""
        ctx = self._ctx(_plugin, "涩图 猫娘", is_group=True)
        ctx.args = "涩图 猫娘"
        out = await _plugin.draw_cmd(ctx)
        assert "私聊" in out and "群里不画" in out
        assert not _env                              # 没派单

    @pytest.mark.asyncio
    async def test_draw_cmd_private_r18_allowed(self, _plugin, _env):
        """私聊放开成年向：正常派单。"""
        ctx = self._ctx(_plugin, "涩图 猫娘")
        ctx.args = "涩图 猫娘"
        out = await _plugin.draw_cmd(ctx)
        assert "在弄了" in out
        assert len(_env) == 1

    @pytest.mark.asyncio
    async def test_draw_cmd_group_sfw_anime_allowed(self, _plugin, _env):
        """群里普通二次元（无 R18 标记）不受硬门影响。"""
        ctx = self._ctx(_plugin, "动漫 猫娘少女", is_group=True)
        ctx.args = "动漫 猫娘少女"
        out = await _plugin.draw_cmd(ctx)
        assert "在弄了" in out
        assert len(_env) == 1

    @pytest.mark.asyncio
    async def test_tool_group_r18_blocked(self, _plugin, _env, monkeypatch):
        """LLM 在群里违规调 ai_draw：工具层拦死，返回引导婉拒文案。"""
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:999:group")
        try:
            out = await _plugin.ai_draw.ainvoke({"prompt": "涩图 猫娘"})
        finally:
            current_chat_id.reset(token)
        assert "群里画不了" in out and "私聊" in out
        assert not _env                              # 没派单

    @pytest.mark.asyncio
    async def test_tool_private_r18_allowed(self, _plugin, _env, monkeypatch):
        """私聊工具路径：正常派单。"""
        from junjun_skills.builtin.memory_skills import current_chat_id
        token = current_chat_id.set("qq:12345:private")
        try:
            out = await _plugin.ai_draw.ainvoke({"prompt": "涩图 猫娘"})
        finally:
            current_chat_id.reset(token)
        assert "在弄了" in out
        assert len(_env) == 1

    def test_has_r18_marker(self, _plugin):
        assert _plugin.has_r18_marker("来张涩图")
        assert _plugin.has_r18_marker("nsfw catgirl")
        assert not _plugin.has_r18_marker("动漫 猫娘少女")
        assert not _plugin.has_r18_marker("蓝色的天空")


class TestAipingProvider:
    """AI Ping 生图（2026-08-12 平台迁移）：同步 /images/generations 协议分流。"""

    def test_registry_has_aiping_aliases(self, _plugin):
        reg = _plugin._model_registry()
        assert reg["kolors"] == "Kolors"
        assert reg["glm-image"] == "GLM-Image"
        assert reg["seedream"] == "Doubao-Seedream-4.0"
        assert reg["zimage"] == _plugin._DEFAULT_MODEL  # 旧别名仍在

    def test_registry_env_override(self, _plugin, monkeypatch):
        monkeypatch.setenv("AI_DRAW_MODEL_KOLORS", "Custom-Kolors-X")
        assert _plugin._model_registry()["kolors"] == "Custom-Kolors-X"
        assert "Custom-Kolors-X" in _plugin._aiping_model_ids()

    @pytest.mark.asyncio
    async def test_generate_branches_to_aiping(self, monkeypatch):
        # 不用 _plugin fixture（它把 generate 换成了假实现，这里要测真分流）
        import junjun_skills.plugins.ai_draw.tools as ad
        called = {}

        async def _aiping(prompt, model):
            called["model"] = model
            return "http://x/ap.png"

        async def _submit(prompt, model, negative=""):
            raise AssertionError("AI Ping 模型不该走 ModelScope 任务流")

        monkeypatch.setattr(ad, "_generate_aiping", _aiping)
        monkeypatch.setattr(ad, "submit_task", _submit)
        url = await ad.generate("猫", "Kolors")
        assert url == "http://x/ap.png"
        assert called["model"] == "Kolors"

    @pytest.mark.asyncio
    async def test_generate_modelscope_branch_untouched(self, monkeypatch):
        import junjun_skills.plugins.ai_draw.tools as ad

        async def _aiping(prompt, model):
            raise AssertionError("ModelScope 模型不该走 AI Ping")

        async def _submit(prompt, model, negative=""):
            return None  # 提交即失败即可——只为验证分流方向

        monkeypatch.setattr(ad, "_generate_aiping", _aiping)
        monkeypatch.setattr(ad, "submit_task", _submit)
        assert await ad.generate("猫", ad._DEFAULT_MODEL) is None

    @pytest.mark.asyncio
    async def test_aiping_downloads_local(self, _plugin, monkeypatch, tmp_path):
        """AI Ping 成品必须落盘本地（2026-08-12 实锤：发远程 URL 让 NapCat 自己下载，
        超时误报失败 -> send_retry 盲补发 -> 用户收两张）。"""
        monkeypatch.setenv("AIPING_API_KEY", "ap-test")
        monkeypatch.setenv("AIPING_BASE_URL", "https://ap.test/api/v1")
        monkeypatch.setattr(_plugin, "TMP_DIR", tmp_path)
        monkeypatch.setattr(_plugin, "_schedule_cleanup", lambda p: None)
        captured = {}
        fake_png = b"\x89PNG" + b"\x00" * 2048

        class _Resp:
            def __init__(self, status=200, payload=None, content=b"", ctype="image/png"):
                self.status_code = status
                self._payload = payload or {}
                self.content = content
                self.text = str(self._payload)
                self.headers = {"content-type": ctype}

            def json(self):
                return self._payload

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, json=None, headers=None):
                captured["post_url"] = url
                captured["payload"] = json
                return _Resp(200, {"data": [{"url": "http://img/ap.png"}]})

            async def get(self, url):
                captured["get_url"] = url
                return _Resp(200, content=fake_png)

        monkeypatch.setattr(_plugin.httpx, "AsyncClient", _Client)
        out = await _plugin._generate_aiping("一只猫", "Kolors")
        assert out and out.endswith(".png")
        p = tmp_path / out.split("\\")[-1].split("/")[-1]
        assert p.exists() and p.read_bytes() == fake_png
        assert captured["post_url"] == "https://ap.test/api/v1/images/generations"
        assert captured["payload"]["model"] == "Kolors"
        assert captured["get_url"] == "http://img/ap.png"

    @pytest.mark.asyncio
    async def test_aiping_download_failure_none(self, _plugin, monkeypatch, tmp_path):
        monkeypatch.setenv("AIPING_API_KEY", "ap-test")
        monkeypatch.setenv("AIPING_BASE_URL", "https://ap.test/api/v1")
        monkeypatch.setattr(_plugin, "TMP_DIR", tmp_path)

        class _Resp:
            def __init__(self, status, payload=None, content=b""):
                self.status_code = status
                self._payload = payload or {}
                self.content = content
                self.text = "err"
                self.headers = {}

            def json(self):
                return self._payload

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                return _Resp(200, {"data": [{"url": "http://img/x.png"}]})

            async def get(self, *a, **kw):
                return _Resp(500)

        monkeypatch.setattr(_plugin.httpx, "AsyncClient", _Client)
        assert await _plugin._generate_aiping("猫", "Kolors") is None

    @pytest.mark.asyncio
    async def test_aiping_http_error_none(self, _plugin, monkeypatch):
        monkeypatch.setenv("AIPING_API_KEY", "ap-test")
        monkeypatch.setenv("AIPING_BASE_URL", "https://ap.test/api/v1")

        class _Resp:
            status_code = 500
            text = "server error"

            def json(self):
                return {}

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **kw):
                return _Resp()

        monkeypatch.setattr(_plugin.httpx, "AsyncClient", _Client)
        assert await _plugin._generate_aiping("猫", "Kolors") is None

    @pytest.mark.asyncio
    async def test_aiping_no_key_none(self, _plugin, monkeypatch):
        monkeypatch.delenv("AIPING_API_KEY", raising=False)
        monkeypatch.delenv("AIPING_BASE_URL", raising=False)
        assert await _plugin._generate_aiping("猫", "Kolors") is None

    @pytest.mark.asyncio
    async def test_dual_key_gate(self, _plugin, monkeypatch):
        """只有 AI Ping key（无 ModelScope key）也应能画——默认模型已走 AI Ping。"""
        monkeypatch.delenv("MODELSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("AIPING_API_KEY", "ap-test")
        assert _plugin._any_provider_key() == "ap-test"
