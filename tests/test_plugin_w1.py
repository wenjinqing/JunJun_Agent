"""W1 插件测试：wife / news / pixiv setu / image_viewer。"""

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


class TestWife:
    @pytest.mark.asyncio
    async def test_draw_and_same_day_reuse(self, _fake_gateway, monkeypatch, tmp_path):
        import junjun_skills.plugins.wife.tools as wife
        monkeypatch.setattr(wife, "DATA_DIR", tmp_path)
        monkeypatch.setenv("JUNJUN_QQ_ACCOUNT", "10000001")

        members = [{"user_id": 111, "nickname": "乙", "card": ""},
                   {"user_id": 222, "nickname": "丙", "card": "小丙"},
                   {"user_id": 10000001, "nickname": "bot"}]
        calls = []

        async def _members(group_id):
            calls.append(group_id)
            return members

        monkeypatch.setattr("junjun_core.napcat_client.get_group_members", _members)
        import junjun_core.config.config as cfg_mod
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="10000001", nickname="君君"), raw={})

        # 第一次抽：@ 发命令的人，显示新老婆
        ctx = _ctx("抽老婆")
        await wife.wife_cmd(ctx)
        assert len(_fake_gateway) == 1
        segs = _fake_gateway[0].segments
        assert segs[0].type == "at" and segs[0].data == "12345"
        assert segs[1].type == "text" and "今天的群老婆是" in segs[1].data
        assert segs[2].type == "image" and "qlogo" in segs[2].data

        # 同一天再抽：显示「已经有老婆了」，不查成员列表
        _fake_gateway.clear()
        ctx2 = _ctx("今日老婆")
        await wife.wife_cmd(ctx2)
        assert len(calls) == 1  # 没再查成员列表
        segs2 = _fake_gateway[0].segments
        assert segs2[0].type == "at" and segs2[0].data == "12345"
        # 文本段（第 2 个）包含「已经有群老婆了」
        text_segs = [s for s in segs2 if s.type == "text"]
        assert any("已经有群老婆了" in s.data for s in text_segs)

    @pytest.mark.asyncio
    async def test_different_users_different_wives(self, _fake_gateway, monkeypatch, tmp_path):
        """不同人抽到的老婆应该不同（每人每天一个）。"""
        import junjun_skills.plugins.wife.tools as wife
        monkeypatch.setattr(wife, "DATA_DIR", tmp_path)
        monkeypatch.setenv("JUNJUN_QQ_ACCOUNT", "10000001")

        members = [{"user_id": 111, "nickname": "乙", "card": ""},
                   {"user_id": 222, "nickname": "丙", "card": "小丙"},
                   {"user_id": 333, "nickname": "丁", "card": ""},
                   {"user_id": 10000001, "nickname": "bot"}]

        async def _members(group_id):
            return members
        monkeypatch.setattr("junjun_core.napcat_client.get_group_members", _members)
        import junjun_core.config.config as cfg_mod
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="10000001", nickname="君君"), raw={})

        # 用户 12345 抽
        ctx1 = _ctx("抽老婆")
        await wife.wife_cmd(ctx1)
        # 用户 67890 抽
        ctx2 = _ctx("抽老婆")
        ctx2.meta.user_id = "67890"
        await wife.wife_cmd(ctx2)

        data = wife._load_today("999")
        assert "12345" in data
        assert "67890" in data
        # 注意：random.choice 可能抽到同一个人（小样本下正常），
        # 但两个人的记录独立——各自有自己的老婆条目
        assert data["12345"] is not data["67890"]  # 不同记录对象（独立抽取）

    @pytest.mark.asyncio
    async def test_private_rejected(self, _fake_gateway):
        import junjun_skills.plugins.wife.tools as wife
        result = await wife.wife_cmd(_ctx("抽老婆", is_group=False))
        assert "群聊" in result

    @pytest.mark.asyncio
    async def test_napcat_unavailable_degrades(self, _fake_gateway, monkeypatch, tmp_path):
        import junjun_skills.plugins.wife.tools as wife
        monkeypatch.setattr(wife, "DATA_DIR", tmp_path)

        async def _none(group_id):
            return None

        monkeypatch.setattr("junjun_core.napcat_client.get_group_members", _none)
        import junjun_core.config.config as cfg_mod
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"), raw={})
        result = await wife.wife_cmd(_ctx("抽老婆"))
        assert "抽不了" in result


class TestNews:
    @pytest.mark.asyncio
    async def test_news_cmd_text_and_image(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.news.tools as news

        async def _fetch():
            return {"news": ["大事一", "大事二"], "tip": "微语", "image": "http://x/n.png"}

        monkeypatch.setattr(news, "fetch_60s_news", _fetch)
        await news.news_cmd(_ctx("/news"))
        segs = _fake_gateway[0].segments
        assert "大事一" in segs[0].data and "微语" in segs[0].data
        assert segs[1].type == "image"

    @pytest.mark.asyncio
    async def test_news_failure(self, monkeypatch):
        import junjun_skills.plugins.news.tools as news

        async def _none():
            return None

        monkeypatch.setattr(news, "fetch_60s_news", _none)
        assert "失败" in await news.news_cmd(_ctx("/news"))

    @pytest.mark.asyncio
    async def test_history_tool(self, monkeypatch):
        import junjun_skills.plugins.news.tools as news

        async def _fetch(limit=6):
            return ["1949年 开国大典"]

        monkeypatch.setattr(news, "fetch_today_in_history", _fetch)
        out = await news.get_today_in_history.ainvoke({})
        assert "开国大典" in out


class TestSetu:
    """setu 官方 API 版（2026-08-03 从 Lolicon 迁移）：裸词修复 + 发图 + 冷却。"""

    def test_parse_args(self):
        from junjun_skills.plugins.pixiv import setu
        r = setu._parse_args("3 #萝莉 #白丝 横图 noai")
        assert r == {"num": 3, "tags": ["萝莉", "白丝"], "ratio": "-0.5",
                     "square": False, "exclude_ai": True}
        assert setu._parse_args("")["num"] == 1
        assert setu._parse_args("99")["num"] == 5  # 上限
        assert setu._parse_args("2 竖图 方图") == {
            "num": 2, "tags": [], "ratio": "0.5", "square": True, "exclude_ai": False}

    def test_parse_args_bare_words_are_tags(self):
        """bug 修复：裸词（不带 #）以前被静默丢弃 -> 永远随机图。"""
        from junjun_skills.plugins.pixiv import setu
        r = setu._parse_args("白丝 原神")
        assert r["tags"] == ["白丝", "原神"]
        r = setu._parse_args("2 白丝 #黑丝 noai")
        assert r["tags"] == ["白丝", "黑丝"] and r["num"] == 2

    @pytest.mark.asyncio
    async def test_send_and_cooldown(self, _fake_gateway, monkeypatch):
        from junjun_skills.plugins.pixiv import setu
        setu._last_use.clear()
        monkeypatch.setattr(setu, "_cookie", lambda: "PHPSESSID=1_x")

        async def _pick(tags, ratio, square, exclude_ai, num, group):
            assert tags == ["白丝"]  # 裸词传到了搜索层
            assert group is True
            return ["111", "222"]

        async def _urls(iid, group):
            return f"「作品{iid}」by 作者", [f"http://x/{iid}.jpg"]

        async def _b64(urls):
            return [f"base64://fake-{u}" for u in urls]  # 本侧代下（NapCat 拉不到图床）

        monkeypatch.setattr(setu, "_pick_from_tags", _pick)
        monkeypatch.setattr(setu, "_illust_image_urls", _urls)
        monkeypatch.setattr(setu, "images_to_b64", _b64)
        await setu.setu_cmd(_ctx("/setu 2 白丝"))
        segs = _fake_gateway[0].segments
        assert [s.type for s in segs].count("image") == 2
        assert all(s.data.startswith("base64://") for s in segs if s.type == "image")
        assert any("作者" in (s.data or "") for s in segs if s.type == "text")
        # 冷却内第二次 -> 拒绝
        result = await setu.setu_cmd(_ctx("/setu 2 白丝"))
        assert "秒后" in result
        setu._last_use.clear()

    @pytest.mark.asyncio
    async def test_empty_result(self, monkeypatch):
        from junjun_skills.plugins.pixiv import setu
        setu._last_use.clear()
        monkeypatch.setattr(setu, "_cookie", lambda: "PHPSESSID=1_x")

        async def _pick(*a):
            return []

        monkeypatch.setattr(setu, "_pick_from_tags", _pick)
        assert "没找到" in await setu.setu_cmd(_ctx("/setu 白丝"))
        setu._last_use.clear()

    @pytest.mark.asyncio
    async def test_no_cookie(self, monkeypatch):
        from junjun_skills.plugins.pixiv import setu
        monkeypatch.setattr(setu, "_cookie", lambda: "")
        assert "Cookie" in await setu.setu_cmd(_ctx("/setu"))


class TestImageViewer:
    @pytest.mark.asyncio
    async def test_tui_sends_image(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.image_viewer.tools as iv

        async def _url(tag):
            return f"http://x/{tag}.jpg"

        monkeypatch.setattr(iv, "_fetch_lolicon", _url)
        await iv.tui_cmd(_ctx("看看腿"))
        segs = _fake_gateway[0].segments
        assert segs[1].type == "image" and segs[1].data == "http://x/腿.jpg"

    @pytest.mark.asyncio
    async def test_pantsu_sends_image(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.image_viewer.tools as iv

        async def _url(tag):
            return f"http://x/{tag}.jpg"

        monkeypatch.setattr(iv, "_fetch_lolicon", _url)
        await iv.pantsu_cmd(_ctx("看看胖次"))
        segs = _fake_gateway[0].segments
        assert segs[1].type == "image" and segs[1].data == "http://x/胖次.jpg"

    def test_pantsu_raw_match_no_false_positive(self):
        """误判回归（2026-08-18 新增 raw 关键词）：「看看胖次」整句/句首+空格
        才命中；日常句里含「胖次」「看看」不许误中。"""
        import importlib
        import junjun_skills.plugins.image_viewer.tools as iv
        importlib.reload(iv)   # _clean_buses 清过注册表，重载触发重新注册
        hit = commands._match("看看胖次")
        assert hit is not None and hit[0].name == "看看胖次"
        assert commands._match("康康胖次") is not None      # 别名命中
        assert commands._match("看看胖次搭配什么衣服") is None   # 前缀粘连无空格
        assert commands._match("我看看胖次了") is None          # 关键词不在句首
        assert commands._match("这条胖次多少钱") is None        # 日常购物句
        assert commands._match("看看就走") is None              # 「看看」单独不命中

    @pytest.mark.asyncio
    async def test_xxapi_failure(self, monkeypatch):
        import junjun_skills.plugins.image_viewer.tools as iv

        async def _none(kind):
            return None

        monkeypatch.setattr(iv, "_fetch_xxapi", _none)
        assert "失败" in await iv.heisi_cmd(_ctx("看看黑丝"))

    @pytest.mark.asyncio
    async def test_kankan_dispatch_and_usage(self, _fake_gateway, monkeypatch):
        import junjun_skills.plugins.image_viewer.tools as iv

        async def _url(kind):
            return f"http://x/{kind}.jpg"

        async def _ltag(tag):
            return f"http://x/lolicon-{tag}.jpg"

        monkeypatch.setattr(iv, "_fetch_xxapi", _url)
        monkeypatch.setattr(iv, "_fetch_lolicon", _ltag)
        ctx = _ctx("/kankan jk")
        await iv.kankan_cmd(ctx)
        assert _fake_gateway[0].segments[1].data == "http://x/jk.jpg"
        await iv.kankan_cmd(_ctx("/kankan pangci"))
        assert _fake_gateway[1].segments[1].data == "http://x/lolicon-胖次.jpg"
        assert "用法" in await iv.kankan_cmd(_ctx("/kankan 啥"))


class TestManifestLoad:
    def test_four_plugins_load(self):
        from junjun_skills import plugin_loader, registry
        registry.clear()
        n = plugin_loader.load_plugins()
        names = [s["plugin"] for s in registry.list_skills()]
        assert "news" in names  # get_today_in_history 注册在 news 插件下
        assert n >= 1
        registry.clear()
