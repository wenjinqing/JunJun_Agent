"""maizone 插件测试：g_tk 算法 / cookie 缓存链 / 发说说 / 看空间 / 监控去重与上限。

全部网络调用 monkeypatch：napcat_client.call、Qzone API helper、LLM。
配置注入参照 test_plugin_w1.py 的 cfg_mod.global_config 方式（raw={"maizone": {...}}）。
"""

import json
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


def _set_config(monkeypatch, maizone: dict):
    """注入全局配置（bot uin=12345 + [maizone] 节）。"""
    import junjun_core.config.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "global_config", cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="12345", nickname="君君"),
        raw={"maizone": maizone, "personality": {"personality": "测试人格", "reply_style": "简短"}}))


@pytest.fixture
def _tools(monkeypatch, tmp_path):
    """导入插件并隔离数据目录 + 假登录态。"""
    import junjun_skills.plugins.maizone.tools as mz
    monkeypatch.setattr(mz, "DATA_DIR", tmp_path)

    async def _cookies(force_refresh=False):
        return {"skey": "testskey", "p_skey": "testpskey"}

    monkeypatch.setattr(mz, "ensure_cookies", _cookies)
    return mz


class TestGtk:
    def test_known_vectors(self):
        """g_tk 哈希对已知输入的确定性（向量由旧插件算法算出后硬编码）。"""
        import junjun_skills.plugins.maizone.tools as mz
        assert mz.generate_gtk("abc") == "193485963"
        assert mz.generate_gtk("@bcdABC123xyz") == "1088769333"
        # 两次计算一致
        assert mz.generate_gtk("testskey") == mz.generate_gtk("testskey")


class TestCookies:
    @pytest.mark.asyncio
    async def test_cache_reuse_and_force_refresh(self, monkeypatch, tmp_path):
        """缓存复用：第二次不再调 NapCat；force_refresh 会重新取并落盘。"""
        import junjun_skills.plugins.maizone.tools as mz
        monkeypatch.setattr(mz, "DATA_DIR", tmp_path)
        _set_config(monkeypatch, {})

        calls = []

        async def _call(action, params=None, timeout=15.0):
            calls.append((action, params))
            if params and params.get("domain") == "qzone.qq.com":
                return {"cookies": "skey=AAA; p_skey=BBB"}
            if params and params.get("domain") == "user.qzone.qq.com":
                return {"cookies": "uin=o12345"}
            return None

        monkeypatch.setattr(mz.napcat_client, "call", _call)

        cookies = await mz.ensure_cookies()
        assert cookies["skey"] == "AAA" and cookies["p_skey"] == "BBB"
        n_after_first = len(calls)
        assert n_after_first == 2  # 两个域名各一次

        # 第二次：命中缓存，不再调 NapCat
        cookies2 = await mz.ensure_cookies()
        assert cookies2["p_skey"] == "BBB"
        assert len(calls) == n_after_first

        # force_refresh：重新调 NapCat
        await mz.ensure_cookies(force_refresh=True)
        assert len(calls) > n_after_first

        # 缓存文件命名 cookies-<uin>.json
        assert (tmp_path / "cookies-12345.json").exists()

    @pytest.mark.asyncio
    async def test_all_layers_fail_returns_none(self, monkeypatch, tmp_path):
        """三层都失败（NapCat 不可用且无缓存）→ None。"""
        import junjun_skills.plugins.maizone.tools as mz
        monkeypatch.setattr(mz, "DATA_DIR", tmp_path)
        _set_config(monkeypatch, {})

        async def _none(action, params=None, timeout=15.0):
            return None

        monkeypatch.setattr(mz.napcat_client, "call", _none)
        assert await mz.ensure_cookies() is None


class TestSendFeed:
    @pytest.mark.asyncio
    async def test_full_chain(self, _tools, monkeypatch):
        """/send_feed 全链路：LLM 生成 + 发布成功 → 回执含内容。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "send_enable": True})

        async def _llm(prompt):
            return "今天天气真好，适合发呆"

        published = []

        async def _publish(cookies, uin, content):
            published.append((uin, content))
            return "tid999"

        monkeypatch.setattr(mz, "_ask_llm", _llm)
        monkeypatch.setattr(mz, "publish_feed", _publish)

        result = await mz.send_feed_cmd(_ctx("/send_feed 天气"))
        assert "今天天气真好" in result
        assert published == [("12345", "今天天气真好，适合发呆")]

    @pytest.mark.asyncio
    async def test_publish_failure_degrades(self, _tools, monkeypatch):
        """发布接口抛错 → 友好降级文本。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "send_enable": True})

        async def _llm(prompt):
            return "内容"

        async def _boom(cookies, uin, content):
            raise RuntimeError("code=-1 服务器繁忙")

        monkeypatch.setattr(mz, "_ask_llm", _llm)
        monkeypatch.setattr(mz, "publish_feed", _boom)
        result = await mz.send_feed_cmd(_ctx("发说说"))
        assert "失败" in result

    @pytest.mark.asyncio
    async def test_cookie_fail_message(self, _tools, monkeypatch):
        """登录态三层都拿不到 → 「空间登录态获取失败」。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "send_enable": True})

        async def _llm(prompt):
            return "内容"

        async def _no_cookies(force_refresh=False):
            return None

        monkeypatch.setattr(mz, "_ask_llm", _llm)
        monkeypatch.setattr(mz, "ensure_cookies", _no_cookies)
        result = await mz.send_feed_cmd(_ctx("/send_feed"))
        assert "空间登录态获取失败" in result

    @pytest.mark.asyncio
    async def test_disabled(self, _tools, monkeypatch):
        """enable=false：不生成不发布，直接回未开启。"""
        mz = _tools
        _set_config(monkeypatch, {})
        called = []

        async def _llm(prompt):
            called.append(prompt)
            return "x"

        monkeypatch.setattr(mz, "_ask_llm", _llm)
        result = await mz.send_feed_cmd(_ctx("/send_feed"))
        assert "没开" in result and not called


class TestReadFeed:
    @pytest.mark.asyncio
    async def test_summary(self, _tools, monkeypatch):
        """/read_feed 摘要：作者/内容/时间都在回复里。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "read_enable": True})

        feeds = [
            {"target_qq": "67890", "tid": "t1", "nickname": "小明",
             "content": "今天去爬山了", "created_time": "昨天17:50"},
            {"target_qq": "22222", "tid": "t2", "nickname": "",
             "content": "加班好痛苦", "created_time": "10分钟前"},
        ]

        async def _fetch(cookies, uin, num):
            assert num == 3  # 参数透传
            return feeds

        monkeypatch.setattr(mz, "fetch_friend_feeds", _fetch)
        result = await mz.read_feed_cmd(_ctx("/read_feed 3"))
        assert "小明" in result and "今天去爬山了" in result and "昨天17:50" in result
        assert "22222" in result  # 无昵称时回退 QQ 号

    @pytest.mark.asyncio
    async def test_empty(self, _tools, monkeypatch):
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "read_enable": True})

        async def _fetch(cookies, uin, num):
            return []

        monkeypatch.setattr(mz, "fetch_friend_feeds", _fetch)
        result = await mz.read_feed_cmd(_ctx("看空间"))
        assert "没有新说说" in result


class TestMonitor:
    def _feeds(self):
        return [{"target_qq": "67890", "tid": "t1", "nickname": "小明",
                 "content": "新手机到手", "created_time": "1小时前"}]

    def _patch_ops(self, mz, monkeypatch):
        calls = {"like": [], "comment": []}

        async def _fetch(cookies, uin, num):
            return self._feeds()

        async def _like(cookies, uin, target_qq, fid):
            calls["like"].append(fid)
            return True

        async def _comment(cookies, uin, target_qq, fid, content):
            calls["comment"].append((fid, content))
            return True

        async def _llm(prompt):
            return "好看！"

        monkeypatch.setattr(mz, "fetch_friend_feeds", _fetch)
        monkeypatch.setattr(mz, "like_feed", _like)
        monkeypatch.setattr(mz, "comment_feed", _comment)
        monkeypatch.setattr(mz, "_ask_llm", _llm)
        return calls

    @pytest.mark.asyncio
    async def test_like_comment_and_dedup(self, _tools, monkeypatch, tmp_path):
        """监控点赞+评论，同一条说说不重复处理；处理记录落盘。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "monitor_enable": True,
                                  "like_enable": True, "comment_enable": True})
        calls = self._patch_ops(mz, monkeypatch)

        await mz.maizone_monitor()
        assert calls["like"] == ["t1"]
        assert calls["comment"] == [("t1", "好看！")]

        # 第二次：processed 已记录，不再点赞/评论
        await mz.maizone_monitor()
        assert calls["like"] == ["t1"]
        assert len(calls["comment"]) == 1

        processed = json.loads((tmp_path / "processed_list.json").read_text(encoding="utf-8"))
        assert "67890_t1" in processed

    @pytest.mark.asyncio
    async def test_daily_comment_limit(self, _tools, monkeypatch):
        """日评论上限：max_reply_per_day=1 时两条说说只评论一条（点赞不限）。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "monitor_enable": True,
                                  "like_enable": True, "comment_enable": True,
                                  "max_reply_per_day": 1})
        calls = self._patch_ops(mz, monkeypatch)

        async def _fetch2(cookies, uin, num):
            return self._feeds() + [{"target_qq": "33333", "tid": "t2",
                                     "nickname": "小红", "content": "晚安",
                                     "created_time": "刚刚"}]

        monkeypatch.setattr(mz, "fetch_friend_feeds", _fetch2)
        await mz.maizone_monitor()
        assert len(calls["comment"]) == 1   # 只评论了第一条
        assert len(calls["like"]) == 2      # 点赞不受评论上限限制
        assert mz._daily_comment_count() == 1

    @pytest.mark.asyncio
    async def test_disabled_noop(self, _tools, monkeypatch):
        """enable=false：监控任务完全不动（不拉列表）。"""
        mz = _tools
        _set_config(monkeypatch, {})
        hit = []

        async def _fetch(cookies, uin, num):
            hit.append(1)
            return self._feeds()

        monkeypatch.setattr(mz, "fetch_friend_feeds", _fetch)
        await mz.maizone_monitor()
        assert not hit


class TestStatus:
    @pytest.mark.asyncio
    async def test_qzone_status(self, _tools, monkeypatch, tmp_path):
        """/qzone_status：cookie 状态 + 今日评论数 + 开关。"""
        mz = _tools
        _set_config(monkeypatch, {"enable": True, "send_enable": True,
                                  "max_reply_per_day": 5})
        # 造一份有效缓存
        (tmp_path / "cookies-12345.json").write_text(
            json.dumps({"skey": "a", "p_skey": "b"}), encoding="utf-8")
        mz._incr_daily_comment()

        result = await mz.qzone_status_cmd(_ctx("/qzone_status"))
        assert "cookie 缓存有效" in result
        assert "今日已评论：1/5" in result
        assert "enable=开" in result and "monitor_enable=关" in result


class TestJsonp:
    def test_strip_jsonp(self):
        """JSONP 剥壳：_Callback(...) 与 undefined 替换。"""
        import junjun_skills.plugins.maizone.tools as mz
        raw = '_Callback({"code":0,"data":{"data":[]}, "x":undefined});'
        payload = json.loads(mz._strip_jsonp(raw))
        assert payload["code"] == 0 and payload["x"] is None
        # 纯 JSON 原样通过
        assert json.loads(mz._strip_jsonp('{"a":1}'))["a"] == 1
