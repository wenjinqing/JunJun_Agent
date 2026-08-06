"""号池测试：文件解析/余额巡检/丢弃与复活/轮转/热加载/models.py 展开集成。

HTTP 全部打桩（_check_key 层），不发真实请求。
"""

import time
from pathlib import Path

import pytest

import junjun_core.config.config as cfg_mod
from junjun_llm.key_pool import SFKeyPool


@pytest.fixture
def pool(tmp_path, monkeypatch):
    old = cfg_mod.global_config
    cfg_mod.global_config = cfg_mod.GlobalConfig(
        bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
        raw={"key_pool": {"min_balance": 0.5, "check_hours": 2, "max_legs": 3}})
    f = tmp_path / "sf_keys.txt"
    f.write_text("# 注释行\nsk-aaa\n\nsk-bbb\nsk-ccc\n", encoding="utf-8")
    p = SFKeyPool(path=f)
    yield p
    cfg_mod.global_config = old


class TestFile:
    def test_parse(self, pool):
        assert pool.healthy_keys() and len(pool._keys) == 3
        assert "sk-aaa" in pool._keys and "# 注释行" not in pool._keys

    def test_missing_file_empty(self, tmp_path):
        p = SFKeyPool(path=tmp_path / "nope.txt")
        assert p.healthy_keys() == []

    def test_hot_reload(self, pool):
        first = pool.healthy_keys()
        assert len(first) == 3
        time.sleep(0.02)
        pool._path.write_text("sk-ddd\nsk-eee\n", encoding="utf-8")
        import os
        os.utime(pool._path, (time.time() + 1, time.time() + 1))  # 确保 mtime 变化
        keys = pool.healthy_keys()
        assert set(keys) == {"sk-ddd", "sk-eee"}


class TestRefresh:
    @pytest.mark.asyncio
    async def test_dead_and_revive(self, pool, monkeypatch):
        # 第一轮：bbb 低余额丢弃，ccc 401 丢弃
        async def fake1(client, base, key):
            return {"sk-aaa": ("alive", "¥9"), "sk-bbb": ("dead", "余额 ¥0.10"),
                    "sk-ccc": ("dead", "key 无效(401)")}[key]
        monkeypatch.setattr(pool, "_check_key", fake1)
        await pool.refresh()
        assert pool.healthy_keys() == ["sk-aaa"]
        assert set(pool._dead) == {"sk-bbb", "sk-ccc"}

        # 第二轮：bbb top up 复活，ccc 仍死，aaa 网络 unknown 保持活
        async def fake2(client, base, key):
            return {"sk-aaa": ("unknown", "http 500"), "sk-bbb": ("alive", "¥5"),
                    "sk-ccc": ("dead", "key 无效(401)")}[key]
        monkeypatch.setattr(pool, "_check_key", fake2)
        await pool.refresh()
        assert set(pool.healthy_keys()) == {"sk-aaa", "sk-bbb"}
        assert set(pool._dead) == {"sk-ccc"}

    @pytest.mark.asyncio
    async def test_exception_keeps_dead_state(self, pool, monkeypatch):
        """网络层异常：已死的保持死（不可复活错杀反过来也不可错放）。"""
        pool._dead = {"sk-bbb": "余额 ¥0.10"}

        async def boom(client, base, key):
            raise OSError("网络炸了")
        monkeypatch.setattr(pool, "_check_key", boom)
        await pool.refresh()
        assert "sk-bbb" in pool._dead  # 保持死
        assert set(pool.healthy_keys()) == {"sk-aaa", "sk-ccc"}


class TestRotation:
    def test_rotation_spreads_start(self, pool):
        seen = []
        for _ in range(3):
            keys = pool.healthy_keys()
            seen.append(keys[0])
        assert len(set(seen)) == 3  # 每次起始 key 不同，负载摊开


class TestGeneration:
    """健康集合变化代数：models 槽缓存的失效信号（2026-08-06 欠费 key 不出链修复）。"""

    def test_first_load_bumps(self, pool):
        assert pool.generation == 0
        pool.healthy_keys()
        assert pool.generation == 1

    def test_unchanged_file_no_bump(self, pool):
        pool.healthy_keys()
        g = pool.generation
        pool.healthy_keys()
        assert pool.generation == g

    def test_file_change_bumps(self, pool):
        pool.healthy_keys()
        g = pool.generation
        pool._path.write_text("sk-zzz\n", encoding="utf-8")
        import os
        os.utime(pool._path, (time.time() + 1, time.time() + 1))  # 确保 mtime 变化
        pool.healthy_keys()
        assert pool.generation == g + 1

    @pytest.mark.asyncio
    async def test_refresh_dead_change_bumps_once(self, pool, monkeypatch):
        """巡检杀死 key：代数 +1；下一轮结果不变不重复 bump（防无谓重建链）。"""
        pool.healthy_keys()
        g = pool.generation

        async def fake(client, base, key):
            return ("dead", "余额 ¥0") if key == "sk-bbb" else ("alive", "¥9")
        monkeypatch.setattr(pool, "_check_key", fake)
        await pool.refresh()
        assert pool.generation == g + 1
        await pool.refresh()
        assert pool.generation == g + 1

    def test_tick_no_rotation_consumed(self, pool):
        """tick 是心跳不是取号：触发加载/懒巡检但不消耗轮转游标。"""
        pool.tick()
        assert pool._rr == 0
        assert len(pool._keys) == 3


class _Resp:
    def __init__(self, code, payload=None, text=""):
        self.status_code = code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, info: _Resp, probe: "_Resp | None" = None):
        self._info = info
        self._probe = probe
        self.probe_called = False

    async def get(self, url, headers=None):
        return self._info

    async def post(self, url, headers=None, json=None):
        self.probe_called = True
        return self._probe


class TestCheckKey:
    """两阶段判定：余额够直接活；余额 0 走代金券探针（2026-08 实测场景）。"""

    @pytest.mark.asyncio
    async def test_rich_balance_alive_no_probe(self, pool):
        client = _FakeClient(_Resp(200, {"data": {"totalBalance": "9.9"}}))
        state, _ = await pool._check_key(client, "https://x/v1", "sk-a")
        assert state == "alive" and not client.probe_called

    @pytest.mark.asyncio
    async def test_zero_balance_voucher_alive(self, pool):
        """余额 0 + 探针 200 -> 代金券账号，活。"""
        client = _FakeClient(_Resp(200, {"data": {"totalBalance": "0"}}),
                             _Resp(200))
        state, reason = await pool._check_key(client, "https://x/v1", "sk-a")
        assert state == "alive" and "代金券" in reason and client.probe_called

    @pytest.mark.asyncio
    async def test_zero_balance_probe_402_dead(self, pool):
        client = _FakeClient(_Resp(200, {"data": {"totalBalance": "0"}}),
                             _Resp(402, text='{"error":"Insufficient balance"}'))
        state, _ = await pool._check_key(client, "https://x/v1", "sk-a")
        assert state == "dead"

    @pytest.mark.asyncio
    async def test_probe_400_insufficient_text_dead(self, pool):
        """有的网关余额不足返回 400 + 文案，也得认。"""
        client = _FakeClient(_Resp(200, {"data": {"totalBalance": "0"}}),
                             _Resp(400, text='{"error":{"message":"账户余额不足"}}'))
        state, _ = await pool._check_key(client, "https://x/v1", "sk-a")
        assert state == "dead"

    @pytest.mark.asyncio
    async def test_probe_429_alive(self, pool):
        client = _FakeClient(_Resp(200, {"data": {"totalBalance": "0"}}),
                             _Resp(429))
        state, _ = await pool._check_key(client, "https://x/v1", "sk-a")
        assert state == "alive"

    @pytest.mark.asyncio
    async def test_info_401_dead_no_probe(self, pool):
        client = _FakeClient(_Resp(401))
        state, _ = await pool._check_key(client, "https://x/v1", "sk-a")
        assert state == "dead" and not client.probe_called


class TestModelsIntegration:
    @pytest.fixture(autouse=True)
    def _cfg(self):
        old = cfg_mod.global_config
        cfg_mod.global_config = cfg_mod.GlobalConfig(
            bot=cfg_mod.BotConfig(platform="qq", qq_account="1", nickname="君君"),
            raw={"key_pool": {"min_balance": 0.5, "check_hours": 2, "max_legs": 3}})
        yield
        cfg_mod.global_config = old

    def test_pool_entry_expands(self, tmp_path, monkeypatch):
        """api_key_env=SF_POOL 的条目展开成 N 条腿 + 静态条目兜底。"""
        import junjun_llm.models as models
        toml = tmp_path / "model_config.toml"
        toml.write_text("""
[task.agent]
temperature = 0.6
max_tokens = 2048
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_LLM_MODEL"
api_key_env = "SF_POOL"
[[task.agent.models]]
base_url_env = "DS_BASE_URL"
model_env = "DS_MODEL"
api_key_env = "DEEPSEEK_API_KEY"
""", encoding="utf-8")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("SF_LLM_MODEL", "Qwen/Qwen3.5-397B-A17B")
        monkeypatch.setenv("DS_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setenv("DS_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        # 池里 4 个健康 key，max_legs=3（上面的 fixture config）
        monkeypatch.setattr("junjun_llm.key_pool.sf_pool.healthy_keys",
                            lambda: ["k1", "k2", "k3", "k4"])
        models.reset_slots()
        slot = models._load_slots()["agent"]
        keys = [s.api_key for s in slot.specs]
        assert keys == ["k1", "k2", "k3", "ds-key"]  # 池腿×3（上限）+ DS 兜底
        assert slot.specs[0].temperature == 0.6
        models.reset_slots()

    def test_empty_pool_falls_back_to_static(self, tmp_path, monkeypatch):
        """池空：号池条目缺席，静态条目照常工作。"""
        import junjun_llm.models as models
        toml = tmp_path / "model_config.toml"
        toml.write_text("""
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_LLM_MODEL"
api_key_env = "SF_POOL"
[[task.agent.models]]
base_url_env = "DS_BASE_URL"
model_env = "DS_MODEL"
api_key_env = "DEEPSEEK_API_KEY"
""", encoding="utf-8")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("SF_LLM_MODEL", "Qwen/x")
        monkeypatch.setenv("DS_BASE_URL", "https://api.deepseek.com")
        monkeypatch.setenv("DS_MODEL", "deepseek-v4-flash")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
        monkeypatch.setattr("junjun_llm.key_pool.sf_pool.healthy_keys", lambda: [])
        models.reset_slots()
        slot = models._load_slots()["agent"]
        assert [s.api_key for s in slot.specs] == ["ds-key"]
        models.reset_slots()

    def test_dead_key_evicted_on_rebuild(self, tmp_path, monkeypatch):
        """巡检标死 -> 号池代数变 -> get_chat_model 重建链，欠费 key 出链
        （2026-08-06 实锤：标死没人重建链，死 key 永远留在 fallback 链上）。
        也覆盖启动竞态：首轮展开含全部 key，巡检完成后下一轮取模型即剔除。"""
        import junjun_llm.models as models
        from junjun_llm.key_pool import sf_pool
        toml = tmp_path / "model_config.toml"
        toml.write_text("""
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_LLM_MODEL"
api_key_env = "SF_POOL"
""", encoding="utf-8")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("SF_LLM_MODEL", "Qwen/x")
        # tick 打桩：不碰真实 data/sf_keys.txt、不触发真实巡检
        monkeypatch.setattr(sf_pool, "tick", lambda: None)
        alive = ["k1", "k2"]
        monkeypatch.setattr(sf_pool, "healthy_keys", lambda: list(alive))
        monkeypatch.setattr(sf_pool, "generation", 1)
        models.reset_slots()
        models.get_chat_model("agent")
        assert [s.api_key for s in models._load_slots()["agent"].specs] == ["k1", "k2"]

        # 巡检杀死 k2：代数 +1，健康列表缩短 -> 下次取模型自动重建链
        alive[:] = ["k1"]
        monkeypatch.setattr(sf_pool, "generation", 2)
        models.get_chat_model("agent")
        assert [s.api_key for s in models._load_slots()["agent"].specs] == ["k1"]
        models.reset_slots()

    def test_unchanged_generation_no_rebuild(self, tmp_path, monkeypatch):
        """号池状态没变：槽缓存保持复用，不每轮重建（防性能回退）。"""
        import junjun_llm.models as models
        from junjun_llm.key_pool import sf_pool
        toml = tmp_path / "model_config.toml"
        toml.write_text("""
[[task.agent.models]]
base_url_env = "SF_LLM_BASE_URL"
model_env = "SF_LLM_MODEL"
api_key_env = "SF_POOL"
""", encoding="utf-8")
        monkeypatch.setattr(models, "MODEL_CONFIG_PATH", toml)
        monkeypatch.setenv("SF_LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        monkeypatch.setenv("SF_LLM_MODEL", "Qwen/x")
        monkeypatch.setattr(sf_pool, "tick", lambda: None)
        monkeypatch.setattr(sf_pool, "healthy_keys", lambda: ["k1"])
        monkeypatch.setattr(sf_pool, "generation", 5)
        models.reset_slots()
        models.get_chat_model("agent")
        cached = models._slots
        models.get_chat_model("agent")
        assert models._slots is cached  # 同一代数：缓存对象不变
        models.reset_slots()
