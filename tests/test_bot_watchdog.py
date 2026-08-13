"""bot/adapter 看门狗决策逻辑测试（2026-08-13 审查 P1）。

supervise_once 依赖注入 spawn/probe——假进程假探针全覆盖决策面，
不拉真进程、不碰网络。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bot_watchdog import (BACKOFF, BOOT_GRACE, MAX_RESTARTS_PER_HOUR,
                                  PROBE_FAIL_LIMIT, ProcState, supervise_once)


class _FakeProc:
    def __init__(self, alive=True):
        self._alive = alive
        self.killed = False
        self.pid = 1234

    def poll(self):
        return None if self._alive else 1

    def kill(self):
        self.killed = True
        self._alive = False


class _FakeSpawn:
    """spawn 工厂：产出的假进程可用 .procs[-1]._alive=False 杀死。"""

    def __init__(self):
        self.procs = []

    def __call__(self, cmd):
        p = _FakeProc(alive=True)
        self.procs.append(p)
        return p


def _st(port=0):
    return ProcState("bot", ["py", "x.py"], port=port)


class TestSupervise:
    def test_first_spawn(self):
        sp = _FakeSpawn()
        assert supervise_once(_st(), 1000.0, spawn=sp, probe=lambda p: True) == "spawn"
        assert len(sp.procs) == 1

    def test_alive_healthy_waits(self):
        sp = _FakeSpawn()
        st = _st()
        supervise_once(st, 1000.0, spawn=sp, probe=lambda p: True)
        assert supervise_once(st, 1005.0, spawn=sp, probe=lambda p: True) == "wait"
        assert len(sp.procs) == 1, "健康时不许重复拉起"

    def test_dead_respawns_with_backoff(self):
        sp = _FakeSpawn()
        st = _st()
        supervise_once(st, 1000.0, spawn=sp, probe=lambda p: True)
        sp.procs[-1]._alive = False
        # 立刻巡到死亡：退避 5s 内只等不拉
        assert supervise_once(st, 1001.0, spawn=sp, probe=lambda p: True) == "wait"
        assert len(sp.procs) == 1
        assert supervise_once(st, 1000.0 + BACKOFF[0] + 0.1,
                              spawn=sp, probe=lambda p: True) == "spawn"
        assert len(sp.procs) == 2
        assert st.next_retry == 1005.1 + BACKOFF[1], "第二次重启退避升档"

    def test_crash_loop_gives_up(self):
        """一小时重启到熔断线 -> give_up 且不再拉起（防刷 QQ 风控）。"""
        sp = _FakeSpawn()
        st = _st()
        now = 1000.0
        supervise_once(st, now, spawn=sp, probe=lambda p: True)
        for _ in range(MAX_RESTARTS_PER_HOUR):
            sp.procs[-1]._alive = False
            now = st.next_retry + 0.1
            supervise_once(st, now, spawn=sp, probe=lambda p: True)
        assert st.gave_up
        sp.procs[-1]._alive = False
        assert supervise_once(st, st.next_retry + 999,
                              spawn=sp, probe=lambda p: True) == "give_up"
        assert len(sp.procs) == MAX_RESTARTS_PER_HOUR, "熔断后不许再拉"

    def test_restart_counter_window_slides(self):
        """重启计数只算近一小时——昨天的崩溃不惩罚今天。"""
        sp = _FakeSpawn()
        st = _st()
        supervise_once(st, 1000.0, spawn=sp, probe=lambda p: True)
        st.restarts = [1000.0] * MAX_RESTARTS_PER_HOUR  # 全是一小时前的
        sp.procs[-1]._alive = False
        later = 1000.0 + 7200
        st.next_retry = 0
        assert supervise_once(st, later, spawn=sp, probe=lambda p: True) == "spawn"
        assert not st.gave_up

    def test_port_probe_kills_zombie(self):
        """进程活着但端口连续探测失败 -> 按死亡处理（半死态）。"""
        sp = _FakeSpawn()
        st = _st(port=8192)
        supervise_once(st, 1000.0, spawn=sp, probe=lambda p: True)
        t = 1000.0 + BOOT_GRACE + 1
        for i in range(PROBE_FAIL_LIMIT - 1):
            assert supervise_once(st, t + i, spawn=sp, probe=lambda p: False) == "wait"
        # 到阈值：当轮杀+直接重拉（探测失败的连续等待本身就是退避）
        assert supervise_once(st, t + PROBE_FAIL_LIMIT,
                              spawn=sp, probe=lambda p: False) == "spawn"
        assert sp.procs[0].killed, "半死态必须真杀"
        assert len(sp.procs) == 2

    def test_grace_period_no_probe(self):
        """启动宽限期内不探测（网关监听要时间，误杀启动中的 bot 是事故）。"""
        sp = _FakeSpawn()
        st = _st(port=8192)
        supervise_once(st, 1000.0, spawn=sp, probe=lambda p: True)
        assert supervise_once(st, 1000.0 + BOOT_GRACE - 1,
                              spawn=sp, probe=lambda p: False) == "wait"
        assert st.probe_fails == 0
