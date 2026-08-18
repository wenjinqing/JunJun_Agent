"""本地 ASR（sherpa-onnx sense-voice）单元测试。

2026-08-18 云端 SiliconFlow 欠费退役改本地推理：管线任何一环失败都必须
静默降级返回 ""（旧行为契约），绝不炸主流程。真实模型 234MB 不进测试，
识别器/解码全部打桩。
"""

import numpy as np
import pytest

import junjun_llm.asr as asr


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """每个用例重置识别器全局态（懒加载单例，跨用例污染会假性通过）。"""
    monkeypatch.setattr(asr, "_recognizer", None)
    monkeypatch.setattr(asr, "_recognizer_failed", False)


class _FakeStream:
    def __init__(self):
        self.result = type("R", (), {"text": "你好呀"})()

    def accept_waveform(self, rate, samples):
        assert rate == 16000


class _FakeRecognizer:
    def create_stream(self):
        return _FakeStream()

    def decode_stream(self, stream):
        pass


class TestDegradation:
    @pytest.mark.asyncio
    async def test_empty_data(self):
        assert await asr.transcribe_bytes(b"") == ""

    @pytest.mark.asyncio
    async def test_oversize_rejected(self, monkeypatch):
        monkeypatch.setattr(asr, "_cfg", lambda: {"max_bytes": 10})
        assert await asr.transcribe_bytes(b"x" * 100) == ""

    @pytest.mark.asyncio
    async def test_model_missing_returns_empty(self, monkeypatch, tmp_path):
        """模型文件缺失：静默降级 + 一次性告警标记，不炸调用方。"""
        async def _samples(data):
            return np.zeros(16000, dtype=np.float32)
        monkeypatch.setattr(asr, "_decode_to_16k", _samples)   # 解码放行才会走到识别器
        monkeypatch.setattr(asr, "_DEFAULT_MODEL_DIR", tmp_path / "nonexistent")
        assert await asr.transcribe_bytes(b"\x00" * 4000) == ""
        assert asr._recognizer_failed is True

    @pytest.mark.asyncio
    async def test_decode_failure_returns_empty(self, monkeypatch):
        async def _none(data):
            return None
        monkeypatch.setattr(asr, "_decode_to_16k", _none)
        assert await asr.transcribe_bytes(b"\x00" * 4000) == ""

    @pytest.mark.asyncio
    async def test_too_short_audio_returns_empty(self, monkeypatch):
        async def _short(data):
            return np.zeros(800, dtype=np.float32)   # <0.1s
        monkeypatch.setattr(asr, "_decode_to_16k", _short)
        assert await asr.transcribe_bytes(b"\x00" * 4000) == ""


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_transcribe_roundtrip(self, monkeypatch):
        async def _samples(data):
            return np.zeros(16000, dtype=np.float32)   # 1s 假音频
        monkeypatch.setattr(asr, "_decode_to_16k", _samples)
        monkeypatch.setattr(asr, "_recognizer", _FakeRecognizer())
        assert await asr.transcribe_bytes(b"\x00" * 4000) == "你好呀"

    @pytest.mark.asyncio
    async def test_recognize_exception_returns_empty(self, monkeypatch):
        async def _samples(data):
            return np.zeros(16000, dtype=np.float32)
        monkeypatch.setattr(asr, "_decode_to_16k", _samples)

        def _boom(samples):
            raise RuntimeError("ORT 炸了")
        monkeypatch.setattr(asr, "_recognize", _boom)
        assert await asr.transcribe_bytes(b"\x00" * 4000) == ""


class TestOrtDllSelfHeal:
    def test_ensure_ort_dlls_idempotent(self):
        """真实环境：onnxruntime capi DLL 已同步到 sherpa_onnx/lib，
        重复调用幂等（大小一致不再覆盖）。"""
        from pathlib import Path
        import sherpa_onnx
        asr._ensure_ort_dlls()
        dll = Path(sherpa_onnx.__file__).parent / "lib" / "onnxruntime.dll"
        assert dll.exists()
        mtime = dll.stat().st_mtime_ns
        asr._ensure_ort_dlls()
        assert dll.stat().st_mtime_ns == mtime
