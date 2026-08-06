"""VLM 识图前压缩测试（2026-08-06 生产实锤：QQ 原图 2-4K 分辨率 base64 后数 MB，
32B VLM 30s 超时三连）。缩到长边 1024 + JPEG q80 后通常 <200KB，识图降到秒级。
失败原样返回——宁可慢，不可丢图。
"""

import io

import pytest

from junjun_memory.vision import _downscale


def _png_bytes(w: int, h: int, color=(120, 80, 200)) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class TestDownscale:
    def test_big_image_shrunk_to_jpeg(self):
        """3000x2000 大图 -> 长边 ≤1024、JPEG 魔数、体积明显下降。"""
        data = _png_bytes(3000, 2000)
        out = _downscale(data)
        assert out[:2] == b"\xff\xd8"  # JPEG magic
        from PIL import Image
        img = Image.open(io.BytesIO(out))
        assert max(img.size) <= 1024
        assert len(out) < len(data)

    def test_small_image_passthrough(self):
        """小图（尺寸达标且 <400KB）原样返回，不白花一次编解码。"""
        data = _png_bytes(320, 240)
        assert _downscale(data) is data

    def test_small_dimension_but_fat_file_still_compressed(self):
        """尺寸小但体积肥（>400KB）的图也要压——体积才是超时主因。"""
        from PIL import Image
        import random
        random.seed(7)
        # 噪点图 PNG 压缩率低，800x600 也能上 MB
        img = Image.frombytes("RGB", (800, 600),
                              bytes(random.randrange(256) for _ in range(800 * 600 * 3)))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        data = buf.getvalue()
        assert len(data) > 400_000
        out = _downscale(data)
        assert out[:2] == b"\xff\xd8"
        assert len(out) < len(data)

    def test_garbage_passthrough(self):
        """非图片字节（PIL 打不开）：原样返回，不炸不丢。"""
        garbage = b"\x89PNG-not-really-an-image" * 100
        assert _downscale(garbage) is garbage

    def test_rgba_converted_without_error(self):
        """带透明通道的 PNG -> JPEG 需转 RGB，不能炸。"""
        from PIL import Image
        img = Image.new("RGBA", (2000, 1500), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out = _downscale(buf.getvalue())
        assert out[:2] == b"\xff\xd8"
