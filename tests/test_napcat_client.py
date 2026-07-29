"""napcat_client 辅助逻辑测试。"""

from junjun_core import napcat_client as nc


class TestFileParam:
    def test_relative_path_resolved(self):
        """相对路径转绝对（NapCat 是独立进程，相对路径它找不到）。"""
        from pathlib import Path
        out = nc._file_param("data/pixiv_novel/x.txt")
        assert Path(out).is_absolute()
        assert out.endswith("x.txt")

    def test_http_passthrough(self):
        assert nc._file_param("https://x.com/a.txt") == "https://x.com/a.txt"

    def test_file_uri_passthrough(self):
        assert nc._file_param("file:///E:/a.txt") == "file:///E:/a.txt"
