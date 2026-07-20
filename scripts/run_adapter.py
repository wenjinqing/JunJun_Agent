# -*- coding: utf-8 -*-
"""启动君君 Adapter（连接 NapCat WS client）。
用法：
    .venv\\Scripts\\python.exe scripts\\run_adapter.py
或：
    cd E:\\JunJun_Agent && .venv\\Scripts\\python.exe -m junjun_adapter_napcat
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from junjun_core import initialize_logging
initialize_logging("INFO")

from junjun_adapter_napcat.main import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
