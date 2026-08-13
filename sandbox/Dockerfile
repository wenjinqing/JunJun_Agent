# run_code 沙箱镜像：python:3.12-slim + 白名单数据包，非 root，无网络运行
#
# 构建：docker build -t junjun-sandbox:latest sandbox/
# 运行方式由 sandbox/server.py 编排（docker run --rm -i ...，代码走 stdin）。
# 容器自身不监听任何端口——无 ENTRYPOINT 服务，跑完即销毁。

FROM python:3.12-slim

# 中文字体（matplotlib/词云不写豆腐块）+ 白名单数据/文档包
# （确认单 pandas/openpyxl/requests；2026-08-13 扩容：图表美化 seaborn、
# 图片处理 pillow、Word 产出 python-docx、PDF 读 pdfplumber / 产 reportlab、
# 中文分词+词云 jieba/wordcloud、二维码 qrcode——均纯 wheel，无编译）
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir \
    pandas openpyxl matplotlib seaborn requests \
    pillow python-docx pdfplumber reportlab \
    jieba wordcloud "qrcode[pil]"

# 非 root 运行账号（uid 10001；Linux 宿主需保证 data/workspace 对该 uid 可写，
# Docker Desktop/Windows 的 bind mount 无此问题）
RUN useradd --create-home --uid 10001 sandbox

# matplotlib 默认启用中文字体（生成的图表代码不用自己设 font.family）
RUN mkdir -p /home/sandbox/.config/matplotlib \
 && echo "font.family: Noto Sans CJK SC" > /home/sandbox/.config/matplotlib/matplotlibrc \
 && chown -R sandbox:sandbox /home/sandbox/.config

USER sandbox
WORKDIR /workspace

# 从 stdin 读代码执行：-I 隔离模式（忽略 PYTHON* 环境变量、不放用户 site-packages、
# 不把 cwd 加进 sys.path 前置位）
ENTRYPOINT ["python", "-I", "-u", "-"]
