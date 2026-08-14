# run_code 沙箱镜像：python:3.12-slim + 白名单数据包，非 root，无网络运行
#
# 构建：docker build -t junjun-sandbox:latest sandbox/
# 运行方式由 sandbox/server.py 编排（docker run --rm -i ...，代码走 stdin）。
# 容器自身不监听任何端口——无 ENTRYPOINT 服务，跑完即销毁。

# 版本钉死（2026-08-13 审查 P2）：基座按 digest 钉（标签可被重指漂移），
# pip 包全量钉进 requirements.txt（含传递依赖，从验证过的镜像 freeze）。
# apt 的 fonts-noto-cjk 随基座 digest 的 Debian 快照基本稳定，不另钉。
FROM python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

# 中文字体（matplotlib/词云不写豆腐块）+ 白名单数据/文档包
# （确认单 pandas/openpyxl/requests；2026-08-13 扩容：图表美化 seaborn、
# 图片处理 pillow、Word 产出 python-docx、PDF 读 pdfplumber / 产 reportlab、
# 中文分词+词云 jieba/wordcloud、二维码 qrcode——均纯 wheel，无编译）
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# 非 root 运行账号（uid 10001；Linux 宿主需保证 data/workspace 对该 uid 可写，
# Docker Desktop/Windows 的 bind mount 无此问题）
RUN useradd --create-home --uid 10001 sandbox

# matplotlib 默认启用中文字体（生成的图表代码不用自己设 font.family）。
# 三个坑全踩过（2026-08-14 实锤）：
# 1) rc 写 ~/.config 依赖 HOME，docker --user 运行 HOME 不可靠，根本没被读到；
#    包内 mpl-data/matplotlibrc 是兜底候选，永远被找到。
# 2) DejaVu 在前指望逐字形回退——mpl 3.11 实际不回退，CJK 照样豆腐；
#    Noto 必须第一（它自带拉丁/数字字形，整图统一无回退依赖）。
# 3) 出厂 rc 里这些键是【未注释】的模板行，追加会产生 duplicate key 警告，
#    必须 sed 原位替换；grep 断言防 rc 模板漂移时静默失效。
RUN MPLRC=$(python -c "import matplotlib, os; print(os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'matplotlibrc'))") \
 && sed -i \
      -e 's/^#\?font\.family\s*:.*/font.family: sans-serif/' \
      -e 's/^#\?font\.sans-serif\s*:.*/font.sans-serif: Noto Sans CJK SC, DejaVu Sans, sans-serif/' \
      -e 's/^#\?axes\.unicode_minus\s*:.*/axes.unicode_minus: False/' \
      "$MPLRC" \
 && grep -q "^font\.sans-serif: Noto Sans CJK SC" "$MPLRC"

USER sandbox
WORKDIR /workspace

# 从 stdin 读代码执行：-I 隔离模式（忽略 PYTHON* 环境变量、不放用户 site-packages、
# 不把 cwd 加进 sys.path 前置位）
ENTRYPOINT ["python", "-I", "-u", "-"]
