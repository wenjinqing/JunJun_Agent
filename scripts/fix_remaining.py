# -*- coding: utf-8 -*-
"""修复脚本：修 bug#2 语法 bug + bug#4 set_main_loop 调用点
用法：E:\JunJun_Agent\.venv\Scripts\python.exe E:\JunJun_Agent\scripts\fix_remaining.py
"""
import os

# ===== 1. 修 bug#2：relationship_mcp_server.py 的字面换行 bug =====
p1 = r'E:\JunJun_Agent\junjun_mcp_server\relationship_mcp_server.py'
with open(p1, 'rb') as f:
    t = f.read().decode('utf-8')

# 把 return "\r\n".join 改成 return "\\n".join
bad = 'return "\r\n".join(lines_out)'
good = 'return "\\n".join(lines_out)'
if bad in t:
    t = t.replace(bad, good, 1)
    with open(p1, 'wb') as f:
        f.write(t.encode('utf-8'))
    print('[OK] bug#2 字面换行已修正')
else:
    # 可能已经是好的，或者换行格式不同
    if 'return "\\n".join(lines_out)' in t:
        print('[SKIP] bug#2 已经是正确的')
    else:
        print('[WARN] bug#2 未找到目标块，请手动检查 relationship_mcp_server.py 第 105-106 行')

# ===== 2. 补 bug#4：server.py 的 set_main_loop 调用 =====
p2 = r'E:\JunJun_Agent\junjun_webui\server.py'
with open(p2, 'rb') as f:
    t = f.read().decode('utf-8')

# 找 start_webui 函数体，在 "if os.environ.get..." 之后插入 set_main_loop
marker = '    if os.environ.get("WEBUI_ENABLED", "false").lower() != "true":\n        logger.info("WebUI 未启用（WEBUI_ENABLED != true）")\n        return None'
if marker in t:
    new = marker + '\n    set_main_loop(asyncio.get_running_loop())'
    t = t.replace(marker, new, 1)
    with open(p2, 'wb') as f:
        f.write(t.encode('utf-8'))
    print('[OK] bug#4 set_main_loop 调用已补上')
else:
    if 'set_main_loop(asyncio.get_running_loop())' in t:
        print('[SKIP] bug#4 已经是正确的')
    else:
        print('[WARN] bug#4 未找到目标块，请手动检查 server.py 的 start_webui 函数')

print('\\n修复完成。建议下一步：')
print('  cd E:\\JunJun_Agent')
print('  .\\.venv\\Scripts\\python.exe -m pytest -q')
print('  .\\.venv\\Scripts\\python.exe scripts\\smoke_agent.py')
