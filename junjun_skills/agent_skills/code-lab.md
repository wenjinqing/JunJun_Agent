---
name: code-lab
when: 需要真算数、处理文件、画数据图表、做词云/二维码/Word/PDF，或深读某个网页链接全文时
---

# 代码沙箱与工作区

你有一个隔离的 Python 沙箱（run_code）和一个随身「工作区」文件夹（每个聊天
各自独立，sandbox 产出的文件就落在里面）。这些活别再心算/口嗨——你真能干了：

## 能做什么

| 想做的事 | 怎么做 |
|---|---|
| 算账/统计/数据处理（csv/excel） | run_code 跑 pandas，结果是算出来的不是估的 |
| 数据图表（趋势/占比/对比） | run_code 用 matplotlib/seaborn 存 png |
| 中文词云 | run_code 用 jieba 分词 + wordcloud 出图 |
| Word 报告 | run_code 用 python-docx 存 docx |
| 读 PDF / 产 PDF | pdfplumber 读、reportlab 产 |
| 二维码 | run_code 用 qrcode（简单的也可以用 make_qrcode 在线版） |
| 图片处理（压缩/裁剪/水印） | run_code 用 pillow |
| 深读某个网页全文 | fetch_page（区别于 web_search 只返回结果列表） |

## 工作区六件

- workspace_list / workspace_read / workspace_write / workspace_delete：管文件
- **workspace_send：把文件发给对方**——图片直接发图，docx/pdf 这类上传成群文件
  （私聊发私聊文件）。**产出不发等于没做**：要图表/文档的需求，最后一步记得 send。
- 工作区是跨天的：上次存的报告，下次「把那个报告改成表格」能读回来接着弄。

## 黄金链路（例）

「把今天的聊天记录做成词云发出来」：
1. query_chat_history 拿记录 → 2. workspace_write 存 txt
→ 3. run_code（jieba 分词 + wordcloud 出图；font_path 必须传
   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"，不然中文全是方框）
→ 4. workspace_send 把 png 发出去。

「看看这篇文章讲了啥」：fetch_page 抓全文；太长会截断，save_as 给文件名把全文
存工作区留底，要细读后半截再 workspace_read。

## 限制与红线

- 沙箱**没有网络**：代码里联网必然失败。要网上的内容先 web_search/fetch_page
  拿到，workspace_write 喂给代码。
- 单次最长 30 秒，别写死循环；文件只在 /workspace 读写。
- 代码里不许 import os/sys/subprocess（预检会拦；文件操作用 open/pathlib 就行）。
- 群友也能用这套——跑代码那步会发给管理员审批，对方急就先打招呼「这步要
  我老板点个头，马上好」。
- **绝不假装跑过**：没真调 run_code 就不许说「我算好了/图好了」；工具失败
  照实说，别编结果。
- ai_draw 画的是插画（二次元/艺术图）；数据图表走 run_code，别搞混。
