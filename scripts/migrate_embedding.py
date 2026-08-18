"""embedding 模型迁移脚本：长期记忆索引整体重 embedding。

背景（2026-08-18）：硅基流动欠费 402，EMBEDDING_* 从 SiliconFlow BAAI/bge-m3
切到 AI Ping Qwen3-Embedding-0.6B（同为 1024 维，EMBED_DIM 不用动）。
long_term 索引头校验对「模型不匹配」的处理是重建空库——直接改配置重启会
丢全部长期记忆，故先用本脚本把存量条目用新模型重 embedding 落盘，再重启。

用法：uv run python scripts/migrate_embedding.py
幂等：metadata.json 模型标签已等于当前 EMBEDDING_MODEL 时跳过。
安全：任一批次 embedding 失败则中止，原文件不动（另请先自行备份 data/memory/）。
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

# GBK 控制台防乱码报错（CLAUDE.md 规矩）
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_BATCH = 64          # 单批条数：AI Ping embeddings 接口一次别塞太多
_RETRY = 3           # 单批失败重试次数


async def _embed_batch(client, texts):
    for attempt in range(1, _RETRY + 1):
        vecs = await client.embed(texts)
        if vecs and len(vecs) == len(texts):
            return vecs
        print(f"  批次 embedding 失败（第 {attempt}/{_RETRY} 次）", flush=True)
        await asyncio.sleep(2 * attempt)
    return None


async def main() -> int:
    import numpy as np

    from junjun_memory.embedding import get_embedding_client, EMBED_DIM
    from junjun_memory.long_term import LongTermMemory

    data_dir = ROOT / "data" / "memory"
    meta_path = data_dir / "metadata.json"
    idx_path = data_dir / "faiss_index.bin"
    if not meta_path.exists():
        print("metadata.json 不存在，无需迁移")
        return 0

    client = get_embedding_client()
    if not client.available:
        print("embedding 客户端不可用（检查 EMBEDDING_* 配置）")
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old_model = meta.get("model")
    new_model = client._model
    items = meta.get("items", [])
    print(f"旧模型: {old_model} | 新模型: {new_model} | 条目: {len(items)}")
    if old_model == new_model:
        print("模型标签一致，无需迁移（幂等跳过）")
        return 0
    if meta.get("dim") != EMBED_DIM:
        print(f"维度 {meta.get('dim')} != EMBED_DIM {EMBED_DIM}，本脚本不处理，中止")
        return 1

    # 逐批重 embedding（全部成功才落盘；任一批失败中止，原文件不动）
    texts = [it["text"] for it in items]
    all_vecs = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        vecs = await _embed_batch(client, batch)
        if vecs is None:
            print(f"第 {i // _BATCH + 1} 批重试耗尽，中止迁移，原文件未改动")
            return 1
        all_vecs.extend(vecs)
        print(f"  已 embedding {len(all_vecs)}/{len(texts)}", flush=True)

    # 与 long_term.add() 同管线：float32 + L2 归一化 + IndexFlatIP（内积=余弦）
    import faiss
    mat = np.array(all_vecs, dtype="float32")
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(mat)
    assert index.ntotal == len(items)

    for it in items:
        it["has_vec"] = True
    new_meta = {"dim": EMBED_DIM, "model": new_model, "items": items}

    # 原子落盘（先 tmp 后替换），并刷新 .bak 备份对——同 LongTermMemory.save()
    import shutil
    tmp_idx = idx_path.with_suffix(".tmp")
    tmp_meta = meta_path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp_idx))
    tmp_meta.write_text(json.dumps(new_meta, ensure_ascii=False), encoding="utf-8")
    tmp_idx.replace(idx_path)
    tmp_meta.replace(meta_path)
    for src in (idx_path, meta_path):
        shutil.copy2(src, src.with_suffix(".bak"))
    print(f"落盘完成: {len(items)} 条全部重向量化为 {new_model}")

    # 验证：新实例重新加载 + 抽查一条检索
    mem = LongTermMemory(data_dir)
    mem.load()
    assert len(mem._items) == len(items), "重载条目数不一致"
    sample = items[0]["text"][:20]
    hits = await mem.search(sample, top_k=1)
    print(f"验证检索 top1: {'命中' if hits else '空'} "
          f"（{hits[0].text[:20] if hits else ''}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
