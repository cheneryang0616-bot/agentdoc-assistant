"""最小单元测试（无需 pytest）：python3 tests/test_core.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.context import ContextManager
from agentdoc.memory.scoring import hybrid_score, time_decay
from agentdoc.rag import HashingEmbedder, VectorStore, fix_dim, tokenize
from agentdoc.rag.ingest import semantic_chunk
from agentdoc.rag.retriever import expand_queries, rrf_fuse

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ✓ {name}")


def test_fix_dim():
    print("[fix_dim] 维度动态补齐 / 截断")
    check("补齐到 384", fix_dim(np.ones(10), 384).shape == (384,))
    check("截断到 384", fix_dim(np.ones(1000), 384).shape == (384,))
    check("刚好 384 不变", fix_dim(np.ones(384), 384).shape == (384,))
    check("补的是 0", float(fix_dim(np.ones(10), 384)[100]) == 0.0)


def test_tokenize():
    print("[tokenize] 混合分词")
    t = tokenize("JVM 的垃圾回收 GC")
    check("含英文词 jvm", "jvm" in t)
    check("含中文 bigram 垃圾", "垃圾" in t)
    check("不含中文单字 垃", "垃" not in t)
    check("不含英文 bigram jvm_", "jvm_" not in "".join(t))


def test_embedder():
    print("[HashingEmbedder] 带符号哈希 + IDF")
    emb = HashingEmbedder(dim=256)
    docs = ["JVM 垃圾回收", "Python 引用计数", "RAG 检索增强"]
    emb.fit(docs)
    v = emb.encode("JVM 垃圾回收")
    check("维度 256", v.shape == (256,))
    check("L2 归一", abs(float(np.linalg.norm(v)) - 1.0) < 1e-5)
    s_same = float(v @ emb.encode("JVM 的垃圾回收"))
    s_diff = float(v @ emb.encode("Python 引用计数"))
    check("相似度：同主题 > 异主题", s_same > s_diff)


def test_scoring():
    print("[scoring] 混合评分 / 时间衰减")
    check("半衰期处衰减 0.5", abs(time_decay(72, 72) - 0.5) < 1e-9)
    check("越久越衰减", time_decay(200) < time_decay(10))
    hi = hybrid_score(0.9, 1.0, 0.5)
    lo = hybrid_score(0.1, 1.0, 0.5)
    check("重要性越高分越高", hi > lo)


def test_chunk():
    print("[ingest] 语义分块")
    md = "# 标题\n\n## 小节A\n第一段内容。\n\n## 小节B\n第二段内容。"
    cs = semantic_chunk(md, "doc1")
    check("分块数 >= 2", len(cs) >= 2)
    check("保留 doc_title", cs[0].doc_title == "标题")
    check("chunk_id 前缀", cs[0].chunk_id.startswith("doc1#"))


def test_retriever():
    print("[retriever] MQE / RRF")
    check("MQE 多查询扩展", len(expand_queries("JVM 是什么")) >= 2)
    emb = HashingEmbedder(dim=256)
    emb.fit(["a b c", "d e f"])
    st = VectorStore()
    st.add("x1", "hello", emb.encode("a b c"), kind="doc", doc_id="d1")
    st.add("x2", "world", emb.encode("d e f"), kind="doc", doc_id="d2")
    check("kind 过滤", st.search(emb.encode("a b c"), 5, kind="doc")[0].id == "x1")


def test_context():
    print("[context] L1 落盘")
    cm = ContextManager(budget=200, offload_chars=20)
    msgs = [{"role": "user", "content": "hi"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1",
                                          "content": "X" * 500}]}]
    out, rep = cm.compress(msgs)
    check("L1 落盘 1 条", rep.l1_offloaded == 1)
    check("结果被替换为占位符", "已落盘" in str(out[1]["content"][0]["content"]))


def main():
    for fn in (test_fix_dim, test_tokenize, test_embedder, test_scoring,
               test_chunk, test_retriever, test_context):
        fn()
    print(f"\n全部通过：{PASS} 项断言")


if __name__ == "__main__":
    main()
