"""实验 5：真实嵌入模型 vs 离线哈希嵌入 —— 检索质量对比。

对比两种嵌入后端在同一份标注集上的 Recall@k，并演示 512 -> 384 维的
"维度异常容错（截断）"对召回的影响。

运行：  python3 experiments/exp5_embedder_compare.py
产出：  results/exp5_embedder_compare.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdoc import config
from agentdoc.rag import HashingEmbedder, Retriever, VectorStore, load_knowledge_base, fix_dim
from experiments.exp4_finetune_embedding import LABELS

KS = (1, 3, 5)


def index_text(emb_name: str, c) -> str:
    if emb_name.startswith("hashing"):
        return f"{c.doc_title} {c.doc_title} {c.doc_title} {c.title} {c.text}"
    return f"{c.doc_title} {c.title} {c.text}"


def recall_at(Q, C, gt, chunk_docs, ks=KS):
    S = Q @ C.T
    hits = {k: 0 for k in ks}
    for i, exp in enumerate(gt):
        order = np.argsort(-S[i])
        for k in ks:
            if any(chunk_docs[j] == exp for j in order[:k]):
                hits[k] += 1
    return {k: hits[k] / len(gt) for k in ks}


def run(emb_name: str, embedder, chunks, queries, gt):
    t0 = time.time()
    C = embedder.encode_batch([index_text(emb_name, c) for c in chunks])
    Q = embedder.encode_batch(queries)
    enc_t = time.time() - t0
    dense = recall_at(Q, C, gt, [c.doc_id for c in chunks])
    # MQE（多查询扩展）：逐条扩展后取最优
    from agentdoc.rag.retriever import expand_queries
    st = VectorStore()
    for i, c in enumerate(chunks):
        st.add(c.chunk_id, c.text, C[i], kind="doc", doc_id=c.doc_id)
    r = Retriever(st, embedder)
    mqe_hits = {k: 0 for k in KS}
    for q, exp in zip(queries, gt):
        hits = r.mqe_search(q, top_k=5)
        for k in KS:
            if any(h.doc_id == exp for h in hits[:k]):
                mqe_hits[k] += 1
    mqe = {k: mqe_hits[k] / len(gt) for k in KS}
    return {"dim": int(C.shape[1]), "encode_sec": round(enc_t, 3),
            "dense": {str(k): round(v, 4) for k, v in dense.items()},
            "mqe": {str(k): round(v, 4) for k, v in mqe.items()}, "dense_raw": dense}


def main() -> None:
    config.ensure_dirs()
    chunks = load_knowledge_base()
    queries, gt = [], []
    for doc, qs in LABELS.items():
        for q in qs:
            queries.append(q)
            gt.append(doc)
    print(f"标注集：{len(queries)} 条 query / {len(chunks)} 个 chunk\n")

    out = {"n_queries": len(queries), "n_chunks": len(chunks), "results": {}}

    # ---- 哈希（离线）----
    h = HashingEmbedder()
    h.fit([index_text("hashing", c) for c in chunks])
    res_h = run("hashing", h, chunks, queries, gt)
    out["results"]["hashing"] = {k: v for k, v in res_h.items() if k != "dense_raw"}

    # ---- 真实模型 BGE ----
    res_b = None
    try:
        from agentdoc.rag.embedder import SentenceTransformerEmbedder
        bge = SentenceTransformerEmbedder()
        res_b = run("bge", bge, chunks, queries, gt)
        out["results"]["bge"] = {k: v for k, v in res_b.items() if k != "dense_raw"}
        out["results"]["bge"]["model"] = bge.model_name
    except Exception as e:
        print(f"[警告] 真实嵌入不可用，跳过：{e}")

    # ---- 维度容错：512 -> 384 ----
    dim_demo = None
    if res_b is not None:
        C = bge.encode_batch([index_text("bge", c) for c in chunks])
        Q = bge.encode_batch(queries)
        C3 = np.vstack([fix_dim(v, 384) for v in C])
        Q3 = np.vstack([fix_dim(v, 384) for v in Q])
        # 重新归一化，模拟"截断后重新入向量库"
        C3 = C3 / np.clip(np.linalg.norm(C3, axis=1, keepdims=True), 1e-9, None)
        Q3 = Q3 / np.clip(np.linalg.norm(Q3, axis=1, keepdims=True), 1e-9, None)
        r384 = recall_at(Q3, C3, gt, [c.doc_id for c in chunks])
        dim_demo = {"original_dim": 512, "truncated_dim": 384,
                    "recall_512": {str(k): round(v, 4) for k, v in res_b["dense_raw"].items()},
                    "recall_384": {str(k): round(v, 4) for k, v in r384.items()}}
        out["dim_tolerance"] = dim_demo

    # ---- 打印 ----
    print(f"{'后端':<10}{'维度':>6}{'编码耗时':>10}  {'Recall@1':>9}{'Recall@3':>9}{'Recall@5':>9}  (dense)")
    for name, r in out["results"].items():
        d = r["dense"]
        print(f"{name:<10}{r['dim']:>6}{r['encode_sec']:>9.2f}s  "
              f"{d['1']:>9.4f}{d['3']:>9.4f}{d['5']:>9.4f}")
    print()
    print(f"{'后端':<10}  {'Recall@1':>9}{'Recall@3':>9}{'Recall@5':>9}  (MQE 多查询扩展)")
    for name, r in out["results"].items():
        m = r["mqe"]
        print(f"{name:<10}  {m['1']:>9.4f}{m['3']:>9.4f}{m['5']:>9.4f}")

    if dim_demo:
        print()
        print("维度容错（512 维截断到 384 维后重新归一化）：")
        for k in ("1", "3", "5"):
            print(f"  Recall@{k}: {dim_demo['recall_512'][k]:.4f} -> {dim_demo['recall_384'][k]:.4f}")

    outp = config.RESULTS_DIR / "exp5_embedder_compare.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {outp}")


if __name__ == "__main__":
    main()
