"""384 维嵌入器。

BaseEmbedder —— 纯离线、确定性的特征哈希嵌入器（Hashing Embedder）：
    * 中文字符 + 中文双字 bigram，英文词 + 英文词 bigram；
    * 哈希到固定维度，L2 归一化；
    * 提供 fix_dim()：动态补齐 / 截断到 384 维（对应简历中的"维度异常容错"）。

ProjectionEmbedder —— 在基础嵌入之上叠加一层可训练线性投影，
    供 exp4「Embedding 微调实验（三元损失）」使用。
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Sequence

import numpy as np

from .. import config

# 必须在 import transformers 之前设置：禁用 TensorFlow/Flax，避免 Keras3 冲突；
# 并默认使用 HuggingFace 镜像（国内可访问）。
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_LATIN = re.compile(r"[A-Za-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """混合分词：英文词（长度>=2） + 中文双字 bigram。

    刻意**不使用**中文单字与英文 bigram：它们噪声大、易造成 384 维哈希碰撞。
    """
    low = text.lower().replace("_", " ")
    latin = [w for w in _LATIN.findall(low) if len(w) >= 2]
    cjk = _CJK.findall(low)
    toks: List[str] = list(latin)
    toks += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return toks


def _hash_sign(tok: str, dim: int):
    """返回 (bucket, sign)：带符号哈希（feature hashing），碰撞时相互抵消。"""
    h = hashlib.md5(tok.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % dim
    sign = 1.0 if (h[4] & 1) else -1.0
    return idx, sign


def _hash_index(tok: str, dim: int) -> int:
    return _hash_sign(tok, dim)[0]


def fix_dim(vec: np.ndarray, dim: int = config.EMBED_DIM) -> np.ndarray:
    """维度异常容错：不足则补齐 0，超出则截断。"""
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.shape[0] == dim:
        return v
    if v.shape[0] < dim:
        return np.concatenate([v, np.zeros(dim - v.shape[0], dtype=np.float32)])
    return v[:dim].copy()


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class HashingEmbedder:
    """特征哈希嵌入器：带符号哈希 + 可选 IDF 加权（需先 fit 语料）。"""

    name = "hashing"

    def __init__(self, dim: int = config.EMBED_DIM, use_idf: bool = True) -> None:
        self.dim = dim
        self.use_idf = use_idf
        self.idf = np.ones(dim, dtype=np.float32)
        self.fitted = False

    def _raw(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            i, s = _hash_sign(t, self.dim)
            vec[i] += s
        return vec

    def fit(self, docs: Sequence[str]) -> "HashingEmbedder":
        """在语料上统计每个桶的文档频率，得到 IDF 权重。"""
        df = np.zeros(self.dim, dtype=np.float32)
        for d in docs:
            seen = set()
            for t in tokenize(d):
                seen.add(_hash_sign(t, self.dim)[0])
            for i in seen:
                df[i] += 1.0
        n = max(1, len(docs))
        self.idf = (np.log((n + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        self.fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        vec = self._raw(text)
        if self.use_idf and self.fitted:
            vec = vec * self.idf
        return l2norm(fix_dim(vec, self.dim))

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.encode(t) for t in texts])


class ProjectionEmbedder:
    """基础嵌入 + 线性投影 W。W 由 exp4 的三元损失训练得到。"""

    name = "base+projection"

    def __init__(self, base: HashingEmbedder, W: np.ndarray | None = None) -> None:
        self.base = base
        d = base.dim
        self.W = np.eye(d, dtype=np.float32) if W is None else W.astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        e = self.base.encode(text)
        return l2norm(self.W @ e)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        E = self.base.encode_batch(texts)          # (n, d)
        return np.vstack([l2norm(self.W @ e) for e in E]) if len(E) else E

    def save(self, path) -> None:
        np.save(path, self.W)

    @classmethod
    def load(cls, base: HashingEmbedder, path) -> "ProjectionEmbedder":
        return cls(base, np.load(path))


# --------------------------------------------------------------------------- #
# 真实嵌入模型（可选依赖 sentence-transformers）
# --------------------------------------------------------------------------- #
def _st_importable() -> bool:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class SentenceTransformerEmbedder:
    """真实嵌入模型（默认 BAAI/bge-small-zh-v1.5，中文检索 SOTA 小模型）。

    依赖：sentence-transformers（pip 装）。模型未缓存时会尝试联网下载，
    网络不通或无权限时抛出异常，由 build_embedder 回退到哈希嵌入。
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str | None = None, device: str = "cpu",
                 allow_download: bool = True) -> None:
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # 优先离线加载（快），失败再决定是否联网下载
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or config.ST_MODEL
        model = None
        if not allow_download:
            try:
                model = SentenceTransformer(self.model_name, device=device,
                                            local_files_only=True)
            except Exception as e:  # 未缓存
                raise RuntimeError(f"嵌入模型 {self.model_name} 未本地缓存: {e}")
        else:
            try:
                model = SentenceTransformer(self.model_name, device=device,
                                            local_files_only=True)
            except Exception:
                model = SentenceTransformer(self.model_name, device=device)
        self.model = model
        self.dim = int(model.get_sentence_embedding_dimension())
        self.fitted = False

    def fit(self, docs: Sequence[str]) -> "SentenceTransformerEmbedder":
        # 预训练模型无需在语料上拟合
        self.fitted = True
        return self

    def encode(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True),
                          dtype=np.float32)


# --------------------------------------------------------------------------- #
# 工厂：自动选择可用的嵌入后端
# --------------------------------------------------------------------------- #
def build_embedder(prefer: str | None = None):
    """按优先级构造嵌入器。

    prefer / 环境变量 AGENTDOC_EMBEDDER:
        hash  —— 强制离线哈希（零依赖、最快）
        st    —— 强制真实模型（未缓存则下载）
        auto  —— 有 sentence-transformers 且模型已缓存则用真实模型，否则哈希
    """
    prefer = (prefer or os.environ.get("AGENTDOC_EMBEDDER") or "auto").lower()
    if prefer in ("hash", "hashing", "offline"):
        return HashingEmbedder()
    if prefer in ("st", "sentence-transformers", "bge", "real"):
        return SentenceTransformerEmbedder(allow_download=True)

    # auto
    if _st_importable():
        try:
            return SentenceTransformerEmbedder(allow_download=False)   # 仅用本地缓存
        except Exception:
            if os.environ.get("AGENTDOC_ALLOW_DOWNLOAD") == "1":
                try:
                    return SentenceTransformerEmbedder(allow_download=True)
                except Exception:
                    pass
    return HashingEmbedder()
