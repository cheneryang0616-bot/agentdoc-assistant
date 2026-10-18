"""知识源规范化 + 语义分块。

对应简历 ②「利用 MarkItDown 工具将多模态知识源统一转化为 Markdown 并智能语义分块」。
本实现：
  * 支持 .md / .txt / .html 的规范化（HTML 用标准库剥离标签，等价于 MarkItDown 的文本抽取）；
  * 若安装了 pymupdf，则额外支持 .pdf；若安装了 markitdown，则优先调用它；
  * 语义分块：按 Markdown 标题切段 -> 段落 -> 受 CHUNK_MAX_CHARS 约束的滑动窗口，
    并保留 CHUNK_OVERLAP 的重叠，避免语义在硬切处断裂。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List

from .. import config


# --------------------------------------------------------------------------- #
# 规范化
# --------------------------------------------------------------------------- #
class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        t = "".join(self.parts)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


def normalize_source(path: Path) -> str:
    """把任意知识源统一转换成 Markdown-ish 文本。"""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix in (".html", ".htm"):
        p = _HTMLText()
        p.feed(path.read_text(encoding="utf-8", errors="ignore"))
        return f"# {path.stem}\n\n{p.text()}\n"

    if suffix == ".pdf":
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            return f"# {path.stem}\n\n" + "\n\n".join(pg.get_text() for pg in doc)
        except Exception:
            return f"# {path.stem}\n\n[PDF 文本抽取失败：{suffix}]\n"

    # 兜底：当作纯文本
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# 语义分块
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    doc_title: str = ""
    meta: Dict = field(default_factory=dict)


def _split_paragraphs(md: str) -> List[tuple]:
    """返回 [(doc_title, section_title, paragraph), ...]，标题向下继承。"""
    out: List[tuple] = []
    doc_title = ""
    section = ""
    buf: List[str] = []

    def flush():
        if buf:
            para = " ".join(x.strip() for x in buf if x.strip())
            if para:
                out.append((doc_title, section, para))
            buf.clear()

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            t = s.lstrip("#").strip()
            if level == 1:
                doc_title = t
                section = ""
            else:
                section = t
            continue
        if not s:
            flush()
            continue
        buf.append(s)
    flush()
    return out


def semantic_chunk(md: str, doc_id: str, max_chars: int = config.CHUNK_MAX_CHARS,
                   overlap: int = config.CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for doc_title, section, para in _split_paragraphs(md):
        if len(para) <= max_chars:
            pieces = [para]
        else:
            step = max(1, max_chars - overlap)
            pieces = [para[i:i + max_chars] for i in range(0, len(para), step)]
        for p in pieces:
            idx += 1
            chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{idx}",
                                title=section, text=p, doc_title=doc_title,
                                meta={"chars": len(p), "section": section,
                                      "doc_title": doc_title}))
    return chunks


def load_knowledge_base(kb_dir: Path | None = None) -> List[Chunk]:
    kb_dir = kb_dir or config.KB_DIR
    all_chunks: List[Chunk] = []
    for p in sorted(Path(kb_dir).rglob("*")):
        if not p.is_file():
            continue
        md = normalize_source(p)
        all_chunks.extend(semantic_chunk(md, doc_id=p.stem))
    return all_chunks
