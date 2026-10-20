"""加载知识库源文档：支持 .md / .pdf，按学科目录名提取 source 标签，跳过 .rsls 占位。"""
import os
from pathlib import Path

import config
from radical_map import normalize_radicals


def _normalize(text: str) -> str:
    """把康熙部首 / CJK 部首补充 还原为标准汉字（OCR 怪字归一化），
    避免扫描件 OCR 产生的怪字污染 embedding、干扰检索与展示。"""
    return normalize_radicals(text)


def _load_pdf(path: str) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def load_documents(data_dir=None):
    data_dir = Path(data_dir or config.DATA_DIR)
    if not data_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在: {data_dir}")
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            fp = Path(root) / f
            ext = fp.suffix.lower()
            if ext in (".rsls",):
                continue
            try:
                if ext == ".md":
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                elif ext == ".pdf":
                    text = _load_pdf(str(fp))
                else:
                    # 其它文本类直接尝试
                    try:
                        text = fp.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
            except Exception as e:
                print(f"[loader] 跳过 {fp}: {e}")
                continue
            text = text.strip()
            text = _normalize(text)
            if not text:
                continue
            # 从目录名提取学科（ai_data -> ai）
            source = fp.parent.name.replace("_data", "")
            docs.append({"text": text, "source": source, "file_path": str(fp)})
            print(f"[loader] 已加载 {fp.name} (source={source}, 字符数={len(text)})")
    return docs
