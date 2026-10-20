"""轻量向量库：稠密(DashScope embedding) + 稀疏(BM25) 混合检索，子块召回、父块作上下文。
用 numpy 实现，检索时做 RRF 融合重排，等价于原项目的“混合检索/双路召回 + reranker”。
所有数据持久化到 kb_store/，避免每次重启重新向量化。"""
import json
import math
import re

import numpy as np

import config
import dashscope


def _tokenize(text):
    # 英文/数字按词，中文按单字（适合 BM25）
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


class BM25:
    def __init__(self, corpus):
        self.docs = [_tokenize(d) for d in corpus]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: math.log((self.N - c + 0.5) / (c + 0.5) + 1)
                    for t, c in self.df.items()}
        self.k1, self.b = 1.5, 0.75

    def scores(self, query):
        q = _tokenize(query)
        out = []
        for d in self.docs:
            score = 0.0
            dl = len(d)
            freq = {}
            for t in d:
                freq[t] = freq.get(t, 0) + 1
            for t in q:
                if t in self.idf:
                    f = freq.get(t, 0)
                    score += self.idf[t] * (f * (self.k1 + 1)) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        out.append(score)
        return np.array(out, dtype=np.float32)


def _ranks_desc(arr):
    a = np.asarray(arr, dtype=float)
    return np.argsort(np.argsort(-a)) + 1  # 1 = 最佳


class VectorStore:
    def __init__(self):
        self.embeddings = None   # (N, D) float32
        self.chunks = []         # list of meta dicts (child_text/parent_text/source/...)
        self.bm25 = None

    # ---------- 构建 ----------
    def build(self, chunks):
        print(f"[vectorstore] 正在向量化 {len(chunks)} 个子块 ...")
        texts = [c["child_text"] for c in chunks]
        # 分批调用 embedding，避免单次过大
        batch, embs = 16, []
        for i in range(0, len(texts), batch):
            embs.extend(dashscope.embed(texts[i:i + batch]))
        self.embeddings = np.array(embs, dtype=np.float32)
        self.chunks = chunks
        self.bm25 = BM25(texts)
        self._save()
        print(f"[vectorstore] 完成，维度={self.embeddings.shape}")

    def _save(self):
        np.save(str(config.KB_EMBED_FILE), self.embeddings)
        with open(config.KB_META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    # ---------- 加载 ----------
    def load(self):
        if not (config.KB_EMBED_FILE.exists() and config.KB_META_FILE.exists()):
            return False
        self.embeddings = np.load(str(config.KB_EMBED_FILE))
        with open(config.KB_META_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.bm25 = BM25([c["child_text"] for c in self.chunks])
        print(f"[vectorstore] 已加载本地索引，共 {len(self.chunks)} 子块")
        return True

    def is_built(self):
        return config.KB_EMBED_FILE.exists() and config.KB_META_FILE.exists()

    # ---------- 检索 ----------
    def search(self, query, top_child=None, top_parent=None, source_filter=None,
               return_trace=False):
        top_child = top_child or config.TOP_CHILD_PER_QUERY
        top_parent = top_parent or config.RETRIEVAL_K
        q_emb = np.array(dashscope.embed([query])[0], dtype=np.float32)
        # 稠密余弦
        sims = self.embeddings @ q_emb
        norms = np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-9)
        sims = sims / norms
        dense_rank = _ranks_desc(sims)
        # 稀疏 BM25
        sp = self.bm25.scores(query)
        sparse_rank = _ranks_desc(sp)
        # RRF 融合
        rrf = 1.0 / (60 + dense_rank) + 1.0 / (60 + sparse_rank)

        n_show = min(5, len(self.chunks))

        def _snip(i, n=60):
            return self.chunks[i]["child_text"][:n]

        dense_top = [
            {"i": int(i), "score": round(float(sims[i]), 4),
             "snippet": _snip(i), "source": self.chunks[i].get("source", "")}
            for i in sorted(range(len(sims)), key=lambda k: -sims[k])[:n_show]
        ]
        sparse_top = [
            {"i": int(i), "score": round(float(sp[i]), 4),
             "snippet": _snip(i), "source": self.chunks[i].get("source", "")}
            for i in sorted(range(len(sp)), key=lambda k: -sp[k])[:n_show]
        ]
        rrf_top = [
            {"i": int(i), "score": round(float(rrf[i]), 5),
             "snippet": _snip(i), "source": self.chunks[i].get("source", "")}
            for i in sorted(range(len(rrf)), key=lambda k: -rrf[k])[:n_show]
        ]

        order = np.argsort(-rrf)
        seen, parents = set(), []
        for idx in order:
            meta = self.chunks[idx]
            if source_filter and meta.get("source") != source_filter:
                continue
            pid = meta["parent_id"]
            if pid in seen:
                continue
            seen.add(pid)
            parents.append({
                "parent_id": pid,
                "content": meta["parent_text"],
                "source": meta.get("source", ""),
                "file_path": meta.get("file_path", ""),
                "score": round(float(rrf[idx]), 5),
                "child_preview": meta["child_text"][:80],
            })
            if len(parents) >= top_parent:
                break

        if not return_trace:
            return parents
        trace = {
            "query": query,
            "top_child": top_child,
            "top_parent": top_parent,
            "dense_top": dense_top,
            "sparse_top": sparse_top,
            "rrf_top": rrf_top,
            "selected_parents": [
                {
                    "parent_id": p["parent_id"],
                    "source": p["source"],
                    "score": p["score"],
                    "child_preview": p["child_preview"],
                    "parent_preview": p["content"][:160],
                }
                for p in parents
            ],
        }
        return parents, trace


_store = None


def get_store():
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load()
    return _store
