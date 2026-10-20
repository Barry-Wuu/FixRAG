"""智能家电维修问答系统 FastAPI 服务：/query 问答，/build 重建索引，/health 健康检查。
运行（WSL）：uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline
import db
import config

app = FastAPI(title="智能家电维修问答系统", version="1.0")


class QueryReq(BaseModel):
    query: str
    top_k: int = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def do_query(req: QueryReq):
    try:
        return pipeline.answer(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
def do_query_stream(req: QueryReq):
    """流式问答（SSE）：逐段推送回答，事件类型 meta / delta / done / error。"""
    def gen():
        try:
            for ev in pipeline.stream_answer(req.query):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/build")
def do_build():
    import loader
    import chunker
    import vectorstore

    docs = loader.load_documents()
    chunks = []
    for i, d in enumerate(docs):
        chunks.extend(chunker.split_parent_child(d["text"], i, d["source"]))
    store = vectorstore.VectorStore()
    store.build(chunks)
    seed_file = os.path.join(os.path.dirname(__file__), "seed_qa.json")
    seeded = db.seed_hot_from_file(seed_file) if os.path.exists(seed_file) else 0
    return {"status": "built", "documents": len(docs), "chunks": len(chunks),
            "seeded_hot_qa": seeded}


@app.post("/seed")
def do_seed():
    """把 seed_qa.json 灌入高频问答库（MySQL 角色）。"""
    seed_file = os.path.join(os.path.dirname(__file__), "seed_qa.json")
    if not os.path.exists(seed_file):
        raise HTTPException(status_code=404, detail="seed_qa.json 不存在")
    n = db.seed_hot_from_file(seed_file)
    return {"status": "seeded", "count": n}


@app.get("/stats")
def do_stats():
    """数据库统计：高频问答库条数、运行时缓存条数、热门缓存。"""
    return db.stats()


@app.get("/db/hot_qa")
def do_hot_qa():
    """高频问答库全表内容（供前端「高频问答库」栏目展示）。"""
    return {"items": db.get_all_hot_full()}


@app.get("/db/qa_cache")
def do_qa_cache():
    """运行时缓存全表内容（供前端「运行时缓存」栏目展示）。"""
    return {"items": db.get_all_cache()}


@app.get("/db/vectorstore")
def do_vectorstore():
    """向量库内容（kb_store/chunks.json + embeddings.npy）：按文件路径分组，返回每个文件的全部子块。"""
    import numpy as np
    if not (config.KB_META_FILE.exists() and config.KB_EMBED_FILE.exists()):
        return {"built": False, "files": [], "chunks": 0, "dim": 0}
    chunks = json.loads(config.KB_META_FILE.read_text(encoding="utf-8"))
    embs = np.load(str(config.KB_EMBED_FILE))
    files = {}
    for c in chunks:
        path = c.get("file_path") or c.get("source", "?")  # source 可能相同，按真实路径分组
        f = files.get(path)
        if not f:
            f = files[path] = {
                "source": c.get("source", "?"),
                "file_path": path,
                "chunks": 0,
                "blocks": [],
            }
        f["chunks"] += 1
        f["blocks"].append({
            "child_id": c.get("child_id", ""),
            "child_text": c.get("child_text", ""),
        })
    file_list = sorted(files.values(), key=lambda x: -x["chunks"])
    return {
        "built": True,
        "files": file_list,
        "chunks": len(chunks),
        "dim": int(embs.shape[1]),
        "bytes": int(embs.nbytes),
    }


@app.on_event("startup")
def startup_cleanup():
    """启动时缓存治理：删除测试数据 → 通用知识缓存 → 补判 null 意图 → 补算 embedding → 合并同义变体。"""
    try:
        test_deleted = db.purge_test_data()
        purged = db.purge_general_cache()
        null_deleted = db.classify_null_intents()
        backfilled = db.backfill_cache_embeddings()
        deduped = db.dedupe_cache()
        if test_deleted or purged or null_deleted or backfilled or deduped:
            print(f"[db] 启动缓存治理：清理测试数据 {test_deleted} 条，删除通用知识 {purged} 条，"
                  f"补判删除 {null_deleted} 条，补算 embedding {backfilled} 条，合并同义变体 {deduped} 条")
    except Exception as e:
        print(f"[db] 启动缓存治理失败（不影响服务）：{e}")


@app.post("/db/qa_cache/purge")
def purge_cache():
    """手动触发缓存治理：清理测试数据、通用知识缓存、补判 null 意图、补算 embedding、合并同义变体。"""
    test_deleted = db.purge_test_data()
    purged = db.purge_general_cache()
    null_deleted = db.classify_null_intents()
    backfilled = db.backfill_cache_embeddings()
    deduped = db.dedupe_cache()
    return {"test_deleted": test_deleted, "purged": purged,
            "null_deleted": null_deleted, "backfilled": backfilled,
            "deduped": deduped}


# 前端页面：挂到根路径，浏览器访问 / 即为网页界面（默认对话，可切换栏目）
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
