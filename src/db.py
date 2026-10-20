"""本地数据库层（SQLite，扮演原项目 MySQL + Redis）：
- hot_qa  表：预置的高频问答对（对应流程图里 MySQL 中“初始化加载的高频问答对”）
- qa_cache表：运行时问答缓存（对应流程图 Redis 高频缓存，命中即直接返回，省去检索/生成）
缓存/高频库同时记录 intent(意图) 与 strategy(策略)，命中时回显真实分类。
纯本地文件库，无需服务器/网络，WSL 断网也能正常访问数据库。
缓存支持语义去重：qa_cache 记录问题 embedding，读写时先按余弦相似度找候选、再经 Jev 判定同义后才合并命中，避免逐条缓存。
"""
import json
import sqlite3

import numpy as np

import config


def _connect():
    config.KB_DIR.mkdir(parents=True, exist_ok=True)
    path = config.KB_DIR / "qa.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS qa_cache ("
        "question TEXT PRIMARY KEY, answer TEXT, hits INTEGER DEFAULT 0, "
        "updated_at TEXT, intent TEXT, strategy TEXT, embedding TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS hot_qa ("
        "question TEXT PRIMARY KEY, answer TEXT, hits INTEGER DEFAULT 0, intent TEXT)"
    )
    # 兼容旧表：补列（不影响已有数据）
    cache_cols = {r[1] for r in conn.execute("PRAGMA table_info(qa_cache)")}
    if "intent" not in cache_cols:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN intent TEXT")
    if "strategy" not in cache_cols:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN strategy TEXT")
    if "embedding" not in cache_cols:
        conn.execute("ALTER TABLE qa_cache ADD COLUMN embedding TEXT")
    hot_cols = {r[1] for r in conn.execute("PRAGMA table_info(hot_qa)")}
    if "intent" not in hot_cols:
        conn.execute("ALTER TABLE hot_qa ADD COLUMN intent TEXT")
    conn.commit()
    return conn


def get_hot(question):
    """查高频问答库（MySQL 角色）。命中则命中数 +1，返回 {answer, intent}。"""
    q = question.strip()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT answer, intent FROM hot_qa WHERE question=?", (q,)).fetchone()
        if row:
            conn.execute("UPDATE hot_qa SET hits = hits + 1 WHERE question=?", (q,))
            conn.commit()
            return {"answer": row[0], "intent": row[1]}
        return None
    finally:
        conn.close()


def _embed_texts(texts):
    """计算问题 embedding（延迟 import dashscope，避免模块级依赖）。"""
    import dashscope
    if isinstance(texts, str):
        texts = [texts]
    return dashscope.embed(texts)


def _cosine(e1, e2):
    a = np.asarray(e1, dtype=np.float32)
    b = np.asarray(e2, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b) + 1e-9)))


def get_cache(question):
    """查运行时缓存（Redis 角色）。先精确匹配，再语义相似匹配（同义变体命中同一条）。
    命中则命中数 +1，返回 {answer, intent, strategy, similar_to?, similarity?}。"""
    q = question.strip()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT answer, intent, strategy FROM qa_cache WHERE question=?",
            (q,)).fetchone()
        if row:
            conn.execute("UPDATE qa_cache SET hits = hits + 1 WHERE question=?", (q,))
            conn.commit()
            return {"answer": row[0], "intent": row[1], "strategy": row[2]}
        # 语义相似匹配：同义变体问题命中同一缓存条目，避免逐条缓存
        rows = conn.execute(
            "SELECT question, answer, intent, strategy, embedding "
            "FROM qa_cache WHERE embedding IS NOT NULL AND embedding != ''").fetchall()
        if rows:
            try:
                q_emb = _embed_texts([q])[0]
            except Exception:
                return None
            best_q, best_sim, best = None, 0.0, None
            for r in rows:
                try:
                    sim = _cosine(json.loads(r[4]), q_emb)
                except Exception:
                    continue
                if sim > best_sim:
                    best_sim, best_q, best = sim, r[0], r
            if best_q and best_sim >= config.CACHE_SIM_THRESHOLD:
                # 相似度只是初筛：再用 Jev 判定两句话是否同义。
                # 仅当 LLM 确认同义才命中缓存；句式相同但关键信息（品牌/型号等）不同 → 不命中。
                is_same = True  # LLM 不可用/失败时降级为纯相似度命中（保持原行为）
                try:
                    from intent import llm_is_synonym
                    syn = llm_is_synonym(q, best_q)
                    if syn is not None:
                        is_same = syn[0] == "same"
                except Exception:
                    pass
                if is_same:
                    conn.execute("UPDATE qa_cache SET hits = hits + 1 WHERE question=?", (best_q,))
                    conn.commit()
                    return {"answer": best[1], "intent": best[2], "strategy": best[3],
                            "similar_to": best_q, "similarity": best_sim}
                # LLM 判定不同义：不命中、不加 hits，返回 miss 标记供前端展示
                return {"miss": True, "similar_to": best_q, "similarity": best_sim}
        return None
    finally:
        conn.close()


def put_cache(question, answer, intent=None, strategy=None):
    """写入运行时缓存（Redis 角色）。同义变体不新增行：语义相似则合并更新原条目。"""
    q = question.strip()
    conn = _connect()
    try:
        try:
            q_emb = _embed_texts([q])[0]
        except Exception:
            q_emb = None
        emb_json = json.dumps(q_emb) if q_emb is not None else None
        if q_emb is not None:
            rows = conn.execute(
                "SELECT question, embedding FROM qa_cache "
                "WHERE embedding IS NOT NULL AND embedding != ''").fetchall()
            best_q, best_sim = None, 0.0
            for r in rows:
                try:
                    sim = _cosine(json.loads(r[1]), q_emb)
                except Exception:
                    continue
                if sim > best_sim:
                    best_sim, best_q = sim, r[0]
            if best_q and best_sim >= config.CACHE_SIM_THRESHOLD:
                # 相似度只是初筛：Jev 确认同义才合并更新原条目；不同义则作为新问题新增独立行
                is_same = True  # LLM 不可用/失败时降级为合并（保持原行为）
                try:
                    from intent import llm_is_synonym
                    syn = llm_is_synonym(q, best_q)
                    if syn is not None:
                        is_same = syn[0] == "same"
                except Exception:
                    pass
                if is_same:
                    # 同义变体：合并更新到最相似条目，不新增行
                    conn.execute(
                        "UPDATE qa_cache SET answer=?, hits=hits+1, updated_at=datetime('now'), "
                        "intent=?, strategy=? WHERE question=?",
                        (answer, intent, strategy, best_q))
                    conn.commit()
                    return
        conn.execute(
            "INSERT INTO qa_cache(question, answer, hits, updated_at, intent, strategy, embedding) "
            "VALUES(?,?,1,datetime('now'),?,?,?) "
            "ON CONFLICT(question) DO UPDATE SET answer=excluded.answer, "
            "hits=hits+1, updated_at=datetime('now'), "
            "intent=excluded.intent, strategy=excluded.strategy",
            (q, answer, intent, strategy, emb_json))
        conn.commit()
    finally:
        conn.close()


def purge_general_cache():
    """删除缓存库中意图为「通用知识」的条目（通用知识不缓存）。返回删除条数。"""
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM qa_cache WHERE intent=?", ("通用知识",))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def backfill_cache_embeddings():
    """为缓存库中缺少 embedding 的行补算向量（一次性），供语义去重/相似命中使用。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question FROM qa_cache WHERE embedding IS NULL OR embedding=''").fetchall()
        if not rows:
            return 0
        import dashscope
        questions = [r[0] for r in rows]
        embs = []
        for i in range(0, len(questions), 16):
            try:
                embs.extend(dashscope.embed(questions[i:i + 16]))
            except Exception:
                break
        for q, e in zip(questions, embs):
            conn.execute("UPDATE qa_cache SET embedding=? WHERE question=?", (json.dumps(e), q))
        conn.commit()
        return len(embs)
    finally:
        conn.close()


def classify_null_intents():
    """为缓存库中 intent 为空的旧行补判意图：
    通用知识（含与家电无关、家电通用知识）→ 删除（通用知识不缓存）；
    专业咨询 → 补上 intent 标签。返回删除条数。"""
    from intent import get_classifier
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question FROM qa_cache WHERE intent IS NULL OR intent=''").fetchall()
        if not rows:
            return 0
        clf = get_classifier()
        deleted = 0
        for r in rows:
            code = clf.predict(r[0])
            if code in (config.LABEL_GENERAL, config.LABEL_GENERAL_TECH):
                conn.execute("DELETE FROM qa_cache WHERE question=?", (r[0],))
                deleted += 1
            else:
                conn.execute("UPDATE qa_cache SET intent=? WHERE question=?", ("专业咨询", r[0]))
        conn.commit()
        return deleted
    finally:
        conn.close()


def dedupe_cache():
    """合并缓存库中**近乎文本级重复**（余弦 ≥ 0.97）的行，保留 hits 最高的一条为 canonical。

    注意：不再对中间区间（0.75~0.97）做批量合并——那里的「是否同义」由
    get_cache / put_cache 在读写时用 Jev 判定（避免『…空调显示E1是什么意思?』被并入
    『…空调显示E2是什么意思?』）。返回合并删除的条数。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question, hits, embedding FROM qa_cache "
            "WHERE embedding IS NOT NULL AND embedding != ''").fetchall()
        if len(rows) < 2:
            return 0
        records = []
        for r in rows:
            try:
                records.append({
                    "question": r[0],
                    "hits": r[1],
                    "emb": np.array(json.loads(r[2]), dtype=np.float32),
                })
            except Exception:
                pass
        removed = 0
        for i in range(len(records)):
            if records[i] is None:
                continue
            for j in range(i + 1, len(records)):
                if records[j] is None:
                    continue
                sim = _cosine(records[i]["emb"], records[j]["emb"])
                if sim >= 0.97:  # 仅近乎文本级重复（同义改写由读写时的 Jev 判定负责）
                    if records[i]["hits"] >= records[j]["hits"]:
                        keep, drop = records[i], records[j]
                    else:
                        keep, drop = records[j], records[i]
                    conn.execute(
                        "UPDATE qa_cache SET hits=? WHERE question=?",
                        (keep["hits"] + drop["hits"], keep["question"]))
                    conn.execute("DELETE FROM qa_cache WHERE question=?", (drop["question"],))
                    conn.commit()
                    removed += 1
                    records[j] = None
        return removed
    finally:
        conn.close()


def purge_test_data():
    """清理缓存库与高频问答库中的测试/占位数据（答案为 x/测试答案，或问题含「测试」）。
    返回两个表合计删除条数。"""
    conn = _connect()
    try:
        cur1 = conn.execute(
            "DELETE FROM qa_cache WHERE answer IN ('x','测试答案A','测试答案') "
            "OR question LIKE '%测试%'")
        cur2 = conn.execute(
            "DELETE FROM hot_qa WHERE answer IN ('x','测试答案A','测试答案') "
            "OR question LIKE '%测试%'")
        conn.commit()
        return cur1.rowcount + cur2.rowcount
    finally:
        conn.close()


def seed_hot(pairs):
    """批量写入高频问答对（来自运营统计/日志）。已存在则跳过。默认意图=专业咨询。"""
    conn = _connect()
    n = 0
    try:
        for q, a in pairs:
            cur = conn.execute(
                "INSERT OR IGNORE INTO hot_qa(question, answer, hits, intent) "
                "VALUES(?,?,0,?)",
                (q.strip(), a, "专业咨询"),
            )
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


def clear_hot():
    """清空高频问答库（重建知识库时调用，避免旧 seed 残留）。"""
    conn = _connect()
    try:
        conn.execute("DELETE FROM hot_qa")
        conn.commit()
    finally:
        conn.close()


def seed_hot_from_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = [(d["question"], d["answer"]) for d in data]
    return seed_hot(pairs)


def get_all_hot():
    """返回 hot_qa 全表，供匹配度(语义余弦)计算使用。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question, answer, intent FROM hot_qa").fetchall()
        return [{"question": r[0], "answer": r[1], "intent": r[2]} for r in rows]
    finally:
        conn.close()


def get_all_hot_full():
    """返回 hot_qa 全表（含命中次数），按命中次数降序。供前端「高频问答库」栏目展示。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question, answer, hits, intent FROM hot_qa "
            "ORDER BY hits DESC, question").fetchall()
        return [{"question": r[0], "answer": r[1], "hits": r[2], "intent": r[3]}
                for r in rows]
    finally:
        conn.close()


def get_all_cache():
    """返回 qa_cache 全表（含命中次数/时间/意图/策略），按更新时间降序。供前端「运行时缓存」栏目展示。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT question, answer, hits, updated_at, intent, strategy "
            "FROM qa_cache ORDER BY updated_at DESC").fetchall()
        return [{"question": r[0], "answer": r[1], "hits": r[2],
                 "updated_at": r[3], "intent": r[4], "strategy": r[5]}
                for r in rows]
    finally:
        conn.close()


def stats():
    conn = _connect()
    try:
        hot = conn.execute("SELECT COUNT(*) FROM hot_qa").fetchone()[0]
        cache = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
        top = conn.execute(
            "SELECT question, hits FROM qa_cache ORDER BY hits DESC LIMIT 5"
        ).fetchall()
        return {"hot_qa": hot, "qa_cache": cache, "top_cache": top}
    finally:
        conn.close()
