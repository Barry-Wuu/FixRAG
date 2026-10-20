"""检索策略选择器与多种检索策略：直接检索 / HyDE / 子问题分解 / 回溯简化。
对应流程图中的“策略选择器”分支。

本模块对外暴露：
- select_strategy(query) -> (strategy, rationale)   返回策略与选择理由
- retrieve(query, store, ...) -> (docs, trace)       返回召回文档与检索过程细节
"""
import json
import os
import re

import config
import dashscope


def select_strategy(query: str):
    """返回 (strategy, rationale)。strategy ∈ direct/hyde/sub_query/backtrack。"""
    forced = os.getenv("FIXRAG_STRATEGY", "auto")  # auto/direct/hyde/sub_query/backtrack
    if forced != "auto":
        return forced, f"环境变量 FIXRAG_STRATEGY 强制指定为 {forced}"
    reasons = []
    if len(query) > 40:
        reasons.append(f"问题长度 {len(query)}>40，较复杂")
    conj = [s for s in ["、", "和", "以及", "与", "分别", "各自"] if s in query]
    if conj:
        reasons.append("含并列连词(" + "/".join(conj) + ")，存在多意图")
    if reasons:
        return "sub_query", "触发子问题分解：" + "；".join(reasons)
    return "direct", "默认直接检索（无长句/并列结构）"


def _retrieve(store, text, top_parent, source_filter):
    return store.search(text, top_parent=top_parent, source_filter=source_filter,
                        return_trace=True)


def retrieve_direct(query, store, top_parent=None, source_filter=None):
    docs, st = _retrieve(store, query, top_parent, source_filter)
    return docs, {"rewritten_query": None, "sub_queries": None,
                  "hypothesis": None, "search_trace": st}


def retrieve_hyde(query, store, top_parent=None, source_filter=None):
    hyp = dashscope.chat([
        {"role": "system", "content": "你是一个检索辅助助手。请针对用户问题，写一段可能出现在家电维修资料/产品手册中的、详实的参考答案（仅用于向量检索），不要解释。"},
        {"role": "user", "content": query},
    ], max_tokens=400)
    docs, st = _retrieve(store, hyp, top_parent, source_filter)
    return docs, {"rewritten_query": None, "sub_queries": None,
                  "hypothesis": hyp, "search_trace": st}


def retrieve_backtrack(query, store, top_parent=None, source_filter=None):
    simp = dashscope.chat([
        {"role": "system", "content": "把用户问题简化/改写为更利于知识库检索的简短问句，只输出改写后的问题。"},
        {"role": "user", "content": query},
    ], max_tokens=200)
    docs, st = _retrieve(store, simp, top_parent, source_filter)
    return docs, {"rewritten_query": simp, "sub_queries": None,
                  "hypothesis": None, "search_trace": st}


def retrieve_subquery(query, store, top_parent=None, source_filter=None):
    resp = dashscope.chat([
        {"role": "system", "content": "把用户问题拆成多个独立子问题以便分别检索。规则：① 凡出现『分别/各自/和/以及』连接的并列诉求、或用多个『?/？』提出多个问题，必须拆成≥2个子问题，每个子问题聚焦一个诉求；② 只输出 JSON 数组，如 [\"子问题1\",\"子问题2\"]；③ 最多3个。"},
        {"role": "user", "content": query},
    ], max_tokens=300, temperature=0)
    subs = _extract_list(resp) or [query]
    # 兜底：模型只回 1 个但原问题明显是多问（≥2 个问号），按问号切分，避免漏拆
    if len(subs) < 2 and (query.count("?") + query.count("？")) >= 2:
        segs = [s.strip() + "？" for s in re.split(r"[?？]", query) if s.strip()]
        if len(segs) >= 2:
            subs = segs[:3]
    merged, seen, sub_traces = [], set(), []
    for sub in subs[:3]:
        d, st = _retrieve(store, sub, top_parent=top_parent, source_filter=source_filter)
        sub_traces.append({"sub_query": sub, "search_trace": st})
        for p in d:
            if p["parent_id"] in seen:
                continue
            seen.add(p["parent_id"])
            merged.append(p)
            if len(merged) >= (top_parent or config.RETRIEVAL_K):
                break
        if len(merged) >= (top_parent or config.RETRIEVAL_K):
            break
    return merged, {"rewritten_query": None, "sub_queries": subs[:3],
                    "hypothesis": None, "sub_traces": sub_traces}


def _extract_list(text):
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return [str(x) for x in arr if isinstance(x, str)]
    except Exception:
        return None


def retrieve(query, store, strategy=None, return_trace=False,
             top_parent=None, source_filter=None):
    if strategy is None:
        strategy, rationale = select_strategy(query)
    else:
        rationale = f"指定策略={strategy}"
    if strategy == "hyde":
        docs, sub = retrieve_hyde(query, store, top_parent, source_filter)
    elif strategy == "sub_query":
        docs, sub = retrieve_subquery(query, store, top_parent, source_filter)
    elif strategy == "backtrack":
        docs, sub = retrieve_backtrack(query, store, top_parent, source_filter)
    else:
        docs, sub = retrieve_direct(query, store, top_parent, source_filter)
    if not return_trace:
        return docs
    trace = {"strategy": strategy, "rationale": rationale, **sub}
    return docs, trace
