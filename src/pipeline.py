"""问答编排：缓存 -> 意图识别 -> 通用知识直答 / 专业咨询(策略检索+RAG) -> 缓存结果。
stream_answer 会产出 think(思考过程) / tool(工具调用细节) / meta / delta / error 事件，
供前端逐步展示，对应流程图的可观测过程。"""
import numpy as np

import config
import dashscope
import db
from radical_map import normalize_radicals
from intent import get_classifier
from cache import get_cache
from strategy import select_strategy, retrieve


_HOT_CACHE = {"items": None, "emb": None}


def _match_hot(query):
    """用 embedding 余弦相似度衡量 query 与高频问答库(hot_qa)的匹配程度。
    返回最佳匹配项字典；库为空返回 None。"""
    items = db.get_all_hot()
    if not items:
        return None
    q = (query or "").strip()
    if _HOT_CACHE["items"] is not items:
        _HOT_CACHE["items"] = items
        _HOT_CACHE["emb"] = np.array(
            dashscope.embed([it["question"] for it in items]), dtype=np.float32)
    q_emb = np.array(dashscope.embed([query])[0], dtype=np.float32)
    sims = _HOT_CACHE["emb"] @ q_emb
    sims = sims / (np.linalg.norm(_HOT_CACHE["emb"], axis=1) *
                   (np.linalg.norm(q_emb) + 1e-9))
    i = int(np.argmax(sims))
    exact = (items[i]["question"].strip() == q)
    return {
        "exact": exact,
        "best_sim": float(sims[i]),
        "question": items[i]["question"],
        "answer": items[i]["answer"],
        "intent": items[i]["intent"],
    }

TECH_SYS = (
    "你是智能家电维修问答系统的技术顾问。用户的问题属于家电相关的通用技术知识"
    "（产品原理、概念、选购建议、使用常识、参数对比等）。请结合你的知识直接给出准确、清晰的解答；"
    "若用户进一步询问具体的故障/维修问题，引导其描述故障现象以便获得专业维修建议。"
)
REJECT_SYS = (
    "你是智能家电维修问答系统的助手，只服务家电维修/家电使用相关的咨询。"
    "用户的问题与家电、家电维修完全无关（如生活常识、天气、娱乐、时政、闲聊等）。"
    "请礼貌拒绝回答，并引导用户咨询家电故障、维修、保养、选购等相关话题。"
)
RAG_SYS = (
    "你是智能家电维修问答系统的专业维修顾问。请严格依据下面的【参考资料】回答用户的家电维修专业咨询问题；"
    "如果资料里没有相关信息，请如实说明无法从资料中找到答案，不要编造维修步骤。"
    "如果用户的问题与家电维修无关，请礼貌拒绝回答并引导用户咨询家电相关问题。"
)


def build_context(docs, max_chars=4000):
    parts = []
    total = 0
    for i, d in enumerate(docs, 1):
        block = f"【资料{i}】（来源：{d['source']}）\n{d['content']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_sources(docs):
    """构造前端“参考文档”面板所需数据：文档路径 + 原文段落 + 检索得分。"""
    out = []
    for d in docs:
        out.append({
            "path": d.get("file_path") or d.get("source") or "",
            "source": d.get("source", ""),
            "score": d.get("score"),
            "paragraph": (d.get("content") or d.get("parent_text") or "").strip(),
        })
    return out


def answer(query: str) -> dict:
    query = normalize_radicals(query or "")
    # 1) 高频问答库（MySQL 角色）：精确命中直接返回
    hot = db.get_hot(query)
    if hot is not None:
        return {
            "answer": hot["answer"],
            "intent": hot.get("intent") or "专业咨询",
            "strategy": None,
            "sources": [],
            "cached": "hot",
        }

    # 2) 运行时缓存（Redis 角色）：命中则回显真实意图/策略
    cache = get_cache()
    hit = cache.get(query)
    if hit is not None and not hit.get("miss"):
        intent = hit.get("intent")
        strategy = hit.get("strategy")
        if not intent:  # 旧数据无 intent，按模型补判
            code = get_classifier().predict(query)
            intent = config.INTENT_NAMES.get(code, "专业咨询")
        return {
            "answer": hit["answer"],
            "intent": intent,
            "strategy": strategy,
            "sources": [],
            "cached": True,
        }
    # 语义相似但 Jev 判定不同义（miss）→ 与未命中同样处理，走完整 RAG 并新增独立缓存

    clf = get_classifier()
    intent_code = clf.predict(query)
    intent_label = config.INTENT_NAMES.get(intent_code, "专业咨询")

    if intent_code in (config.LABEL_GENERAL, config.LABEL_GENERAL_TECH):
        # 与家电无关 → 礼貌拒绝；家电通用知识 → 直接作答（均不检索、不缓存）
        sys_prompt = REJECT_SYS if intent_code == config.LABEL_GENERAL else TECH_SYS
        ans = dashscope.chat([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ])
        # 通用知识（两类）不写缓存（拒绝缓存，避免无关问题占用缓存库）
        return {
            "answer": ans,
            "intent": intent_label,
            "strategy": "direct_llm",
            "sources": [],
            "cached": False,
        }

    # 专业咨询 -> 检索 + RAG
    from vectorstore import get_store
    store = get_store()
    if not store.is_built():
        raise RuntimeError("知识库尚未构建，请先运行 build_kb.py 构建索引。")
    strategy, _ = select_strategy(query)
    docs = retrieve(query, store, strategy=strategy)
    ctx = build_context(docs)
    ans = dashscope.chat([
        {"role": "system", "content": RAG_SYS},
        {"role": "user", "content": f"【参考资料】\n{ctx}\n\n【用户问题】\n{query}"},
    ], max_tokens=1024)
    cache.put(query, ans, intent_label, strategy)
    return {
        "answer": ans,
        "intent": intent_label,
        "strategy": strategy,
        "sources": build_sources(docs),
        "cached": False,
    }


def stream_answer(query: str):
    """流式问答：生成器，yield 事件 dict。
    事件类型：think(LLM 思考内容) / trace(流程步骤) / tool(检索细节) / meta / delta / error。"""
    query = normalize_radicals(query or "")
    # 1) 高频问答库（hot_qa / MySQL 角色）：语义余弦匹配，>=0.85 命中
    hm = _match_hot(query)
    if hm is not None:
        sim = hm["best_sim"]
        hit = hm["exact"] or sim >= 0.85
        lines = ["匹配方式：DashScope embedding 语义余弦相似度（命中阈值 0.85）"]
        if hm["exact"]:
            lines.append(f"精确命中：『{hm['question']}』相似度={sim:.3f}")
            lines.append("≥0.85 → 命中高频问答库(hot_qa)，直接抓取该问答对返回（不再检索）")
        elif sim >= 0.85:
            lines.append(f"最佳语义匹配：『{hm['question']}』相似度={sim:.3f}")
            lines.append("≥0.85 → 语义命中高频问答库(hot_qa)，直接返回该问答对")
        else:
            lines.append(f"最佳语义匹配：『{hm['question']}』相似度={sim:.3f}")
            lines.append(f"{sim:.3f} < 0.85 → 未命中高频问答库，进入下一步")
        yield {"type": "trace", "title": "① 高频问答库检索 (hot_qa / MySQL角色)",
               "lines": lines}
        if hit:
            yield {"type": "meta", "intent": hm["intent"] or "专业咨询",
                   "strategy": None, "cached": "hot", "sources": []}
            yield {"type": "delta", "text": hm["answer"]}
            return

    # 2) 运行时缓存（qa_cache / Redis 角色）：精确命中回显
    cache = get_cache()
    hit = cache.get(query)
    if hit is not None and hit.get("miss"):
        # 语义相似但 Jev 判定不同义（如换品牌/换型号/换对象）：不命中，走完整 RAG 并新增独立缓存
        yield {"type": "trace", "title": "② 运行时缓存检索 (qa_cache / Redis角色)",
               "lines": [f"语义相似候选『{hit.get('similar_to')}』"
                         f"（相似度={hit.get('similarity') or 0:.3f}），"
                         "但 Jev 判定与当前问题不同义 → 不命中，走完整 RAG 并新增独立缓存"]}
    elif hit is not None:
        if hit.get("similar_to"):
            clines = [f"语义相似命中（Jev 判定同义）→ 命中缓存条目『{hit['similar_to']}』"
                      f"（相似度={hit.get('similarity') or 0:.3f}），直接回显答案（不新增缓存）"]
        else:
            clines = ["精确匹配命中运行时缓存，直接回显答案（省去检索与生成）"]
        yield {"type": "trace", "title": "② 运行时缓存检索 (qa_cache / Redis角色)",
               "lines": clines}
        intent = hit.get("intent")
        strategy = hit.get("strategy")
        if not intent:
            code = get_classifier().predict(query)
            intent = config.INTENT_NAMES.get(code, "专业咨询")
        yield {"type": "meta", "intent": intent, "strategy": strategy,
               "cached": True, "sources": []}
        yield {"type": "delta", "text": hit["answer"]}
        return
    else:
        yield {"type": "trace", "title": "② 运行时缓存检索 (qa_cache / Redis角色)",
               "lines": ["未命中运行时缓存，进入意图识别"]}

    # 3) 意图识别（两级判定：先判是否专业咨询(家电维修)，再判是否家电通用知识）
    info = get_classifier().analyze(query)
    ilines = []
    ilines.append(f"随机森林原始判定：{info['rf_label']}"
                  + (f"（P(通用)={info['p_general']:.2f}, P(专业)={info['p_consult']:.2f}）"
                     if info['p_general'] is not None else ""))
    tech_parts = (info['strong_tech'] + info['normal_tech'])
    if tech_parts:
        ilines.append(f"Jieba 术语信号：命中 {len(tech_parts)} 个术语 "
                      f"（强：{info['strong_tech']}，普通：{info['normal_tech']}）→ 术语得分 {info['tech_score']:.2f}")
    else:
        ilines.append(f"Jieba 术语信号：未命中术语 → 术语得分 {info['tech_score']:.2f}")
    ilines.append(f"加权分：0.3×P(专业) + 0.7×术语分 = {info['weighted_score']:.3f}")

    s1 = info['stage1']
    if s1['type'] == 'keyword':
        ilines.append("第一级（是否专业咨询）：命中咨询关键词 → 专业咨询")
    elif s1['type'] == 'strong_tech':
        ilines.append("第一级（是否专业咨询）：强技术术语 → 非咨询信号足够强，跳过 Jev")
    elif s1['type'] == 'llm':
        cprob = s1['prob'] if s1['prob'] is not None else 0.0
        ilines.append(f"第一级（是否专业咨询）：Jev 二分类（{config.INTENT_LLM_MODEL}）→ {s1['choice']}"
                      f"（consult 概率 = {cprob:.2f}）→ 融合分 = 0.5×加权 + 0.5×LLM = {info['final_prob']:.3f}"
                      + (" ≥0.5 → 专业咨询" if info['final_code'] == config.LABEL_CONSULT else " <0.5 → 非专业咨询"))
    elif s1['type'] == 'llm_error':
        ilines.append(f"第一级（是否专业咨询）：Jev 调用失败（网络/限流/超时），降级用加权分 {info['final_prob']:.3f}"
                      + (" ≥0.5 → 专业咨询" if info['final_code'] == config.LABEL_CONSULT else " <0.5 → 非专业咨询"))
    else:
        ilines.append(f"第一级（是否专业咨询）：加权分 {info['final_prob']:.3f} "
                      + ("≥0.5 → 专业咨询" if info['final_code'] == config.LABEL_CONSULT else "<0.5 → 非专业咨询"))

    if info['final_code'] != config.LABEL_CONSULT and info['stage2'] is not None:
        s2 = info['stage2']
        if s2['type'] == 'strong_tech':
            ilines.append("第二级（是否家电通用知识）：强术语 → 家电通用知识（LLM 直接作答）")
        elif s2['type'] == 'llm':
            tprob = s2['prob'] if s2['prob'] is not None else 0.0
            ilines.append(f"第二级（是否家电通用知识）：Jev 二分类（{config.INTENT_LLM_MODEL}）→ {s2['choice']}"
                          f"（tech_general 概率 = {tprob:.2f}）"
                          + (" → 家电通用知识（LLM 直接作答）" if s2['choice'] == 'tech_general'
                             else " → 与家电无关（礼貌拒绝）"))
        elif s2['type'] == 'llm_error':
            ilines.append("第二级（是否家电通用知识）：Jev 调用失败，降级判定（普通术语→家电相关，否则→无关）")
        else:
            ilines.append("第二级（是否家电通用知识）：未命中术语，按降级规则判定")

    if info["overridden"]:
        ilines.append(f"最终纠正：{info['correction']} → {info['final_label']}")
    else:
        ilines.append("最终判定：未触发纠正")
    ilines.append(f"最终意图：{info['final_label']}"
                  + (" → 走 RAG：策略选择 + 混合检索 + 生成"
                     if info['final_code'] == config.LABEL_CONSULT
                     else (" → 家电通用知识：LLM 直接作答（不检索、不缓存）"
                           if info['final_code'] == config.LABEL_GENERAL_TECH
                           else " → 与家电无关：LLM 礼貌拒绝（不检索、不缓存）")))
    yield {"type": "trace", "title": "③ 意图识别 (两级判定：专业咨询(家电维修) → 家电通用知识)",
           "lines": ilines}
    intent_label = info["final_label"]

    if info["final_code"] == config.LABEL_GENERAL:
        # 与家电无关：礼貌拒绝（不检索、不缓存）
        yield {"type": "meta", "intent": intent_label, "strategy": "direct_llm",
               "cached": False, "sources": []}
        parts = []
        for item in dashscope.chat_stream([
            {"role": "system", "content": REJECT_SYS},
            {"role": "user", "content": query},
        ]):
            if item.get("reasoning"):
                yield {"type": "think", "text": item["reasoning"]}
            if item.get("content"):
                parts.append(item["content"])
                yield {"type": "delta", "text": item["content"]}
        return

    if info["final_code"] == config.LABEL_GENERAL_TECH:
        # 家电通用知识：LLM 直接作答（不检索、不缓存）
        yield {"type": "meta", "intent": intent_label, "strategy": "direct_llm",
               "cached": False, "sources": []}
        parts = []
        for item in dashscope.chat_stream([
            {"role": "system", "content": TECH_SYS},
            {"role": "user", "content": query},
        ]):
            if item.get("reasoning"):
                yield {"type": "think", "text": item["reasoning"]}
            if item.get("content"):
                parts.append(item["content"])
                yield {"type": "delta", "text": item["content"]}
        return

    # 4) 专业咨询 -> 策略选择 + 混合检索 + RAG
    from vectorstore import get_store
    store = get_store()
    if not store.is_built():
        yield {"type": "error", "message": "知识库尚未构建，请先运行 build_kb.py 构建索引。"}
        return
    strategy, _ = select_strategy(query)
    docs, rtrace = retrieve(query, store, strategy=strategy, return_trace=True)
    yield {"type": "tool", "stage": "retrieval", "data": {
        "strategy": rtrace["strategy"],
        "rationale": rtrace["rationale"],
        "rewritten_query": rtrace.get("rewritten_query"),
        "hypothesis": rtrace.get("hypothesis"),
        "sub_queries": rtrace.get("sub_queries"),
        "sub_traces": rtrace.get("sub_traces"),
        "search_trace": rtrace.get("search_trace"),
    }}
    ctx = build_context(docs)
    yield {"type": "meta", "intent": intent_label, "strategy": strategy,
           "cached": False, "sources": build_sources(docs)}
    parts = []
    for item in dashscope.chat_stream([
        {"role": "system", "content": RAG_SYS},
        {"role": "user", "content": f"【参考资料】\n{ctx}\n\n【用户问题】\n{query}"},
    ], max_tokens=1024):
        if item.get("reasoning"):
            yield {"type": "think", "text": item["reasoning"]}
        if item.get("content"):
            parts.append(item["content"])
            yield {"type": "delta", "text": item["content"]}
    cache.put(query, "".join(parts).strip(), intent_label, strategy)
