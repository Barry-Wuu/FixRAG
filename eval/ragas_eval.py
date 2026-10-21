# -*- coding: utf-8 -*-
"""RAGAS 指标评测：检索相关性 / 检索召回率 / 生成忠实度 / 生成相关性。

基于 RAGAS 官方指标定义，对本机 智能家电维修问答系统 服务（http://127.0.0.1:8000/query）实际评测：
- 评估 LLM：DashScope qwen-max（非思考模型，FIXRAG_EVAL_LLM 可覆盖；默认关闭深度思考以加快评测）
- Embedding：DashScope text-embedding-v1（计算生成相关性的假设问题与原问题相似度）
- 测试集：6 条专业咨询问题 + 人工标准答案（ground_truth，供检索召回率用）

指标算法（与 RAGAS 对齐）：
1. 检索相关性 context_relevance：检索上下文所有句子中，与问题相关句子的比例；
2. 检索召回率 context_recall：ground_truth 拆成原子陈述后，被检索上下文支持的陈述比例；
3. 生成忠实度 faithfulness：生成答案拆成原子陈述后，被检索上下文支持的陈述比例；
4. 生成相关性 answer_relevancy：基于答案生成的假设问题与原问题的平均余弦相似度。

运行：先启动服务（run.bat），再执行 `.venv\\Scripts\\python.exe eval\\ragas_eval.py`。
注意：为保证每个问题都真实走「检索+生成」（而非命中运行时缓存直接返回），
脚本开始时会**清空 qa_cache 运行时缓存**（不影响高频问答库 hot_qa）。
"""
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config
import dashscope

EVAL_LLM = os.getenv("FIXRAG_EVAL_LLM", "qwen-max")  # 默认非思考模型，避免评测时深度思考拖慢速度
EMBED_MODEL = config.EMBED_MODEL
BASE = "http://127.0.0.1:8000"
HYP_Q_NUM = 3
MAX_CTX_SENT = 8  # 每个父块最多判定的句子数（父块约 1200 字，足够覆盖关键信息）

TESTS = [
    {
        "question": "空调显示E1是什么意思？",
        "ground_truth": "空调显示E1通常表示室内温度传感器故障，或压缩机保护（过载/过热）；不同品牌、型号的故障代码含义并不统一，需以对应机型的说明书或售后说明为准。",
    },
    {
        "question": "空调不制冷常见原因有哪些？",
        "ground_truth": "常见原因包括制冷剂（氟/雪种）不足、滤网或蒸发器脏堵、室外机散热不良、压缩机或风扇故障；可先清洗滤网、检查外机通风，仍未解决则需检修制冷剂和压缩机。",
    },
    {
        "question": "洗衣机不脱水是什么原因？",
        "ground_truth": "常见原因有排水管堵塞或弯折、排水泵故障、衣物不平衡触发保护、门盖开关失灵、电机或电容损坏；可先清理排水管、抖散衣物重放再试。",
    },
    {
        "question": "冰箱冷藏室结冰怎么处理？",
        "ground_truth": "多因温控档位过低、门封条不严漏冷、排水孔堵塞或温控器故障；可将档位调到适中（3-4档）、清理排水孔、检查门封，频繁结冰需检修温控器。",
    },
    {
        "question": "洗衣机显示E4错误代码是什么意思？",
        "ground_truth": "洗衣机故障代码E4通常表示进水异常或水位传感器故障，需检查进水阀、水压与进水管路是否堵塞或弯折。",
    },
    {
        "question": "冰箱不制冷但里面的灯还亮，是什么问题？",
        "ground_truth": "灯亮说明供电正常，不制冷多为压缩机/启动器故障、制冷剂泄漏或温控器失灵，建议联系专业维修人员检测压缩机与制冷剂压力。",
    },
]


def query_service(q):
    req = urllib.request.Request(
        BASE + "/query",
        data=json.dumps({"query": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def _extract_json(text):
    """从 LLM 输出稳健提取 JSON：兼容 ```json 代码块、前后多余文本与偶发截断；
    解析失败时退化为正则提取关键布尔字段（relevant/supported）。"""
    import re
    if not text:
        raise ValueError("empty response")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
        if "```" in t:
            t = t[:t.rfind("```")]
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1 and e > s:
        t = t[s:e + 1]
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r'"(relevant|supported)"\s*:\s*(true|false)', t, re.IGNORECASE)
        if m:
            return {m.group(1).lower(): m.group(2).lower() == "true"}
        raise


def llm_json(messages, default, retries=3):
    for i in range(retries):
        try:
            time.sleep(2)  # 降低请求频率，规避 DashScope 限流
            text = dashscope.chat(messages, model=EVAL_LLM, temperature=0, max_tokens=2048, enable_thinking=False)
            return _extract_json(text)
        except Exception as e:
            print(f"    [llm重试{i+1}] {e}")
            time.sleep(3)
    return default


def split_statements(text):
    """把文本拆成原子事实陈述句（LLM 输出 JSON 数组）。"""
    prompt = (
        "请把下面的文本拆分成若干条最原子的事实陈述句。\n"
        "要求：1) 每条必须是单一句子、单一事实；2) 保留原意不增删信息；\n"
        "3) 只输出 JSON 数组，如 [\"陈述1\", \"陈述2\"]，不要任何额外文字。\n\n"
        f"文本：\n{text}"
    )
    stmts = llm_json([{"role": "user", "content": prompt}], [])
    if not isinstance(stmts, list):
        return []
    return [s.strip() for s in stmts if isinstance(s, str) and s.strip()]


def is_supported(question, statement, contexts):
    """判断陈述是否被检索上下文支持（supported / not_supported）。"""
    ctx_text = "\n\n".join(f"[资料{i+1}]\n{c}" for i, c in enumerate(contexts))
    prompt = (
        "你是RAG评估器。根据【检索上下文】判断【陈述】是否被上下文支持。\n"
        "规则：陈述的所有关键事实都能在上下文中找到依据→supported；\n"
        "上下文中没有依据或与上下文矛盾→not_supported。\n"
        "只输出JSON：{\"verdict\": \"supported\" 或 \"not_supported\"}。\n\n"
        f"【问题】{question}\n\n【检索上下文】\n{ctx_text}\n\n【陈述】\n{statement}"
    )
    data = llm_json([{"role": "user", "content": prompt}], {})
    return data.get("verdict") == "supported"


def sentence_relevant(question, sentence):
    """判断检索上下文中的单个句子是否与问题相关。"""
    prompt = (
        "你是RAG评估器。判断【检索句】是否与【用户问题】相关。\n"
        "规则：检索句包含能帮助回答该问题的信息→yes；与问题无关→no。\n"
        "只输出JSON：{\"verdict\": \"yes\" 或 \"no\"}。\n\n"
        f"【用户问题】{question}\n\n【检索句】\n{sentence}"
    )
    data = llm_json([{"role": "user", "content": prompt}], {})
    return data.get("verdict") == "yes"


def hypothetical_questions(answer):
    """基于答案生成若干假设问题（RAGAS answer_relevancy 用）。"""
    prompt = (
        f"根据下面的【回答】，从回答者视角提出 {HYP_Q_NUM} 个该回答能够回答的假设问题。\n"
        "只输出JSON数组，如 [\"问题1\", \"问题2\", \"问题3\"]，不要额外文字。\n\n"
        f"【回答】\n{answer}"
    )
    qs = llm_json([{"role": "user", "content": prompt}], [])
    if not isinstance(qs, list):
        return []
    return [s.strip() for s in qs if isinstance(s, str) and s.strip()][:HYP_Q_NUM]


def split_sentences(text):
    parts = re.split(r"[。；!?；\n]+", text)
    return [s.strip() for s in parts if len(s.strip()) >= 2]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9
    )


def evaluate_one(t):
    q = t["question"]
    gt = t["ground_truth"]
    resp = query_service(q)
    if resp.get("cached") or not resp.get("sources"):
        print(f"  [跳过] 命中缓存/无检索上下文: {q} (cached={resp.get('cached')})")
        return None
    answer = resp["answer"]
    contexts = [s.get("paragraph", "") for s in resp["sources"] if s.get("paragraph")]
    print(f"  检索上下文 {len(contexts)} 个，回答 {len(answer)} 字")

    # 1) 检索相关性：上下文句子中相关句比例
    rel_yes = rel_total = 0
    for ctx in contexts:
        for sent in split_sentences(ctx)[:MAX_CTX_SENT]:
            rel_total += 1
            if sentence_relevant(q, sent):
                rel_yes += 1
    context_relevance = rel_yes / rel_total if rel_total else 0.0
    print(f"  检索相关性: {rel_yes}/{rel_total}")

    # 2) 检索召回率：ground_truth 陈述中被支持的比例
    gt_stmts = split_statements(gt)
    recall_supported = sum(1 for s in gt_stmts if is_supported(q, s, contexts))
    context_recall = recall_supported / len(gt_stmts) if gt_stmts else 0.0
    print(f"  检索召回率: {recall_supported}/{len(gt_stmts)}")

    # 3) 生成忠实度：答案陈述中被支持的比例
    ans_stmts = split_statements(answer)
    faith_supported = sum(1 for s in ans_stmts if is_supported(q, s, contexts))
    faithfulness = faith_supported / len(ans_stmts) if ans_stmts else 0.0
    print(f"  生成忠实度: {faith_supported}/{len(ans_stmts)}")

    # 4) 生成相关性：假设问题与原问题平均余弦
    hyp_qs = hypothetical_questions(answer)
    if hyp_qs:
        vecs = dashscope.embed([q] + hyp_qs, model=EMBED_MODEL)
        qv = vecs[0]
        answer_relevancy = statistics.mean(cosine(qv, v) for v in vecs[1:])
    else:
        answer_relevancy = 0.0
    print(f"  生成相关性: {answer_relevancy:.3f} ({len(hyp_qs)}个假设问题)")

    return {
        "question": q,
        "context_relevance": context_relevance,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "contexts": len(contexts),
    }


def clear_qa_cache():
    """清空运行时缓存（qa_cache），保证评测问题都走真实 RAG 检索而非缓存命中。"""
    import sqlite3
    path = config.KB_DIR / "qa.db"
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.execute("DELETE FROM qa_cache")
        conn.commit()
        print(f"已清空运行时缓存（删除 {cur.rowcount} 条）")
    finally:
        conn.close()


def main():
    print(f"评估 LLM: {EVAL_LLM} | Embedding: {EMBED_MODEL}")
    clear_qa_cache()
    results = []
    for i, t in enumerate(TESTS, 1):
        print(f"[{i}/{len(TESTS)}] 问题: {t['question']}")
        r = evaluate_one(t)
        if r:
            results.append(r)
        print()
    if not results:
        print("没有可用的评测样本（请先清理缓存后重试）。")
        return
    print("=" * 56)
    print("RAGAS 评测汇总（样本均值）")
    print("=" * 56)
    for key, name in [
        ("context_relevance", "检索相关性"),
        ("context_recall", "检索召回率"),
        ("faithfulness", "生成忠实度"),
        ("answer_relevancy", "生成相关性"),
    ]:
        avg = statistics.mean(r[key] for r in results)
        print(f"{name}: {avg:.3f}")
    print("=" * 56)
    for r in results:
        print(f"  {r['question'][:28]}... | 相关={r['context_relevance']:.2f} "
              f"召回={r['context_recall']:.2f} 忠实={r['faithfulness']:.2f} "
              f"相关生成={r['answer_relevancy']:.2f}")


if __name__ == "__main__":
    main()