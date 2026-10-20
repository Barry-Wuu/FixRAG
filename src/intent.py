"""意图识别：两级判定，每级都是更聚焦的二分类。

训练编码（随机森林为二分类）：0=通用知识，1=专业咨询。
本层在其上扩展为三类意图（智能家电维修问答系统场景）：
- LABEL_GENERAL(0)       与家电/家电维修无关的通用知识（天气/生活/闲聊）→ LLM 礼貌拒绝，不检索不缓存
- LABEL_CONSULT(1)       家电维修/售后的专业咨询（故障现象/故障代码/维修步骤/配件型号/安装保养/保修）→ 走 RAG 检索+生成
- LABEL_GENERAL_TECH(2)  家电产品相关的通用技术知识（原理/概念/选购/使用常识/参数对比）→ LLM 直接作答，不检索不缓存

两级判定流程：
第一级「是否为专业咨询（家电维修）」：
  随机森林 P(consult) + Jieba 咨询关键词（强信号）+ 技术术语分（弱化 consult 的负信号）
  混合加权，必要时再请 Jev（OpenRouter 快速决策模型）单独二分类判定 consult / not_consult；
  命中咨询关键词直接判专业咨询，命中强技术术语且无咨询关键词则跳过 Jev（负信号足够强）。
第二级「是否为家电通用技术知识」（仅当第一级判定非专业咨询）：
  命中强技术术语 → 直接家电通用知识；
  否则请 Jev 单独二分类判定 tech_general / unrelated；
  Jev 不可用时降级：命中普通术语 → 家电通用知识，否则 → 与家电无关。
"""
import json
import urllib.error
import urllib.request
from pathlib import Path

import jieba
import joblib

import config

# 专业咨询强信号词（家电维修/售后）：随机森林对这类问句可能临界误判为通用知识，用关键词兜底纠正
# 命中即直接判专业咨询(走 RAG)。注意：故障现象/维修/售后类词放这里，原理/概念类词放 _STRONG_TECH。
_CONSULT_KEYWORDS = (
    "故障", "坏了", "不制冷", "不制热", "不启动", "不工作", "无法开机", "打不开",
    "不转", "不运转", "漏水", "漏氟", "漏雪种", "漏电", "异响", "噪音大", "噪声大",
    "报警", "报错", "故障代码", "错误代码", "代码", "显示屏", "闪烁", "结冰", "结霜",
    "不显示", "不加热", "不进水", "不排水", "不脱水", "不点火", "打不着", "熄火",
    "维修", "修理", "检修", "检测", "拆机", "清洗", "保养", "安装", "移机",
    "加氟", "加雪种", "换", "更换", "配件", "零件", "备件", "型号", "规格",
    "售后", "保修", "质保", "上门", "师傅", "预约", "报修", "维修费", "维修价格",
    "维修多少钱", "多少钱修", "怎么修", "如何修", "解决办法", "处理方法",
)

# 强技术术语：几乎只出现在「家电原理/通用知识」场景，命中且无咨询关键词即判家电通用知识(LLM直答)
_STRONG_TECH = (
    "变频", "定频", "压缩机", "制冷剂", "冷媒", "氟利昂", "毛细管", "膨胀阀", "节流",
    "蒸发器", "冷凝器", "风机", "电机", "电容", "能效", "能效比", "制冷量", "制热量",
    "匹", "分贝", "电压", "频率", "电阻", "温控", "传感器", "pcb", "主板", "电路板",
    "变频板", "排水管", "进水阀", "水泵", "加热管", "镁棒", "滤网", "翅片", "风道",
    "风冷", "直冷", "混冷", "风冷无霜", "双循环", "单循环", "热泵", "电辅热",
    "物联网", "智能家电", "巡航", "自清洁", "除菌", "杀菌",
)

# 普通技术术语：有家电倾向但不够强，参与加权得分
_NORMAL_TECH = (
    "空调", "冰箱", "洗衣机", "电视", "热水器", "油烟机", "燃气灶", "微波炉",
    "电饭煲", "扫地机器人", "风扇", "加湿器", "净化器", "除湿机", "洗碗机", "烘干机",
    "家电", "电器", "品牌", "功率", "耗电", "省电", "节能", "静音", "容量",
    "尺寸", "重量", "插座", "插头", "遥控", "面板", "滤芯", "滤网",
)

# OpenRouter 快速决策模型（充当 Jev 角色）的两个二分类 choice typed question 定义：
# Decisions API（/api/alpha/decisions，与 TypeSafe systemone 同 schema）直接返回结构化 JSON。
# 每次只做二分类，比一次三分类更聚焦，判定更准。

def _llm_decision(query, option_a, option_b, criteria_a, criteria_b, timeout=30):
    """通用二分类判定：Jev 从 option_a / option_b 二选一，返回 (choice, prob_a) 或 None。
    prob_a 为选中 option_a 的概率（0~1）；调用失败/限流/超时返回 None。"""
    if not config.OPENROUTER_API_KEY or not config.USE_LLM_INTENT:
        return None
    payload = {
        "model": config.INTENT_LLM_MODEL,
        "state": query,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "二选一，将用户问题归入更贴切的一类。",
                "criteria": {option_a: criteria_a, option_b: criteria_b},
            }
        },
    }
    req = urllib.request.Request(
        config.INTENT_LLM_BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + config.OPENROUTER_API_KEY,
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ans = data["answers"]["category"]
        choice = ans.get("choice")
        if choice not in (option_a, option_b):
            return None
        probs = ans.get("probabilities") or {}
        pa = probs.get(option_a)
        return choice, (float(pa) if pa is not None else None)
    except Exception:
        return None
    return None


def _llm_is_consult(query: str, timeout: int = 30):
    """第一级：Jev 单独二分类判定是否为专业咨询（家电维修/售后）。返回 (choice, consult_prob) 或 None。"""
    return _llm_decision(
        query,
        "consult", "not_consult",
        "智能家电维修/售后相关的专业咨询（故障现象、故障代码、维修步骤、配件/型号查询、安装/保养/清洗、售后政策/保修等，需要检索维修资料或给出专业维修建议来回答）",
        "不是家电维修专业咨询（家电通用知识、闲聊、天气、生活、娱乐等其他话题）",
        timeout=timeout,
    )


def _llm_is_tech(query: str, timeout: int = 30):
    """第二级：Jev 单独二分类判定是否为家电通用技术知识。返回 (choice, tech_prob) 或 None。"""
    return _llm_decision(
        query,
        "tech_general", "unrelated",
        "与家电产品/家电使用相关的通用技术知识问题（产品原理、概念、选购、使用常识、参数对比等）",
        "与家电无关的话题（闲聊、天气、生活常识、娱乐、其他行业等）",
        timeout=timeout,
    )


def llm_is_synonym(q1: str, q2: str, timeout: int = 30):
    """Jev 判定两个问题是否同义（即可以用同一条答案回答）。

    用于缓存命中前的语义复核：embedding 余弦相似度只能衡量「文本相近」，
    无法区分『格力空调开机显示E1故障代码怎么办?』与『海尔空调开机显示E1故障代码怎么办?』这类
    句式相同但关键信息（品牌/型号/对象）不同的问题，需要 LLM 二次判定。
    返回 (choice, prob_same) 或 None（不可用/失败/超时）。
    """
    state = f"问题A：{q1}\n问题B：{q2}"
    return _llm_decision(
        state,
        "same", "different",
        "两个问题表达的是同一个意思（同义改写、说法不同），可以用同一条答案回答",
        "两个问题意思不同：句式相同但关键信息（品牌/型号/对象/地点等）不同，或问的根本不是同一件事，需要不同的答案",
        timeout=timeout,
    )


class IntentClassifier:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.model = joblib.load(str(self.model_path))
        self.classes_ = list(getattr(self.model, "classes_", [0, 1]))

    def _tech_signal(self, query: str):
        """用 Jieba 分词 + 术语词典提取技术信号。
        返回 (tech_score, strong_hit_list, normal_hit_list)。
        短英文词（<=2 字符，如 ai）用 jieba 分词 token 匹配，避免 email/main 等误命中；
        长词/中文词用子串匹配（不依赖 jieba 切分精度）。"""
        ql = query.lower()
        tokens = {w.lower() for w in jieba.lcut(query)}
        strong, normal = [], []
        for t in _STRONG_TECH:
            if len(t) <= 2 and t.isascii():
                # 短英文词用 jieba token 精确匹配，避免 email/main 误命中 ai 之类
                if t in tokens:
                    strong.append(t)
            elif t in ql:
                strong.append(t)
        for t in _NORMAL_TECH:
            if len(t) <= 2 and t.isascii():
                if t in tokens:
                    normal.append(t)
            elif t in ql:
                normal.append(t)
        score = min(1.0, len(strong) * 0.65 + len(normal) * 0.25)
        return score, strong, normal

    def predict(self, query: str) -> int:
        """返回 0(与家电无关) / 1(专业咨询/家电维修) / 2(家电通用知识)。"""
        return self.analyze(query)["final_code"]

    def predict_proba(self, query: str):
        """返回 (P(通用知识), P(专业咨询))；模型不支持时返回 (None, None)。"""
        q = (query or "").strip()
        try:
            proba = self.model.predict_proba([q])[0]
        except Exception:
            return None, None
        idx = {c: i for i, c in enumerate(self.classes_)}
        p_general = float(proba[idx.get(config.LABEL_GENERAL, 0)])
        p_consult = float(proba[idx.get(config.LABEL_CONSULT, 1)])
        return p_general, p_consult

    def analyze(self, query: str) -> dict:
        """返回意图识别的完整推理信息，供“思考过程”展示。

        两级判定：
        第一级「是否专业咨询」：
          加权分 weighted = 0.3×P(consult) + 0.7×技术术语分（技术分越高越不可能是咨询）；
          命中咨询关键词 → 直接专业咨询；
          命中强技术术语且无咨询关键词 → 跳过 Jev，判非专业咨询（进入第二级）；
          否则调 Jev 二分类 consult/not_consult，融合分 = 0.5×加权 + 0.5×Jev_consult_prob，
          ≥0.5 → 专业咨询。
        第二级「是否家电通用知识」（仅第一级判非专业咨询时）：
          命中强术语 → 家电通用知识；
          否则调 Jev 二分类 tech_general/unrelated；
          Jev 不可用时降级：命中普通术语 → 家电相关通用知识，否则 → 与家电无关。
        """
        q = (query or "").strip()
        rf_code = int(self.model.predict([q])[0])
        rf_label = config.INTENT_NAMES.get(rf_code, "通用知识")
        p_general, p_consult = self.predict_proba(q)
        tech_score, strong_tech, normal_tech = self._tech_signal(q)
        kw_hit = [k for k in _CONSULT_KEYWORDS if k in q]
        weighted = 0.3 * (p_consult if p_consult is not None else 0.0) + 0.7 * tech_score

        final_code = None
        llm_category, llm_prob, llm_used, llm_error = None, None, False, False
        consult_choice = consult_prob = tech_choice = tech_prob = None
        final_prob = weighted
        correction = ""

            # ---- 第一级：是否专业咨询 ----
        if kw_hit:
            final_code = config.LABEL_CONSULT
            correction = "关键词兜底"
        elif strong_tech:
            # 强技术术语且无咨询关键词：非咨询信号足够强，跳过第一级 Jev
            final_code = None  # 待第二级判定
        else:
            # 混合：随机森林加权 + Jev 二分类判定是否专业咨询
            consult_choice, consult_prob = None, None
            if config.USE_LLM_INTENT:
                result = _llm_is_consult(q)
                llm_used = True
                if result is not None:
                    consult_choice, consult_prob = result
                    llm_category = consult_choice
                    llm_prob = consult_prob
                    final_prob = 0.5 * weighted + 0.5 * (consult_prob or 0.0)
                    correction = "LLM 兜底" if final_code != rf_code else ""
                else:
                    llm_error = True
                    final_prob = weighted
                    correction = "加权" if final_code != rf_code else ""
            else:
                final_prob = weighted
                correction = "加权" if final_code != rf_code else ""
            if final_prob >= 0.5:
                final_code = config.LABEL_CONSULT
                if not correction:
                    correction = "加权" if final_code != rf_code else ""
            else:
                final_code = None  # 非专业咨询，进入第二级

        # ---- 第二级：是否 AI/编程相关（仅非专业咨询时） ----
        if final_code is None:
            if strong_tech:
                final_code = config.LABEL_GENERAL_TECH
                correction = "Jieba 强术语"
            else:
                tech_choice, tech_prob = None, None
                if config.USE_LLM_INTENT:
                    result = _llm_is_tech(q)
                    llm_used = True
                    if result is not None:
                        tech_choice, tech_prob = result
                        llm_category = tech_choice
                        llm_prob = tech_prob
                        correction = "LLM 兜底" if final_code != rf_code else ""
                    else:
                        llm_error = True
                        correction = "加权" if final_code != rf_code else ""
                if tech_choice == "tech_general" or (tech_choice is None and normal_tech):
                    final_code = config.LABEL_GENERAL_TECH
                else:
                    final_code = config.LABEL_GENERAL

        final_label = config.INTENT_NAMES.get(final_code, "通用知识")

        # 两级判定的结构化详情（供前端「思考过程」逐步展示）
        if kw_hit:
            stage1 = {"type": "keyword", "choice": None, "prob": None}
        elif strong_tech:
            stage1 = {"type": "strong_tech", "choice": None, "prob": None}
        elif consult_choice is not None:
            stage1 = {"type": "llm", "choice": consult_choice, "prob": consult_prob}
        elif llm_used:
            stage1 = {"type": "llm_error", "choice": None, "prob": None}
        else:
            stage1 = {"type": "weighted", "choice": None, "prob": None}

        if final_code == config.LABEL_CONSULT:
            stage2 = None
        elif strong_tech:
            stage2 = {"type": "strong_tech", "choice": None, "prob": None}
        elif tech_choice is not None:
            stage2 = {"type": "llm", "choice": tech_choice, "prob": tech_prob}
        elif llm_used:
            stage2 = {"type": "llm_error", "choice": None, "prob": None}
        else:
            stage2 = {"type": "weighted", "choice": None, "prob": None}

        return {
            "rf_label": rf_label, "rf_code": rf_code,
            "p_general": p_general, "p_consult": p_consult,
            "keyword_hit": kw_hit, "overridden": final_code != rf_code,
            "correction": correction,
            "tech_score": tech_score,
            "strong_tech": strong_tech,
            "normal_tech": normal_tech,
            "weighted_score": weighted,
            "llm_category": llm_category,
            "llm_prob": llm_prob, "llm_used": llm_used, "llm_error": llm_error,
            "final_prob": final_prob,
            "stage1": stage1, "stage2": stage2,
            "final_label": final_label, "final_code": final_code,
        }

    def label(self, query: str) -> str:
        return config.INTENT_NAMES.get(self.predict(query), "通用知识")


_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier