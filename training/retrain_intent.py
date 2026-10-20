"""重训意图识别随机森林（可复现）。

- 数据：models/query_intent_dataset.jsonl（家电维修场景二分类语料）
- 配置与原训练一致：TfidfVectorizer(char_wb, 1-3) + RandomForestClassifier(300)
- 产出：models/FIXRAG.rf.joblib（原模型自动备份为 models/FIXRAG.rf.v1.joblib）

运行：python training/retrain_intent.py
"""
import json
import os
import shutil
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SRC = MODELS / "query_intent_dataset.jsonl"
MODEL = MODELS / "FIXRAG.rf.joblib"
BAK = MODELS / "FIXRAG.rf.v1.joblib"

# 扩充样本：弥补模板化数据对“故障代码/品牌/配件/型号”等多样化表述覆盖不足
EXTRA_CONSULT = [
    "格力空调显示E1是什么故障", "美的空调不制冷怎么回事", "海尔洗衣机不脱水怎么办",
    "冰箱冷藏室结冰怎么调", "燃气热水器打不着火原因", "空调加氟一次多少钱",
    "洗衣机型号在哪里看", "空调保修期几年", "油烟机不吸烟怎么修",
    "电视黑屏有声音", "电热水器不加热", "冰箱漏氟维修费",
    "空调遥控器失灵", "洗衣机脱水噪音大", "热水器显示E1",
    "冰箱压缩机不启动", "空调内机漏水", "洗衣机门锁不住",
    "电视没有信号", "油烟机怎么清洗", "燃气灶打不着火",
    "微波炉不加热", "空调移机多少钱", "冰箱怎么除味",
    "洗衣机内筒清洗", "热水器镁棒多久换", "空调清洗价格",
    "冰箱排水孔堵了", "洗衣机进水不止", "电视闪屏",
    "空调外机不转", "冰箱噪音大", "洗衣机不进水",
    "热水器水温不稳", "油烟机灯不亮", "空调制热效果差",
    "洗衣机显示E2", "冰箱灯不亮不制冷", "空调显示F0",
    "海尔空调故障代码E6", "西门子冰箱不制冷", "小米电视闪屏",
    "洗衣机皮带松了", "空调需要加雪种吗", "洗碗机不进水",
    "热水器漏水到地上", "冰箱门封条不严", "电视扬声器没声音",
    "空调频繁启停", "洗衣机有异味", "燃气热水器风压故障",
]
EXTRA_GENERAL = [
    "变频空调和定频区别", "一级能效省电吗", "空调匹数怎么选",
    "波轮和滚筒哪个好", "洗衣机容量怎么选", "冰箱风冷直冷区别",
    "油烟机顶吸侧吸", "电热水器选容量", "燃气热水器升数",
    "洗碗机费水吗", "空气净化器有用吗", "电视最佳观看距离",
    "OLED和LCD区别", "净水器原理", "空调温度26度为什么",
    "洗衣机放多少洗衣液", "微波炉能用金属吗", "冰箱停电保鲜多久",
    "空调长时间不用拔插头吗", "电视HDR是什么", "冰箱能效等级怎么看",
    "洗衣机烘干容量", "空调自清洁有用吗", "油烟机风量越大越好吗",
    "电热水器即热储水区别", "空调除湿制冷区别", "冰箱怎么除异味",
    "洗衣机羊毛程序", "燃气热水器恒温原理", "电视投屏卡顿",
    "水的沸点多少度", "地球到月球多远", "什么是光合作用",
    "为什么天空是蓝色", "圆周率多少", "一年多少节气",
    "中国首都是哪", "DNA全称", "光速多少",
    "如何提高睡眠", "咖啡喝多影响", "鸡蛋怎么煮",
    "为什么有四季", "什么是通货膨胀", "长江多长",
    "太阳系几大行星", "竹子是树还是草", "食盐成分",
    "海拔高度影响沸点吗", "今天天气怎么样", "讲个笑话听听",
    "推荐几部好看的电影", "怎么学好英语", "减肥有什么好办法",
]


def load_rows():
    rows = []
    if SRC.exists():
        for line in open(SRC, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    base_n = len(rows)
    existing = {r["query"] for r in rows}
    for q in EXTRA_CONSULT:
        if q not in existing:
            rows.append({"query": q, "label": "专业咨询"})
    for q in EXTRA_GENERAL:
        if q not in existing:
            rows.append({"query": q, "label": "通用知识"})

    texts = [r["query"] for r in rows]
    y = [0 if r["label"] == "通用知识" else 1 for r in rows]
    print(f"数据集: {base_n} -> {len(rows)} 条")

    def build():
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                                      min_df=2, max_features=20000)),
            ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                           random_state=42)),
        ])

    Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=0.2,
                                          random_state=42, stratify=y)
    p = build()
    p.fit(Xtr, ytr)
    yp = p.predict(Xte)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, average="weighted")
    print(f"[80/20] acc={accuracy_score(yte, yp):.4f} "
          f"prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
    print("混淆矩阵:", confusion_matrix(yte, yp, labels=[0, 1]).tolist())

    final = build()
    final.fit(texts, y)
    if MODEL.exists() and not BAK.exists():
        shutil.copy2(MODEL, BAK)
        print("已备份原模型 ->", BAK)
    joblib.dump(final, MODEL)
    print("已保存 ->", MODEL)

    m = joblib.load(MODEL)
    for q in ["空调不制冷怎么办", "什么是变频空调", "洗衣机怎么清洗", "水的沸点是多少度"]:
        lab = "通用知识" if int(m.predict([q])[0]) == 0 else "专业咨询"
        print(f"  [{lab}] {q}")


if __name__ == "__main__":
    main()
