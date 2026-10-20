"""构建知识库索引：加载 ai_data 文档 -> 父/子分块 -> DashScope 向量化 -> 持久化到 kb_store/。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import loader
import chunker
import vectorstore
import db
import config


def main():
    print("数据目录:", config.DATA_DIR)
    docs = loader.load_documents()
    if not docs:
        raise SystemExit("没有加载到任何文档，请检查 FIXRAG_DATA_DIR。")
    print(f"已加载文档 {len(docs)} 篇")

    chunks = []
    for i, d in enumerate(docs):
        chunks.extend(chunker.split_parent_child(
            d["text"], i, d["source"], d.get("file_path", "")))
    print(f"切分出子块 {len(chunks)} 个")

    store = vectorstore.VectorStore()
    store.build(chunks)
    print("知识库构建完成，已保存到:", config.KB_DIR)

    # 灌入高频问答库（MySQL 角色），来自 seed_qa.json
    seed_file = os.path.join(os.path.dirname(__file__), "seed_qa.json")
    if os.path.exists(seed_file):
        db.clear_hot()  # 先清空，避免旧 seed（如黑马课程问答）残留
        n = db.seed_hot_from_file(seed_file)
        print(f"已向高频问答库灌入 {n} 条问答（seed_qa.json）")


if __name__ == "__main__":
    main()
