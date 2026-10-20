"""中文递归分块：先切父块(parent)，再切子块(child)，子块元数据里保留父块全文。
对应原项目 document_loader.process_documents 的 parent/child 分层策略。"""
import config

SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]


def _split_recursive(text, chunk_size, overlap, seps):
    if len(text) <= chunk_size:
        return [text]
    sep = seps[0]
    if sep == "":
        step = max(1, chunk_size - overlap)
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]
    if sep not in text:
        return _split_recursive(text, chunk_size, overlap, seps[1:])
    pieces = text.split(sep)
    pieces = [p + sep for p in pieces[:-1]] + [pieces[-1]]
    merged, buf = [], ""
    for p in pieces:
        if len(buf) + len(p) <= chunk_size:
            buf += p
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    result = []
    for m in merged:
        if len(m) > chunk_size:
            result.extend(_split_recursive(m, chunk_size, overlap, seps[1:]))
        else:
            result.append(m)
    return result


def split_parent_child(text, doc_index, source, file_path="",
                       parent_size=None, child_size=None, overlap=None):
    parent_size = parent_size or config.PARENT_CHUNK_SIZE
    child_size = child_size or config.CHILD_CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    parents = _split_recursive(text, parent_size, overlap, SEPARATORS)
    out = []
    for j, parent in enumerate(parents):
        parent_id = f"doc_{doc_index}_parent_{j}"
        children = _split_recursive(parent, child_size, overlap, SEPARATORS)
        for k, child in enumerate(children):
            out.append({
                "child_id": f"{parent_id}_child_{k}",
                "parent_id": parent_id,
                "child_text": child,
                "parent_text": parent,
                "source": source,
                "file_path": file_path,
            })
    return out
