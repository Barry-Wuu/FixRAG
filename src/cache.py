"""问答缓存（底层为本地 SQLite，对应流程图 Redis 高频缓存）。
get 返回 {answer, intent, strategy} 或 None；put 支持写入 intent/strategy。
"""
import db


class AnswerCache:
    def get(self, query):
        return db.get_cache(query)

    def put(self, query, answer, intent=None, strategy=None):
        db.put_cache(query, answer, intent=intent, strategy=strategy)

    def __len__(self):
        return db.stats()["qa_cache"]


_cache = None


def get_cache():
    global _cache
    if _cache is None:
        _cache = AnswerCache()
    return _cache
