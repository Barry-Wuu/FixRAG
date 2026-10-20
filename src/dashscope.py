"""LLM 调用封装（OpenAI 兼容接口，标准库 urllib，无额外依赖）：
- embed：DashScope 文本嵌入（text-embedding-v1，1536 维）
- chat / chat_stream：优先 DashScope 生成；若失败（如免费配额耗尽 HTTP 403），
  自动回退到 OpenRouter 免费生成模型（config.OPENROUTER_GEN_MODEL），保证服务不中断。
"""
import json
import urllib.request
import urllib.error

import config


def _post(path: str, payload: dict, timeout: int = 60):
    url = config.DASHSCOPE_BASE_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"DashScope {path} HTTP {e.code}: {body}") from e


def embed(texts, model=None):
    """调用 DashScope 文本嵌入，返回 list[list[float]]（顺序与输入一致）。"""
    model = model or config.EMBED_MODEL
    if isinstance(texts, str):
        texts = [texts]
    resp = _post("/embeddings", {"model": model, "input": texts})
    items = sorted(resp["data"], key=lambda x: x.get("index", 0))
    return [x["embedding"] for x in items]


# ---------------------------------------------------------------------------
# OpenRouter 免费生成模型回退
# ---------------------------------------------------------------------------

def _gen_model_list():
    """解析配置的 OpenRouter 免费模型回退链（逗号分隔）。"""
    return [m.strip() for m in config.OPENROUTER_GEN_MODEL.split(",") if m.strip()]


def _openrouter_chat_any(messages, temperature=0.7, max_tokens=1024):
    """按回退链依次尝试非流式生成，全部失败才抛异常。"""
    last_err = None
    for m in _gen_model_list():
        try:
            return _openrouter_chat(messages, m, temperature, max_tokens)
        except Exception as e:
            print(f"[llm] OpenRouter {m} 失败({str(e)[:80]})，尝试下一个")
            last_err = e
    raise RuntimeError(f"所有 OpenRouter 免费模型均失败: {last_err}")


def _openrouter_stream_any(messages, temperature=0.7, max_tokens=1024):
    """按回退链依次尝试流式生成，全部失败才抛异常。"""
    last_err = None
    for m in _gen_model_list():
        try:
            yield from _openrouter_stream(messages, m, temperature, max_tokens)
            return
        except Exception as e:
            print(f"[llm] OpenRouter 流式 {m} 失败({str(e)[:80]})，尝试下一个")
            last_err = e
    raise RuntimeError(f"所有 OpenRouter 免费模型均失败: {last_err}")


def _openrouter_chat(messages, model=None, temperature=0.7, max_tokens=1024, timeout=180):
    """非流式调用 OpenRouter 免费生成模型，返回文本字符串。"""
    model = model or config.OPENROUTER_GEN_MODEL
    url = config.OPENROUTER_GEN_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        raise RuntimeError(f"OpenRouter /chat/completions HTTP {e.code}: {body}") from e
    return data["choices"][0]["message"]["content"].strip()


def _openrouter_stream(messages, model=None, temperature=0.7, max_tokens=1024):
    """流式调用 OpenRouter 免费生成模型，yield {"content":..., "reasoning":...}。"""
    model = model or config.OPENROUTER_GEN_MODEL
    url = config.OPENROUTER_GEN_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            content = None
            reasoning = None
            try:
                delta = json.loads(data_str)["choices"][0]["delta"]
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            except Exception:
                continue
            if content or reasoning:
                yield {"content": content, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# 对外接口：优先 DashScope，失败自动回退 OpenRouter
# ---------------------------------------------------------------------------

def chat(messages, model=None, temperature=0.7, max_tokens=1024):
    """非流式对话。DashScope 生成额度已用光，优先 OpenRouter 免费模型链；失败再回退 DashScope。"""
    try:
        return _openrouter_chat_any(messages, temperature, max_tokens)
    except Exception as e:
        print(f"[llm] OpenRouter 免费模型均失败({str(e)[:100]})，回退 DashScope")
        try:
            model = model or config.LLM_MODEL
            resp = _post(
                "/chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e2:
            raise RuntimeError(f"OpenRouter 与 DashScope 生成均失败: {e2}") from e2


def _dashscope_stream(messages, model=None, temperature=0.7, max_tokens=1024):
    """DashScope 流式对话生成器（SSE）。yield {"content":..., "reasoning":...}。"""
    model = model or config.LLM_MODEL
    url = config.DASHSCOPE_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            content = None
            reasoning = None
            try:
                delta = json.loads(data_str)["choices"][0]["delta"]
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
            except Exception:
                pass
            if content or reasoning:
                yield {"content": content, "reasoning": reasoning}


def chat_stream(messages, model=None, temperature=0.7, max_tokens=1024):
    """流式对话生成器。DashScope 生成额度已用光，优先 OpenRouter 免费模型链；失败再回退 DashScope。"""
    try:
        yield from _openrouter_stream_any(messages, temperature, max_tokens)
    except Exception as e:
        print(f"[llm] OpenRouter 流式均失败({str(e)[:100]})，回退 DashScope")
        yield from _dashscope_stream(messages, model, temperature, max_tokens)