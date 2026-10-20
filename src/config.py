"""智能家电维修问答系统 配置：路径、模型名、DashScope key 加载（优先环境变量，回退到 .env 文件）。"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # D:\models\edurag


def _read_env_file(path: str) -> dict:
    d = {}
    p = Path(path)
    if not p.exists():
        return d
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def _load_api_key() -> str:
    # 1) 环境变量
    if os.getenv("DASHSCOPE_API_KEY"):
        return os.getenv("DASHSCOPE_API_KEY")
    # 2) 项目内 .env
    for cand in [BASE_DIR / ".env", BASE_DIR / "config" / ".env"]:
        if cand.exists():
            v = _read_env_file(str(cand)).get("DASHSCOPE_API_KEY")
            if v:
                return v
    # 3) 用户自己的 env（按你之前提供的位置）
    for cand in [r"D:\B++\.env", "/mnt/d/B++/.env"]:
        v = _read_env_file(cand).get("DASHSCOPE_API_KEY")
        if v:
            return v
    raise RuntimeError(
        "未找到 DASHSCOPE_API_KEY：请设置环境变量，或把 key 写入项目 .env / D:\\B++\\.env"
    )


# ---- DashScope ----
DASHSCOPE_API_KEY = _load_api_key()
DASHSCOPE_BASE_URL = os.getenv(
    "FIXRAG_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBED_MODEL = os.getenv("FIXRAG_EMBED_MODEL", "text-embedding-v1")  # 1536 维
LLM_MODEL = os.getenv("FIXRAG_LLM_MODEL", "qwen-max")             # 你 key 可用：qwen-max/plus/turbo/long


def _load_openrouter_key() -> str:
    """加载 OpenRouter API key：优先环境变量 OPENROUTER_API_KEY，回退 .env 中的同名键。"""
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY")
    for cand in [BASE_DIR / ".env", r"D:\B++\.env", "/mnt/d/B++/.env"]:
        v = _read_env_file(cand).get("OPENROUTER_API_KEY")
        if v:
            return v
    return ""


# ---- OpenRouter 免费决策模型（充当 Jev 角色，LLM 意图判定兜底）----
OPENROUTER_API_KEY = _load_openrouter_key()
INTENT_LLM_MODEL = os.getenv(
    "FIXRAG_INTENT_LLM_MODEL", "inception/mercury-decide:free"
)  # 快速决策模型（OpenRouter Decisions API / systemone schema）：
    # inception/mercury-decide:free 实测 5/5 分类准确、平均约 1.7s，直接输出结构化 JSON（choice + probabilities）
INTENT_LLM_BASE_URL = os.getenv(
    "FIXRAG_INTENT_LLM_BASE_URL", "https://openrouter.ai/api/alpha/decisions"
)
USE_LLM_INTENT = os.getenv("FIXRAG_USE_LLM_INTENT", "1") == "1"  # 0=关闭 LLM 兜底，只用 随机森林+Jieba 加权

# ---- OpenRouter 免费生成模型（DashScope 免费配额耗尽/失败时自动回退）----
# 逗号分隔的回退链：前一个模型被上游限流(429)/不可用(404)/失败时，自动尝试下一个。
# OpenRouter 免费模型当前普遍限流(429)，暂置空以直连 DashScope；恢复时填回回退链即可
OPENROUTER_GEN_MODEL = os.getenv("FIXRAG_GEN_LLM_MODEL", "")
OPENROUTER_GEN_BASE_URL = os.getenv(
    "FIXRAG_GEN_LLM_BASE_URL", "https://openrouter.ai/api/v1"
)

def _resolve(win_path: str, wsl_path: str, env_name: str) -> Path:
    """优先 env 覆盖；否则在 Windows 路径与 WSL 挂载路径间自动选择存在的那个。"""
    if os.getenv(env_name):
        return Path(os.getenv(env_name))
    wp = Path(win_path)
    if wp.exists():
        return wp
    lp = Path(wsl_path)
    if lp.exists():
        return lp
    return wp  # 都不存在时返回 Windows 默认（后续会报错提示）


# ---- 意图识别模型（已训好的随机森林）----
# 训练时编码：0=通用知识（直接 LLM 答），1=专业咨询（走 RAG）
# 优先仓库内 models/FIXRAG.rf.joblib，回退外部 D:\models\ 路径
_INREPO_MODEL = BASE_DIR / "models" / "FIXRAG.rf.joblib"
_EXTERNAL_MODEL = Path(r"D:\models\FIXRAG.rf.joblib")
_EXTERNAL_MODEL_WSL = Path("/mnt/d/models/FIXRAG.rf.joblib")
if os.getenv("FIXRAG_MODEL_PATH"):
    MODEL_PATH = Path(os.getenv("FIXRAG_MODEL_PATH"))
elif _INREPO_MODEL.exists():
    MODEL_PATH = _INREPO_MODEL
elif _EXTERNAL_MODEL.exists():
    MODEL_PATH = _EXTERNAL_MODEL
elif _EXTERNAL_MODEL_WSL.exists():
    MODEL_PATH = _EXTERNAL_MODEL_WSL
else:
    MODEL_PATH = _INREPO_MODEL
# 意图标签（智能家电维修问答系统场景）：
# 0=与家电/家电维修无关（礼貌拒绝，不检索不缓存）
# 1=专业咨询（家电维修/售后，走 RAG 检索+生成）
# 2=家电通用技术知识（原理/选购/使用常识，LLM 直答，不检索不缓存）
LABEL_GENERAL = 0
LABEL_CONSULT = 1
LABEL_GENERAL_TECH = 2
INTENT_NAMES = {
    LABEL_GENERAL: "与家电无关",
    LABEL_GENERAL_TECH: "家电通用知识",
    LABEL_CONSULT: "专业咨询(家电维修)",
}

# ---- 知识库源文档目录（优先仓库内 data/docs，回退外部原项目路径）----
_INREPO_DATA = BASE_DIR / "data" / "docs"
_EXTERNAL_DATA = Path(
    r"D:\外部家电资料\data\docs"
)
_EXTERNAL_DATA_WSL = Path(
    "/mnt/d/外部家电资料/data/docs"
)
if os.getenv("FIXRAG_DATA_DIR"):
    DATA_DIR = Path(os.getenv("FIXRAG_DATA_DIR"))
elif _INREPO_DATA.exists():
    DATA_DIR = _INREPO_DATA
elif _EXTERNAL_DATA.exists():
    DATA_DIR = _EXTERNAL_DATA
elif _EXTERNAL_DATA_WSL.exists():
    DATA_DIR = _EXTERNAL_DATA_WSL
else:
    DATA_DIR = _INREPO_DATA

# ---- 向量库 / 缓存持久化目录 ----
KB_DIR = Path(os.getenv("FIXRAG_KB_DIR", str(BASE_DIR / "kb_store")))
KB_DIR.mkdir(parents=True, exist_ok=True)
KB_EMBED_FILE = KB_DIR / "embeddings.npy"
KB_META_FILE = KB_DIR / "chunks.json"

# ---- 检索/切块参数（对应原项目 config.ini [retrieval]）----
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "1200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))        # 返回父块数量
TOP_CHILD_PER_QUERY = int(os.getenv("TOP_CHILD", "10")) # 每次检索召回的子块数
CACHE_SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.75"))  # 缓存语义去重/命中阈值（同义变体合并）
