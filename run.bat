@echo off
REM ===== 家电维修问答系统 Windows launcher (uses local Python at D:\B++\py313) =====
cd /d D:\models\edurag

REM Path to your Python 3.13 interpreter
set PY=D:\B++\py313\python.exe

REM Optional: uncomment and set your key / paths (omit to auto-read from D:\B++\.env)
REM set DASHSCOPE_API_KEY=sk-ws-xxxx
REM set FIXRAG_DATA_DIR=D:\models\edurag\data\docs
REM set FIXRAG_MODEL_PATH=D:\models\FIXRAG.rf.joblib

REM 1) Create venv if missing (reuse base packages via --system-site-packages)
if not exist .venv (
    "%PY%" -m venv --system-site-packages .venv
)
call .venv\Scripts\activate

REM 2) Install dependencies (skips fast if already satisfied)
pip install -r requirements.txt

REM 3) Build knowledge base index ONCE (cached to kb_store/).
REM    Skips if index already exists; delete kb_store\embeddings.npy to force rebuild.
if not exist kb_store\embeddings.npy (
    python build_kb.py
) else (
    echo kb_store already exists, skip build_kb.py
)

REM 4) Start server and open the web UI in browser
start "" http://127.0.0.1:8000/
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
