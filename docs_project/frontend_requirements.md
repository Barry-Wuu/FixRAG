# 智能家电维修问答系统 前端需求说明（供 WorkBuddy / 包 实现）

后端服务已在本机运行：`http://127.0.0.1:8000`（CORS 已放开）。前端只需调用后端 REST 接口即可，无需了解内部实现。

## 一、总体要求
- 单页应用，顶部为导航，含两个可切换栏目：
  1. **对话**（默认界面）
  2. **项目说明**
- 技术栈不限（原生 HTML/JS 或 Vue/React 均可），能 `fetch` 调用下面接口即可。

## 二、栏目 1：对话（默认界面）
- 进入页面**默认展示对话界面**。
- 组成：消息列表 + 底部输入框 + 发送按钮。
- 发送时调用 `POST /query`，请求体 `{"query": "用户输入"}`。
- 响应字段：
  - `answer`：回答文本（直接渲染，支持简单 Markdown 加粗/换行更佳）
  - `intent`：意图（如「通用知识」「专业咨询」「高频问答库」「缓存命中」）
  - `strategy`：检索策略（可能为 `null`）
  - `sources`：参考资料列表（可为空）
  - `cached`：是否命中缓存（`true` / `false` / `"hot"`）
- 体验：发送后显示“加载中”，返回后追加回答；支持回车发送；保留多轮对话记录。

请求/响应示例：
```json
POST /query
{"query": "什么是量子纠缠？"}
```
```json
{
  "answer": "量子纠缠是……",
  "intent": "通用知识",
  "strategy": "direct_llm",
  "sources": [],
  "cached": false
}
```

## 三、栏目 2：项目说明
- 通过顶部导航从「对话」切换到此栏目。
- 数据来源：进入时调用 `GET /about`，返回：
  - `title`：项目标题
  - `intro`：项目介绍（Markdown 文本，需渲染）
  - `flow_mermaid`：流程图（Mermaid 语法字符串，需用 Mermaid 渲染成图）
  - `api_endpoints`：接口列表（可选展示）
- 布局：上方展示「项目介绍」，下方展示「流程图」。

流程图渲染示例：
```html
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<pre class="mermaid" id="flow"></pre>
<script>
  fetch('http://127.0.0.1:8000/about').then(r => r.json()).then(d => {
    document.getElementById('intro').innerHTML = renderMarkdown(d.intro);
    document.getElementById('flow').textContent = d.flow_mermaid;
    mermaid.initialize({ startOnLoad: true });
  });
</script>
```
> 也可直接内联本仓库 README 中的 Mermaid 流程图，不依赖 `/about`。项目说明内容须与 README 的「项目介绍 / 流程图」保持一致。

## 四、其它接口（可选）
- `GET /health`：健康检查 → `{"status":"ok"}`
- `GET /stats`：数据库统计（高频问答库/缓存条数、热门缓存）
- `GET /about`：项目介绍 + 流程图
- `POST /seed`、`POST /build`：运维用，前端可不接

## 五、注意事项
- 后端地址固定 `http://127.0.0.1:8000`，CORS 已放开，跨域可直接调。
- 首次提问若未命中缓存，后端会调用 DashScope，可能需数秒，注意 loading 状态与超时（建议 ≥60s）。
- 「项目说明」栏目的介绍与流程图，须与本仓库 README 保持一致（同源，均来自 `src/about.py`）。
