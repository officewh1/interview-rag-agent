# 📘 计算机面试题 RAG 智能辅导助手

基于 LangChain + Ollama 本地部署的中文 RAG + Agent 系统，面向计算机专业面试备考。

---

## ✨ 功能特性

### 基础问答（RAG 模式）
- **混合检索**：BM25（jieba 中文分词）+ 向量检索（bge-small-zh-v1.5）融合，兼顾关键词精确匹配与语义理解
- **意图识别**：自动识别 definition / method / comparison / general 四种问题类型，动态调整 Prompt
- **多轮对话**：基于 ConversationBufferMemory 实现上下文记忆，支持代词追问（"它是什么意思？"）
- **多用户隔离**：session_id 隔离，TTL 自动回收，最多 50 路并发会话
- **可解释检索**：每条回答附带检索来源片段与章节信息

### Agent 模式（5 条路由，自动路由）

| 触发条件 | 路由 | 执行内容 |
|---------|------|---------|
| "帮我复习…" / "总结…" | 章节复习工作流 | 知识点总结 → 高频考点检索 → 3 道模拟题（三步串联） |
| 含 SQL 语句 / "这条查询对吗" | SQL 沙箱 | SQLite 真实执行 + 结果表格 + LLM 语法分析 |
| "我的答案是…" / "我认为…" | 答案评估 | RAG 检索参考答案，LLM 输出评分 + 优点 + 遗漏 + 建议 |
| "最新…" / "趋势…" / "2025" | Web 搜索补充 | 本地知识库 + DuckDuckGo 搜索结果合并生成 |
| 其他 | 纯 RAG | 混合检索 + LLM 生成 |

### 其他功能
- **📒 笔记系统**：保存任意 AI 回复为笔记，SQLite 持久化，支持查看 / 删除
- **章节快捷按钮**：一键触发软件工程 / 数据库 / 机器学习 / 大数据复习工作流
- **一键启动**：`python start.py` 自动拉起后端 + 前端 + 打开浏览器

---

## 🏗️ 架构

```
用户（浏览器）
     │
     ▼
Streamlit UI (app.py)
     │  HTTP
     ▼
FastAPI (api.py)
     │
     ├── /ask        → rag_backend.py（多轮 RAG 问答）
     ├── /agent_ask  → agent.py（路由 → 工具调用 → LLM）
     ├── /notes/*    → notes_db.py（SQLite 笔记）
     └── /clear      → 会话清除
          │
          ├── tools.py        ← Agent 工具集
          │    ├── rag_search
          │    ├── web_search
          │    ├── sql_executor
          │    ├── chapter_summary
          │    └── generate_quiz
          │
          ├── sql_executor.py ← SQLite 沙箱（6 张测试表）
          │
          └── rag_backend.py  ← RAG 核心
               ├── PDF 解析（PyMuPDF）
               ├── 章节切分 + 语义分块
               ├── 向量库（Chroma + bge-small-zh-v1.5）
               ├── BM25（jieba 分词 + pickle 缓存）
               └── LLM（Ollama qwen3:4b）
```

---

## 📦 前置依赖

- **Python** 3.9+
- **Ollama**（本地 LLM 服务）— [安装地址](https://ollama.com/)
- **嵌入模型**：`bge-small-zh-v1.5`（约 184MB，不含在仓库内，见下方安装步骤）
- **面试题 PDF**：放入 `data/` 目录，PDF 内需包含四个章节标题：`软件工程` / `数据库` / `机器学习` / `大数据`

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 下载嵌入模型

```bash
# 方式一：huggingface-cli
pip install huggingface_hub
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./bge-small-zh-v1.5

# 方式二：手动下载
# 访问 https://huggingface.co/BAAI/bge-small-zh-v1.5
# 下载所有文件，放到 ./bge-small-zh-v1.5/ 目录下
```

### 4. 启动 Ollama 并拉取模型

```bash
ollama serve                  # 另开一个终端，保持运行
ollama pull qwen3:4b          # 首次下载约 2.5GB
```

### 5. 放入 PDF

将面试题 PDF 命名为 `章节面试题.pdf`，放入 `data/` 目录。

> ⚠️ 本仓库不提供 PDF 原文件，请使用自己的资料。

### 6. 一键启动

```bash
python start.py
```

浏览器会自动打开 `http://localhost:8501`，按 `Ctrl+C` 停止所有服务。

---

## 📂 目录结构

```
.
├── start.py            # 一键启动入口
├── app.py              # Streamlit 前端
├── api.py              # FastAPI 后端接口
├── agent.py            # Agent 路由与多步骤工作流
├── rag_backend.py      # RAG 核心引擎
├── tools.py            # Agent 工具集（5 个工具）
├── sql_executor.py     # SQLite 内存沙箱
├── notes_db.py         # 笔记持久化层
├── requirements.txt
├── LICENSE
├── README.md
│
├── data/               # 放你的 PDF（不含在仓库内）
├── bge-small-zh-v1.5/  # 嵌入模型（不含在仓库内，见安装步骤）
├── rag_chroma_200/     # 向量库缓存（运行时自动生成）
└── bm25_cache_*.pkl    # BM25 缓存（运行时自动生成）
```

---

## 🔬 设计说明

**为什么用规则路由而不是 LLM Function Calling？**

本项目使用 Qwen3-4B 本地模型，参数量较小，函数调用稳定性不足（容易调错工具或格式出错）。改用关键词规则路由后，每条路由执行路径固定，稳定性高，且每步都有 trace 记录，前端可展开查看完整执行过程。

**BM25 为什么要用 jieba 分词？**

LangChain 默认 BM25 按空格切词。中文无空格，整句会被视为一个 token，BM25 实际上退化为字面完全匹配。接入 jieba 后切出关键词，BM25 才能真正发挥召回作用。

**向量库版本管理**

分块参数（chunk_size / overlap）编码为 MD5 哈希存入 `.config_hash` 文件。每次启动时对比哈希，参数变更自动触发向量库重建，无需手动删缓存。

---

## 📝 License

[MIT](./LICENSE)
