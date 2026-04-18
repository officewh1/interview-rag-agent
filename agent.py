# =========================================
# Agent 编排
#
# 支持四条路由（按优先级）：
#   A. 章节复习请求  → 总结 + 考点 + 模拟题（工作流）
#   B. 含 SQL 代码   → SQL沙箱执行 + RAG解释
#   C. 时效性问题    → RAG + Web搜索合成
#   D. 普通问题      → 纯 RAG
# =========================================

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain.prompts import PromptTemplate
from rag_backend import llm, infer_question_type
from tools import rag_search, web_search, sql_executor, chapter_summary, generate_quiz
from sql_executor import run_sql, format_result, SCHEMA_HINT

# ===============================
# 路由关键词
# ===============================

# 章节名映射（支持多种说法）
_CHAPTER_ALIASES = {
    "软件工程": ["软件工程", "软工", "工程"],
    "数据库":   ["数据库", "database", "sql", "SQL", "mysql", "MySQL"],
    "机器学习": ["机器学习", "machine learning", "ml", "ML", "深度学习", "神经网络"],
    "大数据":   ["大数据", "big data", "hadoop", "Hadoop", "spark", "Spark", "mapreduce"],
}

# 复习意图关键词
_REVIEW_KEYWORDS = [
    "复习", "总结", "梳理", "整理", "帮我学", "讲一讲",
    "系统学", "知识点", "考点", "出几道题", "出题", "模拟题",
]

# SQL 语句检测
_SQL_STMT_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\b",
    re.IGNORECASE,
)
_SQL_QUESTION_KEYWORDS = [
    "sql", "SQL", "这条", "这个查询", "写法", "语句对",
    "查询对", "执行结果", "sql对", "SQL对",
]

# 时效性关键词
_TEMPORAL_KEYWORDS = [
    "最新", "趋势", "现在", "目前", "2024", "2025",
    "新特性", "新功能", "新版本", "进展", "动态", "未来", "前景", "现状",
]

# 答案评估关键词
_EVAL_KEYWORDS = [
    "我的答案", "我的理解", "我认为", "我觉得", "我回答",
    "帮我评估", "帮我批改", "这样对吗", "这样回答对吗",
    "我的回答是", "评分", "打分", "对不对", "答案对吗",
]

# ===============================
# Agent 执行结果
# ===============================

@dataclass
class AgentResult:
    answer: str
    question_type: str
    route: str                            # review / sql / eval / web / rag
    used_web: bool
    chapter: str = ""                     # 复习路由时的目标章节
    quiz: str = ""                        # 生成的模拟题
    sql_result: dict = field(default_factory=dict)
    trace: List[dict] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)

# ===============================
# 路由判断工具函数
# ===============================

def _detect_chapter(text: str) -> Optional[str]:
    """返回文本中提到的章节名，未检测到返回 None"""
    text_lower = text.lower()
    for chapter, aliases in _CHAPTER_ALIASES.items():
        if any(a.lower() in text_lower for a in aliases):
            return chapter
    return None

def _is_review_request(text: str) -> bool:
    return any(k in text for k in _REVIEW_KEYWORDS)

def _is_sql_question(text: str) -> bool:
    has_kw = any(k in text for k in _SQL_QUESTION_KEYWORDS)
    has_stmt = bool(_SQL_STMT_RE.search(text))
    return has_kw or has_stmt

def _needs_web(text: str) -> bool:
    return any(k in text for k in _TEMPORAL_KEYWORDS)

def _is_eval_request(text: str) -> bool:
    return any(k in text for k in _EVAL_KEYWORDS)

def _extract_sql(text: str) -> str:
    code_block = re.search(r"```(?:sql)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    match = _SQL_STMT_RE.search(text)
    if match:
        return text[match.start():].split("\n")[0].strip()
    return ""

# ===============================
# Prompt 模板
# ===============================

_PROMPT_SQL = PromptTemplate(
    input_variables=["question", "sql", "sql_result", "rag_context"],
    template=(
        "你是一个数据库面试辅导助手。\n"
        "用户提交了一条 SQL 语句，请结合执行结果和知识库内容进行分析。\n\n"
        "【用户 SQL】\n{sql}\n\n"
        "【沙箱执行结果】\n{sql_result}\n\n"
        "【知识库参考】\n{rag_context}\n\n"
        "请完成以下分析：\n"
        "1. SQL 语法是否正确？\n"
        "2. 执行结果是否符合预期？\n"
        "3. 有无可以优化的地方？\n\n"
        "用户问题：{question}\n回答："
    ),
)

_PROMPT_WEB = PromptTemplate(
    input_variables=["question", "rag_context", "web_context"],
    template=(
        "你是一个面试知识问答助手。\n"
        "以下提供了两个来源的信息，请综合回答问题。\n\n"
        "【本地知识库（基础知识）】\n{rag_context}\n\n"
        "【网络搜索（最新资料）】\n{web_context}\n\n"
        "要求：先用本地知识库回答基础概念，再结合网络资料补充最新动态。\n\n"
        "问题：{question}\n回答："
    ),
)

_PROMPT_RAG = PromptTemplate(
    input_variables=["question", "rag_context"],
    template=(
        "你是一个面试知识问答助手。\n"
        "请严格基于参考内容回答问题，不得编造。\n\n"
        "【参考内容】\n{rag_context}\n\n"
        "问题：{question}\n回答："
    ),
)

_PROMPT_EVAL = PromptTemplate(
    input_variables=["user_answer", "rag_context"],
    template=(
        "你是一位严格但友善的面试考官。\n"
        "请对面试者的回答进行评估，参考标准答案来自知识库。\n\n"
        "【知识库参考答案】\n{rag_context}\n\n"
        "【面试者的回答】\n{user_answer}\n\n"
        "请按以下格式输出评估报告：\n\n"
        "**综合评分：X / 10**\n\n"
        "**优点**\n- （列出回答中正确、完整的部分）\n\n"
        "**不足 / 遗漏**\n- （列出缺少的关键知识点）\n\n"
        "**改进建议**\n（给出一段完整的参考答案或补充说明）\n"
    ),
)

# ===============================
# 核心执行函数
# ===============================

def agent_ask(question: str, session_id: str = "default") -> AgentResult:
    trace = []
    q_type = infer_question_type(question)
    sql_res = {}
    rag_result = ""
    target_chapter = ""
    quiz_text = ""

    # ── 路由决策 ──────────────────────────────
    if _is_review_request(question):
        route = "review"
        target_chapter = _detect_chapter(question) or "机器学习"
    elif _is_sql_question(question):
        route = "sql"
    elif _is_eval_request(question):
        route = "eval"
    elif _needs_web(question):
        route = "web"
    else:
        route = "rag"

    route_desc = {
        "review": f"章节复习请求 → 三步工作流（总结 → 考点 → 模拟题），章节：{target_chapter}",
        "sql":    "检测到 SQL → 沙箱执行 + RAG 解释",
        "eval":   "答案评估 → RAG 检索参考答案 + LLM 打分",
        "web":    "时效性问题 → RAG + Web 搜索合成",
        "rag":    "普通问题 → 本地 RAG 检索",
    }
    trace.append({"step": "🔀 路由决策", "detail": route_desc[route]})

    # ══════════════════════════════════════════
    # 路由 A：章节复习工作流
    # 步骤：① 知识点总结 → ② 检索高频考点 → ③ 生成模拟题
    # ══════════════════════════════════════════
    if route == "review":

        # 步骤 1：章节知识点总结
        trace.append({"step": "📋 步骤1：章节知识点总结", "detail": f"正在梳理【{target_chapter}】核心知识点..."})
        summary = chapter_summary.invoke({"chapter": target_chapter})
        trace.append({"step": "✅ 总结完成", "detail": summary[:150] + "..."})

        # 步骤 2：检索高频考点
        trace.append({"step": "🔍 步骤2：检索高频考点", "detail": f"查询【{target_chapter}】高频面试考点..."})
        rag_result = rag_search.invoke({"query": f"{target_chapter} 常见面试题 重点", "chapter": target_chapter})
        trace.append({"step": "✅ 考点检索完成", "detail": rag_result[:150] + "..."})

        # 步骤 3：生成模拟面试题
        trace.append({"step": "🎯 步骤3：生成模拟面试题", "detail": f"为【{target_chapter}】生成 3 道模拟题..."})
        quiz_text = generate_quiz.invoke({"chapter": target_chapter, "n": 3})
        trace.append({"step": "✅ 模拟题生成完成", "detail": quiz_text[:150] + "..."})

        # 组合最终答案
        answer = (
            f"## 📚 {target_chapter} 章节复习\n\n"
            f"### 核心知识点\n{summary}\n\n"
            f"---\n\n"
            f"### 🎯 模拟面试题\n{quiz_text}"
        )
        trace.append({"step": "✅ 工作流完成", "detail": "已生成知识总结 + 模拟题"})

    # ══════════════════════════════════════════
    # 路由 B：SQL 沙箱
    # ══════════════════════════════════════════
    elif route == "sql":
        sql_code = _extract_sql(question)
        trace.append({"step": "🗄️ 提取 SQL", "detail": sql_code or "未提取到独立SQL"})

        if sql_code:
            trace.append({"step": "⚡ 沙箱执行", "detail": f"执行：{sql_code}"})
            sql_res = run_sql(sql_code)
            sql_result_str = format_result(sql_res)
            trace.append({
                "step": "✅ 执行完成" if sql_res["success"] else "❌ 执行失败",
                "detail": sql_result_str[:300],
            })
        else:
            sql_result_str = "未提取到 SQL 代码，跳过沙箱执行。"

        trace.append({"step": "🔍 RAG 检索知识背景", "detail": question})
        rag_result = rag_search.invoke({"query": question, "chapter": "数据库"})
        trace.append({"step": "✅ RAG 完成", "detail": rag_result[:150] + "..."})

        trace.append({"step": "🤖 LLM 分析", "detail": "正在生成..."})
        prompt = _PROMPT_SQL.format(
            question=question,
            sql=sql_code or "（未提取到独立SQL）",
            sql_result=sql_result_str,
            rag_context=rag_result,
        )
        answer = llm.invoke(prompt)

    # ══════════════════════════════════════════
    # 路由 C：答案评估
    # ══════════════════════════════════════════
    elif route == "eval":
        # 从用户输入中提取"用户的答案"部分
        # 去掉触发词，剩余内容视为答案主体
        user_answer = question
        for kw in _EVAL_KEYWORDS:
            user_answer = user_answer.replace(kw, "").strip()
        if user_answer.startswith("：") or user_answer.startswith(":"):
            user_answer = user_answer[1:].strip()
        if not user_answer:
            user_answer = question  # 兜底：用整句话检索

        trace.append({"step": "📝 提取用户答案", "detail": user_answer[:200]})

        trace.append({"step": "🔍 RAG 检索参考答案", "detail": question})
        rag_result = rag_search.invoke({"query": question})
        trace.append({"step": "✅ RAG 完成", "detail": rag_result[:150] + "..."})

        trace.append({"step": "🤖 LLM 评估打分", "detail": "正在生成..."})
        prompt = _PROMPT_EVAL.format(
            user_answer=user_answer,
            rag_context=rag_result,
        )
        answer = llm.invoke(prompt)

    # ══════════════════════════════════════════
    # 路由 D：RAG + Web
    # ══════════════════════════════════════════
    elif route == "web":
        trace.append({"step": "🔍 本地 RAG 检索", "detail": question})
        rag_result = rag_search.invoke({"query": question})
        trace.append({"step": "✅ RAG 完成", "detail": rag_result[:150] + "..."})

        trace.append({"step": "🌐 网络搜索", "detail": question})
        web_result = web_search.invoke({"query": question})
        trace.append({"step": "✅ Web 完成", "detail": web_result[:150] + "..."})

        trace.append({"step": "🤖 LLM 合成", "detail": "正在生成..."})
        prompt = _PROMPT_WEB.format(
            question=question,
            rag_context=rag_result,
            web_context=web_result,
        )
        answer = llm.invoke(prompt)

    # ══════════════════════════════════════════
    # 路由 D：纯 RAG
    # ══════════════════════════════════════════
    else:
        trace.append({"step": "🔍 本地 RAG 检索", "detail": question})
        rag_result = rag_search.invoke({"query": question})
        trace.append({"step": "✅ RAG 完成", "detail": rag_result[:150] + "..."})

        trace.append({"step": "🤖 LLM 生成", "detail": "正在生成..."})
        prompt = _PROMPT_RAG.format(question=question, rag_context=rag_result)
        answer = llm.invoke(prompt)

    # ── 统一处理 answer ───────────────────────
    if route != "review":
        if hasattr(answer, "content"):
            answer = answer.content
        answer = str(answer).strip()
        trace.append({"step": "✅ 生成完成", "detail": answer[:100] + "..."})

    # ── 整理 sources ─────────────────────────
    sources = []
    if rag_result:
        for block in rag_result.split("---"):
            block = block.strip()
            if not block:
                continue
            ch_match = re.search(r"章节:(.+?)\s*\|", block)
            ct_match = re.search(r"类型:(.+?)]", block)
            content = re.sub(r"\[章节:.+?\]\n?", "", block).strip()
            sources.append({
                "source":     "local",
                "chapter":    ch_match.group(1) if ch_match else target_chapter or "未知",
                "chunk_type": ct_match.group(1) if ct_match else "general",
                "content":    content[:200],
            })
    if route == "web" and web_result:
        for block in web_result.split("---"):
            block = block.strip()
            if block:
                sources.append({"source": "web", "chapter": "网络", "chunk_type": "realtime", "content": block[:200]})

    return AgentResult(
        answer=answer,
        question_type=q_type,
        route=route,
        used_web=(route == "web"),
        chapter=target_chapter,
        quiz=quiz_text,
        sql_result=sql_res,
        trace=trace,
        sources=sources,
    )


# ===============================
# 本地测试
# ===============================
if __name__ == "__main__":
    cases = [
        ("帮我复习一下机器学习章节",              "review"),
        ("复习数据库",                             "review"),
        ("SELECT name FROM employees 这个对吗",   "sql"),
        ("2025 年大数据有哪些最新趋势？",           "web"),
        ("什么是第三范式？",                        "rag"),
        ("我的答案是：CAP定理指一致性、可用性和分区容错性不能同时满足", "eval"),
    ]
    for q, expected_route in cases:
        print(f"\n{'='*60}\n问：{q}  (期望路由：{expected_route})")
        r = agent_ask(q)
        print(f"实际路由：{r.route}")
        for t in r.trace:
            print(f"  {t['step']}")
        print(f"\n答（前200字）：{r.answer[:200]}")
