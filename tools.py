# =========================================
# 工具定义（供 Agent 调用）
# rag_search / web_search / sql_executor / chapter_summary / generate_quiz
# =========================================

from langchain_core.tools import tool
from rag_backend import hybrid_retriever, llm
from sql_executor import run_sql, format_result, SCHEMA_HINT


# ===============================
# 工具 1：本地 RAG 检索
# ===============================

@tool
def rag_search(query: str, chapter: str = "") -> str:
    """
    在本地面试题 PDF 知识库中检索相关内容。
    chapter 可选值：软件工程 / 数据库 / 机器学习 / 大数据。
    不指定时全库检索。
    示例：rag_search("CAP 定理", chapter="数据库")
    """
    docs = hybrid_retriever.invoke(query)
    if chapter:
        docs = [d for d in docs if d.metadata.get("chapter") == chapter]
    if not docs:
        return "【本地知识库】未找到相关内容。"
    results = []
    for d in docs[:3]:
        ch = d.metadata.get("chapter", "未知")
        ct = d.metadata.get("chunk_type", "general")
        results.append(f"[章节:{ch} | 类型:{ct}]\n{d.page_content[:400]}")
    return "\n---\n".join(results)


# ===============================
# 工具 2：网络搜索
# ===============================

@tool
def web_search(query: str) -> str:
    """
    搜索互联网获取最新技术资料，适用于 PDF 里没有的新知识、时效性问题。
    示例：web_search("数据库最新趋势 2025")
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3, region="cn-zh"):
                title = r.get("title", "")
                body = r.get("body", "")
                results.append(f"【{title}】\n{body[:400]}")
        if not results:
            return "【网络搜索】无结果，请检查网络连接。"
        return "\n---\n".join(results)
    except ImportError:
        return "【网络搜索】未安装 duckduckgo-search，请运行：pip install duckduckgo-search"
    except Exception as e:
        return f"【网络搜索】失败：{e}"


# ===============================
# 工具 3：SQL 沙箱执行
# ===============================

@tool
def sql_executor(sql: str) -> str:
    """
    在本地 SQLite 内存沙箱中执行 SQL 语句，返回执行结果或错误信息。
    沙箱内预置了 employees / departments / students / courses / enrollments / orders 六张测试表。
    示例：sql_executor("SELECT name, salary FROM employees WHERE salary > 9000")
    """
    result = run_sql(sql)
    formatted = format_result(result)
    if not result["success"]:
        return f"执行失败：{result['error']}\n\n{SCHEMA_HINT}"
    return f"执行成功：\n{formatted}\n\n{SCHEMA_HINT}"


# ===============================
# 工具 4：章节知识点总结
# ===============================

@tool
def chapter_summary(chapter: str) -> str:
    """
    对指定章节进行知识点梳理，返回结构化总览。
    chapter 可选值：软件工程 / 数据库 / 机器学习 / 大数据。
    示例：chapter_summary("机器学习")
    """
    # 多关键词检索，尽量覆盖章节全貌
    queries = [f"{chapter}核心概念", f"{chapter}常见考点", f"{chapter}重要知识点"]
    seen, docs = set(), []
    for q in queries:
        for d in hybrid_retriever.invoke(q):
            if d.metadata.get("chapter") == chapter and d.page_content not in seen:
                seen.add(d.page_content)
                docs.append(d)
        if len(docs) >= 9:
            break

    if not docs:
        return f"未找到【{chapter}】相关内容。"

    context = "\n---\n".join(d.page_content[:300] for d in docs[:9])
    prompt = (
        f"你是一个面试辅导助手。请基于以下【{chapter}】章节内容，"
        "整理出结构化的知识点总览。\n\n"
        "要求：\n"
        "1. 列出 5~8 个核心知识点，每点一行，加序号\n"
        "2. 每个知识点用一句话说明核心内容\n"
        "3. 最后用一句话概括本章重点\n\n"
        f"参考内容：\n{context}\n\n"
        f"【{chapter}】知识点总览："
    )
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


# ===============================
# 工具 5：生成模拟面试题
# ===============================

@tool
def generate_quiz(chapter: str, n: int = 3) -> str:
    """
    根据章节内容生成 n 道模拟面试题（附参考答案）。
    chapter 可选值：软件工程 / 数据库 / 机器学习 / 大数据。
    示例：generate_quiz("数据库", n=3)
    """
    docs = []
    seen = set()
    for d in hybrid_retriever.invoke(f"{chapter} 面试题 考点"):
        if d.metadata.get("chapter") == chapter and d.page_content not in seen:
            seen.add(d.page_content)
            docs.append(d)
    if not docs:
        return f"未找到【{chapter}】相关内容，无法生成题目。"

    context = "\n---\n".join(d.page_content[:300] for d in docs[:6])
    prompt = (
        f"你是一个面试辅导助手。请基于以下【{chapter}】内容，"
        f"生成 {n} 道典型面试题并给出参考答案。\n\n"
        "格式要求（严格遵守）：\n"
        "Q1: [面试问题]\n"
        "A1: [参考答案，2~4句话]\n\n"
        "Q2: [面试问题]\n"
        "A2: [参考答案，2~4句话]\n\n"
        "题目类型要多样：包含定义类、方法类、对比类。\n\n"
        f"参考内容：\n{context}\n\n"
        "生成题目："
    )
    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


# ===============================
# 所有工具列表（统一出口）
# ===============================

ALL_TOOLS = [rag_search, web_search, sql_executor, chapter_summary, generate_quiz]
