import re
import streamlit as st
import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

# ===============================
# 工具函数
# ===============================

def _extract_quiz_questions(text: str) -> list:
    """从章节复习回答中提取 Q1/Q2/Q3 题目"""
    questions = re.findall(r'Q\d+:\s*(.+?)(?=\n|$)', text)
    return [q.strip() for q in questions if q.strip()]

def _extract_score(text: str) -> int:
    """从评估报告中提取分数"""
    match = re.search(r'综合评分[：:]\s*(\d+)\s*/\s*10', text)
    return int(match.group(1)) if match else 0

# ===============================
# 初始化
# ===============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_indices" not in st.session_state:
    st.session_state.saved_indices = set()
if "practice_question" not in st.session_state:
    st.session_state.practice_question = ""
if "practice_mode" not in st.session_state:
    st.session_state.practice_mode = False
if "practice_session" not in st.session_state:
    st.session_state.practice_session = {}

st.set_page_config(
    page_title="面试题 RAG 问答助手",
    page_icon="📘",
    layout="wide",
)

# ===============================
# 侧边栏
# ===============================
agent_mode = False

with st.sidebar:
    # ── 模式选择 ──────────────────────────────
    st.markdown("### 模式选择")
    agent_mode = st.toggle(
        "🤖 Agent 模式（知识补全）",
        value=False,
        help="含「最新/趋势/2025」等关键词时，自动触发 Web 搜索",
    )
    if agent_mode:
        st.info("**Agent 模式已开启**\n\n时效性问题将同时检索本地知识库和网络。", icon="🤖")
    else:
        st.info("**RAG 模式**\n\n仅检索本地 PDF 知识库，支持多轮追问。", icon="📚")

    st.divider()

    # ── 章节快捷复习 ──────────────────────────
    st.markdown("### 章节快捷复习")
    chapter_buttons = [
        ("📚 软件工程", "帮我复习一下软件工程章节"),
        ("🗄️ 数据库",   "帮我复习一下数据库章节"),
        ("🤖 机器学习", "帮我复习一下机器学习章节"),
        ("📊 大数据",   "帮我复习一下大数据章节"),
    ]
    for label, prompt in chapter_buttons:
        if st.button(label, use_container_width=True,
                     disabled=st.session_state.practice_mode):
            st.session_state.pending_message = prompt
            st.rerun()

    st.divider()

    # ── 练习模式入口 ──────────────────────────
    st.markdown("### 🎯 练习模式")
    if not st.session_state.practice_mode:
        prac_chapter = st.selectbox(
            "选择章节",
            ["软件工程", "数据库", "机器学习", "大数据"],
            key="prac_chapter_select",
        )
        prac_n = st.radio("题目数量", [3, 5], horizontal=True, key="prac_n_radio")
        if st.button("▶ 开始练习", use_container_width=True, type="primary"):
            with st.spinner("正在生成题目..."):
                try:
                    resp = requests.post(
                        f"{BASE_URL}/practice/questions",
                        json={"chapter": prac_chapter, "n": prac_n},
                        timeout=180,
                    )
                    data = resp.json()
                    qs = data.get("questions", [])
                    if qs:
                        st.session_state.practice_mode = True
                        st.session_state.practice_session = {
                            "chapter":     prac_chapter,
                            "questions":   qs,
                            "current_idx": 0,
                            "results":     [],
                            "phase":       "answering",
                        }
                        st.rerun()
                    else:
                        st.error("题目生成失败，请重试")
                except Exception as e:
                    st.error(f"请求失败：{e}")
    else:
        ps = st.session_state.practice_session
        total = len(ps.get("questions", []))
        current = ps.get("current_idx", 0)
        st.info(
            f"**{ps.get('chapter', '')}**\n\n"
            f"第 {min(current + 1, total)} / {total} 题",
            icon="🎯",
        )
        if st.button("⏹ 结束练习", use_container_width=True):
            st.session_state.practice_mode = False
            st.session_state.practice_session = {}
            st.rerun()

    st.divider()

    # ── 笔记本 ────────────────────────────────
    st.markdown("### 📒 我的笔记")
    try:
        resp = requests.get(
            f"{BASE_URL}/notes",
            params={"session_id": st.session_state.session_id},
            timeout=5,
        )
        notes = resp.json().get("notes", [])
    except Exception:
        notes = []

    if not notes:
        st.caption("暂无笔记，点击回复下方「保存为笔记」添加。")
    else:
        for note in notes:
            with st.expander(f"📌 {note['title'][:24]}"):
                st.caption(f"章节：{note['chapter'] or '未分类'}  |  {note['created_at'][:16]}")
                st.markdown(note["content"][:400] + ("..." if len(note["content"]) > 400 else ""))
                if st.button("🗑️ 删除", key=f"del_note_{note['id']}"):
                    try:
                        requests.delete(f"{BASE_URL}/notes/{note['id']}", timeout=5)
                    except Exception:
                        pass
                    st.rerun()

    st.divider()

    # ── 会话控制 ──────────────────────────────
    st.markdown("### 会话控制")
    st.caption(f"会话 ID：`{st.session_state.session_id[:8]}...`")
    if st.button("🧹 清除对话", use_container_width=True,
                 disabled=st.session_state.practice_mode):
        try:
            requests.post(f"{BASE_URL}/clear",
                          json={"session_id": st.session_state.session_id}, timeout=10)
        except Exception as e:
            st.warning(f"后端清理失败：{e}")
        st.session_state.chat_history = []
        st.session_state.saved_indices = set()
        st.rerun()

# ===============================
# 练习模式主界面
# ===============================

def _render_practice_mode():
    ps = st.session_state.practice_session
    questions = ps["questions"]
    total     = len(questions)
    idx       = ps["current_idx"]
    phase     = ps["phase"]

    st.title(f"🎯 练习模式 · {ps['chapter']}")

    # 进度条
    progress = idx / total if phase != "complete" else 1.0
    st.progress(progress)
    if phase != "complete":
        st.caption(f"第 {idx + 1} 题 / 共 {total} 题")

    st.divider()

    # ── 答题阶段 ──────────────────────────────
    if phase == "answering":
        q = questions[idx]
        st.markdown(f"### 📋 {q}")
        answer = st.text_area(
            "写下你的答案：",
            height=180,
            key=f"prac_answer_{idx}",
            placeholder="尽量完整地回答，系统会对照知识库逐项评分...",
        )
        if st.button("📤 提交答案", type="primary", use_container_width=False):
            if answer.strip():
                with st.spinner("正在评估..."):
                    try:
                        msg = f"针对题目：{q}\n我的答案是：{answer.strip()}"
                        resp = requests.post(
                            f"{BASE_URL}/practice/evaluate",
                            json={"query": msg,
                                  "session_id": st.session_state.session_id},
                            timeout=180,
                        )
                        evaluation = resp.json().get("evaluation", "评估失败")
                        score = _extract_score(evaluation)
                        ps["results"].append({
                            "question":   q,
                            "answer":     answer.strip(),
                            "evaluation": evaluation,
                            "score":      score,
                        })
                        ps["phase"] = "reviewing"
                        st.rerun()
                    except Exception as e:
                        st.error(f"评估失败：{e}")
            else:
                st.warning("请先输入答案")

    # ── 查看评估阶段 ──────────────────────────
    elif phase == "reviewing":
        result = ps["results"][-1]
        score  = result["score"]

        # 分数颜色
        color = "green" if score >= 8 else ("orange" if score >= 5 else "red")
        st.markdown(f"#### 你的答案")
        st.info(result["answer"])
        st.markdown("#### 评估报告")
        st.markdown(result["evaluation"])

        st.divider()
        if idx + 1 < total:
            if st.button("下一题 →", type="primary"):
                ps["current_idx"] += 1
                ps["phase"] = "answering"
                st.rerun()
        else:
            if st.button("查看总结 🏆", type="primary"):
                ps["phase"] = "complete"
                st.rerun()

    # ── 练习完成汇总 ──────────────────────────
    elif phase == "complete":
        results     = ps["results"]
        total_score = sum(r["score"] for r in results)
        max_score   = total * 10

        st.balloons()
        st.success(f"🏆 练习完成！总分 **{total_score} / {max_score}**")

        # 总分进度条
        st.progress(total_score / max_score)

        # 每题得分卡片
        st.markdown("#### 各题得分")
        cols = st.columns(total)
        for i, (col, r) in enumerate(zip(cols, results), 1):
            with col:
                score = r["score"]
                color = "🟢" if score >= 8 else ("🟡" if score >= 5 else "🔴")
                st.metric(f"Q{i}", f"{score}/10", delta=None)
                st.caption(color)

        # 详细展开
        st.markdown("#### 详细回顾")
        for i, r in enumerate(results, 1):
            with st.expander(f"Q{i}：{r['question'][:40]}..."):
                st.markdown(f"**你的答案：** {r['answer']}")
                st.divider()
                st.markdown(r["evaluation"])

        # 薄弱题提示
        weak = [i + 1 for i, r in enumerate(results) if r["score"] < 6]
        if weak:
            st.warning(f"⚠️ 建议重点复习：Q{', Q'.join(map(str, weak))}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 再练一次", use_container_width=True):
                chapter = ps["chapter"]
                n = len(questions)
                with st.spinner("重新生成题目..."):
                    try:
                        resp = requests.post(
                            f"{BASE_URL}/practice/questions",
                            json={"chapter": chapter, "n": n},
                            timeout=60,
                        )
                        qs = resp.json().get("questions", [])
                        st.session_state.practice_session = {
                            "chapter":     chapter,
                            "questions":   qs,
                            "current_idx": 0,
                            "results":     [],
                            "phase":       "answering",
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"失败：{e}")
        with col2:
            if st.button("返回聊天", use_container_width=True):
                st.session_state.practice_mode = False
                st.session_state.practice_session = {}
                st.rerun()


# ===============================
# 普通对话扩展面板
# ===============================

def _render_assistant_extras(item: dict, msg_idx: int):
    if item.get("trace"):
        with st.expander("🔀 查看 Agent 执行步骤"):
            for t in item["trace"]:
                st.markdown(f"**{t['step']}**")
                if t.get("detail"):
                    st.caption(str(t["detail"]))

    sql_res = item.get("sql_result", {})
    if sql_res.get("columns"):
        with st.expander("🗄️ SQL 执行结果"):
            import pandas as pd
            df = pd.DataFrame(sql_res["rows"], columns=sql_res["columns"])
            st.dataframe(df, use_container_width=True)
            st.caption(sql_res.get("note", ""))
    elif sql_res and not sql_res.get("success"):
        with st.expander("🗄️ SQL 执行结果"):
            st.error(f"执行失败：{sql_res.get('error', '未知错误')}")

    if item.get("sources"):
        with st.expander("📚 查看检索证据"):
            for i, s in enumerate(item["sources"], 1):
                badge = "🌐 网络" if s.get("source") == "web" else "📖 本地"
                st.markdown(
                    f"**证据 {i}** {badge}\n"
                    f"- 章节：{s['chapter']}\n"
                    f"- 类型：{s['chunk_type']}\n"
                    f"- 内容：{s['content']}"
                )

    # 练习按钮（仅章节复习回答显示）
    if item.get("route") == "review":
        questions = _extract_quiz_questions(item.get("content", ""))
        if questions:
            st.markdown("**✏️ 针对以上题目单题练习：**")
            cols = st.columns(len(questions))
            for i, (col, q) in enumerate(zip(cols, questions), 1):
                with col:
                    if st.button(f"作答 Q{i}", key=f"practice_{msg_idx}_{i}",
                                 use_container_width=True):
                        st.session_state.practice_question = q
                        st.rerun()

    # 保存为笔记
    already_saved = msg_idx in st.session_state.saved_indices
    if already_saved:
        st.caption("✅ 已保存为笔记")
    else:
        if st.button("📌 保存为笔记", key=f"save_{msg_idx}"):
            content = item["content"]
            title   = content.strip().replace("#", "").strip()[:30]
            chapter = item.get("chapter", "")
            try:
                requests.post(
                    f"{BASE_URL}/notes/save",
                    json={"session_id": st.session_state.session_id,
                          "title": title, "content": content, "chapter": chapter},
                    timeout=10,
                )
                st.session_state.saved_indices.add(msg_idx)
                st.rerun()
            except Exception as e:
                st.error(f"保存失败：{e}")


# ===============================
# 主内容区：练习模式 / 普通对话
# ===============================

if st.session_state.practice_mode:
    _render_practice_mode()

else:
    st.title("📘 面试题 RAG 问答助手")
    st.caption("基于 PDF 的 RAG · BM25/向量混合检索 · 多轮追问 · 笔记系统")

    # 渲染历史对话
    for idx, item in enumerate(st.session_state.chat_history):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant":
                _render_assistant_extras(item, idx)

    # 单题答题面板
    if st.session_state.practice_question:
        pq = st.session_state.practice_question
        st.info(f"📝 **练习题目**\n\n{pq}")
        practice_answer = st.text_area(
            "输入你的答案：", key="practice_answer_input", height=120,
            placeholder="写下你的作答，提交后 Agent 会自动打分...",
        )
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("📤 提交评估", use_container_width=True):
                if practice_answer.strip():
                    msg = f"针对题目：{pq}\n我的答案是：{practice_answer.strip()}"
                    st.session_state.pending_message = msg
                    st.session_state.practice_question = ""
                    st.rerun()
                else:
                    st.warning("请先输入答案再提交")
        with col2:
            if st.button("取消", use_container_width=True):
                st.session_state.practice_question = ""
                st.rerun()
        st.divider()

    # 输入框
    if agent_mode:
        placeholder = "时效性问题示例：2025 年机器学习有哪些最新趋势？"
    else:
        placeholder = "请输入问题，例如：CAP 定理讲的是什么？"

    user_input = st.chat_input(placeholder)

    if "pending_message" in st.session_state:
        user_input = st.session_state.pop("pending_message")
        agent_mode = True

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        endpoint     = "/agent_ask" if agent_mode else "/ask"
        spinner_text = "🤖 Agent 正在规划并检索..." if agent_mode else "🔍 正在检索并生成回答..."

        with st.spinner(spinner_text):
            try:
                response = requests.post(
                    f"{BASE_URL}{endpoint}",
                    json={"query": user_input,
                          "session_id": st.session_state.session_id},
                    timeout=180,
                )
                result = response.json()
            except Exception as e:
                st.error(f"请求失败：{e}")
                st.stop()

        used_web    = result.get("used_web", False)
        answer_text = result["answer"]
        if used_web:
            answer_text += "\n\n> 🌐 已补充网络资料"

        assistant_entry = {
            "role":       "assistant",
            "content":    answer_text,
            "sources":    result.get("sources", []),
            "trace":      result.get("trace", []),
            "sql_result": result.get("sql_result", {}),
            "chapter":    result.get("chapter", ""),
            "route":      result.get("route", ""),
        }
        st.session_state.chat_history.append(assistant_entry)

        new_idx = len(st.session_state.chat_history) - 1
        with st.chat_message("assistant"):
            st.markdown(answer_text)
            _render_assistant_extras(assistant_entry, new_idx)
