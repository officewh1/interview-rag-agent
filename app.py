import streamlit as st
import requests
import uuid

BASE_URL = "http://127.0.0.1:8000"

# ===============================
# 初始化
# ===============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_indices" not in st.session_state:
    st.session_state.saved_indices = set()   # 已保存的消息索引

st.set_page_config(
    page_title="面试题 RAG 问答助手",
    page_icon="📘",
    layout="wide",
)

st.title("📘 面试题 RAG 问答助手")
st.caption("基于 PDF 的 RAG · BM25/向量混合检索 · 多轮追问 · 笔记系统")

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
        if st.button(label, use_container_width=True):
            st.session_state.pending_message = prompt
            st.rerun()

    st.divider()

    # ── 答案评估说明 ──────────────────────────
    st.markdown("### 答案评估")
    st.info(
        "输入你的答案，Agent 会自动打分并指出不足。\n\n"
        "**示例：**\n"
        "我的答案是：CAP 定理指...\n\n"
        "我认为第三范式的定义是...",
        icon="📝",
    )

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
    if st.button("🧹 清除对话", use_container_width=True):
        try:
            requests.post(f"{BASE_URL}/clear", json={"session_id": st.session_state.session_id}, timeout=10)
        except Exception as e:
            st.warning(f"后端清理失败：{e}")
        st.session_state.chat_history = []
        st.session_state.saved_indices = set()
        st.rerun()

# ===============================
# 渲染历史对话
# ===============================

def _render_assistant_extras(item: dict, msg_idx: int):
    """渲染助手消息下方的扩展面板 + 保存按钮"""
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

    # 保存为笔记按钮
    already_saved = msg_idx in st.session_state.saved_indices
    if already_saved:
        st.caption("✅ 已保存为笔记")
    else:
        if st.button("📌 保存为笔记", key=f"save_{msg_idx}"):
            content = item["content"]
            # 用内容前 30 字作标题
            title = content.strip().replace("#", "").strip()[:30]
            chapter = item.get("chapter", "")
            try:
                requests.post(
                    f"{BASE_URL}/notes/save",
                    json={
                        "session_id": st.session_state.session_id,
                        "title": title,
                        "content": content,
                        "chapter": chapter,
                    },
                    timeout=10,
                )
                st.session_state.saved_indices.add(msg_idx)
                st.rerun()
            except Exception as e:
                st.error(f"保存失败：{e}")


for idx, item in enumerate(st.session_state.chat_history):
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant":
            _render_assistant_extras(item, idx)

# ===============================
# 输入框
# ===============================
if agent_mode:
    placeholder = "时效性问题示例：2025 年机器学习有哪些最新趋势？"
else:
    placeholder = "请输入问题，例如：CAP 定理讲的是什么？"

user_input = st.chat_input(placeholder)

# 快捷按钮注入的消息优先
if "pending_message" in st.session_state:
    user_input = st.session_state.pop("pending_message")
    agent_mode = True   # 章节复习走 Agent 路由

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    endpoint = "/agent_ask" if agent_mode else "/ask"
    spinner_text = "🤖 Agent 正在规划并检索..." if agent_mode else "🔍 正在检索并生成回答..."

    with st.spinner(spinner_text):
        try:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json={"query": user_input, "session_id": st.session_state.session_id},
                timeout=180,
            )
            result = response.json()
        except Exception as e:
            st.error(f"请求失败：{e}")
            st.stop()

    used_web = result.get("used_web", False)
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
    }
    st.session_state.chat_history.append(assistant_entry)

    new_idx = len(st.session_state.chat_history) - 1
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        _render_assistant_extras(assistant_entry, new_idx)
