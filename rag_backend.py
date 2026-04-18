# =========================================
# 200 道章节面试题 · RAG
# （语义切分 + BM25/向量混合检索 + 意图对齐 + 多轮记忆）
# =========================================

import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
import pickle
import hashlib
import shutil
import fitz
import jieba

# ============ 强制离线 ============
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============ 项目基目录 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "章节面试题.pdf")
MODEL_PATH = os.path.join(BASE_DIR, "bge-small-zh-v1.5")
PERSIST_DIR = os.path.join(BASE_DIR, "rag_chroma_200")

# ============ 分块配置指纹（参数变化时自动触发重建） ============
_CHUNK_CONFIG = "chunk_size=500,overlap=80,min_len=50"
_CONFIG_HASH = hashlib.md5(_CHUNK_CONFIG.encode()).hexdigest()[:8]
_CONFIG_FILE = os.path.join(PERSIST_DIR, ".config_hash")
BM25_CACHE_PATH = os.path.join(BASE_DIR, f"bm25_cache_{_CONFIG_HASH}.pkl")

# ============ LangChain ============
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.memory import ConversationBufferMemory

# ===============================
# 1️⃣ PDF 解析 + 清洗
# ===============================

def load_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    text = re.sub(r"\s+", " ", text)
    text = text.replace("（", "(").replace("）", ")")

    print(f"✅ PDF解析完成，字符数：{len(text)}")
    return text

raw_text = load_pdf_text(PDF_PATH)

# ===============================
# 2️⃣ 按章节切分
# ===============================

CHAPTERS = ["软件工程", "数据库", "机器学习", "大数据"]

def split_by_chapter(text: str):
    blocks = {}
    for i, chapter in enumerate(CHAPTERS):
        start = text.find(chapter)
        if start == -1:
            continue
        end = text.find(CHAPTERS[i + 1], start) if i + 1 < len(CHAPTERS) else len(text)
        blocks[chapter] = text[start:end]
    return blocks

chapter_texts = split_by_chapter(raw_text)

# ===============================
# 3️⃣ 语义切分
# ===============================

def infer_chunk_type(text: str):
    if any(k in text for k in ["范式", "是什么", "定义"]):
        return "definition"
    if any(k in text for k in ["方法", "如何", "步骤"]):
        return "method"
    if any(k in text for k in ["区别", "对比", "优缺点"]):
        return "comparison"
    return "general"

def infer_question_type(question: str):
    """扩展关键词表，覆盖更多问法"""
    if any(k in question for k in ["什么是", "定义", "是什么", "讲的是", "指的是", "含义", "概念"]):
        return "definition"
    if any(k in question for k in ["如何", "怎么", "怎样", "方法", "步骤", "流程", "有哪些", "解决"]):
        return "method"
    if any(k in question for k in ["区别", "对比", "不同", "优缺点", "比较", "异同"]):
        return "comparison"
    return "general"

# chunk_size 从 300 提升到 500，避免把完整题目切成两半
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["。", "；", "\n"]
)

texts, metadatas = [], []

for chapter, text in chapter_texts.items():
    for chunk in splitter.split_text(text):
        if len(chunk) > 50:
            texts.append(chunk)
            metadatas.append({
                "chapter": chapter,
                "chunk_type": infer_chunk_type(chunk)
            })

# ===============================
# 4️⃣ 向量库（带配置版本检查）
# ===============================

def _config_matches():
    """检查持久化向量库的配置是否与当前一致"""
    if not os.path.exists(_CONFIG_FILE):
        return False
    with open(_CONFIG_FILE, "r") as f:
        return f.read().strip() == _CONFIG_HASH

def _save_config():
    with open(_CONFIG_FILE, "w") as f:
        f.write(_CONFIG_HASH)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True}
)

def _build_new_db():
    print(f"🔨 构建新向量库：{PERSIST_DIR}")
    new_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=PERSIST_DIR,
    )
    _save_config()
    return new_db

def _try_load_db():
    """尝试加载已有向量库，并做一次 smoke test"""
    print(f"♻️  尝试复用已有向量库：{PERSIST_DIR}")
    loaded = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    _ = loaded.similarity_search("测试", k=1)
    print("✅ 已有向量库可用，跳过重建")
    return loaded

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR) and _config_matches():
    try:
        db = _try_load_db()
    except Exception as e:
        print(f"⚠️  已有向量库损坏（{type(e).__name__}: {e}），删除后重建")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        db = _build_new_db()
else:
    if os.path.exists(PERSIST_DIR):
        print("⚠️  检测到分块配置变更，重建向量库")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    db = _build_new_db()

# ===============================
# 5️⃣ 检索器
# ===============================

_STOPWORDS = {
    "什么", "是", "的", "了", "和", "与", "及", "中", "有", "哪些",
    "如何", "怎么", "一个", "一种", "常见", "核心", "进行", "实现",
    "可以", "需要", "之间", "以及", "或者", "但是", "因为", "所以",
}

def chinese_tokenize(text: str):
    """jieba 分词 + 停用词过滤 + 去单字"""
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok or tok in _STOPWORDS or len(tok) < 2:
            continue
        tokens.append(tok)
    return tokens

def _build_bm25():
    """构建 BM25 检索器，支持 pickle 缓存加速启动"""
    if os.path.exists(BM25_CACHE_PATH):
        try:
            with open(BM25_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            cached.k = 3
            print("♻️  BM25 索引从缓存加载")
            return cached
        except Exception:
            pass

    print("🔨 构建 BM25 索引...")
    retriever = BM25Retriever.from_texts(
        texts,
        preprocess_func=chinese_tokenize,
    )
    retriever.k = 3
    try:
        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump(retriever, f)
        print("💾 BM25 索引已缓存")
    except Exception as e:
        print(f"⚠️  BM25 缓存失败（{e}），下次启动将重建")
    return retriever

bm25 = _build_bm25()

vector = db.as_retriever(search_kwargs={"k": 3})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.4, 0.6]
)

# ===============================
# 6️⃣ LLM + Prompt
# ===============================

llm = Ollama(
    model="qwen3:4b",
    temperature=0.0,
    num_ctx=2048
)

INTENT_INSTRUCTIONS = {
    "definition": "请先给出清晰的定义，再说明关键特征。",
    "method":     "请按步骤清晰说明具体方法或流程。",
    "comparison": "请分点对比不同选项的异同、优缺点。",
    "general":    "请基于参考内容准确、简洁地回答。",
}

def build_prompt(q_type: str) -> PromptTemplate:
    """根据问题意图动态构造 prompt。"""
    intent_hint = INTENT_INSTRUCTIONS.get(q_type, INTENT_INSTRUCTIONS["general"])
    return PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "你是一个面试知识问答助手。\n"
            "请严格基于参考内容回答问题，不得编造。\n"
            f"{intent_hint}\n\n"
            "参考内容：\n{context}\n\n"
            "问题：{question}\n回答："
        )
    )

# ===============================
# 7️⃣ 会话管理（TTL 过期 + 容量上限）
# ===============================

_MAX_SESSIONS = 50
_SESSION_TTL = 3600  # 1 小时

_sessions = {}      # session_id -> {"memory": ..., "last_access": float}
_chain_cache = {}   # session_id -> {q_type -> chain}

def _cleanup_sessions():
    """清理过期和超量会话"""
    now = time.time()
    expired = [sid for sid, info in _sessions.items()
               if now - info["last_access"] > _SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)
        _chain_cache.pop(sid, None)

    if len(_sessions) > _MAX_SESSIONS:
        by_time = sorted(_sessions.items(), key=lambda x: x[1]["last_access"])
        for sid, _ in by_time[:len(_sessions) - _MAX_SESSIONS]:
            _sessions.pop(sid, None)
            _chain_cache.pop(sid, None)

def get_memory(session_id: str) -> ConversationBufferMemory:
    _cleanup_sessions()
    if session_id not in _sessions:
        _sessions[session_id] = {
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            ),
            "last_access": time.time(),
        }
    else:
        _sessions[session_id]["last_access"] = time.time()
    return _sessions[session_id]["memory"]

def clear_session(session_id: str):
    """清除指定会话的记忆和缓存"""
    _sessions.pop(session_id, None)
    _chain_cache.pop(session_id, None)

# ===============================
# 8️⃣ Chain 缓存（避免每次调用都重建）
# ===============================

def _get_chain(session_id: str, q_type: str):
    """按 session_id + q_type 缓存 chain 实例，同一会话共享 memory"""
    if session_id not in _chain_cache:
        _chain_cache[session_id] = {}
    if q_type not in _chain_cache[session_id]:
        memory = get_memory(session_id)
        _chain_cache[session_id][q_type] = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": build_prompt(q_type)},
        )
    return _chain_cache[session_id][q_type]

# ===============================
# 9️⃣ 统一问答入口
# ===============================

def ask(question: str, session_id="default"):
    """混合检索 + 多轮记忆问答，通过 session_id 实现多用户会话隔离"""

    q_type = infer_question_type(question)
    chain = _get_chain(session_id, q_type)

    result = chain.invoke({"question": question})
    answer = result["answer"]
    docs = result.get("source_documents", [])

    sources = []
    for d in docs[:3]:
        sources.append({
            "chapter": d.metadata.get("chapter"),
            "chunk_type": d.metadata.get("chunk_type"),
            "content": d.page_content[:200]
        })

    return {
        "question_type": q_type,
        "answer": answer.strip(),
        "sources": sources
    }

# ===============================
# 🔟 本地测试
# ===============================

if __name__ == "__main__":
    print("\n====== 混合检索 + 多轮记忆测试 ======\n")

    # 第一轮
    r1 = ask("CAP 定理讲的是什么？", session_id="test")
    print(r1["answer"])
    print()
    # 第二轮追问（验证上下文记忆）
    r2 = ask("那在实际系统中一般如何取舍？", session_id="test")
    print(r2["answer"])
