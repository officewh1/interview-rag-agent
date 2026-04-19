import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag_backend import ask, clear_session
from agent import agent_ask
from notes_db import save_note, get_notes, delete_note
from tools import generate_quiz

app = FastAPI(title="面试题 RAG 接口")


class Question(BaseModel):
    query: str
    session_id: str


class Session(BaseModel):
    session_id: str


class NoteCreate(BaseModel):
    session_id: str
    title: str
    content: str
    chapter: Optional[str] = ""


class PracticeRequest(BaseModel):
    chapter: str
    n: int = 3


# ── 普通 RAG 问答 ──────────────────────────────
@app.post("/ask")
def ask_question(q: Question):
    """多轮记忆 + 混合检索问答"""
    return ask(question=q.query, session_id=q.session_id)


# ── Agent 问答 ────────────────────────────────
@app.post("/agent_ask")
def agent_ask_endpoint(q: Question):
    result = agent_ask(question=q.query, session_id=q.session_id)
    return {
        "answer":        result.answer,
        "question_type": result.question_type,
        "route":         result.route,
        "used_web":      result.used_web,
        "sql_result":    result.sql_result,
        "sources":       result.sources,
        "trace":         result.trace,
    }


# ── 清除会话 ──────────────────────────────────
@app.post("/clear")
def clear(s: Session):
    clear_session(s.session_id)
    return {"status": "ok"}


# ── 练习模式：获取题目 ────────────────────────
@app.post("/practice/questions")
def practice_questions(req: PracticeRequest):
    """生成指定章节的练习题目列表"""
    raw = generate_quiz.invoke({"chapter": req.chapter, "n": req.n})
    questions = re.findall(r'Q\d+:\s*(.+?)(?=\nA\d+:|\n\n|$)', raw, re.MULTILINE)
    questions = [q.strip() for q in questions if q.strip()]
    return {"questions": questions, "chapter": req.chapter}


# ── 练习模式：评估单题答案 ────────────────────
@app.post("/practice/evaluate")
def practice_evaluate(q: Question):
    """评估练习答案，格式：针对题目：{q}\n我的答案是：{a}"""
    result = agent_ask(question=q.query, session_id=q.session_id)
    return {"evaluation": result.answer}


# ── 笔记：保存 ────────────────────────────────
@app.post("/notes/save")
def note_save(n: NoteCreate):
    note_id = save_note(n.session_id, n.title, n.content, n.chapter)
    return {"status": "ok", "id": note_id}


# ── 笔记：查询 ────────────────────────────────
@app.get("/notes")
def note_list(session_id: str = ""):
    return {"notes": get_notes(session_id)}


# ── 笔记：删除 ────────────────────────────────
@app.delete("/notes/{note_id}")
def note_delete(note_id: int):
    if not delete_note(note_id):
        raise HTTPException(status_code=404, detail="笔记不存在")
    return {"status": "ok"}
