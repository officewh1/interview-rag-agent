# =========================================
# 笔记系统 —— SQLite 持久化
# =========================================

import sqlite3
import os
from datetime import datetime

_DB_PATH = os.path.join(os.path.dirname(__file__), "notes.db")


def _conn():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                title      TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                chapter    TEXT    DEFAULT '',
                created_at TEXT    DEFAULT (datetime('now', 'localtime'))
            )
        """)
        con.commit()


# 启动时建表
init_db()


def save_note(session_id: str, title: str, content: str, chapter: str = "") -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO notes (session_id, title, content, chapter) VALUES (?,?,?,?)",
            (session_id, title, content, chapter),
        )
        con.commit()
        return cur.lastrowid


def get_notes(session_id: str = "") -> list[dict]:
    with _conn() as con:
        if session_id:
            rows = con.execute(
                "SELECT * FROM notes WHERE session_id=? ORDER BY id DESC", (session_id,)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM notes ORDER BY id DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def delete_note(note_id: int) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM notes WHERE id=?", (note_id,))
        con.commit()
        return cur.rowcount > 0
