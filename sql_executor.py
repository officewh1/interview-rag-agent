# =========================================
# SQL 执行沙箱
# 每次调用都用 SQLite 内存库 + 预置测试表
# 完全隔离，不影响任何本地文件
# =========================================

import sqlite3
import re
from typing import Any, Dict, List

# ===============================
# 预置测试表 + 样本数据
# 覆盖数据库章节常见考题场景
# ===============================
_INIT_SQL = """
-- 员工表
CREATE TABLE employees (
    emp_id     INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    dept_id    INTEGER,
    salary     REAL,
    hire_date  TEXT,
    manager_id INTEGER
);
INSERT INTO employees VALUES
(1, '张伟',   1, 12000, '2019-03-01', NULL),
(2, '李娜',   1,  9500, '2020-06-15', 1),
(3, '王芳',   2,  8000, '2021-01-20', 1),
(4, '赵磊',   2,  7500, '2021-08-10', 3),
(5, '陈静',   3, 11000, '2018-11-05', NULL),
(6, '刘洋',   3,  6500, '2022-03-30', 5),
(7, '孙鹏',   1, 15000, '2017-07-01', NULL),
(8, '周梅',   2,  9000, '2020-09-12', 3);

-- 部门表
CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    location  TEXT
);
INSERT INTO departments VALUES
(1, '研发部', '北京'),
(2, '测试部', '上海'),
(3, '产品部', '深圳');

-- 学生表
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    age        INTEGER,
    major      TEXT,
    gpa        REAL
);
INSERT INTO students VALUES
(1001, '小明', 20, '计算机科学', 3.8),
(1002, '小红', 21, '软件工程',   3.5),
(1003, '小张', 19, '数据科学',   3.9),
(1004, '小李', 22, '计算机科学', 2.9),
(1005, '小王', 20, '软件工程',   3.2);

-- 课程表
CREATE TABLE courses (
    course_id   INTEGER PRIMARY KEY,
    course_name TEXT NOT NULL,
    credits     INTEGER,
    teacher     TEXT
);
INSERT INTO courses VALUES
(101, '数据库原理',   3, '王老师'),
(102, '操作系统',     4, '李老师'),
(103, '数据结构',     4, '张老师'),
(104, '机器学习',     3, '陈老师');

-- 选课成绩表
CREATE TABLE enrollments (
    student_id INTEGER,
    course_id  INTEGER,
    score      REAL,
    PRIMARY KEY (student_id, course_id)
);
INSERT INTO enrollments VALUES
(1001, 101, 92), (1001, 102, 85), (1001, 103, 88),
(1002, 101, 78), (1002, 104, 91),
(1003, 101, 95), (1003, 102, 90), (1003, 103, 93), (1003, 104, 87),
(1004, 101, 60), (1004, 103, 55),
(1005, 102, 72), (1005, 104, 80);

-- 订单表
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product     TEXT,
    quantity    INTEGER,
    unit_price  REAL,
    order_date  TEXT
);
INSERT INTO orders VALUES
(1, 201, '笔记本电脑', 1, 6999, '2024-01-10'),
(2, 202, '手机',       2, 3999, '2024-01-15'),
(3, 201, '平板',       1, 2999, '2024-02-01'),
(4, 203, '耳机',       3,  299, '2024-02-10'),
(5, 202, '笔记本电脑', 1, 6999, '2024-03-05'),
(6, 204, '手机',       1, 3999, '2024-03-20');
"""

# ===============================
# 危险操作黑名单（禁止执行）
# ===============================
_BLOCKED = re.compile(
    r"\b(ATTACH|DETACH|PRAGMA|\.import|\.read)\b",
    re.IGNORECASE,
)

# ===============================
# 核心执行函数
# ===============================
def run_sql(sql: str, max_rows: int = 30) -> Dict[str, Any]:
    """
    在内存 SQLite 沙箱里执行 SQL。
    每次调用都是全新的数据库 + 预置数据，互不影响。

    返回：
        success   : bool
        columns   : 列名列表（仅 SELECT）
        rows      : 数据行列表（仅 SELECT，最多 max_rows 条）
        rowcount  : 影响行数（DML 语句）
        error     : 错误信息（失败时）
        note      : 补充说明
    """
    sql = sql.strip().rstrip(";")

    if _BLOCKED.search(sql):
        return {"success": False, "error": "包含不允许的操作，已拒绝执行。", "columns": [], "rows": []}

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_INIT_SQL)
        cursor = conn.execute(sql)

        # SELECT 类：返回结果集
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            all_rows = cursor.fetchall()
            rows = [list(r) for r in all_rows[:max_rows]]
            note = f"共 {len(all_rows)} 行，显示前 {len(rows)} 行。" if len(all_rows) > max_rows else f"共 {len(all_rows)} 行。"
            return {"success": True, "columns": columns, "rows": rows, "rowcount": 0, "note": note}

        # DML 类：返回影响行数
        conn.commit()
        return {"success": True, "columns": [], "rows": [], "rowcount": cursor.rowcount, "note": f"执行成功，影响 {cursor.rowcount} 行。"}

    except sqlite3.Error as e:
        return {"success": False, "error": str(e), "columns": [], "rows": []}
    finally:
        conn.close()


def format_result(result: Dict[str, Any]) -> str:
    """把 run_sql 的结果格式化成可读字符串，供 LLM 参考。"""
    if not result["success"]:
        return f"执行失败：{result['error']}"

    if result["columns"]:
        header = " | ".join(result["columns"])
        sep = "-" * len(header)
        body_lines = [" | ".join(str(v) for v in row) for row in result["rows"]]
        body = "\n".join(body_lines) if body_lines else "（无数据）"
        return f"{header}\n{sep}\n{body}\n{result.get('note', '')}"

    return result.get("note", "执行成功。")


# ===============================
# 可用表说明（给用户和 LLM 参考）
# ===============================
SCHEMA_HINT = """
沙箱内预置表（均有样本数据，可直接查询）：
  employees(emp_id, name, dept_id, salary, hire_date, manager_id)  -- 员工表
  departments(dept_id, dept_name, location)                         -- 部门表
  students(student_id, name, age, major, gpa)                       -- 学生表
  courses(course_id, course_name, credits, teacher)                 -- 课程表
  enrollments(student_id, course_id, score)                         -- 选课成绩表
  orders(order_id, customer_id, product, quantity, unit_price, order_date) -- 订单表
"""


# ===============================
# 本地测试
# ===============================
if __name__ == "__main__":
    tests = [
        "SELECT name, salary FROM employees WHERE salary > 9000 ORDER BY salary DESC",
        "SELECT d.dept_name, AVG(e.salary) AS avg_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id GROUP BY d.dept_name",
        "SELECT s.name, COUNT(en.course_id) AS course_count FROM students s LEFT JOIN enrollments en ON s.student_id = en.student_id GROUP BY s.student_id",
        "SELECT * FROM nonexistent_table",  # 测试错误处理
    ]
    for sql in tests:
        print(f"\nSQL: {sql}")
        result = run_sql(sql)
        print(format_result(result))
        print("-" * 60)
