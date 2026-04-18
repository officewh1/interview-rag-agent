"""
一键启动：后端 (uvicorn) + 前端 (streamlit)
运行方式：python start.py
"""

import subprocess
import sys
import time
import signal
import os
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON  = sys.executable


def port_ready(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(port: int, label: str, max_wait: int = 60):
    print(f"  等待 {label} 启动", end="", flush=True)
    for _ in range(max_wait):
        if port_ready("127.0.0.1", port):
            print(" ✓")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(" 超时！")
    return False


processes = []


def shutdown(sig=None, frame=None):
    print("\n正在关闭服务...")
    for p in processes:
        p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("已关闭。")
    sys.exit(0)


signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

print("=" * 50)
print("  面试题 RAG 问答助手  —  一键启动")
print("=" * 50)

# ── 启动后端 ──────────────────────────────────
print("\n[1/2] 启动后端 (FastAPI :8000) ...")
backend = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=BASE_DIR,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
processes.append(backend)

if not wait_for_port(8000, "FastAPI"):
    print("后端启动失败，退出。")
    shutdown()

# ── 启动前端 ──────────────────────────────────
print("[2/2] 启动前端 (Streamlit :8501) ...")
frontend = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", "app.py",
     "--server.port", "8501",
     "--server.headless", "true"],
    cwd=BASE_DIR,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
processes.append(frontend)

if not wait_for_port(8501, "Streamlit"):
    print("前端启动失败，退出。")
    shutdown()

# ── 自动打开浏览器 ────────────────────────────
import webbrowser
webbrowser.open("http://localhost:8501")

print("\n启动成功！")
print("  前端地址：http://localhost:8501")
print("  后端地址：http://localhost:8000")
print("\n按 Ctrl+C 停止所有服务。\n")

# ── 保持运行，监控子进程 ──────────────────────
while True:
    time.sleep(2)
    for p in processes:
        if p.poll() is not None:
            print(f"子进程意外退出 (pid={p.pid})，正在关闭...")
            shutdown()
