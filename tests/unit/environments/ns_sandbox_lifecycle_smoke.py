"""Run inside the sandbox image with the patched /app/main.py mounted read-only."""

import importlib.util
import tempfile
import time
import uuid
from pathlib import Path

import psutil

spec = importlib.util.spec_from_file_location("sandbox_server", "/app/main.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
client = server.app.test_client()


def execute(session, code, timeout=5):
    response = client.post(
        "/execute",
        headers={"X-Session-ID": session},
        json={"generated_code": code, "timeout": timeout, "language": "ipython"},
    )
    assert response.status_code == 200, response.data
    return response.json


session = str(uuid.uuid4())
assert execute(session, "x = 41")["process_status"] == "completed"
assert execute(session, "print(x + 1)")["stdout"].strip() == "42"
assert client.delete(f"/sessions/{session}").status_code == 200
assert not server.shell_manager.shells
assert client.delete(f"/sessions/{session}").status_code == 404
print("PASS: state persists across turns; deletion is idempotent")

session = str(uuid.uuid4())
execute(session, "x = 1")
with tempfile.TemporaryDirectory() as directory:
    pidfile = str(Path(directory) / "child.pid")
    code = (
        "import subprocess, pathlib; "
        "p = subprocess.Popen(['sleep', '120']); "
        f"pathlib.Path({pidfile!r}).write_text(str(p.pid)); p.wait()"
    )
    started = time.monotonic()
    result = execute(session, code, timeout=0.5)
    assert result["process_status"] == "timeout", result
    assert time.monotonic() - started < 5
    pid = int(Path(pidfile).read_text())
    time.sleep(0.2)
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
    assert execute(session, "print('recovered')")["stdout"].strip() == "recovered"
    client.delete(f"/sessions/{session}")
print("PASS: timeout kills child solver, returns promptly, fresh session works")

for _ in range(20):
    session = str(uuid.uuid4())
    execute(session, "print(1)")
    client.delete(f"/sessions/{session}")
assert not server.shell_manager.shells
session = str(uuid.uuid4())
execute(session, "x = 1")
execute(session, "import os; os._exit(7)")
assert execute(session, "print('restarted')")["stdout"].strip() == "restarted"
client.delete(f"/sessions/{session}")
assert not server.shell_manager.shells
print("PASS: repeated rollouts and shell crash leave no retained sessions")


session = str(uuid.uuid4())
with tempfile.TemporaryDirectory() as directory:
    pidfile = str(Path(directory) / "background.pid")
    code = (
        "import subprocess, pathlib; "
        "p = subprocess.Popen(['sleep', '120']); "
        f"pathlib.Path({pidfile!r}).write_text(str(p.pid))"
    )
    assert execute(session, code)["process_status"] == "completed"
    pid = int(Path(pidfile).read_text())
    assert client.delete(f"/sessions/{session}").status_code == 200
    time.sleep(0.2)
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )
assert not server.shell_manager.shells
print("PASS: normal completion cleanup kills background descendants")


session = str(uuid.uuid4())
execute(session, "x=1")
server.SESSION_TIMEOUT = 1200
server.shell_manager.shells[session]["last_used"] -= 1300
next_session = str(uuid.uuid4())
execute(next_session, "print(1)")
assert session not in server.shell_manager.shells
client.delete(f"/sessions/{next_session}")
assert not server.shell_manager.shells
print("PASS: idle expiry cleans a session whose caller lost its ID")
