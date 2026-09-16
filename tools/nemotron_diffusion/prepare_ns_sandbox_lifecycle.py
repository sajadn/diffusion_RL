# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a checked patch of the submitted sandbox server, without changing its image."""

import argparse
import hashlib
import subprocess
from pathlib import Path


def patch_server(source: str) -> str:
    old = "def shell_worker(conn):\n    shell = TerminalInteractiveShell()"
    assert source.count(old) == 1, "Unexpected sandbox shell_worker implementation"
    source = source.replace(
        old,
        "def shell_worker(conn):\n    os.setsid()\n    shell = TerminalInteractiveShell()",
        1,
    )
    start = source.index("    def stop_shell(self, shell_id):")
    end = source.index("    def run_cell(", start)
    source = (
        source[:start]
        + """    def stop_shell(self, shell_id):
        with self.manager_lock:
            entry = self.shells.pop(shell_id, None)
        if entry is None:
            return
        proc, conn = entry["proc"], entry["conn"]
        # Each shell is a session leader. Include subprocesses such as CBC and pip.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Also covers a newly forked shell that has not reached setsid yet.
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=2)
        conn.close()
        if not proc.is_alive():
            proc.close()

"""
        + source[end:]
    )
    old = """                    with self.manager_lock:
                        self.shells.pop(shell_id, None)
                    self.start_shell(shell_id)"""
    assert source.count(old) == 2, "Unexpected sandbox restart implementation"
    source = source.replace(
        old,
        """                    self.stop_shell(shell_id)
                    self.start_shell(shell_id)""",
    )
    start = source.index("            # no reply yet -> try gentle interrupt")
    end = source.index("\ndef log_session_count", start)
    source = (
        source[:start]
        + """            # A timeout ends the whole execution process group. Interrupting only
            # IPython leaves child solvers running after the tool has returned.
            self.stop_shell(shell_id)
            self.start_shell(shell_id)
            with self.manager_lock:
                self.shells[shell_id]["restart_pending"] = True
            return {
                "status": "timeout_killed",
                "id": exec_id,
                "shell_was_restarted": True,
                "shell_was_recently_restarted": shell_was_recently_restarted,
            }

"""
        + source[end:]
    )
    compile(source, "sandbox/main.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("output", type=Path)
    parser.add_argument("--launcher-output", type=Path)
    args = parser.parse_args()
    original = subprocess.check_output(
        ["unsquashfs", "-cat", args.image, "app/main.py"]
    )
    # This patch replaces lifecycle methods in the inspected image. Refuse a
    # different server version until its implementation has been reviewed.
    expected_sha256 = "02bce0cfee05853b208cddcc27731b6ee03e4ed096ef97a2c37df69c58aa5162"
    if hashlib.sha256(original).hexdigest() != expected_sha256:
        raise ValueError("Sandbox server changed; review its lifecycle before patching")
    patched = patch_server(original.decode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patched)
    print("source_sha256=" + hashlib.sha256(original).hexdigest())
    print("patched_sha256=" + hashlib.sha256(patched.encode()).hexdigest())
    if args.launcher_output is not None:
        launcher = subprocess.check_output(
            ["unsquashfs", "-cat", args.image, "start-with-nginx.sh"], text=True
        )
        expected_launcher = (
            "91b71856b5bf1bc640e1199139e88818b278e22ff05e9bfd7733b5f93a884313"
        )
        if hashlib.sha256(launcher.encode()).hexdigest() != expected_launcher:
            raise ValueError(
                "Sandbox launcher changed; review its cleanup before patching"
            )
        launcher = launcher.replace(
            "    pkill -f nginx || true",
            "    # Containers share the host PID namespace; target this nginx via its PID file.\n"
            "    nginx -s quit || true",
        )
        launcher = launcher.replace(
            'export NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}',
            'export NUM_WORKERS=${NUM_WORKERS:-$(nproc --all)}\n'
            'if [[ -n "${NEMO_SKILLS_SANDBOX_BLAS_THREADS:-}" ]]; then\n'
            '    export OPENBLAS_NUM_THREADS="$NEMO_SKILLS_SANDBOX_BLAS_THREADS"\n'
            '    export OMP_NUM_THREADS="$NEMO_SKILLS_SANDBOX_BLAS_THREADS"\n'
            '    export MKL_NUM_THREADS="$NEMO_SKILLS_SANDBOX_BLAS_THREADS"\n'
            '    export NUMEXPR_NUM_THREADS="$NEMO_SKILLS_SANDBOX_BLAS_THREADS"\n'
            'fi',
        )
        subprocess.run(["bash", "-n"], input=launcher, text=True, check=True)
        args.launcher_output.parent.mkdir(parents=True, exist_ok=True)
        args.launcher_output.write_text(launcher)
        args.launcher_output.chmod(0o755)
        print("launcher_sha256=" + hashlib.sha256(launcher.encode()).hexdigest())


if __name__ == "__main__":
    main()
