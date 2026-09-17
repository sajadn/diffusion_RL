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
"""Per-problem execution driver for LiveCodeBench grading.

Run as a subprocess, one per problem: reads a JSON job on stdin, runs the extracted
solution against every test case, writes a JSON verdict to stdout. Kept in its own
process so a runaway solution can be killed without taking the grader down, and so
resource limits apply only to model-authored code.
"""

import io
import json
import os
import resource
import signal
import sys
from contextlib import redirect_stdout
from decimal import Decimal, InvalidOperation


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def stdout_matches(actual, expected):
    """Match upstream LiveCodeBench's line-wise text / exact Decimal comparison.

    Reference: LiveCodeBench/LiveCodeBench, lcb_runner/evaluation/testing_util.py,
    grade_stdio. Internal blank lines and nonnumeric token spacing are significant.
    """
    actual_lines = [line.strip() for line in actual.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line == expected_line:
            continue
        try:
            actual_values = [Decimal(token) for token in actual_line.split()]
            expected_values = [Decimal(token) for token in expected_line.split()]
            if actual_values != expected_values:
                return False
        except InvalidOperation:
            return False
    return True


def jsonable(v):
    if isinstance(v, tuple):
        return [jsonable(x) for x in v]
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    return v


def run_stdin(code, tests, per_test_timeout):
    for t in tests:
        buf = io.StringIO()
        g = {"__name__": "__main__"}
        previous_stdin = sys.stdin
        signal.alarm(per_test_timeout)
        try:
            sys.stdin = io.TextIOWrapper(
                io.BytesIO(t["input"].encode("utf-8")), encoding="utf-8"
            )
            with redirect_stdout(buf):
                exec(compile(code, "<solution>", "exec"), g)
        except Timeout:
            return False, "timeout"
        except SystemExit:
            pass
        except BaseException as e:
            return False, "%s: %s" % (type(e).__name__, str(e)[:120])
        finally:
            signal.alarm(0)
            sys.stdin = previous_stdin
        if not stdout_matches(buf.getvalue(), t["output"]):
            return False, "wrong_answer"
    return True, "ok"


def run_functional(code, tests, func_name, per_test_timeout):
    g = {"__name__": "__main__"}
    # Same helper namespace and import order as upstream testing_util.import_string.
    modules = (
        "string",
        "re",
        "datetime",
        "collections",
        "heapq",
        "bisect",
        "copy",
        "math",
        "random",
        "statistics",
        "itertools",
        "functools",
        "operator",
        "io",
        "sys",
        "json",
        "builtins",
        "typing",
    )
    preamble = "".join(f"from {module} import *\n" for module in modules)
    preamble += "".join(f"import {module}\n" for module in modules[:-2])
    preamble += "sys.setrecursionlimit(50000)\n"
    try:
        exec(compile(preamble + code, "<solution>", "exec"), g)
    except BaseException as e:
        return False, "import_error: %s: %s" % (type(e).__name__, str(e)[:120])
    if "Solution" not in g:
        return False, "no_Solution_class"
    try:
        obj = g["Solution"]()
        fn = getattr(obj, func_name)
    except BaseException as e:
        return False, "no_func: %s" % str(e)[:120]

    for t in tests:
        args = []
        for line in t["input"].split("\n"):
            line = line.strip()
            if line:
                args.append(json.loads(line))
        expected = json.loads(t["output"])
        signal.alarm(per_test_timeout)
        try:
            got = fn(*args)
        except Timeout:
            return False, "timeout"
        except BaseException as e:
            return False, "%s: %s" % (type(e).__name__, str(e)[:120])
        finally:
            signal.alarm(0)
        if jsonable(got) != jsonable(expected):
            return False, "wrong_answer"
    return True, "ok"


def main():
    job = json.load(sys.stdin)
    # cap memory and CPU for model-authored code
    mem = job.get("mem_limit_mb", 4096) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):
        pass
    signal.signal(signal.SIGALRM, _alarm)

    code = job["code"]
    if not code.strip():
        out = {"passed": False, "reason": "empty_code"}
    elif job["mode"] == "functional":
        # Functional grading uses return values, not stdout. Discard prints from
        # module initialization, constructors, and calls without buffering them.
        with open(os.devnull, "w") as output, redirect_stdout(output):
            ok, reason = run_functional(
                code, job["tests"], job["func_name"], job["per_test_timeout"]
            )
        out = {"passed": ok, "reason": reason}
    else:
        ok, reason = run_stdin(code, job["tests"], job["per_test_timeout"])
        out = {"passed": ok, "reason": reason}
    sys.__stdout__.write(json.dumps(out))


if __name__ == "__main__":
    main()
