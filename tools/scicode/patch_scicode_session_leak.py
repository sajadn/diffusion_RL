#!/usr/bin/env python3
"""Backport the SciCode sandbox-session cleanup into the eval container's NeMo-Skills.

The container ships NeMo-Skills at git 1a67193, whose scicode evaluator does:

    sandbox = get_sandbox(**eval_config.sandbox)
    output_dict, _ = await sandbox.execute_code(code, ...)
    return elem_idx, output_dict

`execute_code` defaults to language="ipython" with session_id=None, which makes the
sandbox mint a fresh IPython session per call. That session id is discarded and never
deleted, and the sandbox client is never closed. The local sandbox is a single-process
Flask app, so on test_aai (~338 subproblem executions) the sessions accumulate in one
process until tool calls stop returning -- the same way the ns_tools rollouts wedged at
~253 active sessions.

Upstream fixed this by deleting the session and closing the client in a finally block.
This applies that fix in place. Patching nemo_skills inside the container follows the
same pattern as the other eval pipelines here; the container root is a writable enroot
overlay, so the edit lasts exactly as long as the job.

Idempotent: running it on an already-fixed evaluator is a no-op.
"""

import os
import sys

OLD = """    sandbox = get_sandbox(**eval_config.sandbox)
    output_dict, _ = await sandbox.execute_code(code, timeout=eval_config.timeout, max_output_characters=100000)

    return elem_idx, output_dict
"""

NEW = """    sandbox = get_sandbox(**eval_config.sandbox)
    session_id = None
    try:
        output_dict, session_id = await sandbox.execute_code(
            code, timeout=eval_config.timeout, max_output_characters=100000
        )
        return elem_idx, output_dict
    finally:
        try:
            if session_id is not None:
                await sandbox.delete_session(str(session_id))
        finally:
            await sandbox.close()
"""


def main() -> int:
    from nemo_skills.evaluation.evaluator import scicode as scicode_evaluator

    path = scicode_evaluator.__file__
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    if "await sandbox.delete_session" in source:
        print(f"already patched: {path}")
        return 0

    if OLD not in source:
        print(
            f"ERROR: {path} does not contain the expected unpatched block. "
            "The container's NeMo-Skills version changed; re-check the evaluator "
            "before trusting SciCode results.",
            file=sys.stderr,
        )
        return 1

    patched = source.replace(OLD, NEW)

    # site-packages entries are symlinks into the uv archive cache. Replace the link
    # with a real file so the edit cannot leak into anything else reading that cache.
    if os.path.islink(path):
        os.unlink(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"patched session cleanup into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
