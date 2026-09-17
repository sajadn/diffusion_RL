#!/usr/bin/env python3
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
"""Request-parameter fixes to NeMo-Skills' OpenAI client needed to drive a diffusion vLLM server.

These live in `nemo_skills/inference/model/openai.py`, which overrides
`_build_chat_request_params` from the base class.

1. **Drop `seed`.** vLLM's diffusion path raises in `_validate_diffusion()` when
   `self.seed is not None`, so every request 400s with "Diffusion models only support
   temperature 0 (greedy) or 1 ...". NeMo-Skills sets `"seed": random_seed`
   unconditionally (default 0). `++inference.random_seed=null` does not help either:
   the field is typed `int`, so Hydra fails with "Error merging override".

2. **Forward `extra_body`.** The base class builds `"extra_body": extra_body`, but
   this subclass's override accepts the argument and then never puts it in the
   request. Anything passed via `++inference.extra_body.*` is therefore silently
   dropped -- including `thinking_token_budget`, which is the whole point of running
   on vLLM. Symptom when unfixed: generations run to the full token cap with no
   `</think>` anywhere, exactly as if no budget had been set.

3. **Separate repetition and presence penalties.** The container client maps
   repetition_penalty=1.0 (neutral) to presence_penalty=1.0 (non-neutral).
   Send repetition_penalty through vLLM's extra_body and keep presence neutral.

Fix 1 is gated on NEMO_SKILLS_OMIT_SEED=1 because the seed is what makes the AR and
SGLang paths reproducible. Fix 2 is unconditional -- dropping a parameter the caller
explicitly asked for is a bug on any backend.

Idempotent.
"""

import os
import sys

MARKER = "# patched: diffusion-compatible request params"

OLD = '            "seed": random_seed,'
NEW = (
    "            " + MARKER + "\n"
    '            **({} if os.environ.get("NEMO_SKILLS_OMIT_SEED") == "1" else {"seed": random_seed}),\n'
    '            **({"extra_body": extra_body} if extra_body else {}),'
)


PENALTY_OLD = '            params["presence_penalty"] = repetition_penalty'
PENALTY_NEW = (
    '            params["presence_penalty"] = 0.0\n'
    '            params["extra_body"] = {"repetition_penalty": repetition_penalty, **(extra_body or {})}'
)


def patch_source(source: str) -> str:
    """Patch the pinned client's request builder, including earlier patched copies."""
    if MARKER not in source:
        if source.count(OLD) != 1:
            raise ValueError("NeMo-Skills seed mapping changed; recheck the client")
        source = source.replace(OLD, NEW, 1)
    if PENALTY_NEW not in source:
        if source.count(PENALTY_OLD) != 1:
            raise ValueError("NeMo-Skills penalty mapping changed; recheck the client")
        source = source.replace(PENALTY_OLD, PENALTY_NEW, 1)
    return source


def main() -> int:
    from nemo_skills.inference.model import openai as ns_openai

    path = ns_openai.__file__
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        patched = patch_source(source)
    except ValueError as error:
        print(f"ERROR: {path}: {error}", file=sys.stderr)
        return 1
    if patched == source:
        print(f"already patched: {path}")
        return 0

    if os.path.islink(path):
        os.unlink(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(
        f"patched seed omission, extra_body, and neutral presence penalty into {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
