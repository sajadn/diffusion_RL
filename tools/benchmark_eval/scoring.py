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
"""Shared record validation and reasoning extraction for offline scoring."""

import json
from pathlib import Path


def strip_thinking(text: str) -> tuple[str, bool]:
    r"""Return (answer_text, closed) with any <think> block removed.

    With thinking ENABLED the prompt ends in an open "<think>\n", so the response is
    reasoning followed by "</think>" and then the answer. IFBench verifiers count
    words/keywords/format over the whole response, so the reasoning must go. If
    "</think>" never appears the response is entirely reasoning (truncated) and is
    unscoreable -- reported, not silently scored as a failure of the model's format.
    """
    marker = "</think>"
    idx = text.rfind(marker)
    if idx == -1:
        return "", False
    return text[idx + len(marker) :].strip(), True


def resolve_thinking_mode(records_path: Path, override: str) -> bool:
    """Whether generation ran with thinking enabled.

    This CANNOT be inferred from the response text: with thinking disabled the
    "<think></think>" pair lives in the prompt, so the response carries no tags at
    all -- textually identical to a thinking-on response that was truncated before
    emitting "</think>". Stripping the latter is required; stripping the former
    would delete the entire answer. So read the generating job's own setting.
    """
    if override in ("on", "off"):
        return override == "on"
    metrics_path = records_path.parent / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            f"cannot resolve thinking mode: {metrics_path} not found; "
            "pass --thinking on|off explicitly"
        )
    settings = json.loads(metrics_path.read_text()).get("settings") or {}
    if "enable_thinking" not in settings:
        raise SystemExit(
            f"{metrics_path} has no settings.enable_thinking (generated before the "
            "toggle existed); pass --thinking on|off explicitly"
        )
    return str(settings["enable_thinking"]).lower() in ("true", "1")


def load_records(path: Path, key: str = "source_id") -> list[dict]:
    """Reject empty, missing-ID, or duplicate records instead of changing denominators."""
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"No records in {path}")
    seen = set()
    for record in records:
        value = record.get(key)
        if value is None or str(value) == "" or str(value) in seen:
            raise ValueError(f"Missing or duplicate {key}: {value!r} in {path}")
        seen.add(str(value))
    return records
