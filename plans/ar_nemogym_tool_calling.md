# AR GRPO on NeMo-Gym v30: tool-calling and thinking issues

> ## RESOLVED 2026-09-08 -- READ THIS FIRST
>
> **Sections 3 and 4 below are superseded.** The tool-name and tool-format problems were both
> caused by a STALE LOCAL CHECKPOINT, not by the model, the prompt, the template, or the parser.
>
> The local `Nemotron-Labs-Diffusion-8B-v1.5-sft` was downloaded 2026-07-08. HF replaced the
> weights twice afterwards (2026-07-23 -> step_10000; 2026-08-06 -> ada step_9000). Because
> "all 309 tensor names are unchanged", the file SIZE stayed identical at 16,979,144,720 bytes,
> so a size check could not see the swap. Only sha256 could:
> ```
> stale     0e8485daa7c0c92b747c094685368d218ff2129a5160f5a59fc507148dd3b136
> current   ae7be4d9d94572012888748d822a1f63b0704458343c5065b9d7b0193fa6dd8e
> ```
>
> Measured on the verbatim Cascade-2 `math_tool` prompt, `model.ar_generate`, temp 1.0, n=8,
> identical template/seeds/budget:
> ```
> refreshed (current HF)   XML 6/8   -> stateful_python_code_exec   CORRECT
> ada_step9000 (== HF)     XML 7/8   -> stateful_python_code_exec   CORRECT
> stale local              0 XML     -> "python"                    WRONG
> yongganf cd2_step3000    0 XML     -> "python"                    WRONG
> ```
> `cd2_step3000` ran ada's modeling code and tokenizer and still failed, so the WEIGHTS are the
> cause -- not the modeling code, tokenizer, decode path, thinking budget, or parser.
>
> **Consequences**
> - The bare-JSON chat templates (`nemotron_labs_diffusion_bare_json*.jinja`) and the
>   `llama3_json` parser were fitted to stale weights and now point the model AWAY from its
>   trained format. The parent config's `step3p5` + the checkpoint's own `chat_template.jinja`
>   are correct again.
> - The `python -> stateful_python_code_exec` alias in `ns_tools/app.py` should be harmless but
>   is no longer load-bearing; keep it only if rollouts still show unknown-tool errors.
> - Every format-matrix number in section 4 (variants A-E, F-I) was measured on stale weights
>   and should not be used to choose a template or parser.
> - Any RL run or eval before 2026-09-08 used stale weights.
>
> **Still true and worth reporting upstream:** `sft_cascade2_full_shuffled.jsonl` serializes
> `tool_calls` without a `name` field (37/37 in an 84-record head sample; `id`/`type` wrongly
> nested inside `function`), so `chat_template.jinja` renders `<tool_call>\n<function=>` with an
> EMPTY name. That explains `cd2_step3000`, which was trained on it.
>
> Refreshed checkpoint:
> `/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/Nemotron-Labs-Diffusion-8B-v1.5-sft-20260902`
> Verification recipe: sha256 vs the LFS oid from `/api/models/<repo>/tree/main?recursive=1`,
> and `/api/models/<repo>/commits/main` for weight-replacing commits.

> ### SECOND ROOT CAUSE, found the same day: ns_tools had no sandbox
>
> Independent of the checkpoint. `ns_tools` spawns an MCP `python_tool` server which executes
> code via nemo_skills `LocalSandbox`. "Local" means locally HOSTED, not in-process -- it POSTs
> to `http://127.0.0.1:6000/execute`. **Nothing in NeMo-RL, ray.sub, the submit script, or Gym
> ever started that service.** Every tool call therefore returned "Error: Tool execution failed",
> no `function_call_output` was appended, and simple_agent's `while True` could not reach a
> second turn. Multi-turn was impossible regardless of prompt, parser or weights.
>
> Verified on a compute node that the sandbox itself is healthy: `/health` 200, `/execute` runs
> sympy, IPython session state persists between calls.
>
> **Fix:** `resources_servers/ns_tools/app.py` now calls `_start_local_sandbox()` in
> `setup_webserver()` before `_start_python_tool_server()`. Idempotent (skips if `/health`
> already answers), spawns only on 127.0.0.1, waits 60s for readiness.
>
> Gotchas: `local_sandbox_server.py` calls `resource.setrlimit(RLIMIT_AS)` at module import, so
> it cannot even be imported on a dfw login node (hard limit 8GB) but works on compute nodes;
> its entrypoint is a bare `app.run(port=6000)` so the port is hardcoded; and
> `NSToolsResourcesServer` is a pydantic model, so the Popen handle must be kept in a local
> rather than an undeclared private attr.

> ### THIRD FINDING: the prohibition degrades the tool name to EMPTY
>
> Run 18144254 (refreshed checkpoint, sandbox patched, but using the checkpoint's OWN
> chat_template unmodified) produced:
> ```
> step1  XML <function= 87/128   names {<EMPTY>: 88}   multi-turn 0
> step2  XML <function= 77/128   names {<EMPTY>: 77}   multi-turn 0
> step3  XML <function= 65/128   names {<EMPTY>: 65, "code": 1}   multi-turn 1/128
> ```
> Format is fixed (XML, ~60-68% call rate vs ~19% before) but EVERY name is empty.
>
> Cause: ns_tools rows carry only a user turn plus a `tools` field -- no system message. The
> template's `else` branch then injects "You are a helpful and harmless assistant.\n\nYou are
> not allowed to use any tools." ALONGSIDE the `<tools>` declaration. Under that contradiction
> the model degrades to `<function=>`.
>
> The single multi-turn sample shows the mechanism outright. After a routing error
> (`{"error": "Unknown tool: code"}`) the model retried with:
> ```
> <tool_call>
> <function=>
> <parameter=code>50*12 - 600</parameter>
> <parameter=name>stateful_python_code_exec</parameter>
> </function>
> </tool_call>
> ```
> It KNOWS the tool name -- it writes it -- but places it in a PARAMETER because it has no
> learned pattern for the `<function=NAME>` slot (that slot was empty throughout the SFT text;
> see the second root cause above). Given a clean prompt it does fill the slot: the HF probe
> with an empty system message produced `<function=stateful_python_code_exec>` 6-7 times of 8.
>
> **Fix:** `examples/configs/chat_templates/nld_v15sft_20260902_emptysys.jinja` -- the
> checkpoint's own template with exactly ONE line changed (line 28: default system message ->
> `""`). Verified by rendering a real ns_tools row: prohibition gone, `<tools>` still declared.
> Set on BOTH `policy.tokenizer.chat_template` and
> `policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template` -- the training
> side loads `.jinja` paths at algorithms/utils.py:328, the serving side via the
> `load_chat_template` fix in vllm_worker_async.py.

> ### FOURTH FINDING: a truncated tool call poisons the whole trajectory
>
> Run 18147448 (refreshed ckpt + sandbox + emptysys template) produced ZERO rollouts in 2h and
> hit the walltime. The driver log shows:
> ```
> 21x json.decoder.JSONDecodeError: Unterminated string at line 1 column 10 (char 9)
>     vllm/entrypoints/chat_utils.py:1857 _postprocess_messages
> 10x RuntimeError: NemoGym rollout failed (HTTP 500) in ns_tools_simple_agent
> -> "STALLING: Need 8 trajectories for step 0, but only 5 are ready"  (forever)
> ```
>
> Chain:
> 1. `max_new_tokens: 4096` with `thinking_token_budget: 2048` leaves ~2048 for the call.
>    This model writes its whole derivation INSIDE the `code` argument as `#` comments
>    (seen directly in HF probe seed 1007), so the call is cut mid-string.
>    Note `rollouts.py:1154` maps max_new_tokens -> max_output_tokens, so the cap DOES bind
>    (the old "NeMo-Gym ignores max_new_tokens" gap is fixed).
> 2. step3p5 accumulates arguments incrementally (`arguments += new_args`,
>    step3p5_tool_parser.py:641), so a cut leaves partial JSON like `{"code": "import sympy`.
> 3. simple_agent catches its OWN json.loads failure, but the malformed function_call stays in
>    `new_outputs`, and every later turn re-POSTs the whole conversation. vLLM re-parses each
>    call's arguments server-side -> HTTP 500 -> `raise_for_status()` at simple_agent app.py:93
>    -> trajectory dies -> AsyncTrajectoryCollector loses it -> ReplayBuffer stalls forever.
>
> **Two fixes, both applied:**
> - `max_new_tokens: 4096 -> 8192` in `ar_v30_ns_refreshed.yaml` (fewer truncations).
> - `simple_agent/app.py`: on a malformed-arguments call, set `.arguments = "{}"` IN PLACE
>   before appending the error output, so the conversation stays parseable and one bad call
>   can no longer stall the run. The original text is preserved in the tool output the model
>   sees.
>
> This is the failure mode to watch for generally: ONE malformed tool call in ONE trajectory
> stalls the entire async run, because the buffer waits for a group that can never complete.

> ### FIFTH FINDING: the sandbox itself is the next bottleneck
>
> With format, names, execution and truncation all fixed, run 18156660 STILL wedged --
> and this time with no errors at all:
> ```
> Unterminated string: 0   HTTP 500: 0   Traceback: 0
> (NemoGym pid=...) [worker unknown] INFO: active_sessions=253   <- flat, stopped growing
> GPU utilization on the generation node: 0 %
> 5/8 trajectory groups complete -> ReplayBuffer stalls -> walltime burned on idle GPUs
> ```
> `nemo_skills/code_execution/local_sandbox/local_sandbox_server.py` ends in a bare
> single-process `app.run(port=6000)`. Its `WORKER_NUM` env var implies a multi-worker
> deployment that the package does not ship. Every rollout opens a persistent IPython
> session, and although the server exposes `DELETE /sessions/{id}`, nothing ever calls it --
> so sessions accumulate monotonically until the server stops serving.
>
> **Mitigation for now:** keep ns_tools rollout concurrency small.
> `ar_v30_ns_esys_small.yaml` uses 4 prompts x 4 generations = 16 concurrent rollouts
> (vs 8 x 16 = 128, which wedged). The proper fixes are to serve the sandbox multi-worker
> and to reclaim sessions when a rollout finishes.
>
> Positive result from that run: the sandbox patch works. `active_sessions` climbing 0 -> 253
> is direct evidence that tool calls were reaching the sandbox and executing.

> ### RESULT: multi-turn works (run 18158751, 2026-09-09)
>
> ```
>                      MULTI-TURN   tool names                       tool results
> old prod run          0 / 1920    web_search / python / junk       (sandbox never ran)
> run 18158751 step1   16 / 16      stateful_python_code_exec x249   REAL x247, err 0
> run 18158751 step2   16 / 16      stateful_python_code_exec x364   REAL x362, err 0
> ```
> Format XML 16/16, zero empty names, zero JSON errors, zero HTTP 500s. Segment counts run
> 2..50, i.e. genuine iterative tool use rather than a single call.
>
> Working config: `examples/configs/ar_v30_ns_esys_small.yaml`
> (4 prompts x 4 generations = 16 concurrent rollouts; max_new_tokens 8192; the emptysys
> template on BOTH `policy.tokenizer.chat_template` and the vLLM serving kwargs; step3p5 tool
> and reasoning parsers; checkpoint `...-v1.5-sft-20260902`).
>
> **Concurrency ceiling, measured (run 18162496, same fixes throughout):**
> ```
> config                       rollouts/step  result             multi-turn   tool results
> ar_v30_ns_esys_small.yaml    16             COMPLETED 31min    16/16 100%   609/613 real
> ar_v30_ns_esys_64.yaml       64             COMPLETED          62/64 96.9%  1696/1710 real
> ar_v30_ns_refreshed.yaml     128            WEDGES (twice)     0            n/a
> ```
> **Use `ar_v30_ns_esys_64.yaml` in production.** Step1 and step2 are identical, so it is
> stable. At 64 the sandbox reclaims sessions (83 -> 77 observed, i.e.
> NEMO_SKILLS_SANDBOX_SESSION_TIMEOUT=600 IS working); at 128 it saturates, nothing goes idle,
> and no rollout ever completes.
>
> Note the mechanism, since it bounds any future scaling: `ShellManager.start_shell` spawns
> `mp.Process(target=shell_worker)` PER SESSION, so 253 sessions = 253 IPython subprocesses on
> one node. Multi-worker would NOT fix it -- `ShellManager.shells` is a plain dict on a
> module-level singleton with no shared store, so a session created on worker A is invisible to
> worker B and the model's variables would vanish between turns without sticky X-Session-ID
> routing. The real fix is pooling/reusing a bounded set of shells, upstream in nemo_skills.
>
> ### CONCURRENCY RESOLVED (2026-09-09): the sandbox container lifted the ceiling
>
> Moving the sandbox from a hand-rolled in-process spawn to upstream's launcher component
> (ray.sub `NEMO_SKILLS_SANDBOX=1`, uwsgi+nginx with ~128 workers) removed the 64-rollout cap:
> ```
> shape        nodes  sandbox            step1            step2            result
> 8x8  = 64      3    in-process 1 proc  62/64  (96.9%)   62/64            COMPLETED
> 8x8  = 64      3    container 128 wkr  64/64  (100%)    62/64  (96.9%)   COMPLETED
> 8x16 = 128     3    in-process 1 proc  0 rollouts in 2h -- WEDGED TWICE
> 8x16 = 128     2    container 128 wkr  127/128 (99.2%)  121/128 (94.5%)  COMPLETED
> 16x8 = 128     2    container 128 wkr  123/128 (96.1%)  123/128 (96.1%)  COMPLETED
> ```
> Both 128 shapes work, on TWO nodes (1 train + 1 gen, dp 1) -- generation is tool-bound, not
> compute-bound, and 2-node requests backfill far more easily on a full cluster.
>
> Wrong tool names are now rare but nonzero: 16x8 step1 had 13 `python` out of 1743 (0.7%),
> step2 a single `stateful_python_codeExec` typo out of 1592. The refreshed checkpoint strongly
> prefers the correct name; the old failure mode survives in the tail at temperature 1.0. Not
> worth restoring the deleted alias at that rate.
>
> **One open item, and it is a modelling question rather than infrastructure:** some
> trajectories reach 50 segments, exactly `max_steps: 50` (ns_tools.yaml:52) -- the model does
> not reliably stop calling the tool. That drives cost per rollout directly.

Status: FINDINGS (2026-08-27). Companion to `multiturn_diffusion_support.md`, which
covers the diffusion-side multi-span crash. This file covers the **AR** path.

Everything below was measured on `nd8b_ar_v30_10t6g` (production, steps 1-130) and on
nine 3-node toys run 2026-08-26/27. Numbers are stated with their sample size. Claims
that are inference rather than measurement are marked INFERRED.

---

## 0. Structural facts you need before reading anything else

12 agents serve 18 datasets, via **two** agent classes:

| class | file | loop? |
|---|---|---|
| `simple_agent` | `responses_api_agents/simple_agent/app.py` (222 lines, `while True` at :83) | yes |
| `tool_simulation_agent` | `responses_api_agents/tool_simulation_agent/app.py` (90 lines) | **no** |

`tool_simulation_agent.run()` is: POST model once -> POST `/verify` -> return. No loop,
no tool dispatch, no response injection. It is single-step **by construction**.

Datasets bound to it: `tau_pivot` (37,377 rows, 23.1%) and `search_pivot` (12,002, 7.4%)
= **30.5% of the corpus cannot close a tool loop, ever**.

`simple_agent` can loop, but only if the row declares tools. Only 4 datasets do:
`math_tir_skywork_no_omni`, `math_tir_turing`, `math_tir_holdout_small_igor` (all ->
`ns_tools_simple_agent`) and `workbench` (-> `workplace_assistant_simple_agent`).
Total 9,193 rows = **5.7% of the corpus can actually loop**.

The other 12 datasets run `simple_agent` with no tools: `all_fn_calls` is always empty,
so `if not all_fn_calls and all_output_messages: break` exits on pass 1. Functionally
single-turn.

Routing is per-row, in the data: every row carries
`agent_ref = {"type": "responses_api_agents", "name": "<instance>"}`.
`env.nemo_gym.config_paths` only *registers* which instances exist.

---

## 1. CLARIFICATION (not a bug): "multi-segment trajectory" is an invalid success
metric on 2 of 12 agents

Symptom: 0/384 and 0/1920 samples with two trainable segments in `token_loss_mask`,
across many configurations, on `search_pivot` and `tau_pivot`.

Root cause: those agents are `tool_simulation_agent`. The count was pinned at zero by
the harness, not by the model or the format.

Cost: an entire 5-way format matrix was scored on a metric that could not move. Use
**reward** for `tool_simulation_agent` agents (they grade by comparing the single
predicted action to `expected_action`), and reserve segment counts for `simple_agent`
agents that declare tools.

Status: nothing to fix in the code. This is a measurement error on my part, recorded
so it is not repeated. The metric itself is correct and is what later confirmed the
loop closing on `ns_tools` -- it is only meaningless on agents whose class has no loop.

---

## 2. ISSUE: `ns_tools` prompts declare tools AND forbid tool use  [FIXED]

Symptom: 128/128 `ns_tools` prompts contained both a `<tools>` block declaring
`stateful_python_code_exec` and the sentence "You are not allowed to use any tools."

Root cause: `chat_template.jinja` has two INDEPENDENT branches writing into the same
system block:

    {%- if messages[0]["role"] == "system" %}    system_message = messages[0].content
    {%- else %}                                  system_message = "...not allowed to use any tools."

    {%- if tools is iterable and tools | length > 0 %}   renders "# Tools ... <tools>"

Branch A never checks `tools`; branch B never checks the system message. The default
fires whenever a row ships no system turn.

Scope -- it needs BOTH "declares tools" AND "no system message". Measured on val_broad:

| ships own system msg | datasets |
|---|---|
| yes (4) | search_pivot, tau_pivot, workbench, calendar_v2 |
| no (14) | the rest |

Of the 14, only the 3 `math_tir_*` also declare tools. So the contradiction hits
**exactly `ns_tools`**. `workbench` has tools but ships its own system message, so it
escaped -- that is the control proving the trigger is the intersection.

Fix: make the default conditional on `tools`
(`chat_templates/nemotron_labs_diffusion_bare_json_tools_sysfix.jinja`). Leaves the 12
tool-less datasets with the (correct) prohibition and the 4 self-supplied ones untouched.

Effect (ns_tools, 384 samples each): call rate 3% -> 12%, reward 0.052 -> 0.096.

---

## 3. ISSUE: the model calls a tool name that does not exist  [SUPERSEDED -- stale checkpoint, see banner]

Symptom: `<tool_response>{"error": "Unknown tool: python"}</tool_response>`.

Measured tool names emitted (ns_tools, 104 calls): `python` 98, `code_execution` 2,
`stateful_python_code_exec` 4. In variant I: **45/45 wrote `python`**.

The prompt declares the real name -- `<name>stateful_python_code_exec</name>` with its
full description. The model ignores it. INFERRED: its SFT (math_tir = tool-integrated
reasoning) used `python`.

Mechanism: `simple_agent` POSTs to `/{output_function_call.name}`, so the tool name is a
URL path. `ns_tools/app.py` has a wildcard route `app.post("/{tool_name}")`, so FastAPI
accepts anything; the rejection happens in application code:

    if tool_name not in self._tool_name_map:  return {"error": f"Unknown tool: {tool_name}"}
    result = await self.tool_manager.execute_tool(raw_name=tool_name, ...)

`_tool_name_map` was an identity map built from `list_all_tools()`.

Fix (2 parts, both required) in `resources_servers/ns_tools/app.py`:
  1. add aliases `python`/`code_execution` -> `stateful_python_code_exec` to the map
  2. dispatch `raw_name=self._tool_name_map[tool_name]` (was the raw key)
Part 1 alone passes the check then hands an unknown name to nemo-skills.

Effect: unknown-tool errors 13 -> 0, reward 0.096 -> 0.107.

WHAT DID NOT WORK: adding "do NOT write python or code_execution" to the prompt
(variant H) made it **worse** -- wrong names 13 -> 21, reward 0.096 -> 0.086.

NOT FIXED: after aliasing, calls now fail downstream with
`<tool_response>Error: Tool execution failed</tool_response>`. Routing works; execution
does not. Next thing to check is whether the nemo-skills python tool server
(`_start_python_tool_server()`, a separate process) is up and reachable.

---

## 4. ISSUE: tool-call format mismatch  [SUPERSEDED -- stale checkpoint, see banner]

The SFT format is genuinely ambiguous and varies by domain in `Nemotron-Cascade-2-SFT-Data`:
  - `conversational_agent`: `<tool_call><function=NAME><parameter=KEY>` XML (7/7 turns)
  - `terminal_agent`: no tags at all, a JSON response schema instructed in the user turn
So "OpenAI format" as reported by the SFT owner describes the **messages container**
(`{role, content}`), not how a call is written inside `content`.

Measured on the v30 policy: it emits bare JSON after `</think>`, wrapper omitted.
On `search_pivot` under the JSON-wrapped template: 78% emitted `{"name":...}`, only
31% included `<tool_call>`; 69 samples had body-without-wrapper, 27 wrapper-without-body.

Parsers (all in the vllm fork, `vllm/tool_parsers/`):

| parser | reads | wrapper | trailing prose |
|---|---|---|---|
| `step3p5` | XML via expat, streaming | required | n/a |
| `hermes` | JSON, `json.loads` whole content | REQUIRED (bails if absent) | fatal ("Extra data") |
| `llama3_json` | JSON, brace scan + `raw_decode` | optional | ignored |

Format matrix, `search_pivot`, 384 samples each, thinking budget 2048, scored on reward:

| cfg | template | parser | meanRew | callJSON | wrapper | XML |
|---|---|---|---|---|---|---|
| D | bare JSON | `llama3_json` | **0.036** | 79% | 20% | 0% |
| C | JSON in `<tool_call>` | `llama3_json` | 0.018 | 76% | 30% | 0% |
| E | XML + explicit error reminders | `step3p5` | 0.008 | 20% | 64% | 12% |
| A | XML (checkpoint default) | `step3p5` | 0.000 | 18% | 61% | 11% |
| B | JSON in `<tool_call>` | `hermes` | 0.000 | 78% | 31% | 0% |

B is the proof that the wrapper requirement is decisive: 78% emission, 0% scored.

INFERRED, NOT VERIFIED: the JSON templates still render the tool *declaration* in XML
(`<tools><function><name>...`). Only the call sites were changed. That inconsistency may
explain residual wrapper leakage (20-31% even when told "do NOT wrap"). Untested.

---

## 5. Infrastructure bugs hit along the way  [ALL FIXED]

**5a. `chat_template` given as a path became the template.**
`http_server_serving_chat_kwargs.chat_template` takes template *content*. vLLM's own
entrypoints resolve path-or-content via `load_chat_template()`
(`openai/api_server.py:395`, `entrypoints/llm.py:363`); NeMo-RL passed the raw string
through. A path has no jinja syntax, so it rendered to itself: every prompt collapsed to
the path followed by `<|im_end|>` padding, with no `<|im_start|>` at all. The run
trained happily on garbage for 3 steps and reported COMPLETED.
Fix: call `load_chat_template` in `nemo_rl/models/generation/vllm/vllm_worker_async.py`.

**5b. `tool_parser: llama` is not a registered name.** Registry names are `llama3_json`
and `llama4_json` (both -> `Llama3JsonToolParser`). Wrong name = hard failure at engine
init: `--enable-auto-tool-choice requires tool_parser:'llama' which has not been registered`.

**5c. `llama3_json` refused to initialise.** Its constructor raised because
`<|python_tag|>` is absent from the Nemotron vocab -- even though `extract_tool_calls`
already treats the token as optional (`if not (self.bot_token in model_output or "{" in
model_output)`). Fix: warn and fall back to brace-only detection.

**5d. `thinking_token_budget` rejected on every request.** It gates on the ENGINE-level
`vllm_config.reasoning_config.enabled`, built from `EngineArgs.reasoning_parser`
(`arg_utils.py:_set_default_reasoning_config_args`). Setting
`http_server_serving_chat_kwargs.reasoning_parser` configures only the SERVING layer.
Symptom: 410 `VLLMValidationError`, 78,850 `EngineGenerateError`, zero steps, 20+ min of
wall clock per job. Fix: also set `policy.generation.vllm_kwargs.reasoning_parser`, which
is splatted into engine kwargs at `vllm_worker.py:587`.

**5e. Slurm reports COMPLETED when the driver dies.** Every failure mode above exited 0.
Judge by elapsed time and by `grep -c "Total step time"`, never by job state.

---

## 6. ISSUE: unfinished thinking  [FIXED by the budget]

Without a budget (`nd8b_ar_v30_10t6g`, step 40, n=1920):

    completed  845 (44%) | empty 389 (20%) | UNFINISHED 686 (36%)

"Unfinished" = no `</think>` anywhere in the generated tokens, i.e. the model spent its
whole generation budget reasoning and never produced an answer. Per agent:

    math_with_judge 86% | code_gen 80% | instruction_following 72% | mcqa 32%
    structured_outputs 25% | ns_tools 13% | search_pivot 10% | reasoning_gym 4%
    workplace_assistant 3% | single_step_tool_use 1% | terminal_pivot 1% | calendar 0%

That is very likely what drives `math_with_judge` (reward 0.163 -> 0.125 over steps
10-50) and `code_gen` (negative reward): most samples never emit an answer at all.

With `thinking_token_budget: 2048`: 4-5% on `search_pivot` (whose own baseline was 10%,
so the honest like-for-like is 10% -> 4%, not 36% -> 4%).

Note the reasoning channel was never disabled -- the template defaults
`enable_thinking = True` and opens `<think>` in the prompt, so generations begin with a
bare `</think>`. What was missing was the budget, not the channel.

---

## 7. ISSUE: `structured_outputs` scores a hard 0.000  [FIXED, unverified in a run]

144/144 (AR) and 120/120 (BJG-Fast) samples scored exactly 0. Not difficulty:

    resources_servers/structured_outputs/app.py
      parse_content: json.loads(content)        # RAW response text, no stripping
      evaluate_structured_output_response: strictify_schema -> required = all properties,
                                           additionalProperties = False

Every generation begins with `</think>` (see 6), so `json.loads` dies on character 0
before the schema is consulted. A sample producing clean, correct JSON still scored 0.

Fix: `reasoning_parser: step3p5` in the shared v30 parent (committed, `08c6e14ed`), so
the reasoning block is separated from content before grading.

Scope: this only affects graders that parse the WHOLE response strictly.
`math_with_judge` (math_verify LaTeX extraction) and `code_gen` (`extract_code`) scan for
a target and tolerate a prefix. `instruction_following` is also at 0.000 in BJG-Fast and
has NOT been checked for the same pattern.

---

## 8. Open items

- **Tool execution fails after routing is fixed** (see 3). Highest priority.
- **Turn truncation.** In multi-segment `ns_tools` samples both trainable segments were
  exactly 4096 tokens. 2048 thinking + 4096 `max_new_tokens` does not fit a two-turn
  trajectory. Raising `max_new_tokens` is untested.
- **`tau_pivot` has never been run with the corrected format.** It is 23.1% of the
  corpus and expects a tool call in 66% of turns (265/400 sampled `expected_action` are
  `{type,name,arguments}`; the other 135 are `{type: message}` -- so its 2% attempt rate
  is a shortfall against 66%, not against 100%).
- **XML tool declaration inside JSON templates** (see 4).
- **`instruction_following` at 0.000** -- unchecked for the strict-parse pattern.

---

## 9. HAZARD: these fixes are dangerous on the diffusion path

`multiturn_diffusion_support.md` section 7 warns that AR-correct tool fixes "manufacture
exactly the multi-span shape that kills the run" on diffusion --
`_validate_single_completion_span` raises when scored positions are not one contiguous
run, and that killed 5 of 11 `nd8b_bjgfast_v30` legs.

Two of the fixes here are **global**, not per-config:
  - the `ns_tools/app.py` alias (Gym source), and
  - `reasoning_parser: step3p5` in the shared v30 parent.
The alias makes tool calls actually succeed, which produces a second turn. A diffusion
run touching `ns_tools` would now get multi-span samples it cannot validate.
Land the diffusion stopgap before running BJG-Fast against a patched `ns_tools`.
(`reasoning_parser` alone is safe -- it does not create extra turns.)

---

## 10. Where things live

    configs        examples/configs/ar_v30_sp_matrix_{A..E}.yaml     search_pivot format matrix
                   examples/configs/ar_v30_ns_matrix_{F,G,H,I}.yaml  ns_tools variants
                   examples/configs/ar_v30_sp_think2k_base.yaml      shared base
    templates      examples/configs/chat_templates/*.jinja
    subsets        /lustre/.../v30_deterministic/{search_pivot_only,ns_tools_only}/
    runs           /lustre/.../runs/diffusion_rl/ar_v30_{spx_*,ns_*}/
    scorers        ~/score_matrix.py, ~/score_reward.py

Uncommitted code changes: `nemo_rl/models/generation/vllm/vllm_worker_async.py`,
`nemo_rl/models/megatron/train.py` (cp_group, inert at CP=1),
fork `vllm/tool_parsers/llama_tool_parser.py`, Gym `resources_servers/ns_tools/app.py`.
