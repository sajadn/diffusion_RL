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
"""Validate confidence experiment configuration and use the established runtime."""

import argparse
import os
from pathlib import Path
import runpy
import sys

repo = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(repo), str(repo / "tools/nemotron_diffusion")]
from omegaconf import OmegaConf
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.algorithms.confidence_transition import validate_correction_config
from nemo_rl.utils.checkpoint import CheckpointManager

register_omegaconf_resolvers()
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args, overrides = parser.parse_known_args()
records = (
    Path(os.environ["NRL_CONFIDENCE_EXPERIMENT_DIR"])
    if os.environ.get("NRL_CONFIDENCE_EXPERIMENT_DIR")
    else Path("/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl")
    / os.environ["RUN_NAME"]
    / "confidence_records"
)
records.mkdir(parents=True, exist_ok=True)
os.environ["NRL_CONFIDENCE_EXPERIMENT_DIR"] = str(records)
worker_path = f"{repo}/tools/nemotron_diffusion:/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_runtime_patches:{repo}"
extra = [
    f"policy.megatron_cfg.env_vars.PYTHONPATH={worker_path}",
    f"++policy.megatron_cfg.env_vars.NRL_CONFIDENCE_EXPERIMENT_DIR={records}",
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS='1'",
]
sys.argv.extend(extra)
c = OmegaConf.to_container(
    parse_hydra_overrides(load_config(args.config), overrides + extra), resolve=True
)
validate_correction_config(c["grpo"], c["policy"], c["loss_fn"])
experiment = c["policy"]["logprob_estimation"]["confidence_experiment"]
assert experiment["mode"] in ("filter", "importance")
estimator = c["policy"]["logprob_estimation"]
assert (
    estimator["mask_token_id"] == 100
    and estimator["block_size"] == 16
    and estimator["confidence_threshold"] == 0.9
)
assert c["policy"]["generation"]["temperature"] == 1
assert (
    float(experiment.get("filter_bound", 4)) > 1
    and float(experiment.get("importance_clip", 2)) >= 1
)
assert c["grpo"]["seq_logprob_error_threshold"] is None
assert c["grpo"]["use_dynamic_sampling"] is False
if experiment.get("start_checkpoint"):
    weights = Path(experiment["start_checkpoint"]) / "policy/weights"
    assert (weights / "iter_0000000/common.pt").is_file()
    original = CheckpointManager.get_resume_paths
    CheckpointManager.get_resume_paths = staticmethod(
        lambda checkpoint: (weights, None)
        if checkpoint is None
        else original(checkpoint)
    )
    print(f"Loading diagnostic weights only: {weights}", flush=True)
for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
    os.environ.pop(name, None)
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
print(
    f"CONFIDENCE EXPERIMENT mode={experiment['mode']} nodes={c['cluster']['num_nodes']} batch={c['policy']['train_global_batch_size']} lr={c['policy']['megatron_cfg']['optimizer']['lr']} records={records}",
    flush=True,
)
if os.environ.get("EXPERIMENT_CHECK_ONLY") == "1":
    raise SystemExit(0)
os.chdir(repo)
runpy.run_path(str(repo / "examples/run_grpo.py"), run_name="__main__")
