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
"""Validate the shared diffusion training and generation configuration."""

from omegaconf import DictConfig, OmegaConf

from block_diffusion.config import DiffusionExperimentConfig
from block_diffusion.generation.validation import validation_variants
from block_diffusion.positive_diffusion import resolve_positive_diffusion_config


def validate_experiment(
    config: DictConfig,
    diffusion: DiffusionExperimentConfig,
    *,
    require_policy_scores: bool = True,
) -> None:
    """Validate the supported research subset of the upstream GRPO schema."""
    variants = validation_variants(
        OmegaConf.to_container(config.policy.generation, resolve=True)
    )
    if variants and config.policy.generation.refit_transport is not None:
        raise ValueError("Multi-mode validation requires native IPC/NCCL refit")
    if variants and diffusion.runtime != "reference_vllm":
        raise ValueError("Multi-mode validation requires reference_vllm generation")
    g, p, loss = config.grpo, config.policy, config.loss_fn
    if p.get("diffusion_aux_loss") is not None:
        resolve_positive_diffusion_config(
            OmegaConf.to_container(p.diffusion_aux_loss, resolve=True),
            mask_token_id=diffusion.schedule.mask_token_id,
            block_size=diffusion.schedule.block_size,
        )
    if require_policy_scores and loss.force_on_policy_ratio:
        raise ValueError(
            "Diffusion GRPO requires recomputed prev_logprobs; force_on_policy_ratio must be false"
        )
    if (
        min(
            g.max_num_steps,
            g.num_prompts_per_step,
            g.max_val_samples,
            g.val_batch_size,
            p.train_micro_batch_size,
            p.logprob_batch_size,
            p.generation.max_new_tokens,
        )
        <= 0
        or g.num_generations_per_prompt < 2
    ):
        raise ValueError(
            "Require positive batch/step sizes and at least two generations per prompt"
        )
    if (
        p.train_global_batch_size
        != g.num_prompts_per_step * g.num_generations_per_prompt
    ):
        raise ValueError(
            "The research driver requires one global optimizer update per rollout group"
        )
    if p.optimizer.name != "torch.optim.AdamW" or p.optimizer.kwargs.lr <= 0:
        raise ValueError(
            "The research driver supports AdamW with positive learning rate"
        )
    if p.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if (
        p.dtensor_cfg.enabled
        or p.sequence_packing.enabled
        or p.dynamic_batching.enabled
    ):
        raise ValueError(
            "DTensor, packing, and dynamic batching are not supported by this research driver"
        )
    if not p.megatron_cfg.enabled:
        raise ValueError("Diffusion GRPO requires the Megatron training backend")
    m = p.megatron_cfg
    world = config.cluster.num_nodes * config.cluster.gpus_per_node
    if not p.generation.colocated.enabled:
        if not g.async_grpo.enabled or diffusion.runtime != "reference_vllm":
            raise ValueError("Dedicated generation requires async reference vLLM")
        resources = p.generation.colocated.resources
        if config.cluster.num_nodes == 1:
            if resources.num_nodes not in (None, 1):
                raise ValueError("Single-node inference must use one node")
            if (
                resources.gpus_per_node is None
                or not 0 < resources.gpus_per_node < world
            ):
                raise ValueError("Dedicated inference needs a positive GPU subset")
            world -= resources.gpus_per_node
        else:
            if (
                resources.num_nodes is None
                or not 0 < resources.num_nodes < config.cluster.num_nodes
            ):
                raise ValueError("Dedicated inference needs a positive node subset")
            if resources.gpus_per_node != config.cluster.gpus_per_node:
                raise ValueError("Dedicated inference must use complete nodes")
            world -= resources.num_nodes * resources.gpus_per_node
    tp = m.tensor_model_parallel_size
    if not diffusion.distributed or tp < 1 or world % tp:
        raise ValueError(
            "Megatron requires distributed training and a TP size dividing the GPU count"
        )
    if (
        m.pipeline_model_parallel_size != 1
        or m.context_parallel_size != 1
        or m.expert_model_parallel_size != 1
        or m.sequence_parallel
        or m.peft.enabled
        or m.fp8_cfg.enabled
        or m.fp4_cfg.enabled
        or m.model_overrides
        or m.use_fused_linear_logprobs
    ):
        raise ValueError(
            "Megatron Diffusion GRPO supports dense BF16 DP/TP with PP=CP=EP=1, without sequence parallelism, PEFT, quantization, or model overrides"
        )
    if (
        m.distributed_data_parallel_config.overlap_grad_reduce
        or m.distributed_data_parallel_config.overlap_param_gather
    ):
        raise ValueError(
            "Megatron Diffusion GRPO requires synchronous gradient and parameter collectives"
        )
    if m.activation_checkpointing and m.recompute_granularity != "full":
        raise ValueError(
            "Megatron Diffusion GRPO currently requires full activation recomputation"
        )
    if m.optimizer.optimizer != "adam" or m.optimizer.use_precision_aware_optimizer:
        raise ValueError("Megatron Diffusion GRPO requires the standard Adam optimizer")
    if (
        m.optimizer.lr != p.optimizer.kwargs.lr
        or m.optimizer.weight_decay != p.optimizer.kwargs.weight_decay
    ):
        raise ValueError(
            "Megatron and policy optimizer learning rate/weight decay must agree"
        )
    if p.max_total_sequence_length % diffusion.schedule.block_size:
        raise ValueError(
            "Megatron context length must align to the diffusion block size"
        )
    local_batch = p.train_global_batch_size // (world // tp)
    if local_batch % p.train_micro_batch_size or local_batch % p.logprob_batch_size:
        raise ValueError(
            "Megatron requires complete training and logprob microbatches per DP replica"
        )
    if p.precision != "bfloat16" or p.dtensor_cfg.lora_cfg.enabled or p.draft.enabled:
        raise ValueError(
            "The research driver requires a full-parameter bf16-compute policy"
        )
    if g.use_dynamic_sampling or g.overlong_filtering or g.reward_shaping.enabled:
        raise ValueError(
            "Dynamic sampling, overlong filtering, and reward shaping are unsupported"
        )
    if g.async_grpo.enabled:
        if diffusion.runtime == "reference_vllm" and p.generation.colocated.enabled:
            raise ValueError("Async vLLM requires dedicated generation GPUs")
        if g.async_grpo.in_flight_weight_updates:
            raise ValueError("Shared-policy inference must drain before weight updates")
        if (
            diffusion.runtime == "megatron"
            and g.async_grpo.recompute_kv_cache_after_weight_updates
        ):
            raise ValueError(
                "Uncached diffusion inference has no KV cache to recompute"
            )
        if not loss.use_importance_sampling_correction:
            raise ValueError("Async GRPO requires importance sampling correction")
        if g.reward_scaling.enabled or (config.get("data_plane") or {}).get(
            "enabled", False
        ):
            raise ValueError(
                "Async Diffusion GRPO does not support reward scaling or the data plane"
            )
    if g.max_rollout_turns != 1 or g.val_num_generations_per_prompt != 1:
        raise ValueError(
            "The research driver supports single-turn fixed-step runs with one validation generation per prompt"
        )
    if g.val_period != -1 and g.val_period <= 0:
        raise ValueError("val_period must be -1 (disabled) or a positive step interval")
    if g.adv_estimator.name != "grpo" or g.stop_at_validation_metric is not None:
        raise ValueError(
            "The research driver supports GRPO advantages without early stopping"
        )
    if (
        loss.use_kl_in_reward
        or not loss.token_level_loss
        or loss.sequence_level_importance_ratios
        or loss.positive_example_nll_weight != 0
    ):
        raise ValueError(
            "The research worker requires token-level loss without reward-side KL, sequence ratios, or auxiliary NLL"
        )
    if loss.truncated_importance_sampling_type == "seq-mask-tis":
        raise ValueError(
            "seq-mask-tis requires whole-response filtering and is unsupported by per-level diffusion training"
        )
    if config.data.default.env_name == "sudoku6x6":
        for split in (config.data.train, config.data.validation):
            if split.size <= 0 or split.repeat <= 0:
                raise ValueError("Dataset size and repeat must be positive")
        if config.data.train.size * config.data.train.repeat < g.num_prompts_per_step:
            raise ValueError("Training dataset must contain a full prompt batch")
        if config.data.default.prompt_style not in ("question", "sudoku_answer_tag"):
            raise ValueError("Unsupported Sudoku prompt style")
        if (
            any(
                split.dataset_name != "sudoku6x6"
                for split in (config.data.train, config.data.validation)
            )
            or "sudoku6x6" not in config.env
        ):
            raise ValueError(
                "Training, validation, and environment must all select sudoku6x6"
            )
    elif config.data.default.env_name == "math":
        if (
            config.data.default.processor != "math_hf_data_processor"
            or config.data.validation is None
            or "math" not in config.env
        ):
            raise ValueError(
                "Math requires the native processor and an explicit validation split"
            )
    elif config.data.default.env_name == "justgrpo_code":
        if (
            diffusion.runtime != "megatron_llada"
            or config.data.default.processor != "justgrpo_code"
            or config.data.validation is None
            or "justgrpo_code" not in config.env
        ):
            raise ValueError(
                "Code requires LLaDA, the reference processor, and explicit validation"
            )
    else:
        raise ValueError("Supported environments are sudoku6x6, math, and justgrpo_code")
    if g.max_num_epochs <= 0:
        raise ValueError("max_num_epochs must be positive")
    world_size = world
    if diffusion.distributed:
        data_parallel_size = world_size // p.megatron_cfg.tensor_model_parallel_size
        if world_size < 1:
            raise ValueError("Distributed training requires at least one policy GPU")
        if diffusion.runtime == "megatron_llada":
            if p.train_global_batch_size % data_parallel_size:
                raise ValueError(
                    "LLaDA requires complete completion shards per DP rank"
                )
        elif g.num_prompts_per_step % data_parallel_size:
            raise ValueError(
                "Distributed runs require a whole prompt group per data-parallel rank"
            )
        if g.max_val_samples < data_parallel_size:
            raise ValueError(
                "Validation needs at least one puzzle per data-parallel rank"
            )
    if (
        diffusion.runtime != "megatron_llada"
        and diffusion.validation_sampling != diffusion.sampling
    ):
        raise ValueError("Validation must use the training sampling settings")
    if g.max_val_samples < g.val_batch_size:
        raise ValueError("max_val_samples must include at least one val_batch_size")
    if diffusion.runtime == "megatron_llada":
        if g.async_grpo.enabled or not p.generation.colocated.enabled:
            raise ValueError(
                "LLaDA shared workers require synchronous colocated generation"
            )
        if p.worker_extension_cls_fqn != (
            "just_grpo.algorithms.llada_policy_worker.LLaDAJustGRPOPolicyWorker"
        ):
            raise ValueError("LLaDA requires the full-canvas policy worker")
        if (
            p.generation.stop_token_ids
            or p.generation.max_new_tokens != diffusion.schedule.block_size
        ):
            raise ValueError(
                "LLaDA must generate the complete canvas without early stopping"
            )
    expected_backend = (
        "megatron" if diffusion.runtime in ("megatron", "megatron_llada") else "vllm"
    )
    if p.generation.backend != expected_backend:
        raise ValueError("policy.generation.backend must match the diffusion runtime")
    if diffusion.runtime in ("megatron", "megatron_llada"):
        if tp != 1:
            raise ValueError(
                "In-process Megatron diffusion generation currently requires TP=1"
            )
        if diffusion.generation_python is not None:
            raise ValueError(
                "Megatron diffusion generation uses the policy interpreter"
            )
        if p.generation.mcore_generation_config.expose_http_server:
            raise ValueError(
                "Research Megatron inference requires HTTP serving disabled"
            )
        if p.generation.refit_transport != "mcore":
            raise ValueError("Shared Megatron inference requires native mcore refit")
        if p.generation.mcore_generation_config.cuda_graph_impl != "none":
            raise ValueError(
                "Uncached Megatron diffusion generation requires cuda_graph_impl=none"
            )
    else:
        worker_name = (
            "ReferenceVllmAsyncWorker"
            if g.async_grpo.enabled
            else "ReferenceVllmWorker"
        )
        if p.generation.worker_extension_cls_fqn != (
            f"block_diffusion.generation.reference_vllm.{worker_name}"
        ):
            raise ValueError("Reference vLLM requires its diffusion worker extension")
        params = diffusion.sampling
        expected_diffusion = {
            "canvas_length": params.block_size,
            "max_denoising_steps": params.max_steps,
            "temperature": params.temperature,
            "selection_policy": params.selection_policy,
            "confidence_threshold": params.threshold,
            "return_entropy": params.returns_entropy,
            "return_reveal_steps": params.returns_reveal_steps,
        }
        if params.selection_policy == "entropy_budget":
            expected_diffusion["entropy_bound"] = params.entropy_bound
        if params.emit_full_blocks is not None:
            expected_diffusion["emit_full_blocks"] = params.emit_full_blocks
        if (
            OmegaConf.to_container(
                p.generation.vllm_kwargs.diffusion_config, resolve=True
            )
            != expected_diffusion
        ):
            raise ValueError("vLLM diffusion_config must match algorithm sampling")
        vllm = p.generation.vllm_cfg
        if (
            vllm.tensor_parallel_size != 1
            or vllm.pipeline_parallel_size != 1
            or vllm.expert_parallel_size != 1
        ):
            raise ValueError("Reference vLLM requires single-GPU engine replicas")
        if bool(vllm.async_engine) != bool(g.async_grpo.enabled):
            raise ValueError("vLLM async_engine must match async GRPO")
        if vllm.get("expose_http_server", False):
            raise ValueError(
                "Reference diffusion generation does not expose HTTP serving"
            )
        if vllm.precision != "bfloat16" or vllm.logprobs_mode != "processed_logprobs":
            raise ValueError(
                "Reference generation requires bf16 and processed at-unmask logprobs"
            )
    if (
        p.generation.top_p != 1
        or p.generation.val_top_p != 1
        or p.generation.top_k not in (None, -1)
        or p.generation.val_top_k not in (None, -1)
        or p.generation.stop_strings is not None
    ):
        raise ValueError(
            "Diffusion generation supports unfiltered sampling with model EOS only"
        )
    for params in (diffusion.sampling, diffusion.validation_sampling):
        if p.generation.max_new_tokens % params.block_size:
            raise ValueError("max_new_tokens must be a multiple of each canvas size")
    if (
        diffusion.sampling.temperature != p.generation.temperature
        or diffusion.validation_sampling.temperature != p.generation.val_temperature
    ):
        raise ValueError("Diffusion temperatures must match policy.generation settings")
