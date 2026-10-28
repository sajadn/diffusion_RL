"""Custom LLaDA validation owns greedy decoding, not the upstream Gym fields."""

from pathlib import Path

import pytest

from just_grpo.config import validate_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@pytest.mark.parametrize(
    "dataset,smoke,auxiliary",
    [
        (dataset, smoke, auxiliary)
        for dataset in ("gsm8k", "math", "acecode")
        for smoke in (False, True)
        for auxiliary in (False, True)
        # GSM8K has no separate auxiliary vLLM smoke recipe.
        if not (dataset == "gsm8k" and smoke and auxiliary)
    ],
)
def test_vllm_recipes_keep_greedy_validation_in_custom_adapter(
    monkeypatch, dataset, smoke, auxiliary
):
    register_omegaconf_resolvers()
    for variable in (
        "LLADA_MATH_DATA_DIR",
        "LLADA_CODE_DATA_DIR",
        "JUSTGRPO_REFERENCE_DIR",
    ):
        monkeypatch.setenv(variable, "/data")
    monkeypatch.setenv("POLICY_PYTHON", "/usr/bin/python3")
    size, stage = ("2n8g", "smoke") if smoke else ("16n8g", "long")
    stem = (
        f"just_grpo-acecode-llada8b-{size}-vllm-humaneval-{stage}"
        if dataset == "acecode"
        else f"just_grpo-{dataset}-llada8b-canvas256-{size}-vllm-{stage}"
    )
    suffix = "-diffaux" if auxiliary else ""
    recipe = Path(__file__).parents[2] / "configs/recipes" / f"{stem}{suffix}.yaml"
    config = load_config(recipe)
    diffusion = validate_config(config)
    if dataset in ("math", "acecode") and not smoke:
        assert config.grpo.max_num_steps == 300
    assert diffusion.runtime == "reference_vllm"
    assert diffusion.validation_runtime == "megatron_llada"
    assert (
        config.policy.generation.val_temperature
        == config.policy.generation.temperature
        == 1
    )
    assert diffusion.sampling.temperature == 1
    assert diffusion.sampling.selection_policy == "leftmost"
    assert diffusion.validation_sampling.temperature == 0
    assert len(diffusion.validation_modes) == 4
    assert all(mode.temperature == 0 for mode in diffusion.validation_modes.values())
