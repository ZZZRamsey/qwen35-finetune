"""Regression coverage for the pruned public source dependency closure."""


def test_parallel_plans_import_without_pruned_model_packages():
    from nemo_automodel.components.distributed.optimized_tp_plans import (
        PARALLELIZE_FUNCTIONS,
    )

    dense = "nemo_automodel.components.models.qwen3_5.model."
    assert callable(PARALLELIZE_FUNCTIONS[dense + "Qwen3_5ForConditionalGeneration"])
    assert callable(PARALLELIZE_FUNCTIONS[dense + "Qwen3_5ForCausalLM"])
    removed = ("baichuan", "llama", "mistral3_vlm", "qwen2", "qwen3")
    for name in removed:
        prefix = f"nemo_automodel.components.models.{name}."
        assert not any(key.startswith(prefix) for key in PARALLELIZE_FUNCTIONS)


def test_registry_exposes_config_resolution_hook_for_model_init():
    from nemo_automodel._transformers.registry import resolve_custom_config_cls

    assert resolve_custom_config_cls("qwen3_5") is None
    assert resolve_custom_config_cls("qwen3_5_moe") is None


def test_pruned_mistral_ai_model_is_not_imported_by_public_tp_plans():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "nemo_automodel/components/distributed/optimized_tp_plans.py"
    ).read_text()
    assert "transformers.models.mistral3" not in source
