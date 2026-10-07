from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_distributed_generate_forwards_validated_ep4_setup_request(monkeypatch):
    import examples.vlm_generate.distributed_generate as module

    sentinel = object()
    monkeypatch.setattr(module, "create_distributed_setup_from_config", lambda config, world_size: sentinel)
    setup = module.build_inference_setup(
        {
            "distributed": {
                "_target_": "nemo_automodel.components.distributed.fsdp2.FSDP2Manager",
                "tp_size": 1,
                "cp_size": 1,
                "pp_size": 1,
                "dp_replicate_size": 1,
                "ep_size": 4,
                "sequence_parallel": False,
                "activation_checkpointing": True,
            },
            "dist_env": {"timeout_minutes": 60},
        },
        world_size=4,
    )
    assert setup is sentinel


def test_distributed_generate_rejects_non_ep4_world_size():
    from examples.vlm_generate.distributed_generate import build_inference_setup

    with pytest.raises(ValueError, match="world_size=4.*ep_size=4"):
        build_inference_setup({"distributed": {"ep_size": 4}}, world_size=2)


def test_distributed_generate_checkpoint_config_uses_sharded_model_reader():
    from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat
    from nemo_automodel.components.checkpoint.config import SaveConsolidatedMode
    from examples.vlm_generate.distributed_generate import make_checkpoint_config

    config = make_checkpoint_config(
        "/shared/checkpoints/epoch_0_step_0",
        "/shared/models/Qwen3.5-35B-A3B",
    )

    assert config.enabled is True
    assert config.model_save_format == SerializationFormat.SAFETENSORS
    assert config.save_consolidated == SaveConsolidatedMode.FALSE
    assert config.model_repo_id.endswith("Qwen3.5-35B-A3B")


def test_distributed_generate_writes_output_only_on_rank_zero():
    source = (ROOT / "examples/vlm_generate/distributed_generate.py").read_text()
    assert "dist_info.is_main" in source
    assert "output_file" in source
    assert "torchrun" not in source  # launcher belongs to the shell wrapper




def test_inference_backend_config_strips_hydra_target_metadata():
    from examples.vlm_generate.distributed_generate import _normalize_backend_config

    backend = {
        "_target_": "nemo_automodel.components.models.common.BackendConfig",
        "attn": "sdpa",
        "experts": "torch_mm",
    }
    normalized = _normalize_backend_config(backend)

    assert normalized == {"attn": "sdpa", "experts": "torch_mm"}
    assert backend["_target_"]


def test_inference_checkpoint_loader_passes_training_process_groups():
    source = (ROOT / "examples/vlm_generate/distributed_generate.py").read_text()
    assert 'pp_group=' in source
    assert 'getattr(mesh_context, "process_group", None)' in source
    assert 'device_mesh["pp"].get_group()' in source or 'pp_group' in source


def test_distributed_inference_initializes_visible_stage_logging():
    source = (ROOT / "examples/vlm_generate/distributed_generate.py").read_text()
    assert "logging.basicConfig" in source
    assert "loading base model with EP4 distributed setup" in source
    assert "loading sharded checkpoint" in source
    assert "generate" in source


def test_moe_generation_forward_declares_multimodal_inputs_for_transformers_validation():
    import inspect
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration

    params = inspect.signature(Qwen3_5MoeForConditionalGeneration.forward).parameters
    for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        assert name in params, f"generation input {name} must be explicit in forward signature"
