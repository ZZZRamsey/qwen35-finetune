from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_4b_multinode_config_is_full_parameter_and_world_size_8_compatible():
    path = ROOT / "configs/qwen35_4b_full.yaml"
    config = yaml.safe_load(path.read_text())
    assert "peft" not in config
    assert config["step_scheduler"]["local_batch_size"] == 1
    assert config["step_scheduler"]["global_batch_size"] == 8
    assert config["step_scheduler"]["global_batch_size"] % 8 == 0
    assert config["step_scheduler"]["max_steps"] == 1
    assert config["freeze_config"]["freeze_vision_tower"] is False
    assert config["freeze_config"]["freeze_language_model"] is False
    assert "QWEN_CHECKPOINT_DIR" in config["checkpoint"]["checkpoint_dir"]


def test_4b_multinode_inference_entry_only_writes_on_rank_zero():
    path = ROOT / "examples/vlm_generate/multinode_generate.py"
    source = path.read_text()
    assert "get_rank()" in source
    assert "if rank == 0" in source
    assert "output_file" in source
    assert "dist.barrier()" in source


def test_4b_multinode_inference_loads_sharded_model_on_every_rank():
    path = ROOT / "examples/vlm_generate/multinode_generate.py"
    source = path.read_text()
    assert "create_distributed_setup_from_config" in source
    assert "_load_model_and_checkpoint" in source
    assert "--config" in source
    assert "load_model_from_checkpoint" not in source
    assert "rank == 0" in source
    assert "dist.barrier()" in source
