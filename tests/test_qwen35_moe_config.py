from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_moe_full_smoke_disables_adamw_foreach_for_finite_updates():
    config = yaml.safe_load((ROOT / "configs/qwen35_35b_moe_ep4.yaml").read_text())
    assert config["optimizer"]["foreach"] is False


def test_moe_smoke_uses_reference_scale_distributed_timeout():
    for name in (
        'qwen35_35b_moe_ep4.yaml',
    ):
        config = yaml.safe_load((ROOT / 'configs' / name).resolve().read_text())
        assert config['dist_env']['timeout_minutes'] == 60


def test_moe_packed_smoke_config_declares_ep4_and_vlm_packing():
    config_path = ROOT / "configs/qwen35_35b_moe_ep4.yaml"
    assert config_path.is_file(), "MoE packed smoke config must be checked in before submission"
    config = yaml.safe_load(config_path.read_text())
    assert config["distributed"]["ep_size"] == 4
    assert config["model"]["backend"]["experts"] == "torch_mm"
    assert config["model"]["backend"]["dispatcher"] == "torch"
    assert config["optimizer"]["foreach"] is False
    assert config["packed_sequence"]["pack_size"] == 512
    assert config["packed_sequence"]["packing_format"] == "neat"
    assert config["packed_sequence"]["pretokenize"] is True


def test_moe_all_multimodal_smoke_config_unfreezes_all_towers():
    config_path = ROOT / "configs/qwen35_35b_moe_ep4.yaml"
    assert config_path.is_file(), "All-multimodal MoE smoke config must be checked in before submission"
    config = yaml.safe_load(config_path.read_text())
    freeze = config["freeze_config"]
    assert freeze["freeze_vision_tower"] is False
    assert freeze["freeze_audio_tower"] is False
    assert freeze["freeze_language_model"] is False
    assert config["distributed"]["ep_size"] == 4
    assert config["model"]["backend"]["experts"] == "torch_mm"
    assert config["optimizer"]["foreach"] is False
    assert config["step_scheduler"]["max_steps"] == 1
    assert config["smoke_audit"]["enabled"] is True
    assert config["smoke_audit"]["require_finite_gradient"] is True
    assert config["smoke_audit"]["require_parameter_update"] is True
