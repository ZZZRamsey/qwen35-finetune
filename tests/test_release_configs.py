from pathlib import Path
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("name,model,ep", [
    ("qwen35_4b_full", "Qwen/Qwen3.5-4B", 1),
    ("qwen35_35b_moe_ep4", "Qwen/Qwen3.5-35B-A3B", 4),
])
def test_full_parameter_config_is_portable_and_topology_matches(name, model, ep):
    path = ROOT / "configs" / f"{name}.yaml"
    assert path.is_file(), "portable full-parameter config is missing"
    text = path.read_text()
    cfg = yaml.safe_load(text)
    assert cfg["model"]["pretrained_model_name_or_path"] == model
    assert cfg["processor"]["pretrained_model_name_or_path"] == model
    assert not cfg.get("peft")
    assert all(value is False for value in cfg["freeze_config"].values())
    assert cfg["distributed"]["ep_size"] == ep
    assert all(cfg["distributed"][key] == 1 for key in ("tp_size", "cp_size", "pp_size"))
    assert cfg["checkpoint"]["save_consolidated"] is False
    assert cfg["smoke_audit"]["require_parameter_update"] is True
    assert "/data/" not in text and "/Users/" not in text
    if ep == 4:
        assert cfg["model"]["backend"]["experts"] == "torch_mm"
        assert cfg["model"]["backend"]["dispatcher"] == "torch"
        assert cfg["optimizer"]["foreach"] is False
        pack = cfg["packed_sequence"]
        assert pack["pack_size"] == pack["max_length"] == 512
        assert pack["packing_format"] == "neat"
        assert pack["drop_long_samples"] is False
