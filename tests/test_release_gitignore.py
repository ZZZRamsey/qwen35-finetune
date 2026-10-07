"""Runtime ignore rules must not hide vendored model implementation files."""
from pathlib import Path
import subprocess


def test_model_source_is_not_ignored(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=tmp_path).returncode == 0
    assert not ignored("nemo_automodel/components/models/qwen3_5/model.py")
    assert not ignored("nemo_automodel/components/models/qwen3_5_moe/model.py")
    assert ignored("models/Qwen3.5-4B/config.json")
    assert ignored("runs/smoke/checkpoint.txt")
