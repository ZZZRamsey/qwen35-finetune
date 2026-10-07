"""Capacity gate must fail before expensive model loading, without writing files."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/storage_preflight.py"

def run_gate(minimum):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), str(ROOT), str(minimum)], capture_output=True, text=True)

def test_rejects_insufficient_storage():
    result = run_gate(10**12)
    assert result.returncode == 28
    assert "STORAGE_PREFLIGHT_FAILED" in result.stderr

def test_reports_available_storage():
    result = run_gate(0)
    assert result.returncode == 0
    assert "STORAGE_PREFLIGHT_OK" in result.stdout

def test_rejects_nonfinite_or_negative_threshold():
    for value in ("nan", "inf", "-1", "invalid"):
        result = run_gate(value)
        assert result.returncode == 2
        assert "invalid minimum" in result.stderr

def test_bootstrap_checks_capacity_before_launch(tmp_path):
    import os
    model = tmp_path / "model"
    model.mkdir()
    runs = tmp_path / "runs"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QWEN_", "SENSECORE_"))}
    env.update(QWEN_PYTHON=sys.executable, QWEN_MODEL_PATH=str(model),
               QWEN_RUNS_ROOT=str(runs), QWEN_RUN_ID="capacity-check",
               QWEN_MIN_FREE_GIB=str(10**12))
    result = subprocess.run(["bash", str(ROOT / "scripts/run_sco_bootstrap.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 28
    assert "STORAGE_PREFLIGHT_FAILED" in result.stderr
    assert not runs.exists()


def test_submit_forwards_explicit_storage_threshold_without_submission():
    import os
    env = dict(os.environ, SCO_BIN="/bin/echo", QWEN_MIN_FREE_GIB="280")
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_sco.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "QWEN_MIN_FREE_GIB=280" in result.stdout
