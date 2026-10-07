"""Exercise the real launcher planning/admission path without reserving GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/launch.py"

@pytest.fixture
def launch_env(tmp_path):
    model = tmp_path / "base model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    checkpoint = tmp_path / "saved checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    image = tmp_path / "image.png"
    image.write_bytes(b"test")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("QWEN_", "SENSECORE_"))}
    env.update(QWEN_PYTHON=sys.executable, QWEN_MODEL_PATH=str(model),
               QWEN_RUNS_ROOT=str(tmp_path / "runs"), QWEN_RUN_ID="unique-smoke",
               QWEN_DRY_RUN="1", QWEN_CHECKPOINT=str(checkpoint),
               QWEN_IMAGE=str(image), QWEN_MIN_FREE_GIB="0")
    return env

def run(env, mode="train", model="4b", *args):
    assert LAUNCH.is_file(), "portable launcher is missing"
    return subprocess.run([sys.executable, "-B", str(LAUNCH), mode, model, *args],
                          env=env, text=True, capture_output=True)

def plan(env, mode="train", model="4b", *args):
    result = run(env, mode, model, *args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)

def test_single_node_uses_standalone_and_forwards_training_overrides(launch_env):
    p = plan(launch_env, "train", "4b", "--optimizer.lr=0.00002", "--checkpoint.restore_from=/shared/previous")
    cmd = p["command"]
    assert "--standalone" in cmd and "--nnodes=1" in cmd
    assert cmd[-2:] == ["--optimizer.lr=0.00002", "--checkpoint.restore_from=/shared/previous"]
    assert f'--model.pretrained_model_name_or_path={launch_env["QWEN_MODEL_PATH"]}' in cmd
    assert f'--processor.pretrained_model_name_or_path={launch_env["QWEN_MODEL_PATH"]}' in cmd
    assert f'--checkpoint.checkpoint_dir={launch_env["QWEN_RUNS_ROOT"]}/unique-smoke/checkpoints' in cmd
    assert not Path(launch_env["QWEN_RUNS_ROOT"]).exists(), "dry-run must not create run artifacts"

@pytest.mark.parametrize("model,nproc", [("4b", "4"), ("35b", "2")])
def test_two_node_rendezvous_and_rank_local_cache(launch_env, model, nproc):
    launch_env.update(QWEN_NNODES="2", QWEN_NODE_RANK="1", QWEN_NPROC=nproc,
                      QWEN_MASTER_ADDR="trainer-0", QWEN_MASTER_PORT="29701")
    p = plan(launch_env, "train", model)
    cmd = p["command"]
    for expected in ("--nnodes=2", "--node-rank=1", f"--nproc-per-node={nproc}",
                     "--master-addr=trainer-0", "--master-port=29701"):
        assert expected in cmd
    assert "--standalone" not in cmd
    assert p["environment"]["HF_HOME"].endswith("runtime/node-1/hf")
    assert p["world_size"] == (8 if model == "4b" else 4)

@pytest.mark.parametrize("address", ["", "localhost", "127.0.0.1", "::1", "127.3.4.5"])
def test_multinode_rejects_missing_or_loopback_master(launch_env, address):
    launch_env.update(QWEN_NNODES="2", QWEN_MASTER_ADDR=address)
    assert run(launch_env).returncode == 2

@pytest.mark.parametrize("key,value", [("QWEN_NPROC", "0"), ("QWEN_NODE_RANK", "1"),
                                       ("QWEN_MASTER_PORT", "65536"), ("QWEN_RUN_ID", "../escape")])
def test_invalid_launch_parameters_fail_before_run_creation(launch_env, key, value):
    launch_env[key] = value
    assert run(launch_env).returncode == 2
    assert not Path(launch_env["QWEN_RUNS_ROOT"]).exists()

def test_ep4_rejects_world_size_eight(launch_env):
    launch_env["QWEN_NPROC"] = "8"
    result = run(launch_env, "train", "35b")
    assert result.returncode == 2
    assert "world_size=4" in result.stderr

@pytest.mark.parametrize("model,module", [("4b", "multinode_generate.py"), ("35b", "distributed_generate.py")])
def test_inference_selects_loader_and_forwards_quoted_arguments(launch_env, model, module):
    p = plan(launch_env, "infer", model, "--prompt", "What color is this?", "--max-new-tokens", "7")
    cmd = p["command"]
    assert any(item.endswith(module) for item in cmd)
    assert cmd[cmd.index("--checkpoint-path") + 1] == launch_env["QWEN_CHECKPOINT"]
    assert cmd[cmd.index("--base-model-path") + 1] == launch_env["QWEN_MODEL_PATH"]
    assert cmd[cmd.index("--image") + 1] == launch_env["QWEN_IMAGE"]
    assert cmd[-4:] == ["--prompt", "What color is this?", "--max-new-tokens", "7"]

@pytest.mark.parametrize("key", ["QWEN_MODEL_PATH", "QWEN_IMAGE", "QWEN_CHECKPOINT"])
def test_inference_rejects_missing_input(launch_env, key):
    launch_env[key] = "/definitely/not/present"
    assert run(launch_env, "infer").returncode == 2

def test_storage_admission_fails_before_import_or_run_creation(launch_env):
    launch_env.update(QWEN_MIN_FREE_GIB=str(10**12), QWEN_DRY_RUN="0")
    result = run(launch_env)
    assert result.returncode == 28
    assert "STORAGE_PREFLIGHT_FAILED" in result.stderr
    assert not Path(launch_env["QWEN_RUNS_ROOT"]).exists()

def test_existing_run_is_not_overwritten(launch_env):
    run_root = Path(launch_env["QWEN_RUNS_ROOT"]) / launch_env["QWEN_RUN_ID"]
    run_root.mkdir(parents=True)
    marker = run_root / "keep.txt"
    marker.write_text("must survive")
    launch_env["QWEN_DRY_RUN"] = "0"
    result = run(launch_env)
    assert result.returncode == 2 and "existing run" in result.stderr.lower()
    assert marker.read_text() == "must survive"

def test_sco_environment_fallbacks_are_consumed_and_stale_rank_env_removed(launch_env):
    launch_env.update(SENSECORE_PYTORCH_NNODES="2", SENSECORE_PYTORCH_NODE_RANK="1",
                      SENSECORE_PYTORCH_MASTER_ADDR="trainer-0", SENSECORE_PYTORCH_MASTER_PORT="29711",
                      QWEN_NPROC="2", WORLD_SIZE="99", RANK="88", LOCAL_RANK="77")
    p = plan(launch_env, "infer", "35b")
    assert "--node-rank=1" in p["command"]
    assert "--master-port=29711" in p["command"]
    assert not {"WORLD_SIZE", "RANK", "LOCAL_RANK"} & p["environment"].keys()

def test_dense_larger_world_requires_compatible_explicit_batch(launch_env):
    launch_env.update(QWEN_NNODES="2", QWEN_NPROC="8", QWEN_MASTER_ADDR="trainer-0")
    result = run(launch_env)
    assert result.returncode == 2
    assert "global_batch_size" in result.stderr
    p = plan(launch_env, "train", "4b", "--step_scheduler.global_batch_size=16")
    assert p["world_size"] == 16

def test_child_failure_is_propagated(launch_env):
    launch_env.update(QWEN_DRY_RUN="0", QWEN_PYTHON="/bin/false")
    assert run(launch_env).returncode == 1

def test_two_nodes_share_one_run_without_overwriting(launch_env):
    launch_env.update(QWEN_NNODES="2", QWEN_NPROC="4", QWEN_MASTER_ADDR="trainer-0",
                      QWEN_DRY_RUN="0", QWEN_PYTHON="/bin/true", QWEN_NODE_RANK="0")
    first = run(launch_env)
    assert first.returncode == 0, first.stderr
    launch_env["QWEN_NODE_RANK"] = "1"
    second = run(launch_env)
    assert second.returncode == 0, second.stderr
    root = Path(launch_env["QWEN_RUNS_ROOT"]) / launch_env["QWEN_RUN_ID"]
    assert (root / "runtime/node-0").is_dir()
    assert (root / "runtime/node-1").is_dir()
    assert run(launch_env).returncode == 2, "same node must not launch twice"

def test_multinode_join_rejects_mismatched_launch(launch_env):
    launch_env.update(QWEN_NNODES="2", QWEN_NPROC="4", QWEN_MASTER_ADDR="trainer-0",
                      QWEN_DRY_RUN="0", QWEN_PYTHON="/bin/true", QWEN_NODE_RANK="0")
    assert run(launch_env).returncode == 0
    launch_env.update(QWEN_NODE_RANK="1", QWEN_MASTER_PORT="29799")
    result = run(launch_env)
    assert result.returncode == 2
    assert "mismatch" in result.stderr.lower()

def test_rank_one_waits_for_rank_zero(launch_env):
    launch_env.update(QWEN_NNODES="2", QWEN_NPROC="4", QWEN_MASTER_ADDR="trainer-0",
                      QWEN_DRY_RUN="0", QWEN_PYTHON="/bin/true", QWEN_NODE_RANK="1",
                      QWEN_JOIN_TIMEOUT="0")
    result = run(launch_env)
    assert result.returncode == 2
    assert "node 0" in result.stderr.lower()
    assert not Path(launch_env["QWEN_RUNS_ROOT"]).exists()

def test_dry_run_does_not_dump_inherited_credentials(launch_env):
    launch_env['PRIVATE_SERVICE_TOKEN'] = 'do-not-print-this-value'
    result = run(launch_env)
    assert result.returncode == 0
    assert 'do-not-print-this-value' not in result.stdout
    assert 'PRIVATE_SERVICE_TOKEN' not in result.stdout

@pytest.mark.parametrize('script,mode,model', [
    ('run_qwen35_4b_full.sh', 'train', '4b'),
    ('run_qwen35_4b_multinode.sh', 'train', '4b'),
    ('run_smoke.sh', 'train', '4b'),
    ('run_sco_bootstrap.sh', 'train', '4b'),
    ('run_sco_infer_bootstrap.sh', 'infer', '4b'),
    ('infer_qwen35_4b_multinode.sh', 'infer', '4b'),
    ('run_sco_moe_infer_bootstrap.sh', 'infer', '35b'),
    ('infer_qwen35_35b_ep4.sh', 'infer', '35b'),
])
def test_wrappers_delegate_to_canonical_planner(launch_env, script, mode, model):
    launch_env.update(QWEN_NNODES='1', QWEN_NPROC='4')
    args = [] if mode == 'train' else ['--prompt', 'What color?']
    result = subprocess.run(['bash', str(ROOT / 'scripts' / script), *args],
                            env=launch_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    expected = plan(launch_env, mode, model, *args)
    assert actual['command'] == expected['command']
    assert actual['world_size'] == expected['world_size']
    assert not Path(launch_env['QWEN_RUNS_ROOT']).exists()

def test_children_import_release_source_without_editable_install(launch_env):
    p = plan(launch_env)
    assert p['environment']['PYTHONPATH'].split(os.pathsep)[0] == str(ROOT)
    env = dict(launch_env, **p['environment'])
    result = subprocess.run([sys.executable, '-B', '-c', 'import nemo_automodel; print(nemo_automodel.__file__)'],
                            cwd=ROOT / 'examples/vlm_finetune', env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(ROOT / 'nemo_automodel/__init__.py') in result.stdout

def test_dense_batch_override_supports_space_syntax(launch_env):
    launch_env.update(QWEN_NNODES='2', QWEN_NPROC='8', QWEN_MASTER_ADDR='trainer-0')
    assert plan(launch_env, 'train', '4b', '--step_scheduler.global_batch_size', '16')['world_size'] == 16


@pytest.mark.parametrize("mode", ["train", "infer"])
def test_relative_paths_resolve_before_worker_changes_directory(launch_env, tmp_path, mode):
    for key in ("QWEN_MODEL_PATH", "QWEN_RUNS_ROOT", "QWEN_CHECKPOINT", "QWEN_IMAGE"):
        launch_env[key] = os.path.relpath(launch_env[key], tmp_path)
    result = subprocess.run([sys.executable, "-B", str(LAUNCH), mode, "4b"],
                            cwd=tmp_path, env=launch_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    p = json.loads(result.stdout)
    assert p["run_root"] == str(tmp_path / "runs" / "unique-smoke")
    cmd = p["command"]
    if mode == "infer":
        for flag in ("--base-model-path", "--checkpoint-path", "--image", "--output-file"):
            assert Path(cmd[cmd.index(flag) + 1]).is_absolute()
    else:
        assert f"--model.pretrained_model_name_or_path={tmp_path / 'base model'}" in cmd


def test_multinode_requires_explicit_shared_run_id(launch_env):
    launch_env.pop("QWEN_RUN_ID")
    launch_env.update(QWEN_NNODES="2", QWEN_MASTER_ADDR="trainer-0")
    result = run(launch_env)
    assert result.returncode == 2
    assert "QWEN_RUN_ID" in result.stderr


@pytest.mark.parametrize("arg", ["--nnodes=2", "--node-rank", "--nproc-per-node=8", "--master_addr=host"])
def test_rejects_torchrun_flags_passed_as_recipe_overrides(launch_env, arg):
    result = run(launch_env, "train", "4b", arg)
    assert result.returncode == 2
    assert "QWEN_" in result.stderr
