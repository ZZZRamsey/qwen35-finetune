#!/usr/bin/env python3
"""Portable planner/launcher for the verified Qwen3.5 dense and EP4 paths."""
from __future__ import annotations
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MODELS = {
    "4b": {"config": "qwen35_4b_full.yaml", "default_nproc": 4, "infer": "examples/vlm_generate/multinode_generate.py"},
    "35b": {"config": "qwen35_35b_moe_ep4.yaml", "world_size": 4, "default_nproc": 4, "infer": "examples/vlm_generate/distributed_generate.py"},
}
def fail(message: str, code: int = 2):
    print(f"LAUNCHER_ERROR: {message}", file=sys.stderr); raise SystemExit(code)
def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        return int(value)
    except ValueError:
        fail(f"{name} must be an integer, got {value!r}")

def first_env_int(primary: str, fallback: str, default: int) -> int:
    value = os.environ.get(primary)
    if value is None:
        value = os.environ.get(fallback, str(default))
    try:
        return int(value)
    except ValueError:
        fail(f"{primary}/{fallback} must be an integer, got {value!r}")
def clean_child_env(env: dict[str,str]) -> dict[str,str]:
    for key in ("WORLD_SIZE","RANK","LOCAL_RANK"): env.pop(key,None)
    return env

def admit_run(run_root: Path, runtime_root: Path, node_rank: int, manifest: dict) -> None:
    """Node zero owns creation; peers join the same immutable launch manifest."""
    manifest_path = run_root / "launch.json"
    if node_rank == 0:
        try:
            run_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            fail(f"existing run must not be overwritten: {run_root}")
        pending = run_root / "launch.json.tmp"
        pending.write_text(json.dumps(manifest, sort_keys=True))
        pending.replace(manifest_path)
    else:
        timeout = env_int("QWEN_JOIN_TIMEOUT", 120)
        deadline = time.monotonic() + max(0, timeout)
        while not manifest_path.is_file():
            if time.monotonic() >= deadline:
                fail(f"timed out waiting for node 0 to create {manifest_path}")
            time.sleep(0.1)
        if json.loads(manifest_path.read_text()) != manifest:
            fail("launch manifest mismatch with node 0; use identical inputs and rendezvous")
    try:
        runtime_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        fail(f"node {node_rank} already joined this run: {runtime_root}")
def main(argv: list[str]) -> int:
    if len(argv)<3 or argv[1] not in {"train","infer"} or argv[2].lower() not in MODELS:
        fail("usage: launch.py train|infer 4b|35b [overrides]")
    mode, model_name=argv[1],argv[2].lower(); spec=MODELS[model_name]; env=dict(os.environ)
    topology_flags = {"--nnodes", "--node-rank", "--nproc-per-node", "--master-addr", "--master-port"}
    for arg in argv[3:]:
        if arg.split("=", 1)[0].replace("_", "-") in topology_flags:
            fail("configure topology with QWEN_NNODES/QWEN_NODE_RANK/QWEN_NPROC/QWEN_MASTER_ADDR/QWEN_MASTER_PORT, not recipe flags")
    for key in ("QWEN_MODEL_PATH", "QWEN_CHECKPOINT", "QWEN_IMAGE", "QWEN_CONFIG", "QWEN_RUNS_ROOT"):
        if env.get(key):
            env[key] = str(Path(env[key]).expanduser().resolve())
    model_path_value = env.get("QWEN_MODEL_PATH", "")
    model_path = Path(model_path_value).expanduser().resolve() if model_path_value else None
    runs_root=Path(env.get("QWEN_RUNS_ROOT", str(ROOT/"runs"))).expanduser().resolve(); run_id=env.get("QWEN_RUN_ID", f"qwen35_{model_name}_run")
    if not SAFE_RUN_ID.fullmatch(run_id): fail(f"invalid QWEN_RUN_ID: {run_id!r}")
    nnodes = first_env_int("QWEN_NNODES", "SENSECORE_PYTORCH_NNODES", 1)
    node_rank = first_env_int("QWEN_NODE_RANK", "SENSECORE_PYTORCH_NODE_RANK", 0)
    nproc = env_int("QWEN_NPROC", spec["default_nproc"])
    port = first_env_int("QWEN_MASTER_PORT", "SENSECORE_PYTORCH_MASTER_PORT", 29500)
    master = env.get("QWEN_MASTER_ADDR", env.get("SENSECORE_PYTORCH_MASTER_ADDR", ""))
    if nnodes<1 or node_rank<0 or node_rank>=nnodes or nproc<1 or not 1<=port<=65535: fail(f"invalid distributed parameters nnodes={nnodes} node_rank={node_rank} nproc={nproc} port={port}")
    if nnodes>1 and (not master or master in {"localhost","::1"} or master.startswith("127.")): fail("multi-node launch requires a non-loopback QWEN_MASTER_ADDR")
    if nnodes > 1 and not env.get("QWEN_RUN_ID"):
        fail("multi-node launch requires the same explicit QWEN_RUN_ID on every node")
    world_size=nnodes*nproc
    if model_name=="35b" and world_size!=spec["world_size"]: fail(f"Qwen3.5-35B-A3B EP4 requires world_size=4, got world_size={world_size}")
    config=Path(env.get("QWEN_CONFIG", str(ROOT/"configs"/spec["config"]))).expanduser().resolve()
    if not config.is_file(): fail(f"missing config: {config}")
    if mode == "train" and model_name == "4b":
        import yaml
        cfg = yaml.safe_load(config.read_text())
        scheduler = dict(cfg["step_scheduler"])
        for index, arg in enumerate(argv[3:], start=3):
            for key in ("global_batch_size", "local_batch_size"):
                prefix = f"--step_scheduler.{key}="
                if arg.startswith(prefix) or arg == prefix[:-1]:
                    try:
                        scheduler[key] = int(arg[len(prefix):] if arg.startswith(prefix) else argv[index + 1])
                    except (ValueError, IndexError):
                        fail(f"{key} must be an integer")
        global_batch, local_batch = scheduler["global_batch_size"], scheduler["local_batch_size"]
        if local_batch < 1 or global_batch < 1 or global_batch % (local_batch * world_size):
            fail("global_batch_size must be positive and divisible by local_batch_size * world_size")
    if mode=="infer":
        for key in ("QWEN_MODEL_PATH","QWEN_CHECKPOINT","QWEN_IMAGE"):
            value=env.get(key)
            if not value or not Path(value).exists(): fail(f"missing inference input {key}: {value or '<unset>'}")
    elif model_path is None or not model_path.is_dir(): fail(f"missing QWEN_MODEL_PATH: {model_path or '<unset>'}")
    run_root=runs_root/run_id
    preflight=[sys.executable,"-B",str(ROOT/"scripts/storage_preflight.py"),str(run_root),env.get("QWEN_MIN_FREE_GIB","0")]
    result=subprocess.run(preflight,text=True,capture_output=True,env=clean_child_env(dict(env)))
    if result.returncode: sys.stderr.write(result.stderr); return result.returncode
    dry=bool(int(env.get("QWEN_DRY_RUN","0")))
    runtime_root=run_root/(f"runtime/node-{node_rank}" if nnodes>1 else "runtime")
    if not dry:
        manifest = dict(mode=mode, model=model_name, model_path=str(model_path),
                        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                        nnodes=nnodes, nproc=nproc, master=master, port=port,
                        arguments=argv[3:])
        if mode == "infer":
            manifest.update(checkpoint=env["QWEN_CHECKPOINT"], image=env["QWEN_IMAGE"])
        admit_run(run_root, runtime_root, node_rank, manifest)
    child_env=clean_child_env(dict(env)); child_env.update(HF_HOME=str(runtime_root/"hf"),HF_DATASETS_CACHE=str(runtime_root/"hf"/"datasets"),HF_HUB_CACHE=str(runtime_root/"hf"/"hub")); child_env["QWEN_CHECKPOINT_DIR"]=str(run_root/"checkpoints")
    child_env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    python=env.get("QWEN_PYTHON",sys.executable); dist=[python,"-m","torch.distributed.run"]
    if nnodes==1: dist += ["--standalone", "--nnodes=1", f"--nproc-per-node={nproc}"]
    else: dist += [f"--nnodes={nnodes}",f"--node-rank={node_rank}",f"--nproc-per-node={nproc}",f"--master-addr={master}",f"--master-port={port}"]
    if mode=="train": command=dist+[str(ROOT/"examples/vlm_finetune/finetune.py"),"--config",str(config),f"--model.pretrained_model_name_or_path={model_path}",f"--processor.pretrained_model_name_or_path={model_path}",f"--checkpoint.checkpoint_dir={run_root/'checkpoints'}"]+argv[3:]
    else: command=dist+[str(ROOT/spec["infer"]),"--config",str(config),"--checkpoint-path",env["QWEN_CHECKPOINT"],"--base-model-path",env["QWEN_MODEL_PATH"],"--image",env["QWEN_IMAGE"],"--output-file",str(run_root/"response.txt")]+argv[3:]
    # Do not disclose inherited service tokens or other ambient credentials.
    public_env = {key: child_env[key] for key in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "QWEN_CHECKPOINT_DIR", "PYTHONPATH")}
    plan={"command":command,"environment":public_env,"world_size":world_size,"run_root":str(run_root),"config":str(config)}
    if dry: print(json.dumps(plan,sort_keys=True)); return 0
    return subprocess.run(command,cwd=ROOT,env=child_env).returncode
if __name__=="__main__": raise SystemExit(main(sys.argv))
