"""Distributed image generation for a sharded Qwen3.5-4B checkpoint.

Every rank builds the same FSDP2 topology, loads its own checkpoint shard, and
executes the multimodal forward/generate.  Only rank 0 writes the response so
shared output files are never written concurrently.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from PIL import Image
from transformers import AutoProcessor

from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-format", choices=("text", "json"), default="text")
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def _load_training_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a mapping: {config_path}")
    return config


def _normalize_backend_config(backend: Any) -> Any:
    if isinstance(backend, dict):
        return {key: value for key, value in backend.items() if key != "_target_"}
    return backend


def _build_inference_setup(config: dict[str, Any], world_size: int):
    distributed = config.get("distributed", {})
    if not isinstance(distributed, dict):
        raise TypeError("distributed config must be a mapping")
    for name in ("tp_size", "cp_size", "pp_size", "ep_size"):
        if int(distributed.get(name, 1) or 1) != 1:
            raise ValueError(f"dense 4B inference requires {name}=1")
    return create_distributed_setup_from_config(config, world_size=world_size)


def _rank_for_checkpoint(setup) -> tuple[int, int, int]:
    mesh = setup.mesh_context.device_mesh
    if mesh is None:
        return 0, 0, 0
    dp_mesh = get_flat_mesh(mesh, "dp")
    dp_rank = dp_mesh.get_local_rank() if dp_mesh.size() > 1 else 0
    tp_rank = mesh["tp"].get_local_rank() if "tp" in mesh.mesh_dim_names and mesh["tp"].size() > 1 else 0
    pp_rank = mesh["pp"].get_local_rank() if "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1 else 0
    return dp_rank, tp_rank, pp_rank


def _make_checkpoint_config(checkpoint_path: str, base_model_path: str) -> CheckpointingConfig:
    return CheckpointingConfig(
        enabled=True,
        checkpoint_dir=checkpoint_path,
        model_save_format="safetensors",
        model_cache_dir="",
        model_repo_id=base_model_path,
        save_consolidated=False,
        is_peft=False,
    )


def _load_model_and_checkpoint(*, base_model_path: str, checkpoint_path: str, config: dict[str, Any], setup):
    model_cfg = config.get("model", {})
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        "distributed_setup": setup,
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
        "local_files_only": True,
    }
    backend = _normalize_backend_config(model_cfg.get("backend"))
    if backend is not None:
        model_kwargs["backend"] = backend
    LOGGER.info("loading dense 4B model on rank %s", dist.get_rank())
    model = NeMoAutoModelForImageTextToText.from_pretrained(base_model_path, **model_kwargs)

    dp_rank, tp_rank, pp_rank = _rank_for_checkpoint(setup)
    mesh_context = setup.mesh_context
    device_mesh = mesh_context.device_mesh
    pp_group = None
    if device_mesh is not None and "pp" in device_mesh.mesh_dim_names and device_mesh["pp"].size() > 1:
        pp_group = device_mesh["pp"].get_group()
    checkpointer = Checkpointer(
        config=_make_checkpoint_config(checkpoint_path, base_model_path),
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        process_group=getattr(mesh_context, "process_group", None),
        pp_group=pp_group,
    )
    LOGGER.info("loading sharded checkpoint: %s/model", checkpoint_path)
    checkpointer.load_model(model, os.path.join(checkpoint_path, "model"))
    return model


def _generate_response(model, processor, image_path: str, prompt: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
    prompt_tokens = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0, prompt_tokens:], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=os.environ.get("QWEN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s rank=%(process)d %(message)s",
        force=True,
    )
    dist_info = initialize_distributed(backend="nccl", timeout_minutes=60)
    rank = dist_info.rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    local_status = torch.ones(1, dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu")
    response = ""
    try:
        config = _load_training_config(args.config)
        setup = _build_inference_setup(config, dist_info.world_size)
        model = _load_model_and_checkpoint(
            base_model_path=args.base_model_path,
            checkpoint_path=args.checkpoint_path,
            config=config,
            setup=setup,
        )
        processor = AutoProcessor.from_pretrained(args.base_model_path, local_files_only=True)
        response = _generate_response(model, processor, args.image, args.prompt, args.max_new_tokens)
        LOGGER.info("rank %s completed multimodal generation", dist_info.rank)
    except Exception:
        local_status.zero_()
        LOGGER.exception("rank %s failed during distributed inference", dist_info.rank)

    dist.all_reduce(local_status, op=dist.ReduceOp.MIN)
    if rank == 0 and int(local_status.item()) == 1:
        if args.output_format == "json":
            output = json.dumps(
                {"prompt": args.prompt, "image_url": args.image, "response": response},
                indent=2,
            )
        else:
            output = response
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        LOGGER.info("rank 0 wrote %s", output_path)
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    if int(local_status.item()) != 1:
        raise RuntimeError("distributed inference failed on at least one rank")


if __name__ == "__main__":
    main()
