"""EP4 distributed checkpoint reload and image generation for Qwen3.5 MoE.

This module is launched by a distributed shell wrapper.  It deliberately
contains no launcher logic so it can also be imported by unit tests.
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


def _distributed_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("distributed", {})
    if not isinstance(section, dict):
        raise TypeError("distributed config must be a mapping")
    return section


def build_inference_setup(config: dict[str, Any], world_size: int):
    """Build the same FSDP2/EP4 topology used by the MoE training recipe."""
    distributed = _distributed_section(config)
    ep_size = int(distributed.get("ep_size", 1))
    if world_size != 4 or ep_size != 4:
        raise ValueError(f"Qwen3.5-35B-A3B inference requires world_size=4 and ep_size=4; got world_size={world_size}, ep_size={ep_size}")
    for name in ("tp_size", "pp_size", "cp_size"):
        if int(distributed.get(name, 1)) != 1:
            raise ValueError(f"EP4 inference requires {name}=1")
    return create_distributed_setup_from_config(config, world_size=world_size)


def make_checkpoint_config(checkpoint_path: str | os.PathLike[str], base_model_path: str | os.PathLike[str]):
    """Return a config for loading a non-consolidated sharded safetensors model."""
    return CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.fspath(checkpoint_path),
        model_save_format="safetensors",
        model_cache_dir="",
        model_repo_id=os.fspath(base_model_path),
        save_consolidated=False,
        is_peft=False,
    )


def _rank_for_checkpoint(setup) -> tuple[int, int, int]:
    """Derive checkpoint coordinates from the runtime mesh, as training does."""
    mesh = setup.mesh_context.device_mesh
    if mesh is None:
        return 0, 0, 0
    dp_mesh = get_flat_mesh(mesh, "dp")
    dp_rank = dp_mesh.get_local_rank() if dp_mesh.size() > 1 else 0
    tp_rank = mesh["tp"].get_local_rank() if "tp" in mesh.mesh_dim_names and mesh["tp"].size() > 1 else 0
    pp_rank = mesh["pp"].get_local_rank() if "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1 else 0
    return dp_rank, tp_rank, pp_rank


def _load_training_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"training config must be a mapping: {config_path}")
    return config


def _normalize_backend_config(backend: Any) -> Any:
    """Remove Hydra-only metadata before passing a YAML backend mapping to the model loader."""
    if isinstance(backend, dict):
        return {key: value for key, value in backend.items() if key != "_target_"}
    return backend


def _load_model_and_checkpoint(
    *,
    base_model_path: str,
    checkpoint_path: str,
    training_config: dict[str, Any],
    setup,
):
    model_cfg = training_config.get("model", {})
    backend = _normalize_backend_config(model_cfg.get("backend"))
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        "distributed_setup": setup,
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
        "local_files_only": True,
    }
    if backend is not None:
        model_kwargs["backend"] = backend
    LOGGER.info("loading base model with EP4 distributed setup")
    model = NeMoAutoModelForImageTextToText.from_pretrained(base_model_path, **model_kwargs)

    dp_rank, tp_rank, pp_rank = _rank_for_checkpoint(setup)
    mesh_context = setup.mesh_context
    device_mesh = mesh_context.device_mesh
    pp_group = None
    if device_mesh is not None and "pp" in device_mesh.mesh_dim_names and device_mesh["pp"].size() > 1:
        pp_group = device_mesh["pp"].get_group()
    checkpoint_config = make_checkpoint_config(checkpoint_path, base_model_path)
    checkpointer = Checkpointer(
        config=checkpoint_config,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        moe_mesh=mesh_context.moe_mesh,
        process_group=getattr(mesh_context, "process_group", None),
        pp_group=pp_group,
    )
    LOGGER.info("loading sharded checkpoint: %s/model", checkpoint_path)
    checkpointer.load_model(model, os.path.join(checkpoint_path, "model"))
    return model


def generate_response(model, processor, image_path: str, prompt: str, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
    prompt_tokens = inputs["input_ids"].shape[-1]
    return processor.decode(outputs[0, prompt_tokens:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("QWEN_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s rank=%(process)d %(name)s: %(message)s",
        force=True,
    )

    dist_info = initialize_distributed(backend="nccl", timeout_minutes=60)
    if dist_info.world_size != 4:
        raise ValueError(f"expected four inference ranks, got {dist_info.world_size}")
    config = _load_training_config(args.config)
    setup = build_inference_setup(config, world_size=dist_info.world_size)
    model = _load_model_and_checkpoint(
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        training_config=config,
        setup=setup,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_path, local_files_only=True)
    response = generate_response(model, processor, args.image, args.prompt, args.max_new_tokens)
    if dist_info.is_main:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(response + "\n", encoding="utf-8")
        LOGGER.info("wrote response to %s", output_path)
    dist.barrier()


if __name__ == "__main__":
    main()
