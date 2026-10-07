# Source manifest

This release is an allowlisted source tree assembled from the user's verified
Qwen3.5 project. It contains no model weights, datasets, checkpoints, logs,
virtual environments, scheduler output, credentials, or cluster-specific
secrets.

| Release path | Purpose | Provenance / notes |
| --- | --- | --- |
| `configs/qwen35_4b_full.yaml` | Portable dense full-parameter config | Normalized from the verified 4B run; local model and output paths are environment-resolved |
| `configs/qwen35_35b_moe_ep4.yaml` | Portable 35B-A3B EP4 full-parameter config | Normalized from the verified EP4 run; includes neat packing settings |
| `scripts/launch.py` | Common storage-gated train/infer planner and launcher | Release-specific wrapper and admission logic |
| `scripts/storage_preflight.py` | Capacity check before model import | Release-specific utility |
| `scripts/*.sh` | Portable wrappers and SCO site-adapter example | No scheduler credentials or hard-coded hosts |
| `examples/vlm_finetune/finetune.py` | NeMo AutoModel VLM recipe entrypoint | Retains upstream Apache headers |
| `examples/vlm_generate/*.py` | Dense and EP4 distributed image inference | Retains upstream headers where present; rank-0 output only |
| `smoke_data.py` | Synthetic image/text lifecycle dataset | Release test fixture, not production data |
| `nemo_automodel/` | Dependency-closed NeMo AutoModel source used by the selected recipes | Includes the upstream source snapshot needed by the selected entrypoints; no weights or caches |
| `tests/` | CPU/config/launcher/packing/visual-audit regression tests | Adapted to the two canonical release configs |
| `LICENSE`, `NOTICE`, `requirements.txt` | Legal and runtime metadata | Apache 2.0 source notice plus tested dependency overlay |

The public commit intentionally excludes internal planning documents and
machine-specific absolute paths. Dependency versions are documented as tested
overlay evidence, not as a promise that every CUDA/platform combination is
interchangeable.
