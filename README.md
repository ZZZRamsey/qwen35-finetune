# Qwen3.5 Fine-Tuning Framework

A compact release of the verified NeMo AutoModel paths used for multimodal
Qwen3.5 fine-tuning and image inference.

## Supported paths

| Path | Parallel layout | Scope verified in the source validation runs |
| --- | --- | --- |
| Qwen3.5-4B dense | 1 node × 4 GPUs, and 2 nodes × 4 GPUs | Full-parameter multimodal training, sharded checkpoint save/resume, packing, image inference |
| Qwen3.5-35B-A3B | 2 nodes × 2 GPUs, EP4 | Full-parameter multimodal one-step smoke, visual-gradient/update audit, sharded checkpoint save/resume, packing, image inference |

The published tree has CPU launcher/config/import tests. The GPU/SCO results
above are inherited validation evidence from the source project; this release
was not presented as a new GPU benchmark. One-step smoke tests do not establish
convergence, production quality, or real-data accuracy.

## Installation

Use a Python environment with the versions in `requirements.txt`, a compatible
PyTorch CUDA build, and a NeMo AutoModel-compatible runtime. Install this
repository as a source checkout (no editable install is required):

```bash
export QWEN_ROOT=/path/to/qwen35-finetune
export QWEN_PYTHON=/path/to/python
export QWEN_MODEL_PATH=/path/to/local/Qwen3.5-4B
export QWEN_RUNS_ROOT=/path/to/shared/runs
export QWEN_MIN_FREE_GIB=100  # Example only: size for your checkpoint/optimizer budget.
```

The model directory must be a local, complete Hugging Face snapshot. Training
and inference paths are intentionally local-files-only once the snapshot is
prepared.

## Training

A one-step 4B full-parameter launch on one node:

```bash
QWEN_NPROC=4 QWEN_RUN_ID=qwen35_4b_one_step \
  bash scripts/run_qwen35_4b_full.sh
```

The verified 4B multi-node shape is two nodes with four processes per node.
Set the same run id, model path, runs root, master address, port, and config on
both nodes; use rank 0 and rank 1 respectively. `QWEN_RUNS_ROOT` must be on
a shared filesystem mounted at the same absolute path on both nodes. Choose a
new shared run ID for each launch (do not generate it independently per node):

```bash
# node 0
QWEN_RUN_ID=qwen35_4b_multinode_001 \
QWEN_NNODES=2 QWEN_NPROC=4 QWEN_NODE_RANK=0 \
QWEN_MASTER_ADDR='node-0-host' QWEN_MASTER_PORT=29500 \
  bash scripts/run_qwen35_4b_multinode.sh

# node 1
QWEN_RUN_ID=qwen35_4b_multinode_001 \
QWEN_NNODES=2 QWEN_NPROC=4 QWEN_NODE_RANK=1 \
QWEN_MASTER_ADDR='node-0-host' QWEN_MASTER_PORT=29500 \
  bash scripts/run_qwen35_4b_multinode.sh
```

For the 35B-A3B EP4 path, use two nodes with two processes per node:

```bash
# Run on both nodes; change QWEN_NODE_RANK to 1 on the second node.
QWEN_RUN_ID=qwen35_35b_ep4_001 \
QWEN_NNODES=2 QWEN_NPROC=2 QWEN_NODE_RANK=0 \
QWEN_MASTER_ADDR='node-0-host' QWEN_MASTER_PORT=29501 \
QWEN_MODEL_KIND=35b \
QWEN_MODEL_PATH=/path/to/local/Qwen3.5-35B-A3B \
  bash scripts/run_sco_bootstrap.sh
```

`QWEN_CHECKPOINT_DIR` is derived inside the new run directory. Resume into a
**new** run ID with an explicit, complete checkpoint from the previous run and
a step/epoch budget larger than the saved counters. For example (one node):

```bash
QWEN_RUN_ID=qwen35_4b_resume_002 QWEN_NPROC=4 \
  bash scripts/run_qwen35_4b_full.sh \
  --checkpoint.restore_from=/path/to/previous-run/checkpoints/epoch_0_step_1 \
  --step_scheduler.max_steps=2 --step_scheduler.num_epochs=2
```

For multi-node resume, apply the same absolute restore path and new run ID on
both nodes. Do not use bare `LATEST` with a new empty checkpoint directory:
it cannot find the previous run. Use the actual saved directory name rather
than assuming the example's step suffix.

The launcher checks `QWEN_MIN_FREE_GIB` before loading models (default `0`,
which does **not** reserve space or prevent checkpoint ENOSPC). Set a nonzero
threshold appropriate for model weights, optimizer state, retention, and other
concurrent jobs. It rejects unsafe run-id reuse and
uses an immutable shared launch manifest so a second node cannot silently join
with different topology or rendezvous settings.

Topology is configured through the `QWEN_*` environment variables above, not
raw torchrun CLI flags. Inference options such as `--prompt` and
`--max-new-tokens` are CLI arguments, not `QWEN_PROMPT` variables.

## Packing

The 35B canonical config enables neat packed sequences with a 512-token pack
budget, pre-tokenization, a 0.9 packing ratio target, and explicit handling for
long samples. The same packed-sequence implementation is exercised by the
focused tests and can be adapted through the `packed_sequence` config block.

## Image inference

Inference consumes a sharded checkpoint and writes the response only from rank
0. Example for the dense path:

```bash
# On the second node set QWEN_NODE_RANK=1; use the same new inference run ID.
QWEN_RUN_ID=qwen35_4b_infer_003 \
QWEN_MODEL_PATH=/path/to/local/Qwen3.5-4B \
QWEN_CHECKPOINT=/path/to/run/checkpoints/epoch_0_step_1 \
QWEN_IMAGE=/path/to/image.png QWEN_NNODES=2 QWEN_NPROC=4 \
QWEN_NODE_RANK=0 QWEN_MASTER_ADDR='node-0-host' \
  bash scripts/infer_qwen35_4b_multinode.sh \
  --prompt 'What is the color of this image?'
```

Use `scripts/infer_qwen35_35b_ep4.sh` for the EP4 loader and set the same
four-rank rendezvous variables (`QWEN_NNODES=2 QWEN_NPROC=2`) on both nodes,
with a new common inference run ID. The launcher requires the model,
checkpoint, and image paths to exist before starting distributed workers.

## SCO / gpu8 integration

`submit_sco.sh` is deliberately a thin site-adapter example. It forwards the
storage threshold and invokes `run_sco_bootstrap.sh`; scheduler-specific
arguments must be supplied by the site wrapper or environment. The repository
does not contain credentials, scheduler dumps, cluster hostnames, or a claim
that generic SCO syntax is portable.

## Tests

From a checkout with the tested runtime dependencies:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests
```

The tests cover YAML portability, packed-sequence behavior, visual gradient
and update audits, distributed inference helpers, storage preflight, launcher
argument forwarding, multi-node run admission, and import/syntax boundaries.
They do not download model weights or reserve GPUs.

## Storage and cleanup

Keep model snapshots, checkpoints, logs, and temporary validation files outside
Git. The launcher writes runtime caches below each run directory. Remove failed
or obsolete run directories only after confirming they are not needed for
resume or comparison; never delete the base model snapshot or a valid training
checkpoint accidentally.

## License

The selected framework source is distributed under Apache License 2.0; see
`LICENSE` and `NOTICE`. Model weights, tokenizer files, and datasets are not
included and remain subject to their own terms.
