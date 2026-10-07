#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "${QWEN_PYTHON:-python}" -B "$ROOT/scripts/launch.py" infer "${QWEN_MODEL_KIND:-4b}" "$@"
