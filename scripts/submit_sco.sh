#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCO_BIN="${SCO_BIN:-sco}"
QWEN_MIN_FREE_GIB="${QWEN_MIN_FREE_GIB:-0}"
export QWEN_MIN_FREE_GIB
# Site-specific submit arguments may be supplied through SCO_ARGS as a bash array.
if declare -p SCO_ARGS >/dev/null 2>&1; then
  exec "$SCO_BIN" "${SCO_ARGS[@]}" "QWEN_MIN_FREE_GIB=$QWEN_MIN_FREE_GIB" "$ROOT/scripts/run_sco_bootstrap.sh" "$@"
fi
exec "$SCO_BIN" "QWEN_MIN_FREE_GIB=$QWEN_MIN_FREE_GIB" "$ROOT/scripts/run_sco_bootstrap.sh" "$@"
