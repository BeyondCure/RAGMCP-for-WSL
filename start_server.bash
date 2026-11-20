#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

ENV_NAME="${COSTUME_MCP_ENV:-costume-mcp}"

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
    if [[ -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
  done
fi
if [[ -z "$CONDA_BIN" ]]; then
  echo "error: conda not found; install Miniconda or set CONDA_EXE" >&2
  exit 1
fi

CONDA_BASE="$("$CONDA_BIN" info --base)"
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: conda env '$ENV_NAME' not found. Create it first:" >&2
  echo "  conda create -n $ENV_NAME python=3.12 -y" >&2
  echo "  conda run -n $ENV_NAME pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PYTHON" runMCPServer.py "$@"