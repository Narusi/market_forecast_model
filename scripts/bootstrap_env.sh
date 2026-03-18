#!/usr/bin/env bash
set -euo pipefail

MIN_PYTHON="3.11"
REQ_FILE="requirements/core.txt"
WHEELHOUSE_DIR="wheelhouse"

python_bin="${PYTHON_BIN:-python3}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Error: $python_bin was not found in PATH." >&2
  exit 1
fi

python_version="$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if [ "$(printf '%s\n' "$MIN_PYTHON" "$python_version" | sort -V | head -n1)" != "$MIN_PYTHON" ]; then
  echo "Error: Python $MIN_PYTHON+ is required, found $python_version." >&2
  exit 1
fi

echo "Using $python_bin (version $python_version)."

if [ -d "$WHEELHOUSE_DIR" ] && find "$WHEELHOUSE_DIR" -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) | grep -q .; then
  echo "Found local wheelhouse at $WHEELHOUSE_DIR. Installing offline..."
  "$python_bin" -m pip install --no-index --find-links="$WHEELHOUSE_DIR" -r "$REQ_FILE"
else
  echo "No usable wheelhouse found. Falling back to online install from $REQ_FILE..."
  "$python_bin" -m pip install -r "$REQ_FILE"
fi
