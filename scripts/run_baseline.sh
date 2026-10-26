#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root"
exec "$root/.venv/bin/python" -m shield_learning.baseline "$@"
