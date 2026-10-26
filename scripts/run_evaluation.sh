#!/usr/bin/env bash
# Masked vs unmasked baselines over the structured + seeded throw set. No training.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root"
exec "$root/.venv/bin/python" -m shield_learning.evaluation "$@"
