#!/usr/bin/env bash
# Paired oracle-vs-calibrated-perception evaluation (arms A-D). No training.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root"
exec "$root/.venv/bin/python" -m shield_learning.paired_evaluation "$@"
