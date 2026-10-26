#!/usr/bin/env bash
# Reuse the existing venv. This does not build the simulator or native bridge.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
export PIP_CACHE_DIR="$root/.cache/pip"
if [[ ${1:-} == --offline-fallback ]]; then
  "$root/.venv/bin/python" -m pip install --no-index --find-links .cache/wheels \
    numpy==1.26.4 setuptools==78.1.0 wheel==0.45.1 typing_extensions==4.13.0
  "$root/.venv/bin/python" -m pip install --no-index --no-deps --find-links .cache/wheels gymnasium==0.28.1
  echo 'Restored the tested fallback. This does NOT satisfy pyproject.toml or all Gymnasium dependencies.'
else
  "$root/.venv/bin/python" -m pip install -e .
fi
exec "$root/.venv/bin/python" -m pip check
