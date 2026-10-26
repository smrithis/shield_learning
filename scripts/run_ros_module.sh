#!/usr/bin/env bash
# Run a shield_learning module in the existing ros_new environment (read-only use of smrithi_ws).
# Usage: bash scripts/run_ros_module.sh perception_calibration|perception_validation|runtime_validation
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ws=${SHIELD_WS:-/home/shield/code/smrithi_ws}
ros_env=${SHIELD_ROS_ENV:-/home/shield/mambaforge/envs/ros_new}
export PATH="$ros_env/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$root/.cache/numba"
export MPLCONFIGDIR="$root/.cache/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$root/results"
source "$ws/devel/setup.bash"
export SHIELD_WS="$ws"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
exec "$ros_env/bin/python" -m "shield_learning.$1" "${@:2}"
