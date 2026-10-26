#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ws=${SHIELD_WS:-/home/shield/code/smrithi_ws}
ros_env=${SHIELD_ROS_ENV:-/home/shield/mambaforge/envs/ros_new}
export PATH="$ros_env/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$root/.cache/numba"
export MPLCONFIGDIR="$root/.cache/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
source "$ws/devel/setup.bash"
export SHIELD_WS="$ws"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
exec "$ros_env/bin/python" -m shield_learning.backend "$@"
