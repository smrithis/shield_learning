#!/usr/bin/env bash
# Coverage figure from results/evaluation.json using ros_new's matplotlib.
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ros_env=${SHIELD_ROS_ENV:-/home/shield/mambaforge/envs/ros_new}
export MPLCONFIGDIR="$root/.cache/matplotlib" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root"
cd "$root"
exec "$ros_env/bin/python" -m shield_learning.plot_evaluation "$@"
