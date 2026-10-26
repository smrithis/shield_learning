#!/usr/bin/env bash
set -eo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ws=${SHIELD_WS:-/home/shield/code/smrithi_ws}
ros_env=${SHIELD_ROS_ENV:-/home/shield/mambaforge/envs/ros_new}
mkdir -p "$root/build"
g++ -std=c++17 -O2 -fPIC -shared "$root/native/controller_bridge.cpp" \
  -I"$ws/src/belief_planner/include" -I"$ros_env/include" \
  -L"$ws/devel/lib" -L"$ros_env/lib" \
  -Wl,-rpath,"$ws/devel/lib" -Wl,-rpath,"$ros_env/lib" \
  -lexisting_ruckig_controller -lruckig -o "$root/build/libshield_learning_controller.so"
