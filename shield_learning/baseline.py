"""Reproducible SHIELD baselines; no learned policy or training dependencies."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from gymnasium.utils.env_checker import check_env
from .env import ShieldInterceptionEnv, ROOT

# Node 45: corrected FK error 5.6e-9 m; nearest well-matched goal to the default
# throw's predicted crossing. Used for designed (controlled) interception throws.
CONTROLLED_ACTION = 44


def run(episodes=5):
    dependencies = subprocess.run([sys.executable, "-m", "pip", "check"],
                                  capture_output=True, text=True, timeout=30)
    gym_version = importlib.metadata.version("gymnasium")
    dependency_status = {
        "pip_check_pass": dependencies.returncode == 0,
        "pip_check_output": dependencies.stdout.strip(),
        "requested_gymnasium_range": ">=1.0,<2",
        "requested_gymnasium_range_satisfied": int(gym_version.split(".")[0]) == 1,
    }
    with ShieldInterceptionEnv() as env:
        check_env(env, skip_render_check=True)
        verification = env.verify()
        rows = []
        for mode in ("original_planner", "discrete_goals"):
            for episode in range(episodes):
                obs, _ = env.reset(seed=episode, options={"randomize": episode != 0})
                initial = {k: v.tolist() for k, v in obs.items()}
                total, steps, actions, nodes = 0.0, 0, [], []
                while True:
                    if mode == "original_planner":
                        obs, reward, terminated, truncated, info = env.step_planner()
                    else:
                        action = env.baseline_action()
                        actions.append(action)
                        obs, reward, terminated, truncated, info = env.step(action)
                    if info["selected_node_index"] is not None and info["selected_node_index"] not in nodes:
                        nodes.append(info["selected_node_index"])
                    total += reward
                    steps += 1
                    if terminated or truncated:
                        break
                rows.append(dict(mode=mode, seed=episode, randomized=episode != 0,
                                 initial_observation=initial, actions=actions, selected_nodes=nodes,
                                 steps=steps, return_=total, terminated=terminated, truncated=truncated,
                                 swept=env.swept_check(), **info))
        controlled = [env.controlled_interception(CONTROLLED_ACTION, moving=moving) for moving in (True, False)]
        # Compare with the prior session's source manifest, without writing upstream.
        manifest = json.loads((ROOT / "results/source_hashes_before.json").read_text())
        changed, missing = [], []
        for name, expected in manifest.items():
            path = Path(name)
            if not path.is_file():
                missing.append(name)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                changed.append(name)
        report = {"gymnasium_checker_pass": True, "episodes": rows,
                  "controlled_interceptions": controlled,
                  "dependency_status": dependency_status,
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("gymnasium", "numpy")},
                  "backend_metadata": env.backend_metadata,
                  "source_integrity": {"checked_files": len(manifest), "changed": changed, "missing": missing},
                  "physical_interception_verified": "geometry only: designed throws, sub-tick swept check; "
                                                    "no contact dynamics"}
    (ROOT / "results/verification_current.json").write_text(json.dumps(verification, indent=2) + "\n")
    (ROOT / "results/baselines.json").write_text(json.dumps(report, indent=2) + "\n")
    for mode in ("original_planner", "discrete_goals"):
        selected = [row for row in rows if row["mode"] == mode]
        print(f'{mode}: {len(selected)} episodes, {sum(r["is_success"] for r in selected)} sampled overlaps, '
              f'{sum(r["swept"]["swept_first_contact_time_s"] is not None for r in selected)} swept contacts; '
              f'minimum surface distance {min(r["minimum_surface_distance_m"] for r in selected):.6f} m sampled, '
              f'{min(r["swept"]["swept_min_surface_distance_m"] for r in selected):.6f} m swept')
    for result in controlled:
        swept = result["episode_swept"]
        print(f'controlled {"moving" if result["moving"] else "static"} throw at node {result["goal_node_index"]}: '
              f'{result["termination_reason"]} sampled at {result["sampled_termination_time_s"]:.3f} s, swept first '
              f'contact {swept["swept_first_contact_time_s"]:.5f} s, shield speed '
              f'{result["shield_speed_at_designed_contact_m_s"]:.3f} m/s')
    print(f'source integrity: {len(manifest)} files, {len(changed)} changed, {len(missing)} missing')
    print("Geometry-only interception. Contact dynamics remain unmodelled. See results/verification_current.json.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="episodes per baseline, seeds 0..N-1")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    run(args.episodes)


if __name__ == "__main__":
    main()
