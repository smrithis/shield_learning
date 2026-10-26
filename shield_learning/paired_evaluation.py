"""Paired oracle-vs-noisy evaluation of the original planner on identical throws. No training.

Arms (all with the default 5 cm goal mask, original planner via step_planner):
  A oracle_ideal       exact belief every 4 ms; previous bridge timing (decide each step)
  B oracle_runtime     exact belief every 4 ms; controller.cpp request loop and gates
  C timed_truth        exact state, but only at calibrated publication times + latency
  D calibrated         calibrated measurements -> unchanged SlidingKalman beliefs
B-D share request-delay draws, and C/D share the whole perception stream, per seed,
so A->B isolates runtime timing, B->C perception timing/visibility, C->D estimation.
"""
import argparse
import collections
import json
import math

import numpy as np

from .env import ShieldInterceptionEnv, ROOT
from .scenarios import randomized, structured

ARMS = {"A_oracle_ideal": dict(),
        "B_oracle_runtime": dict(original_runtime=True, substeps=1),
        "C_timed_truth": dict(perception="timed_truth", original_runtime=True, substeps=1),
        "D_calibrated": dict(perception="calibrated", original_runtime=True, substeps=1)}
CAPTURE = 0.17 + 0.0381


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def episode(env, scenario, seed, swept):
    env.reset(seed=seed, options={"randomize": False, "throw": scenario["throw"],
                                  "launch_time_s": scenario["launch_time_s"]})
    while True:
        _, _, terminated, truncated, info = env.step_planner()
        if terminated or truncated:
            break
    d = env.episode_diagnostics()
    trace = env.perception_trace()
    published = [b for b in trace["beliefs"] if b["published"]]
    row = {"is_success": info["is_success"], "termination_reason": info["termination_reason"],
           "min_surface_distance_m": info["minimum_surface_distance_m"], "failure_cause": d["failure_cause"],
           "coverage_gap_m": d.get("coverage_gap_m"), "crossing_time_s": d.get("ball_plane_crossing_time_s"),
           "first_command_s": d["commands"][0][0] if d["commands"] else None,
           "goal_node_index": d.get("goal_node_index"), "goal_switches": info["goal_switches"],
           "joint_error_at_crossing_rad": d.get("joint_error_at_crossing_rad"),
           "plan_requests": info.get("runtime", {}).get("plan_requests"),
           "ignored_after_finish": info.get("runtime", {}).get("ignored_after_finish"),
           "measurements": len(trace["measurements"]), "published_beliefs": len(published),
           "first_belief_available_s": published[0]["available_s"] if published else None,
           "filter_restarts": trace.get("resets"),
           "median_belief_position_error_m": float(np.median([b["position_error_m"] for b in published]))
           if published else None}
    if swept:
        s = env.swept_check(subdivisions=100)
        row["swept_contact"] = s["swept_first_contact_time_s"] is not None
    return row


def summarize(rows, seeds):
    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    # Coverage from the ideal oracle arm (geometry only; identical across arms).
    covered = {r["scenario"] for r in by_arm["A_oracle_ideal"] if (r["coverage_gap_m"] or 9) <= CAPTURE}
    out = {"in_coverage_scenarios": len(covered)}
    for arm, group in sorted(by_arm.items()):
        inside = [r for r in group if r["scenario"] in covered]
        k = sum(r["is_success"] for r in inside)
        first = [r["first_command_s"] for r in inside if r["first_command_s"] is not None]
        out[arm] = {"episodes": len(group), "successes": sum(r["is_success"] for r in group),
                    "in_coverage_episodes": len(inside), "in_coverage_successes": k,
                    "in_coverage_success_rate": k / len(inside), "wilson95": wilson(k, len(inside)),
                    "failure_causes_in_coverage": dict(collections.Counter(r["failure_cause"] for r in inside
                                                                           if not r["is_success"])),
                    "median_first_command_s": float(np.median(first)) if first else None,
                    "median_goal_switches": float(np.median([r["goal_switches"] for r in inside]))}
        swept = [r for r in group if "swept_contact" in r]
        if swept:
            out[arm]["sampled_vs_swept_disagreements"] = sum(r["swept_contact"] != r["is_success"] for r in swept)
            out[arm]["swept_checked"] = len(swept)
    # Paired transitions on in-coverage (scenario, seed) cells; arm A has one deterministic run.
    index = {(r["arm"], r["scenario"], r["seed"]): r["is_success"] for r in rows}
    pairs = {}
    for a, b in (("A_oracle_ideal", "B_oracle_runtime"), ("B_oracle_runtime", "C_timed_truth"),
                 ("C_timed_truth", "D_calibrated"), ("A_oracle_ideal", "D_calibrated")):
        c = collections.Counter()
        for s in covered:
            for seed in seeds:
                x = index[(a, s, 0 if a == "A_oracle_ideal" else seed)]
                y = index[(b, s, seed)]
                c[f'{"hit" if x else "miss"}->{"hit" if y else "miss"}'] += 1
        pairs[f"{a} -> {b}"] = dict(c)
    out["paired_in_coverage"] = pairs
    return out


def run(seeds=5, random_count=60, scenario_seed=2026):
    rows, scenarios = [], None
    for arm, kwargs in ARMS.items():
        with ShieldInterceptionEnv(max_ticks=1000, **kwargs) as env:
            if scenarios is None:
                meta = env.backend_metadata
                R, t = meta["base_from_plane_rotation"], meta["base_from_plane_translation"]
                masked = [tuple(p) for p, f in zip(meta["goal_task_positions_m"], env.goal_flagged) if f]
                scenarios = [s for s in structured(R, t, masked) + randomized(R, t, random_count, scenario_seed)
                             if s["feasible"]]
            for scenario in scenarios:
                for seed in ([0] if arm == "A_oracle_ideal" else range(seeds)):
                    row = episode(env, scenario, seed, swept=(arm == "D_calibrated" and seed == 0))
                    rows.append({"arm": arm, "scenario": scenario["name"], "category": scenario["category"],
                                 "seed": seed, "reaction_time_s": scenario["reaction_time_s"],
                                 "speed_m_s": scenario["speed_m_s"], **row})
            print(arm, sum(r["is_success"] for r in rows if r["arm"] == arm), "/",
                  sum(r["arm"] == arm for r in rows), flush=True)
    report = {"seeds": seeds, "scenario_seed": scenario_seed, "arms": ARMS,
              "summary": summarize(rows, range(seeds)), "episodes": rows}
    (ROOT / "results/paired_evaluation.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report["summary"], indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--random", type=int, default=60)
    args = parser.parse_args()
    run(args.seeds, args.random)


if __name__ == "__main__":
    main()
