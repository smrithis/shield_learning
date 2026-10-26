"""Masked vs unmasked baseline evaluation over diverse, reproducible throws. No training."""
import argparse
import collections
import csv
import json

import numpy as np

from .env import ShieldInterceptionEnv, ROOT
from .scenarios import randomized, structured

POLICIES = ("original_planner", "discrete_goals")
CAPTURE = 0.17 + 0.0381  # Shield half-width + ball radius (MuJoCo evaluator box).


def run_episode(env, scenario, policy):
    env.reset(seed=0, options={"randomize": False, "throw": scenario["throw"],
                               "launch_time_s": scenario["launch_time_s"]})
    holds = 0
    while True:
        if policy == "original_planner":
            _, _, terminated, truncated, info = env.step_planner()
        else:
            try:
                action = env.baseline_action()
            except RuntimeError as error:
                if "cannot predict" not in str(error):
                    raise
                holds += 1
                _, _, terminated, truncated, info = env.step_hold()
            else:
                _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    swept = env.swept_check(subdivisions=100)
    diagnostics = env.episode_diagnostics()
    return {"termination_reason": info["termination_reason"], "is_success": info["is_success"],
            "elapsed_s": info["elapsed_s"], "sampled_min_surface_distance_m": info["minimum_surface_distance_m"],
            "swept_min_surface_distance_m": swept["swept_min_surface_distance_m"],
            "swept_contact": swept["swept_first_contact_time_s"] is not None,
            "swept_first_contact_time_s": swept["swept_first_contact_time_s"],
            "swept_first_contact_ball_in_shield_m": swept["swept_first_contact_ball_in_shield_m"],
            "swept_max_relative_motion_between_samples_m": swept["swept_max_relative_motion_between_samples_m"],
            "evaluator_vs_mujoco_max_outside_disagreement_m": swept["evaluator_vs_mujoco_max_outside_disagreement_m"],
            "goal_switches": info["goal_switches"], "hold_steps": holds,
            "selected_goal_flagged": info["selected_goal_flagged"], **diagnostics}


def annotate_headroom(rows, fk, reach, flagged):
    """Mark failures where some allowed goal covers the crossing and is reachable from home first.

    Approximate oracle: static shield at the goal, reach time from home to within 0.01 rad.
    It bounds what better goal *selection* alone could add; it is not a guarantee.
    """
    fk = np.asarray(fk)
    for r in rows:
        r["faster_covering_goal_exists"] = None
        if r["is_success"] or r.get("crossing_yz_m") is None:
            continue
        allowed = ~np.asarray(flagged) if r["mask"] == "masked" else np.ones(len(fk), dtype=bool)
        gap = np.max(np.abs(fk - np.asarray(r["crossing_yz_m"])), axis=1)
        time = np.array([np.inf if x is None else x for x in reach])
        r["faster_covering_goal_exists"] = bool(np.any(allowed & (gap <= CAPTURE)
                                                       & (time <= r["ball_plane_crossing_time_s"])))


def summarize(rows):
    out = {}
    for key in sorted({(r["mask"], r["policy"]) for r in rows}):
        group = [r for r in rows if (r["mask"], r["policy"]) == key]
        failures = [r for r in group if not r["is_success"]]
        covered = [r for r in group if r.get("coverage_gap_m") is not None and r["coverage_gap_m"] <= CAPTURE]
        flagged = [r for r in group if r["selected_goal_flagged"]]
        by_category = collections.defaultdict(lambda: [0, 0])
        for r in group:
            by_category[r["category"]][0] += r["is_success"]
            by_category[r["category"]][1] += 1
        out[f"{key[0]}/{key[1]}"] = {
            "episodes": len(group), "successes": sum(r["is_success"] for r in group),
            "success_rate": sum(r["is_success"] for r in group) / len(group),
            "in_coverage_successes": f'{sum(r["is_success"] for r in covered)}/{len(covered)}',
            "out_of_coverage_successes": f'{sum(r["is_success"] for r in group if r not in covered)}/'
                                         f'{len(group) - len(covered)}',
            "swept_contacts": sum(r["swept_contact"] for r in group),
            "sampled_vs_swept_disagreements": sum(r["is_success"] != r["swept_contact"] for r in group),
            "failure_causes": dict(collections.Counter(r["failure_cause"] for r in failures)),
            "failure_median_closest_separation_m": float(np.median([r["swept_min_surface_distance_m"] for r in failures]))
            if failures else None,
            "flagged_goal_selections": len(flagged),
            "flagged_goal_successes": sum(r["is_success"] for r in flagged),
            "flagged_goal_failure_causes": dict(collections.Counter(r["failure_cause"] for r in flagged
                                                                    if not r["is_success"])),
            "failures_where_faster_covering_goal_exists": sum(bool(r.get("faster_covering_goal_exists"))
                                                              for r in failures),
            "failures_with_goal_reach_time_after_crossing": sum(
                (r.get("goal_reach_time_from_home_s") or 0) > (r.get("ball_plane_crossing_time_s") or 1e9)
                for r in failures),
            "distinct_goals_selected": len({r.get("goal_node_index") for r in group} - {None}),
            "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_category.items())},
        }
    paired = collections.Counter()
    index = {(r["scenario"], r["policy"], r["mask"]): r for r in rows}
    for (name, policy, mask), r in index.items():
        if mask == "masked":
            other = index[(name, policy, "unmasked")]
            paired[f'{policy}: unmasked {"hit" if other["is_success"] else "miss"} -> '
                   f'masked {"hit" if r["is_success"] else "miss"}'] += 1
    out["paired_mask_effect"] = dict(sorted(paired.items()))
    return out


def coverage(scenarios, rows, metadata):
    feasible = [s for s in scenarios if s["feasible"]]
    crossings = [r for r in rows if r.get("crossing_yz_m") is not None and r["mask"] == "masked"
                 and r["policy"] == POLICIES[0]]
    goals = np.asarray(metadata["goal_fk_task_positions_m"])
    return {"scenarios_defined": len(scenarios), "scenarios_feasible": len(feasible),
            "infeasible_release_height": [s["name"] for s in scenarios if not s["feasible"]],
            "categories": dict(collections.Counter(s["category"] for s in feasible)),
            "plane_y_range_m": [min(s["plane_y_m"] for s in feasible), max(s["plane_y_m"] for s in feasible)],
            "plane_z_range_m": [min(s["plane_z_m"] for s in feasible), max(s["plane_z_m"] for s in feasible)],
            "speed_range_m_s": [min(s["speed_m_s"] for s in feasible), max(s["speed_m_s"] for s in feasible)],
            "launch_angle_range_deg": [min(s["launch_angle_deg"] for s in feasible),
                                       max(s["launch_angle_deg"] for s in feasible)],
            "side_offset_range_deg": [min(s["side_offset_deg"] for s in feasible),
                                      max(s["side_offset_deg"] for s in feasible)],
            "launch_time_range_s": [min(s["launch_time_s"] for s in feasible), max(s["launch_time_s"] for s in feasible)],
            "reaction_time_range_s": [min(s["reaction_time_s"] for s in feasible),
                                      max(s["reaction_time_s"] for s in feasible)],
            "incidence_range_deg": [min(r["crossing_incidence_deg"] for r in crossings),
                                    max(r["crossing_incidence_deg"] for r in crossings)],
            "crossings_inside_valid_goal_coverage": f'{sum(r["coverage_gap_m"] <= CAPTURE for r in crossings)}/'
                                                    f'{len(crossings)}',
            "distinct_goals_selected_any_run": len({r.get("goal_node_index") for r in rows} - {None}),
            "goal_grid_y_range_m": [float(goals[:, 0].min()), float(goals[:, 0].max())],
            "goal_grid_z_range_m": [float(goals[:, 1].min()), float(goals[:, 1].max())]}


def run(random_count=60, seed=2026):
    rows, scenarios, metadata = [], None, None
    reach_path = ROOT / "results/goal_reach_times.json"
    reach = dict(zip(*(lambda d: (d["goal_node_indices"], d["reach_time_s"]))(json.loads(reach_path.read_text())))) \
        if reach_path.exists() else {}
    for mask in ("masked", "unmasked"):
        with ShieldInterceptionEnv(max_ticks=1000, apply_goal_mask=mask == "masked") as env:
            metadata = env.backend_metadata
            if scenarios is None:
                R, t = metadata["base_from_plane_rotation"], metadata["base_from_plane_translation"]
                masked_targets = [tuple(p) for p, f in zip(metadata["goal_task_positions_m"], env.goal_flagged) if f]
                scenarios = structured(R, t, masked_targets) + randomized(R, t, random_count, seed)
            for scenario in scenarios:
                if not scenario["feasible"]:
                    continue
                for policy in POLICIES:
                    result = run_episode(env, scenario, policy)
                    rows.append({"scenario": scenario["name"], "category": scenario["category"], "mask": mask,
                                 "policy": policy, **{k: scenario[k] for k in (
                                     "plane_y_m", "plane_z_m", "speed_m_s", "launch_angle_deg", "side_offset_deg",
                                     "release_distance_m", "launch_time_s", "reaction_time_s")}, **result,
                                 "goal_reach_time_from_home_s": reach.get(result.get("goal_node_index"))})
    flagged = np.asarray(metadata["goal_fk_error_m"]) > 0.05
    annotate_headroom(rows, metadata["goal_fk_task_positions_m"], [reach.get(i) for i in metadata["goal_node_indices"]],
                      flagged)
    report = {"seed": seed, "random_count": random_count, "mask_threshold_m": 0.05,
              "masked_actions": [int(i) for i in np.flatnonzero(np.asarray(metadata["goal_fk_error_m"]) > 0.05)],
              "goal_fk_task_positions_m": metadata["goal_fk_task_positions_m"],
              "summary": summarize(rows), "coverage": coverage(scenarios, rows, metadata),
              "scenarios": scenarios, "episodes": rows}
    (ROOT / "results/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = list(dict.fromkeys(k for r in rows for k in r
                                 if k not in ("commands", "swept_first_contact_ball_in_shield_m")))
    with open(ROOT / "results/evaluation_episodes.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": report["summary"], "coverage": report["coverage"]}, indent=2))


def resummarize():
    """Recompute summary/headroom from saved episodes without re-running them."""
    path = ROOT / "results/evaluation.json"
    report = json.loads(path.read_text())
    reach = json.loads((ROOT / "results/goal_reach_times.json").read_text())
    flagged = np.asarray(reach["goal_fk_error_m"]) > report["mask_threshold_m"]
    annotate_headroom(report["episodes"], report["goal_fk_task_positions_m"], reach["reach_time_s"], flagged)
    report["summary"] = summarize(report["episodes"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


def fixed_goal_oracle():
    """Simulate every allowed goal as a fixed action on each masked-baseline failure.

    An upper bound on what goal selection alone (one goal per episode, issued at t=0)
    could achieve on these throws, including hits while the shield is still moving.
    """
    path = ROOT / "results/evaluation.json"
    report = json.loads(path.read_text())
    scenarios = {s["name"]: s for s in report["scenarios"]}
    failed = sorted({r["scenario"] for r in report["episodes"] if r["mask"] == "masked" and not r["is_success"]})
    results = {}
    with ShieldInterceptionEnv(max_ticks=1000) as env:
        allowed = np.flatnonzero(env.action_masks())
        for name in failed:
            hits = []
            for action in allowed:
                env.reset(seed=0, options={"randomize": False, "throw": scenarios[name]["throw"],
                                           "launch_time_s": scenarios[name]["launch_time_s"]})
                while True:
                    _, _, terminated, truncated, info = env.step(int(action))
                    if terminated or truncated:
                        break
                if info["is_success"]:
                    hits.append(int(action))
            check = None
            if hits:  # Confirm one oracle hit with the sub-tick check.
                env.reset(seed=0, options={"randomize": False, "throw": scenarios[name]["throw"],
                                           "launch_time_s": scenarios[name]["launch_time_s"]})
                while not env.step(hits[0])[2]:
                    pass
                check = env.swept_check()["swept_first_contact_time_s"] is not None
            results[name] = {"category": scenarios[name]["category"], "intercepting_actions": hits,
                             "first_hit_swept_confirmed": check}
    report["fixed_goal_oracle"] = {
        "definition": "each allowed (masked env) goal as a fixed action from t=0; sampled overlap",
        "failed_scenarios": len(failed), "recoverable": sum(bool(v["intercepting_actions"]) for v in results.values()),
        "scenarios": results}
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report["fixed_goal_oracle"].items() if k != "scenarios"}, indent=2))
    for name, v in results.items():
        print(name, v["category"], len(v["intercepting_actions"]), v["first_hit_swept_confirmed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", action="store_true", help="fixed-goal oracle on masked failures")
    parser.add_argument("--summarize-only", action="store_true", help="recompute summary from saved episodes")
    parser.add_argument("--random", type=int, default=60, help="seeded random scenarios in addition to the grid")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.oracle:
        fixed_goal_oracle()
    elif args.summarize_only:
        resummarize()
    else:
        run(args.random, args.seed)


if __name__ == "__main__":
    main()
