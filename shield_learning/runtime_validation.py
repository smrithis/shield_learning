"""Replay recorded beliefs and planner requests (exp_TD 015-021) through the planner.

Validates the request-loop semantics used by `original_runtime`: the planner
receives every recorded `/projectile` belief in receipt order (projectile_callback),
and each recorded `/get_best_action/goal` calls `plan(goal.cur_time)` once, which
consumes the belief. Two hypotheses for *when* plan() runs are compared:
`request` (at the goal's receipt) and `execution` (just before its result's
receipt, i.e. the round trip is queueing before planning). Recorded
SUCCEEDED/ABORTED outcomes and returned joint targets are compared with each
replay. Read-only; ros_new environment.
"""
import json
import sys

import numpy as np
import rosbag

from .perception import WS
from .perception_calibration import BAGS, ROOT, TRIALS

sys.path.insert(0, str(WS / "src/belief_planner/belief-space-planner/scripts"))
from existing_belief_planner import ExistingBeliefPlanner  # noqa: E402
from planner_types import ProjectileBelief  # noqa: E402

TREE = WS / "src/belief_planner/belief-space-planner/scripts/action_tree_3d_demo.pkl"


def replay(trial, planner, when="execution"):
    events, goals = [], {}
    with rosbag.Bag(str(BAGS / f"{trial}.bag")) as bag:
        for topic, m, t in bag.read_messages(topics=["/projectile", "/get_best_action/goal", "/get_best_action/result"]):
            events.append((t.to_nsec(), topic, m))
            if topic == "/get_best_action/goal":
                goals[m.goal_id.id] = m
    if when == "execution":  # run each plan() at its result's receipt instead of the request's
        events = [e for e in events if e[1] != "/get_best_action/goal"] + [
            (t - 1, "/get_best_action/goal", goals[m.status.goal_id.id]) for t, topic, m in events
            if topic == "/get_best_action/result" and m.status.goal_id.id in goals]
    events.sort(key=lambda e: e[0])
    planner.reset()
    planner.belief = planner.mu = planner.P = planner.proj_mean = None
    outcome = {}
    for _, topic, m in events:
        if topic == "/projectile":
            planner.update_belief(ProjectileBelief(np.array(m.mu), np.array(m.P).reshape(6, 6), m.header.stamp.to_sec()))
        elif topic == "/get_best_action/goal":
            result = planner.plan(m.goal.cur_time)
            outcome[m.goal_id.id] = None if result is None else np.r_[result.position, result.position2]
    rows = []
    for _, topic, m in events:
        if topic == "/get_best_action/result" and m.status.goal_id.id in outcome:
            replayed = outcome[m.status.goal_id.id]
            recorded_ok = m.status.status == 3
            row = {"recorded_succeeded": recorded_ok, "replay_succeeded": replayed is not None}
            if recorded_ok and replayed is not None:
                row["target_error_rad"] = float(np.max(np.abs(np.r_[m.result.position, m.result.position2] - replayed)))
            rows.append(row)
    return rows


def main():
    planner = ExistingBeliefPlanner(str(TREE))
    report = {}
    for when in ("request", "execution"):
        report[when] = {}
        for trial in TRIALS:
            rows = replay(trial, planner, when)
            errors = [r["target_error_rad"] for r in rows if "target_error_rad" in r]
            report[when][trial] = {
                "results": len(rows),
                "outcome_agreement": sum(r["recorded_succeeded"] == r["replay_succeeded"] for r in rows),
                "recorded_successes": sum(r["recorded_succeeded"] for r in rows),
                "replay_successes": sum(r["replay_succeeded"] for r in rows),
                "both_succeeded": len(errors), "exact_target_matches": sum(e < 1e-9 for e in errors),
                "max_target_error_rad": max(errors) if errors else None}
        keys = ("results", "outcome_agreement", "recorded_successes", "replay_successes", "both_succeeded",
                "exact_target_matches")
        trials = list(report[when].values())
        report[when]["total"] = {k: sum(v[k] for v in trials) for k in keys}
        report[when]["total"]["max_target_error_rad"] = max(v["max_target_error_rad"] or 0 for v in trials)
    (ROOT / "results/runtime_validation.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({w: report[w]["total"] for w in report}, indent=1))


if __name__ == "__main__":
    main()
