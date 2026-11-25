"""Remaining stored-goal FK outliers after the frame correction (backend env only).

Read-only: reports which goals miss their stored plane target, why (reusing the
2026-10-04 study's per-goal classification), and where each goal's shield
actually goes. Nothing in the tree is changed or regenerated.
"""
import ast
import csv
import json

import mujoco
import numpy as np

from .backend import PLANNER_DIR, ROOT, WS

PRIOR_STUDY = WS / "analysis/2026-10/stored_goal_outliers_2026-10-04/bad_goals.csv"
THRESHOLDS = (1e-3, 1e-2, 5e-2, 1e-1)


def ik_clip_limits():
    """qmin/qmax literals from run_IK.py (narrower q5 than the XML joint range)."""
    limits = {}
    for node in ast.parse((PLANNER_DIR / "scripts/run_IK.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in ("qmin", "qmax"):
            limits[node.targets[0].id] = np.array(ast.literal_eval(node.value.args[0]))
    return limits["qmin"], limits["qmax"]


def analyze(backend, tolerance=1e-3):
    evaluator = backend.evaluator
    goals = backend.planner.goal_nodes
    lo, hi = ik_clip_limits()
    ik_rotation = np.array([[np.cos(np.deg2rad(135)), -np.sin(np.deg2rad(135)), 0],
                            [np.sin(np.deg2rad(135)), np.cos(np.deg2rad(135)), 0], [0, 0, 1]])
    targets = np.array([np.r_[0., n.task_position] for n in goals])
    prior = {int(r["node_id"]): r for r in csv.DictReader(open(PRIOR_STUDY))} if PRIOR_STUDY.exists() else {}
    model, data = evaluator.model, evaluator.data
    selections = {}
    baselines = ROOT / "results/baselines.json"
    if baselines.exists():
        for row in json.loads(baselines.read_text())["episodes"]:
            for index in row.get("selected_nodes", [row["selected_node_index"]]):
                selections.setdefault(index, []).append(f'{row["mode"]}:{row["seed"]}')
    rows = []
    for action, node in enumerate(goals):
        q = np.asarray(node.full_6dof_config)
        center, _ = evaluator.shield_geometry(q)
        _, rotation_world = evaluator.world_shield_geometry(q)
        error = center - targets[action]
        if np.linalg.norm(error) <= tolerance:
            continue
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)
        bound = np.minimum(np.abs(q - lo), np.abs(q - hi))
        others = np.linalg.norm(targets[:, 1:] - center[1:], axis=1)
        others[action] = np.inf
        nearest = int(np.argmin(others))
        cos = np.clip((np.trace(ik_rotation.T @ rotation_world) - 1) / 2, -1, 1)
        previous = prior.get(int(node.idx), {})
        rows.append({
            "action": action, "node_index": int(node.idx),
            "target_Y_m": targets[action][1], "target_Z_m": targets[action][2],
            "fk_plane_X_m": center[0], "fk_Y_m": center[1], "fk_Z_m": center[2],
            "xyz_error_m": float(np.linalg.norm(error)), "yz_error_m": float(np.linalg.norm(error[1:])),
            "orientation_error_deg": float(np.degrees(np.arccos(cos))),
            "exact_ik_clip_joints": [int(j + 1) for j in np.flatnonzero(bound <= 1e-6)],
            "near_ik_clip_joints_0p02rad": [int(j + 1) for j in np.flatnonzero(bound <= 0.02)],
            "model_contacts_at_q": int(data.ncon),
            "nearest_other_goal_action": nearest, "nearest_other_goal_yz_distance_m": float(others[nearest]),
            "prior_study_classification": previous.get("classification", "below prior 1 cm threshold"),
            "baseline_selections": selections.get(int(node.idx), []),
        })
    norms = np.array([r["xyz_error_m"] for r in rows])
    summary = {
        "goal_count": len(goals), "tolerance_m": tolerance, "outlier_count": len(rows),
        "count_over_m": {str(x): int(np.sum(norms > x)) for x in THRESHOLDS},
        "outliers_with_exact_ik_clip": sum(bool(r["exact_ik_clip_joints"]) for r in rows),
        "outliers_with_model_contacts": sum(r["model_contacts_at_q"] > 0 for r in rows),
        "prior_study_nodes_reproduced": sorted(set(prior) & {r["node_index"] for r in rows}),
        "prior_study_nodes_missing": sorted(set(prior) - {r["node_index"] for r in rows}),
        "baseline_episodes_selecting_outlier": sum(len(r["baseline_selections"]) for r in rows),
        "outlier_actions_over_1cm": [r["action"] for r in rows if r["xyz_error_m"] > 1e-2],
        "outlier_actions": [r["action"] for r in rows],
        "non_goal_tree_nodes": "root + 3 intermediate nodes carry no task_position or 6-DoF config; not FK-checkable",
        "tree_modified": False,
    }
    return summary, rows


def write(summary, rows):
    with open(ROOT / "results/goal_outliers.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    (ROOT / "results/goal_outliers.json").write_text(json.dumps({"summary": summary, "outliers": rows}, indent=2) + "\n")
