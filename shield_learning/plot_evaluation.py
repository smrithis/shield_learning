"""Plane-coverage figure for results/evaluation.json (ros_new matplotlib; not a learning dependency)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "results/evaluation.json").read_text())
    reach = json.loads((ROOT / "results/goal_reach_times.json").read_text())
    errors = np.asarray(reach["goal_fk_error_m"])
    episodes = report["episodes"]
    fk = np.asarray(report["goal_fk_task_positions_m"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, mask in zip(axes, ("unmasked", "masked")):
        flagged = errors > report["mask_threshold_m"]
        ax.scatter(fk[~flagged, 0], fk[~flagged, 1], s=10, c="0.6", label="goal (FK shield centre)")
        ax.scatter(fk[flagged, 0], fk[flagged, 1], s=40, marker="x", c="tab:purple",
                   label=f"FK error > {report['mask_threshold_m']} m" + (" (masked)" if mask == "masked" else ""))
        rows = [e for e in episodes if e["mask"] == mask and e["policy"] == "original_planner"
                and e.get("crossing_yz_m") is not None]
        for success, colour, label in ((True, "tab:green", "intercepted"), (False, "tab:red", "missed")):
            pts = np.array([r["crossing_yz_m"] for r in rows if r["is_success"] == success]).reshape(-1, 2)
            ax.scatter(pts[:, 0], pts[:, 1], s=28, facecolors="none", edgecolors=colour,
                       label=f"ball crossing, {label} ({len(pts)})")
        ax.set_title(f"original planner, {mask}")
        ax.set_xlabel("plane Y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("plane Z (m)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/evaluation_coverage.png", dpi=130)


if __name__ == "__main__":
    main()
