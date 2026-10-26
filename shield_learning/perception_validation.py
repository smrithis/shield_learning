"""Fidelity check: calibrated synthetic perception vs the real exp_TD 015-021 flights.

The generator is run on each real throw's own reference trajectory (many seeds) and
compared with the recording on measurement statistics, published beliefs, belief
jumps and crossing-prediction error versus time to crossing. The reference is the
drag-absorbing fit from perception_calibration, so both sides share one "truth".
"""
import json
import sys

import numpy as np

from .perception import WS, MeasurementGenerator
from .perception_calibration import ROOT

sys.path.insert(0, str(WS / "src/belief_planner/belief-space-planner/scripts"))
from belief3d_live import propagate_until  # noqa: E402
from existing_belief_planner import ExistingBeliefPlanner  # noqa: E402

BINS = [0.0, 0.1, 0.2, 0.3, 0.6]
SEEDS = 200


def binned(rows):
    out = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        e = [r["crossing_error_m"] for r in rows if lo <= r["time_to_crossing_s"] < hi]
        out[f"{lo:.1f}-{hi:.1f}s"] = {"n": len(e), "median_m": float(np.median(e)) if e else None,
                                      "p90_m": float(np.percentile(e, 90)) if e else None}
    return out


def main():
    planner = ExistingBeliefPlanner(str(WS / "src/belief_planner/belief-space-planner/scripts/action_tree_3d_demo.pkl"))
    R, t = planner.R, planner.t
    report = json.loads((ROOT / "results/perception_calibration_report.json").read_text())
    generator = MeasurementGenerator()
    synthetic, per_trial = [], {}
    for trial in report["trials"]:
        p0, v0, a = (np.asarray(trial[k]) for k in ("reference_p0", "reference_v0", "reference_acceleration_m_s2"))
        t0 = trial["reference_t0_s"]
        lo, hi = trial["flight_window_s"]
        truth = lambda s: (p0 + v0 * (s - t0) + 0.5 * a * (s - t0) ** 2, v0 + a * (s - t0))
        times = np.linspace(lo, hi + 0.5, 20001)
        plane_x = np.array([(R.T @ (truth(s)[0] - t))[0] for s in times])
        crossing_time = times[np.flatnonzero(np.diff(np.sign(plane_x)))[0]]
        true_crossing = (R.T @ (truth(crossing_time)[0] - t))[1:]
        first, last = trial["throw_span_s"]
        stats = {"measurements": [], "outlier_fraction": [], "published": [], "jumps": [], "resets": []}
        for seed in range(SEEDS):
            ep = generator.episode(truth, hi, np.random.default_rng([seed, 99]), start_s=lo)
            m = ep["measurements"]
            stats["measurements"].append(len(m))
            stats["outlier_fraction"].append(np.mean([x["outlier"] for x in m]) if m else 0.0)
            pub = [b for b in ep["beliefs"] if b["published"]]
            in_throw = [b for b in pub if m and m[0]["stamp_s"] <= b["stamp_s"] <= m[-1]["stamp_s"]]
            stats["published"].append(len(in_throw))
            stats["jumps"].append(sum(np.linalg.norm(x["mu"][:3] - y["mu"][:3]) > 0.5
                                      for x, y in zip(in_throw[:-1], in_throw[1:])))
            stats["resets"].append(ep["resets"])
            for b in pub:
                if b["stamp_s"] <= crossing_time:
                    o, _, _ = propagate_until(b["mu"], b["P"], stop_x=0.0, R_bp=R, t_bp=t)
                    if o is not None:
                        synthetic.append({"trial": trial["trial"], "time_to_crossing_s": crossing_time - b["stamp_s"],
                                          "crossing_error_m": float(np.linalg.norm(np.asarray(o).ravel()[:2] - true_crossing))})
        per_trial[trial["trial"]] = {
            # Measurement/published counts from the real track start; outlier fraction, jumps and
            # restarts over the clean throw span (outliers are defined against it).
            "real": {"measurements": trial["tracked_measurements"],
                     "outlier_fraction": trial["outliers"] / trial["flight_measurements"],
                     "published": trial["tracked_published_beliefs"], "jumps": trial["belief_jumps_over_0p5m"],
                     "resets": trial["track_restarts_in_throw"]},
            "synthetic_median": {k: float(np.median(v)) for k, v in stats.items()},
            "synthetic_p10_p90": {k: np.percentile(v, [10, 90]).tolist() for k, v in stats.items()}}
    real = report["real_published_crossing_errors"]
    keys = ("measurements", "outlier_fraction", "published", "jumps", "resets")
    totals = {k: {"real_mean": float(np.mean([v["real"][k] for v in per_trial.values()])),
                  "synthetic_mean_of_medians": float(np.mean([v["synthetic_median"][k] for v in per_trial.values()])),
                  "real_within_synthetic_p10_p90": sum(v["synthetic_p10_p90"][k][0] <= v["real"][k]
                                                       <= v["synthetic_p10_p90"][k][1] for v in per_trial.values())}
              for k in keys}
    out = {"seeds_per_trial": SEEDS, "summary": totals, "per_trial": per_trial,
           "crossing_error_vs_time_to_crossing": {"real": binned(real), "synthetic": binned(synthetic)},
           "accepted_valid_crossing_predictions_per_throw": {"real": len(real) / len(report["trials"]),
                                                             "synthetic": len(synthetic) / (SEEDS * len(report["trials"]))}}
    (ROOT / "results/perception_validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("summary", "crossing_error_vs_time_to_crossing",
                                          "accepted_valid_crossing_predictions_per_throw")}, indent=1))


if __name__ == "__main__":
    main()
