"""Calibrate measurement disturbances from exp_TD bags 015-021 (ros_new env, read-only).

There is no independent projectile truth in the bags. Each throw's reference is
found inside the flight window already established by
analysis/2026-10/riae_replay_validation_2026-10-05: RANSAC with a gravity-only ballistic model
selects inliers, then a constant-acceleration fit on those inliers absorbs drag
(the simulated projectile has none), so drag curvature is not counted as sensor
noise. Residuals to that reference define inlier noise and outliers; it is a
model-based reference, not ground truth.

Writes shield_learning/data/perception_calibration.json (committed) and
results/perception_calibration_report.json (+ figure).
"""
import csv
import hashlib
import importlib.util
import json
import sys

import numpy as np
import rosbag

from .perception import CALIBRATION, PROBE, WS, SlidingKalmanNode, load_sliding, ray_basis

ROOT = CALIBRATION.parents[2]
RIAE = WS / "analysis/2026-10/riae_replay_validation_2026-10-05"
BAGS = WS / "analysis/data/exp_TD/bags"
TRIALS = ["success_015", "success_016", "miss_017", "success_018", "miss_019", "success_020", "success_021"]
G = np.array([0.0, 0.0, -9.81])
INLIER_M = 0.08          # residual threshold separating inliers from outliers
FRAME_PERIOD = 0.01      # camera_fps = 100 in the node
NUM_FRAME = 4            # smallest window uniquely matching recorded outputs (probe.infer)
SEED = 20261025


def base_to_left():
    """Camera pose in odom_combined from params/zed2i.yaml (row-major 4x4)."""
    text = (WS / "src/shield_perception/params/zed2i.yaml").read_text()
    body = text[text.index("base_to_left:"):]
    values = [float(x) for x in body[body.index("[") + 1:body.index("]")].replace("\n", " ").split(",")]
    return np.array(values).reshape(4, 4)


def ballistic(t, p, w=None):
    A = np.c_[np.ones_like(t), t]
    y = p - 0.5 * G * t[:, None] ** 2
    return np.linalg.lstsq(A, y, rcond=None)[0]


def predict(fit, t0, t):
    acceleration = fit[2] if len(fit) > 2 else G
    return fit[0] + np.outer(t - t0, fit[1]) + 0.5 * np.outer((t - t0) ** 2, acceleration)


def quadratic(t, p):
    return np.linalg.lstsq(np.c_[np.ones_like(t), t, 0.5 * t ** 2], p, rcond=None)[0]


def reference(t, p, rng):
    """RANSAC gravity-only ballistic reference toward the robot (vx < -2 m/s)."""
    best = None
    for _ in range(4000):
        idx = np.sort(rng.choice(len(t), 3, replace=False))
        if t[idx[-1]] - t[idx[0]] < 0.03:
            continue
        fit = ballistic(t[idx] - t[idx[0]], p[idx])
        if fit[1][0] > -2:
            continue
        inl = np.linalg.norm(p - predict(fit, t[idx[0]], t), axis=1) < INLIER_M
        if best is None or inl.sum() > best[1].sum():
            best = (t[idx[0]], inl)
    t0, inl = best
    for _ in range(5):  # gravity-only refit on inliers, restricted to their contiguous span
        fit = ballistic(t[inl] - t0, p[inl])
        r = np.linalg.norm(p - predict(fit, t0, t), axis=1)
        span = (t >= t[inl].min()) & (t <= t[inl].max())
        inl = span & (r < INLIER_M)
    gravity_inliers = int(inl.sum())
    for _ in range(5):  # constant-acceleration (drag-absorbing) reference on the same span
        fit = quadratic(t[inl] - t0, p[inl])
        inl = span & (np.linalg.norm(p - predict(fit, t0, t), axis=1) < INLIER_M)
    return t0, fit, span, inl, gravity_inliers


def receipts(trial):
    """Per-topic (header_ns, receipt_ns) and planner request/result round trips."""
    out = {"/projectile": [], "/projectile_measurement": []}
    goals, results = {}, []
    with rosbag.Bag(str(BAGS / f"{trial}.bag")) as bag:
        for topic, m, t in bag.read_messages(topics=list(out) + ["/get_best_action/goal", "/get_best_action/result"]):
            if topic in out:
                out[topic].append((m.header.stamp.to_nsec(), t.to_nsec()))
            elif topic.endswith("goal"):
                goals[m.goal_id.id] = t.to_nsec()
            elif m.status.goal_id.id in goals:
                results.append((goals[m.status.goal_id.id], t.to_nsec(), m.status.status))
    return out, sorted(results)


def main():
    sys.path.insert(0, str(WS / "src/belief_planner/belief-space-planner/scripts"))
    from belief3d_live import propagate_until
    from existing_belief_planner import ExistingBeliefPlanner
    planner = ExistingBeliefPlanner(str(WS / "src/belief_planner/belief-space-planner/scripts/action_tree_3d_demo.pkl"))
    spec = importlib.util.spec_from_file_location("shield_existing_sliding_probe_read", PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    sliding = load_sliding()
    T = base_to_left()
    camera = T[:3, 3]
    rng = np.random.default_rng(SEED)
    tracks = list(csv.DictReader(open(RIAE / "confirmed_tracks.csv")))
    pooled = {k: [] for k in ("ray_residual", "distance", "outlier_ray", "intervals", "jitter",
                              "first_distance", "last_x", "belief_latency", "first_plan", "later_plan",
                              "aborted_plan", "lag1", "request_flight", "first_inlier_distance")}
    accept_counts = np.zeros((2, 2))   # [prev missed/accepted][now missed/accepted]
    outlier_counts = np.zeros((2, 2))  # [prev inlier/outlier][now inlier/outlier]
    trials, replica = [], {"compared": 0, "max_state_error": 0.0, "max_cov_error": 0.0, "gate_mismatch": 0}
    real_crossing = []
    for trial in TRIALS:
        ns, t, p, debug, start = probe.read(trial)
        window = json.loads((RIAE / trial / "metrics.json").read_text())
        flight = np.flatnonzero((t >= window["flight_start_s"] - 1e-6) & (t <= window["flight_end_s"] + 1e-6))
        t0, fit, span, inl, gravity_inliers = reference(t[flight], p[flight], rng)
        idx = flight[span]
        ref = predict(fit, t0, t[idx])
        res = p[idx] - ref
        dist = np.linalg.norm(ref - camera, axis=1)
        ray = np.array([ray_basis(r, camera) @ e for r, e in zip(ref, res)])
        good = inl[span]
        pooled["ray_residual"] += ray[good].tolist()
        pooled["distance"] += dist[good].tolist()
        pooled["outlier_ray"] += ray[~good].tolist()
        if good.sum() > 2:
            g = ray[good][:, 0]
            pooled["lag1"].append(float(np.corrcoef(g[:-1], g[1:])[0, 1]))
        for a, b in zip(good[:-1], good[1:]):
            outlier_counts[int(not a), int(not b)] += 1
        dts = np.diff(t[idx])
        k = np.maximum(1, np.round(dts / FRAME_PERIOD)).astype(int)
        pooled["intervals"] += dts.tolist()
        pooled["jitter"] += (dts - k * FRAME_PERIOD).tolist()
        for missed in k - 1:  # frame sequence: accepted, `missed` misses, accepted
            seq = [1] + [0] * int(missed) + [1]
            for a, b in zip(seq[:-1], seq[1:]):
                accept_counts[a, b] += 1
        # Detection begins where the real tracker began following this throw: the start of
        # the confirmed filter track containing the first inlier (earlier, noisier samples
        # of the same track count), evaluated on the reference trajectory.
        first_inlier = idx[np.flatnonzero(good)[0]]
        track_start = max(float(r["start_bag_s"]) for r in tracks
                          if r["trial"] == trial and int(r["first_raw_index"]) <= first_inlier)
        start_time = max(track_start, t0 - 0.3)
        start_point = predict(fit, t0, np.array([start_time]))[0]
        pooled["first_distance"].append(float(np.linalg.norm(start_point - camera)))
        pooled["first_inlier_distance"].append(float(dist[np.flatnonzero(good)[0]]))
        tracked = (t >= start_time - 1e-9) & (t <= t[idx][-1] + 1e-9)
        pooled["last_x"].append(float(ref[-1, 0]))
        topics, plans = receipts(trial)
        pooled["belief_latency"] += [(r - h) * 1e-9 for h, r in topics["/projectile"]]
        succeeded = [(r - g) * 1e-9 for g, r, s in plans if s == 3]
        if succeeded:
            pooled["first_plan"].append(succeeded[0])
            pooled["later_plan"] += succeeded[1:]
        pooled["aborted_plan"] += [(r - g) * 1e-9 for g, r, s in plans if s != 3]
        if topics["/projectile"]:  # every request issued while beliefs were being published
            lo, hi = topics["/projectile"][0][1], topics["/projectile"][-1][1]
            pooled["request_flight"] += [(r - g) * 1e-9 for g, r, s in plans if lo <= g <= hi]
        # Replica of the node frame loop on the recorded stream, restarting at the
        # recovered track boundaries (rejected frames are not recorded).
        starts = {int(r["first_raw_index"]) for r in tracks if r["trial"] == trial}
        published = {h for h, _ in topics["/projectile"]}
        node = SlidingKalmanNode(sliding, NUM_FRAME)
        for i in range(len(ns)):
            if i in starts:
                node.buffer = []
            estimate, pub = node.frame((ns[i] - start) * 1e-9, p[i])
            if estimate is not None and ns[i] in debug:
                mu, P = debug[ns[i]]
                replica["compared"] += 1
                replica["max_state_error"] = max(replica["max_state_error"], float(np.max(np.abs(estimate[0] - mu))))
                replica["max_cov_error"] = max(replica["max_cov_error"], float(np.max(np.abs(estimate[1] - P))))
                replica["gate_mismatch"] += int(pub != (ns[i] in published))
        # Real crossing-prediction error of published beliefs vs the reference crossing.
        times = np.linspace(t[idx][0] - 0.2, t[idx][-1] + 0.4, 4001)
        plane_x = (planner.R.T @ (predict(fit, t0, times) - planner.t).T)[0]
        cross = np.flatnonzero(np.diff(np.sign(plane_x)))
        ref_cross = None
        if len(cross):
            tc = times[cross[0]]
            ref_cross = (planner.R.T @ (predict(fit, t0, np.array([tc]))[0] - planner.t))[1:]
            for h, _ in topics["/projectile"]:
                s = (h - start) * 1e-9
                if not window["flight_start_s"] <= s <= tc:
                    continue
                mu, P = debug[h]
                o, _, _ = propagate_until(mu, P, stop_x=0.0, R_bp=planner.R, t_bp=planner.t)
                if o is not None:
                    real_crossing.append({"trial": trial, "time_to_crossing_s": float(tc - s),
                                          "crossing_error_m": float(np.linalg.norm(np.asarray(o).ravel()[:2] - ref_cross))})
        trials.append({"trial": trial, "flight_measurements": int(span.sum()), "inliers": int(good.sum()),
                       "outliers": int((~good).sum()), "release_speed_m_s": float(np.linalg.norm(fit[1])),
                       "reference_t0_s": float(t0), "reference_p0": fit[0],
                       "reference_v0": fit[1], "reference_acceleration_m_s2": fit[2],
                       "flight_window_s": [window["flight_start_s"], window["flight_end_s"]],
                       # Counted over the throw itself (first to last reference-span measurement).
                       "throw_span_s": [float(t[idx][0]), float(t[idx][-1])],
                       "published_beliefs_in_throw": int(sum(t[idx][0] <= (h - start) * 1e-9 <= t[idx][-1]
                                                             for h, _ in topics["/projectile"])),
                       "belief_jumps_over_0p5m": int(sum(np.linalg.norm(debug[a][0][:3] - debug[b][0][:3]) > 0.5
                                                         for (a, _), (b, _) in zip(topics["/projectile"][:-1],
                                                                                   topics["/projectile"][1:])
                                                         if t[idx][0] <= (a - start) * 1e-9
                                                         and (b - start) * 1e-9 <= t[idx][-1])),
                       "track_restarts_in_throw": int(sum(t[idx][0] < float(r["start_bag_s"]) <= t[idx][-1]
                                                          for r in tracks if r["trial"] == trial)),
                       # Same quantities from the track start (comparable with the generator).
                       "tracked_span_s": [start_time, float(t[idx][-1])],
                       "tracked_measurements": int(tracked.sum()),
                       "tracked_published_beliefs": int(sum(start_time - 1e-9 <= (h - start) * 1e-9 <= t[idx][-1] + 1e-9
                                                            for h, _ in topics["/projectile"])),
                       "gravity_only_inliers": gravity_inliers,
                       "inlier_rms_m": float(np.sqrt(np.mean(np.sum(res[good] ** 2, axis=1)))),
                       "reference_plane_crossing_yz_m": ref_cross,
                       "bag_sha256": hashlib.sha256((BAGS / f"{trial}.bag").read_bytes()).hexdigest()})
    ray = np.asarray(pooled["ray_residual"])
    d = np.asarray(pooled["distance"])
    # Each reference fits 9 parameters (3 per axis); correct the pooled variance.
    dof = np.sqrt(len(ray) / (len(ray) - 3 * len(TRIALS)))
    sigma_ray = ray.std(axis=0) * dof
    # sigma_depth = k d^2 (stereo), sigma_lateral = k d (pixel); MLE scale for each form.
    k_depth = float(np.sqrt(np.mean(ray[:, 0] ** 2 / d ** 4)))
    k_lat = float(np.sqrt(np.mean(ray[:, 1:] ** 2 / d[:, None] ** 2)))
    loglik = lambda r, s: float(np.sum(-np.log(s) - 0.5 * (r / s) ** 2))
    model_choice = {
        "depth_distance_squared_loglik": loglik(ray[:, 0], k_depth * d ** 2),
        "depth_constant_loglik": loglik(ray[:, 0], np.full_like(d, ray[:, 0].std())),
        "lateral_linear_loglik": loglik(ray[:, 1:].ravel(), np.repeat(k_lat * d, 2)),
        "lateral_constant_loglik": loglik(ray[:, 1:].ravel(), np.full(2 * len(d), ray[:, 1:].std())),
    }
    a = accept_counts / accept_counts.sum(axis=1, keepdims=True)
    o = outlier_counts / outlier_counts.sum(axis=1, keepdims=True)
    calibration = {
        "source": {"bags": [f"exp_TD/bags/{tr}.bag" for tr in TRIALS],
                   "bag_sha256": {x["trial"]: x["bag_sha256"] for x in trials},
                   "estimator_reconstruction": str(PROBE), "flight_windows": str(RIAE),
                   "reference": "RANSAC gravity-only ballistic fit; not independent truth",
                   "inlier_threshold_m": INLIER_M, "seed": SEED},
        "camera_position_odom_m": camera, "frame_period_s": FRAME_PERIOD, "num_frame": NUM_FRAME,
        "stamp_jitter_s": float(np.std(pooled["jitter"])),
        "acceptance_markov": {"after_accepted": float(a[1, 1]), "after_missed": float(a[0, 1])},
        "outlier_markov": {"after_inlier": float(o[0, 1]), "after_outlier": float(o[1, 1])},
        "inlier_noise": {"model": "constant" if model_choice["depth_constant_loglik"] >
                         model_choice["depth_distance_squared_loglik"] else "distance_scaled",
                         "sigma_ray_m": sigma_ray, "depth_per_m2": k_depth, "lateral_per_m": k_lat,
                         "dof_correction": dof, "model_choice": model_choice},
        "outlier_residual_ray_m": pooled["outlier_ray"],
        "visibility": {"first_detection_camera_distance_m": pooled["first_distance"],
                       "first_inlier_camera_distance_m": pooled["first_inlier_distance"],
                       "last_detection_x_m": pooled["last_x"]},
        "belief_latency_s": pooled["belief_latency"],
        "planning_round_trip_s": {"first_success": pooled["first_plan"], "later_success": pooled["later_plan"],
                                  "aborted_median": float(np.median(pooled["aborted_plan"])),
                                  # runtime_validation: plan() runs at the end of the round trip
                                  # (queueing, ~11% ~100 ms waits), on the latest belief then.
                                  "request_during_flight": pooled["request_flight"]},
        "exposure_offset_s": {"default": 0.0, "note": "constant exposure-to-stamp delay is unobservable "
                                                      "from positions; sweep it"},
    }
    CALIBRATION.parent.mkdir(exist_ok=True)
    write = lambda: CALIBRATION.write_text(json.dumps(calibration, indent=1,
                                                      default=lambda x: np.asarray(x).tolist()) + "\n")
    # Bags cannot separate rejected frames (count toward the 5-frame reset) from frames
    # the node loop never processed. Fit that one fraction to the real restart rate.
    real_restarts = float(np.mean([x["track_restarts_in_throw"] for x in trials]))
    fit = []
    for fraction in np.linspace(0.0, 1.0, 11):
        calibration["missed_frame_processed_fraction"] = float(fraction)
        write()
        from .perception import MeasurementGenerator
        generator = MeasurementGenerator()
        restarts = []
        for x in trials:
            p0, v0, acc, t0 = (np.asarray(x["reference_p0"]), np.asarray(x["reference_v0"]),
                               np.asarray(x["reference_acceleration_m_s2"]), x["reference_t0_s"])
            truth = lambda s: (p0 + v0 * (s - t0) + 0.5 * acc * (s - t0) ** 2, v0 + acc * (s - t0))
            for seed in range(100):
                ep = generator.episode(truth, x["flight_window_s"][1], np.random.default_rng([seed, 7]),
                                       start_s=x["flight_window_s"][0])
                restarts.append(ep["resets"])
        fit.append((abs(np.mean(restarts) - real_restarts), float(fraction), float(np.mean(restarts))))
    best = min(fit)
    calibration["missed_frame_processed_fraction"] = best[1]
    calibration["missed_frame_fit"] = {"real_restarts_per_throw": real_restarts,
                                       "grid": [{"fraction": f, "synthetic_restarts": r} for _, f, r in fit]}
    write()
    report = {"trials": trials, "node_replica_vs_recorded": replica,
              "inlier_count": len(ray), "outlier_count": len(pooled["outlier_ray"]),
              "inlier_ray_std_m": ray.std(axis=0), "inlier_lag1_depth_autocorrelation": pooled["lag1"],
              "accepted_interval_s_percentiles_5_50_95": np.percentile(pooled["intervals"], [5, 50, 95]),
              "real_published_crossing_errors": real_crossing}
    out = ROOT / "results/perception_calibration_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=lambda x: np.asarray(x).tolist()) + "\n")
    print(json.dumps({k: v for k, v in calibration.items() if k not in (
        "outlier_residual_ray_m", "belief_latency_s")}, indent=1, default=lambda x: np.round(np.asarray(x), 4).tolist()))
    print(json.dumps({k: report[k] for k in ("node_replica_vs_recorded", "inlier_count", "outlier_count",
                                              "inlier_ray_std_m", "inlier_lag1_depth_autocorrelation",
                                              "accepted_interval_s_percentiles_5_50_95")},
                     indent=1, default=lambda x: np.round(np.asarray(x), 4).tolist()))
    for x in trials:
        print(x["trial"], "inliers", x["inliers"], "(gravity-only", x["gravity_only_inliers"], ") outliers",
              x["outliers"], "speed", round(x["release_speed_m_s"], 2),
              "accel", np.round(x["reference_acceleration_m_s2"], 2), "rms", round(x["inlier_rms_m"], 4))


if __name__ == "__main__":
    main()
