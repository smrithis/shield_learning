"""Calibrated SHIELD perception: ground truth -> camera frames -> measurements -> beliefs.

Backend environment only. Three streams stay separate:
  * ground truth: the analytic projectile (`state_at`), never shown to the planner;
  * measurements: accepted detections with noise/outliers, stamped like the node;
  * beliefs: the unchanged SlidingKalman estimate, published only as the node does.

The estimator is the existing `sliding()` reconstruction (riae_replay_validation
probe.py), validated against all 397 recorded `/projectile_filtered_debug`
outputs of exp_TD 015-021. `SlidingKalmanNode` replicates only the frame loop of
track_ball_meancovariance_node.cpp (lines 772-933): buffer of <=10 accepted
measurements, `num_frame` readiness, clear after 5 consecutive unusable frames,
and publication only when |vx| > 2 m/s.
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

WS = Path(os.environ.get("SHIELD_WS", "/home/shield/code/smrithi_ws")).resolve()
PROBE = WS / "analysis/2026-10/riae_replay_validation_2026-10-05/probe.py"
CALIBRATION = Path(__file__).resolve().parent / "data/perception_calibration.json"
GRAVITY = np.array([0.0, 0.0, -9.81])
# Node constants (track_ball_meancovariance_node.cpp).
MAX_BUFFER, RESET_FRAMES, PUBLISH_MIN_ABS_VX = 10, 5, 2.0
WORKSPACE = ((0.8, 9.0), (-2.0, 2.0), (0.0, 2.8))  # filterPoints() base-frame box


def load_sliding():
    """The existing, validated SlidingKalman reconstruction (unchanged)."""
    spec = importlib.util.spec_from_file_location("shield_existing_sliding_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sliding


class SlidingKalmanNode:
    """Frame-loop replica of the perception node around the unchanged estimator."""

    def __init__(self, sliding, num_frame=4):
        self.sliding = sliding
        self.num_frame = max(3, int(num_frame))
        self.buffer = []
        self.reset_count = 0
        self.clear_stamps = []  # clears that discarded buffered measurements

    def clear(self, stamp):
        if self.buffer:
            self.clear_stamps.append(stamp)
        self.buffer = []

    def frame(self, stamp, position=None):
        """One camera frame. Returns (debug_estimate, published) or (None, False)."""
        if position is None:
            self.reset_count += 1
            if self.reset_count >= RESET_FRAMES:
                self.clear(stamp)
                self.reset_count = 0
            return None, False
        self.reset_count = 0
        self.buffer.append((float(stamp), np.asarray(position, dtype=float)))
        if len(self.buffer) > MAX_BUFFER:
            self.buffer.pop(0)
        if len(self.buffer) < self.num_frame:
            return None, False
        t = np.array([s for s, _ in self.buffer])
        p = np.array([q for _, q in self.buffer])
        if t[2] - t[0] <= 1e-6:
            return None, False
        mu, P = self.sliding(t - t[0], p)
        return (mu, P), bool(abs(mu[3]) > PUBLISH_MIN_ABS_VX)


def ray_basis(point, camera):
    """Depth (along camera ray), horizontal-lateral and vertical-lateral unit axes."""
    u = np.asarray(point) - camera
    u = u / np.linalg.norm(u)
    h = np.cross(u, [0.0, 0.0, 1.0])
    h = h / np.linalg.norm(h)
    return np.stack([u, h, np.cross(h, u)])


def in_workspace(p):
    return all(lo < p[i] < hi for i, (lo, hi) in enumerate(WORKSPACE))


class MeasurementGenerator:
    """Samples one episode's perception from the calibration; deterministic given rng.

    `scale` multiplies inlier noise; `outliers`, `dropouts`, `latency` toggle those
    processes; `exposure_offset_s` (unobservable in the bags, default 0) delays the
    stamp relative to the true exposure time.
    """

    def __init__(self, calibration=None, *, scale=1.0, outliers=True, dropouts=True, latency=True,
                 exposure_offset_s=0.0):
        self.cal = json.loads(Path(calibration or CALIBRATION).read_text())
        self.scale, self.outliers, self.dropouts, self.latency = float(scale), outliers, dropouts, latency
        self.exposure_offset = float(exposure_offset_s)
        self.camera = np.asarray(self.cal["camera_position_odom_m"])
        self.sliding = load_sliding()

    def episode(self, truth, horizon_s, rng, start_s=0.0):
        """truth(t) -> (position, velocity) in odom_combined for episode time t.

        Returns frames (all camera frames, accepted or not), measurements and beliefs,
        each with its own timestamps; ground truth is sampled at exposure times.
        """
        cal = self.cal
        period = cal["frame_period_s"]
        frames, measurements, beliefs = [], [], []
        node = SlidingKalmanNode(self.sliding, cal["num_frame"])
        visible_from = float(rng.choice(cal["visibility"]["first_detection_camera_distance_m"]))
        visible_until_x = float(rng.choice(cal["visibility"]["last_detection_x_m"]))
        accept = cal["acceptance_markov"]
        outlier = cal["outlier_markov"]
        noise = cal["inlier_noise"]
        pool = np.asarray(cal["outlier_residual_ray_m"])
        # Missing 10 ms slots are either processed-and-rejected frames (count toward the
        # 5-frame reset) or frames the slower node loop never processed (no effect).
        processed = cal.get("missed_frame_processed_fraction", 1.0)
        accepted_prev, outlier_prev, seen = True, False, False
        t = start_s + float(rng.uniform(0.0, period))
        while t <= horizon_s:
            exposure = t - self.exposure_offset
            stamp = t + float(rng.normal(0.0, cal["stamp_jitter_s"]))
            position, _ = truth(exposure)
            distance = float(np.linalg.norm(np.asarray(position) - self.camera))
            visible = in_workspace(position) and position[0] >= visible_until_x and (seen or distance <= visible_from)
            ok = visible
            if visible and seen and self.dropouts:
                p_accept = accept["after_accepted"] if accepted_prev else accept["after_missed"]
                ok = bool(rng.random() < p_accept)
            record = {"exposure_s": exposure, "stamp_s": stamp, "visible": visible, "accepted": ok}
            if ok:
                seen = True
                basis = ray_basis(position, self.camera)
                is_outlier = False
                if self.outliers:
                    p_out = outlier["after_outlier"] if outlier_prev else outlier["after_inlier"]
                    is_outlier = bool(rng.random() < p_out)
                if is_outlier:
                    residual = pool[rng.integers(len(pool))] @ basis
                else:
                    if noise["model"] == "constant":
                        sigma = np.asarray(noise["sigma_ray_m"]) * self.scale
                    else:
                        sigma = np.array([noise["depth_per_m2"] * distance ** 2, noise["lateral_per_m"] * distance,
                                          noise["lateral_per_m"] * distance]) * self.scale
                    residual = (rng.normal(0.0, 1.0, 3) * sigma) @ basis
                z = np.asarray(position) + residual
                measurements.append({"stamp_s": stamp, "exposure_s": exposure, "position": z,
                                     "true_position": np.asarray(position), "outlier": is_outlier})
                outlier_prev = is_outlier
                estimate, published = node.frame(stamp, z)
            elif visible and seen and rng.random() >= processed:
                estimate, published = None, False  # frame skipped by the node loop
            else:
                estimate, published = node.frame(stamp, None)
            accepted_prev = ok if seen else True
            if estimate is not None:
                delay = float(rng.choice(cal["belief_latency_s"])) if self.latency else 0.0
                beliefs.append({"stamp_s": stamp, "available_s": stamp + delay, "mu": estimate[0],
                                "P": estimate[1], "published": published})
            frames.append(record)
            t += period
        if measurements:  # restarts inside the tracked span (not the clear after the ball leaves view)
            first, last = measurements[0]["stamp_s"], measurements[-1]["stamp_s"]
            resets = sum(first < s < last for s in node.clear_stamps)
        else:
            resets = 0
        return {"frames": frames, "measurements": measurements, "beliefs": beliefs, "resets": resets}
