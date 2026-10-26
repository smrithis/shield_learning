"""SHIELD worker. Uses ros_new dependencies; never imports learning libraries."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .controller import Controller
from .frames import SweptChecker, UrdfShield, ik_world_from_plane, plane_frame_evaluator
from .perception import MeasurementGenerator
from .runtime import LocalRuntime, load_simulator

ROOT = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("SHIELD_WS", "/home/shield/code/smrithi_ws")).resolve()
PLANNER_DIR = WS / "src/belief_planner/belief-space-planner"
SIM_DIR = WS / "src/shield_simulation/scripts"
URDF_PATH = WS / "src/shield_planner/shield_planner/robots/abb/urdf/abb_irb1600_6_12_realshield_no_lidar.urdf"
sys.path.insert(0, str(PLANNER_DIR / "scripts"))

from existing_belief_planner import ExistingBeliefPlanner
from planner_types import ProjectileBelief, PlannerResult
from std_msgs.msg import Float64MultiArray


GRAVITY = 9.81
PERCEPTION_MODES = ("oracle", "timed_truth", "calibrated")
RUCKIG_FINISHED = 1
# Box half-width plus ball radius: a ball centre this far (max |dY|,|dZ|) from the shield
# centre in the plane can still overlap a plane-aligned shield. Diagnostic only.
CAPTURE_HALF_WIDTH = 0.17 + 0.0381


class MaskedPlanner(ExistingBeliefPlanner):
    """Unmodified planner that ignores masked goal nodes in both scoring paths."""
    masked = frozenset()

    def plan(self, cur_time):
        if not self.masked:
            return super().plan(cur_time)
        goals = self.goal_nodes
        saved = [(node, node.time_to_goals) for node in self.terminal_nodes]
        self.goal_nodes = [n for n in goals if n.idx not in self.masked]
        for node, times in saved:
            node.time_to_goals = {k: v for k, v in times.items() if k not in self.masked}
        try:
            return super().plan(cur_time)
        finally:
            self.goal_nodes = goals
            for node, times in saved:
                node.time_to_goals = times


class Backend:
    dt = 0.004

    def __init__(self):
        self.planner = MaskedPlanner(str(PLANNER_DIR / "scripts/action_tree_3d_demo.pkl"))
        self.runtime = LocalRuntime({
            "~model_path": str(PLANNER_DIR / "irb_1600/irb1600_6_12_realshield.xml"),
            "~output_path": str(ROOT / "results/unused_ros_evaluator.json"),
        })
        arm = load_simulator(SIM_DIR / "sim_arm_bridge.py", self.runtime)
        projectile = load_simulator(SIM_DIR / "sim_projectile_publisher.py", self.runtime)
        geometry = load_simulator(SIM_DIR / "shield_projectile_evaluator.py", self.runtime)
        self.arm = arm.SimArmBridge()
        self.projectile_class = projectile.SimProjectilePublisher
        # Upstream passes raw MuJoCo-world FK as plane coordinates; correct only that.
        self.evaluator = plane_frame_evaluator(geometry.ShieldProjectileEvaluator)()
        self.runtime.start(self.arm)
        self.controller = None
        self.done = True
        self.node_to_action = {node.idx: i for i, node in enumerate(self.planner.goal_nodes)}
        self.config = {}
        self.goal_fk_error = [float(np.linalg.norm(self.evaluator.shield_geometry(
            np.asarray(n.full_6dof_config))[0] - np.r_[0., n.task_position])) for n in self.planner.goal_nodes]
        self.swept = SweptChecker(self.runtime.params["~model_path"], self.evaluator.ball_radius)
        self.launch_time = 0.0
        self.planning_pools = json.loads(
            (ROOT / "shield_learning/data/perception_calibration.json").read_text())["planning_round_trip_s"]
        self.perception = "oracle"
        self.flagged_nodes = set()
        self.mask_threshold = 0.05
        self.valid = np.ones(len(self.goal_fk_error), dtype=bool)

    def ball_state(self, t):
        """Projectile state at episode time t; launched at self.launch_time (<= 0)."""
        return self.projectile.state_at(t - self.launch_time)

    def metadata(self):
        return {
            "n_actions": len(self.planner.goal_nodes),
            "goal_node_indices": [int(n.idx) for n in self.planner.goal_nodes],
            "goal_joint_positions": [np.asarray(n.full_6dof_config).tolist() for n in self.planner.goal_nodes],
            "dt": self.dt,
            "frame": "odom_combined",
            "shield_geometry_frame_correction": "MuJoCo world -> planner plane via T_WP = T_WB T_BP",
            "goal_fk_error_m": self.goal_fk_error,
            "base_from_plane_rotation": self.planner.R, "base_from_plane_translation": self.planner.t,
            "goal_task_positions_m": [np.asarray(n.task_position).tolist() for n in self.planner.goal_nodes],
            "goal_fk_task_positions_m": [self.evaluator.shield_geometry(np.asarray(n.full_6dof_config))[0][1:]
                                         for n in self.planner.goal_nodes],
            "evaluation": "existing finite-box sphere overlap sampled every 4 ms; not contact dynamics",
            "tree_path": str(PLANNER_DIR / "scripts/action_tree_3d_demo.pkl"),
            "model_path": self.runtime.params["~model_path"],
        }

    def reset(self, seed=0, options=None, config=None):
        self.config = config or {}
        self.substeps = int(self.config.get("substeps", 10))
        self.max_ticks = int(self.config.get("max_ticks", 500))
        if self.substeps < 1 or self.max_ticks < 1:
            raise ValueError("substeps and max_ticks must be positive")
        threshold = self.config.get("goal_mask_fk_error_m", 0.05)
        self.mask_threshold = float(threshold)
        if not self.mask_threshold > 0:
            raise ValueError("goal_mask_fk_error_m must be positive")
        flagged = np.asarray(self.goal_fk_error) > self.mask_threshold
        self.valid = ~flagged if self.config.get("apply_goal_mask", True) else np.ones_like(flagged)
        self.flagged_nodes = {int(n.idx) for n, f in zip(self.planner.goal_nodes, flagged) if f}
        self.planner.masked = frozenset(int(n.idx) for n, ok in zip(self.planner.goal_nodes, self.valid) if not ok)
        options = options or {}
        allowed = {"randomize", "throw", "launch_time_s"}
        if set(options) - allowed:
            raise ValueError(f"Unknown reset options: {set(options) - allowed}")
        throw = options.get("throw", {})
        allowed_throw = {"x0", "y0", "z0", "speed", "launch_angle_deg", "side_angle_deg"}
        if set(throw) - allowed_throw:
            raise ValueError("Unknown projectile parameter")
        if not all(np.isfinite(float(x)) for x in throw.values()):
            raise ValueError("Projectile parameters must be finite")
        if float(throw.get("speed", 6.5)) <= 0 or float(throw.get("z0", 1.1)) < 0:
            raise ValueError("Throw needs positive speed and nonnegative height")
        launch_time = float(options.get("launch_time_s", 0.0))
        if not -1.0 <= launch_time <= 0.0:
            raise ValueError("launch_time_s must be in [-1, 0]: the ball is already in flight at reset")
        self.launch_time = launch_time
        self.runtime.time = 1.0
        self.arm.reset(None)
        if self.controller:
            self.controller.close()
        self.controller = Controller(self.arm.positions)
        for key in allowed_throw:
            self.runtime.params.pop("~" + key, None)
        self.runtime.params.update({"~" + k: float(v) for k, v in throw.items()})
        self.runtime.params["~randomize_each_throw"] = bool(options.get("randomize", True))
        self.runtime.params["~random_seed"] = int(seed)
        self.projectile = self.projectile_class()
        self.planner.reset()
        # Upstream reset only resets the node index; discard all episode-local state.
        self.planner.belief = self.planner.mu = self.planner.P = self.planner.proj_mean = None
        self.planner.compute_values_times.clear()
        self.planner.compute_values_call_count = 0
        self.ticks = 0
        self.done = False
        self.last_command_node = None
        self.last_goal_action = None
        self.path_length = 0.0
        self.velocity_cost = 0.0
        self.min_surface = self.surface_distance()
        self.initial_q = np.array(self.arm.positions)
        self.q_history = [self.initial_q.copy()]
        self.max_displacement = 0.0
        self.goal_switches = 0
        self.command_count = 0
        self.commands = []
        self._reset_perception(int(seed))
        self.initial_mean = self.observation()["projectile_mean"].copy()
        info = self.info()
        info["action_mask"] = self.valid.tolist()
        return {"observation": self.observation(), "info": info}

    def _reset_perception(self, seed):
        """Ground truth stays in `ball_state`; beliefs come from the configured perception."""
        cfg = self.config
        self.perception = cfg.get("perception", "oracle")
        if self.perception not in PERCEPTION_MODES:
            raise ValueError(f"perception must be one of {PERCEPTION_MODES}")
        self.original_runtime = bool(cfg.get("original_runtime", False))
        self.trace = None
        self.belief_queue, self.queue_index, self.latest_belief = [], 0, None
        if self.perception != "oracle":
            options = {k: cfg[f"perception_{k}"] for k in ("scale", "outliers", "dropouts", "latency")
                       if f"perception_{k}" in cfg}
            generator = MeasurementGenerator(exposure_offset_s=float(cfg.get("exposure_offset_s", 0.0)), **options)
            truth = lambda t: (np.asarray(self.ball_state(t)[:3]), np.asarray(self.ball_state(t)[3:]))
            self.trace = generator.episode(truth, self.max_ticks * self.dt + 0.05,
                                           np.random.default_rng([seed, 1]))
            for belief in self.trace["beliefs"]:
                belief["true_state"] = np.asarray(self.ball_state(belief["stamp_s"] - generator.exposure_offset))
            self.belief_queue = [b for b in self.trace["beliefs"] if b["published"]]
        # Request-loop state of controller.cpp (used when original_runtime is on).
        rng = np.random.default_rng([seed, 2])
        pools = self.planning_pools
        latency = bool(cfg.get("planning_latency", True)) and self.original_runtime
        # Per-request queueing delay before plan() runs (runtime_validation: 8971/8976 outcomes).
        self.plan_latencies = rng.choice(pools["request_during_flight"], 4096).tolist() if latency else [0.0] * 4096
        self.rt = {"pending": None, "next_request": 0, "executing": False, "exec_start": None,
                   "finished": None, "wait": True, "successes": 0, "requests": 0, "ignored": 0,
                   "cycle": 0, "measure": True}

    def _deliver_beliefs(self):
        """Hand beliefs whose receipt time has passed to the planner, as projectile_callback does."""
        if self.perception == "oracle":
            return
        while self.queue_index < len(self.belief_queue) and \
                self.belief_queue[self.queue_index]["available_s"] <= self.elapsed + 1e-12:
            belief = self.belief_queue[self.queue_index]
            self.queue_index += 1
            mean = belief["true_state"] if self.perception == "timed_truth" else belief["mu"]
            self.latest_belief = (belief, np.asarray(mean))
            self.planner.update_belief(ProjectileBelief(np.asarray(mean), np.asarray(belief["P"]), belief["stamp_s"]))

    def _command(self, result):
        """Send a planner/goal result to the controller unless the original gates forbid it."""
        if self.original_runtime and self.rt["finished"] is not None:
            self.rt["ignored"] += 1  # controller.cpp: execution ended, stop then home; no retarget.
            return 0
        switched = 0
        if result.selected_node_index != self.last_command_node:
            switched = int(self.last_command_node is not None)
            self.controller.command(result)
            self.last_command_node = int(result.selected_node_index)
            self.command_count += 1
            self.goal_switches += switched
            self.commands.append((self.ticks, self.last_command_node))
        if not self.rt["executing"]:
            self.rt["executing"], self.rt["exec_start"] = True, self.ticks
        self.rt["wait"] = bool(result.wait)
        self.rt["measure"] = True  # controller.cpp calls updateState before applying a new goal
        return switched

    def _runtime_tick(self):
        """One 4 ms cycle of controller.cpp's request loop (idle loop and executeTrajectory).

        A request carries cur_time from when it is sent; the action server runs plan()
        after the sampled queueing delay, on the latest belief at that moment, and the
        result is applied on arrival. The next request goes out on the following cycle.
        """
        rt, k = self.rt, self.ticks
        switched = 0
        if rt["pending"] is not None:
            due, cur_time = rt["pending"]
            if k < due:
                return switched
            rt["pending"] = None
            result = self.planner.plan(cur_time)  # consumes the latest belief, as the action server does
            if result is not None:
                rt["successes"] += 1
                switched = self._command(result)
            rt["next_request"] = k + 1
            return switched
        if rt["finished"] is not None or k < rt["next_request"]:
            return switched
        if rt["executing"] and rt["wait"]:
            # shouldRequestReplan: tree (wait) results replan only near the first-3-joint target.
            target = np.asarray(self.planner.graph[self.last_command_node].position[:3])
            if np.linalg.norm(target - np.asarray(self.arm.positions[:3])) >= 0.1:
                return switched
        cur_time = (k - rt["exec_start"]) * self.dt if rt["executing"] else 0.0
        delay = int(np.ceil(self.plan_latencies[rt["requests"] % len(self.plan_latencies)] / self.dt - 1e-9))
        rt["requests"] += 1
        rt["pending"] = (k + delay, cur_time)
        return self._runtime_tick() if delay == 0 else switched

    @property
    def elapsed(self):
        return self.ticks * self.dt

    def observation(self):
        joints = {"joint_position": np.asarray(self.arm.positions, dtype=np.float64),
                  "joint_velocity": np.asarray(self.arm.velocities, dtype=np.float64)}
        if getattr(self, "perception", "oracle") == "oracle":
            state = self.ball_state(self.elapsed)
            msg = self.projectile.projectile_message(state, self.runtime.Time.now())
            return {"projectile_mean": np.asarray(msg.mu, dtype=np.float64),
                    "projectile_covariance": np.asarray(msg.P, dtype=np.float64).reshape(6, 6),
                    "belief_valid": np.ones(1), "belief_age_s": np.zeros(1), **joints}
        if self.latest_belief is None:  # nothing published yet: explicit flag, no invented state
            return {"projectile_mean": np.zeros(6), "projectile_covariance": np.zeros((6, 6)),
                    "belief_valid": np.zeros(1), "belief_age_s": np.zeros(1), **joints}
        belief, mean = self.latest_belief
        return {"projectile_mean": mean.astype(np.float64), "projectile_covariance": np.asarray(belief["P"]),
                "belief_valid": np.ones(1), "belief_age_s": np.array([self.elapsed - belief["stamp_s"]]), **joints}

    def surface_distance(self):
        ball = np.asarray(self.ball_state(self.elapsed)[:3])
        return float(self.evaluator.distance_at(ball, np.asarray(self.arm.positions))[0])

    def info(self, reason=None):
        return {"elapsed_s": self.elapsed, "is_success": reason == "interception",
                "termination_reason": reason, "minimum_surface_distance_m": self.min_surface,
                "joint_path_length_rad": self.path_length,
                "integrated_squared_velocity": self.velocity_cost,
                "max_joint_displacement_rad": self.max_displacement,
                "goal_switches": self.goal_switches, "command_count": self.command_count,
                "selected_node_index": self.last_command_node,
                "selected_goal_node_index": self.last_goal_node(),
                "selected_goal_fk_error_m": self.goal_fk_error[self.node_to_action[self.last_goal_node()]]
                if self.last_goal_node() is not None else None,
                "selected_goal_flagged": self.last_goal_node() in self.flagged_nodes,
                "goal_mask_applied": bool(not self.valid.all()),
                "perception": getattr(self, "perception", "oracle"),
                "original_runtime": bool(getattr(self, "original_runtime", False)),
                # Privileged: for diagnostics/teachers only, never part of the observation.
                "privileged_true_projectile_state": np.asarray(self.ball_state(self.elapsed)),
                "observation_frame": "odom_combined", "reward_is_physical_contact": False}

    def last_goal_node(self):
        """Last commanded node that is a goal (the planner also commands intermediate nodes)."""
        goals = [n for _, n in getattr(self, "commands", []) if n in self.node_to_action]
        return goals[-1] if goals else None

    def _update_belief(self):
        if self.perception != "oracle":
            self._deliver_beliefs()
            return
        obs = self.observation()
        self.planner.update_belief(ProjectileBelief(obs["projectile_mean"],
                                                   obs["projectile_covariance"], self.elapsed))

    def baseline_action(self):
        """Existing planner's direct-goal scoring, restricted to the Gym action space."""
        if self.done:
            raise RuntimeError("reset() required")
        self._update_belief()
        if self.planner.mu is None:
            if self.last_goal_action is None:
                raise RuntimeError("Existing planner cannot predict a valid crossing for this throw")
            return self.last_goal_action
        self.planner.compute_goal_values(self.planner.mu, self.planner.P)
        node = max((n for n, ok in zip(self.planner.goal_nodes, self.valid) if ok), key=lambda n: n.value)
        return self.node_to_action[node.idx]

    def goal_result(self, action):
        node = self.planner.goal_nodes[action]
        return PlannerResult(np.asarray(node.position), np.zeros(3), np.zeros(3),
                             np.asarray(node.full_6dof_config[3:6]), 0.0, False, node.idx, node.value)

    def step(self, action=None, planner=False, hold=False):
        if self.done:
            raise RuntimeError("Episode has ended; call reset()")
        runtime_loop = planner and self.original_runtime
        if hold or runtime_loop:
            result = None  # hold: no valid prediction yet; runtime loop: decided per tick below
        elif planner:
            self._update_belief()
            result = self.planner.plan(self.elapsed)
        else:
            if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(self.planner.goal_nodes):
                raise ValueError("Action must index an existing interception goal")
            if not self.valid[action]:
                raise ValueError(f"Action {action} is masked: goal FK error "
                                 f"{self.goal_fk_error[action]:.4f} m > {self.mask_threshold} m")
            result = self.goal_result(action)
            self.last_goal_action = action
        switched = 0
        if result is not None:
            switched = self._command(result)
        motion = 0.0
        reason = None
        for _ in range(self.substeps):
            # Also catches a deliberately overlapping initial geometry test fixture.
            if self.min_surface <= 0:
                reason = "interception"
                break
            if runtime_loop:
                if self.perception == "oracle":
                    self._update_belief()  # oracle: a fresh exact belief every 4 ms cycle
                else:
                    self._deliver_beliefs()
                switched += self._runtime_tick()
            elif self.perception != "oracle":
                self._deliver_beliefs()
            q_before = np.asarray(self.arm.positions)
            measured = q_before
            if self.original_runtime and self.rt["executing"]:
                # executeTrajectory feeds measured joints to Ruckig only every 15 cycles (and
                # on a new goal); other cycles run open-loop on Ruckig's own state.
                measured = q_before if self.rt["measure"] or self.rt["cycle"] % 15 == 0 else None
                self.rt["measure"] = False
                self.rt["cycle"] += 1
            velocity = self.controller.tick(measured)
            if self.original_runtime and self.rt["executing"] and self.rt["finished"] is None \
                    and self.controller.last_result == RUCKIG_FINISHED:
                self.rt["finished"] = self.ticks
            self.arm.command_callback(Float64MultiArray(data=velocity.tolist()))
            self.runtime.tick(self.arm.dt)
            self.ticks += 1
            q_after = np.asarray(self.arm.positions)
            self.q_history.append(q_after.copy())
            increment = float(np.linalg.norm(q_after - q_before))
            self.path_length += increment
            cost = self.dt * float(np.dot(self.arm.velocities, self.arm.velocities))
            self.velocity_cost += cost
            motion += cost
            self.max_displacement = max(self.max_displacement, float(np.linalg.norm(q_after - self.initial_q)))
            self.min_surface = min(self.min_surface, self.surface_distance())
            if self.min_surface <= 0:
                reason = "interception"
                break
            if self.ball_state(self.elapsed)[2] < 0:
                reason = "ground"
                break
            if self.ticks >= self.max_ticks:
                reason = "time_limit"
                break
        terminated = reason in ("interception", "ground")
        truncated = reason == "time_limit"
        self.done = terminated or truncated
        terms = {"interception": float(self.config.get("success_reward", 10.0)) if reason == "interception" else 0.0,
                 "motion": -float(self.config.get("motion_weight", 0.01)) * motion,
                 "goal_switch": -float(self.config.get("switch_weight", 0.01)) * switched}
        info = self.info(reason)
        info["reward_terms"] = terms
        info["policy_interface"] = "hold" if hold else "ExistingBeliefPlanner.plan" if planner else "discrete_goal"
        if self.original_runtime:
            info["runtime"] = {"plan_requests": self.rt["requests"], "plan_successes": self.rt["successes"],
                               "ignored_after_finish": self.rt["ignored"],
                               "finished_s": None if self.rt["finished"] is None else self.rt["finished"] * self.dt,
                               # controller.cpp would start goHome 1 s after finishing; not modelled.
                               "home_sequence_due": self.rt["finished"] is not None
                               and self.elapsed - self.rt["finished"] * self.dt > 1.0}
        return {"observation": self.observation(), "reward": sum(terms.values()),
                "terminated": terminated, "truncated": truncated, "info": info}

    def goal_reach_times(self, horizon_s=3.0, tolerance_rad=0.01):
        """Time from home until all joints stay within tolerance of each goal (fixed action)."""
        far = {"x0": 100., "y0": 100., "z0": 1000., "speed": 1., "launch_angle_deg": 0., "side_angle_deg": 0.}
        times = []
        for action, node in enumerate(self.planner.goal_nodes):
            self.reset(options={"randomize": False, "throw": far},
                       config={"max_ticks": int(horizon_s / self.dt), "apply_goal_mask": False})
            while not self.done:
                self.step(action)
            error = np.max(np.abs(np.asarray(self.q_history) - np.asarray(node.full_6dof_config)), axis=1)
            outside = np.flatnonzero(error >= tolerance_rad)
            reached = error[-1] < tolerance_rad
            times.append(float((outside[-1] + 1) * self.dt) if reached and len(outside) else 0.0 if reached else None)
        return times

    def perception_trace(self):
        """Separate streams for the last episode: truth, measurements, beliefs (all published flags)."""
        if self.trace is None:
            return {"perception": self.perception, "measurements": [], "beliefs": []}
        R, t = self.planner.R, self.planner.t
        x0, vx = R[:, 0] @ (np.asarray(self.ball_state(0.0)[:3]) - t), R[:, 0] @ np.asarray(self.ball_state(0.0)[3:])
        crossing_time = -x0 / vx if vx else None
        true_crossing = (R.T @ (np.asarray(self.ball_state(crossing_time)[:3]) - t))[1:] if crossing_time else None
        from belief3d_live import propagate_until
        beliefs = []
        for b in self.trace["beliefs"]:
            row = {k: b[k] for k in ("stamp_s", "available_s", "published")}
            row.update(mu=b["mu"], P=b["P"], true_state=b["true_state"],
                       position_error_m=float(np.linalg.norm(b["mu"][:3] - b["true_state"][:3])),
                       velocity_error_m_s=float(np.linalg.norm(b["mu"][3:] - b["true_state"][3:])))
            if true_crossing is not None and b["published"] and b["stamp_s"] <= crossing_time:
                o, _, _ = propagate_until(b["mu"], b["P"], stop_x=0.0, R_bp=R, t_bp=t)
                row["time_to_crossing_s"] = crossing_time - b["stamp_s"]
                row["crossing_error_m"] = None if o is None else float(np.linalg.norm(np.asarray(o).ravel()[:2] - true_crossing))
            beliefs.append(row)
        return {"perception": self.perception, "frames": len(self.trace["frames"]),
                "visible_frames": sum(f["visible"] for f in self.trace["frames"]),
                "resets": self.trace["resets"], "true_crossing_time_s": crossing_time,
                "measurements": self.trace["measurements"], "beliefs": beliefs}

    def episode_diagnostics(self):
        """Analytic plane crossing of the ball and why the episode did or did not intercept.

        Gaps are Chebyshev (max |dY|, |dZ|) distances in the plane, compared with the
        shield half-width plus ball radius. This is a diagnostic attribution, not a
        second success criterion.
        """
        R, t = self.planner.R, self.planner.t
        start = np.asarray(self.ball_state(0.0))
        # The plane x axis is horizontal, so the ball's plane-x is linear in time.
        x0, vx = R[:, 0] @ (start[:3] - t), R[:, 0] @ start[3:]
        crossing_time = -x0 / vx if vx != 0 and -x0 / vx >= 0 else None
        if crossing_time is not None and self.ball_state(crossing_time)[2] < 0:
            crossing_time = None
        success = self.min_surface <= 0
        result = {"ball_plane_crossing_time_s": crossing_time, "launch_time_s": self.launch_time,
                  "commands": [(k * self.dt, n) for k, n in self.commands]}
        if crossing_time is None:
            result["failure_cause"] = None if success else "no_plane_crossing"
            return result
        state = np.asarray(self.ball_state(crossing_time))
        crossing = (R.T @ (state[:3] - t))[1:]
        speed = float(np.linalg.norm(state[3:]))
        incidence = float(np.degrees(np.arccos(min(1.0, abs(R[:, 0] @ state[3:]) / speed))))
        before = [n for k, n in self.commands if k * self.dt <= crossing_time and n in self.node_to_action]
        goal = before[-1] if before else self.last_goal_node()
        qs = np.asarray(self.q_history)
        k = min(crossing_time / self.dt, len(qs) - 1)
        lo = int(np.floor(k))
        q_cross = qs[lo] + (k - lo) * (qs[min(lo + 1, len(qs) - 1)] - qs[lo])
        shield = self.evaluator.shield_geometry(q_cross)[0][1:]
        fk = np.array([self.evaluator.shield_geometry(np.asarray(n.full_6dof_config))[0][1:]
                       for n in self.planner.goal_nodes])
        cheb = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)), axis=-1))
        coverage = np.max(np.abs(fk[self.valid] - crossing), axis=1)
        nearest = int(np.flatnonzero(self.valid)[np.argmin(coverage)])
        result.update({"crossing_yz_m": crossing, "crossing_speed_m_s": speed,
                       "crossing_incidence_deg": incidence,
                       "nearest_valid_goal_action": nearest, "coverage_gap_m": float(np.min(coverage)),
                       "shield_gap_at_crossing_m": cheb(shield, crossing),
                       "episode_ended_before_crossing": bool(self.elapsed < crossing_time)})
        if goal is not None:
            action = self.node_to_action[goal]
            node = self.planner.goal_nodes[action]
            q_goal = np.asarray(node.full_6dof_config)
            settled = np.flatnonzero(np.max(np.abs(qs - q_goal), axis=1) < 0.01)
            result.update({"goal_action": action, "goal_node_index": goal,
                           "goal_flagged": goal in self.flagged_nodes,
                           "goal_fk_error_m": self.goal_fk_error[action],
                           "target_gap_m": cheb(node.task_position, crossing),
                           "goal_fk_gap_m": cheb(fk[action], crossing),
                           "joint_error_at_crossing_rad": float(np.max(np.abs(q_cross - q_goal))),
                           "arm_within_0_01_rad_time_s": float(settled[0] * self.dt) if len(settled) else None})
        if success:
            cause = None
        elif goal is None:
            cause = "no_goal_commanded"
        elif result["coverage_gap_m"] > CAPTURE_HALF_WIDTH:
            cause = "outside_goal_coverage"
        elif result["target_gap_m"] > CAPTURE_HALF_WIDTH:
            cause = "goal_selection"
        elif result["goal_fk_gap_m"] > CAPTURE_HALF_WIDTH:
            cause = "goal_fk_error"
        elif result["shield_gap_at_crossing_m"] > CAPTURE_HALF_WIDTH or result["joint_error_at_crossing_rad"] > 0.01:
            cause = "arm_not_arrived"
        else:
            cause = "near_miss_geometry"
        result["failure_cause"] = cause
        return result

    def verify(self):
        """Measure integration, FK, frame consistency, and evaluator sign before rollouts."""
        self.reset(options={"randomize": False})
        q0 = np.asarray(self.arm.positions).copy()
        center0, rotation = self.evaluator.shield_geometry(q0)
        R = self.planner.R
        t = self.planner.t
        assert np.array_equal(R, self.evaluator.base_from_plane_rotation)
        assert np.array_equal(t, self.evaluator.base_from_plane_translation)
        roundtrip = float(np.max(np.abs(R.T @ (R @ center0 + t - t) - center0)))
        assert roundtrip < 1e-7
        center_base = R @ center0 + t
        inside = self.evaluator.distance_at(center_base, q0)[0]
        outside_base = R @ (center0 + rotation @ (self.evaluator.half_extents + 1.0)) + t
        outside = self.evaluator.distance_at(outside_base, q0)[0]
        assert inside < 0 < outside
        # Analytic face/edge/corner fixtures, accounting for rounded upstream R.
        geometry_cases = []
        for axis in range(3):
            for sign in (-1, 1):
                for gap in (-0.01, 0.0, 0.01):
                    local = np.zeros(3)
                    local[axis] = sign * (self.evaluator.half_extents[axis] + self.evaluator.ball_radius + gap)
                    ball = np.linalg.solve(R.T, center0 + rotation @ local) + t
                    measured = self.evaluator.distance_at(ball, q0)[0]
                    geometry_cases.append(abs(measured - gap))
        for direction in (np.array([1., 1., 0.]), np.ones(3)):
            offset = direction / np.linalg.norm(direction) * (self.evaluator.ball_radius + .01)
            local = self.evaluator.half_extents + offset
            ball = np.linalg.solve(R.T, center0 + rotation @ local) + t
            geometry_cases.append(abs(self.evaluator.distance_at(ball, q0)[0] - .01))
        # Plane-frame shield rotation inherits the 8-decimal planner R (|R^T R - I| ~ 1e-8).
        assert max(geometry_cases) < 1e-8
        self.arm.command_callback(Float64MultiArray(data=[0.1, 0., 0., 0., 0., 0.]))
        for _ in range(25):
            self.runtime.tick(self.arm.dt)
        delta = np.asarray(self.arm.positions) - q0
        assert np.allclose(delta, [0.01, 0., 0., 0., 0., 0.], atol=1e-12)
        center1, _ = self.evaluator.shield_geometry(np.asarray(self.arm.positions))
        assert np.linalg.norm(center1 - center0) > 0.001
        # Exercise the original timeout branch too.
        for _ in range(30):
            self.runtime.tick(self.arm.dt)
        assert np.allclose(self.arm.velocities, 0)
        # T_WP from the MuJoCo base pose and planner T_BP must equal run_IK.py's plane.
        R_ik, t_ik = ik_world_from_plane()
        plane_frame_error = max(float(np.max(np.abs(self.evaluator.world_from_plane_rotation - R_ik))),
                                float(np.max(np.abs(self.evaluator.world_from_plane_translation - t_ik))))
        assert plane_frame_error < 1e-7
        urdf = UrdfShield(URDF_PATH)
        residuals, legacy, urdf_position, urdf_rotation = [], [], [], []
        for q in [q0] + [np.asarray(n.full_6dof_config) for n in self.planner.goal_nodes]:
            center, rotation_plane = self.evaluator.shield_geometry(q)
            expected_center, expected_rotation = urdf.pose(q)
            urdf_position.append(np.linalg.norm(R @ center + t - expected_center))
            urdf_rotation.append(np.max(np.abs(R @ rotation_plane - expected_rotation)))
        for node in self.planner.goal_nodes:
            q = np.asarray(node.full_6dof_config)
            target = np.r_[0., node.task_position]
            residuals.append(self.evaluator.shield_geometry(q)[0] - target)
            legacy.append(self.evaluator.world_shield_geometry(q)[0] - target)
        assert max(urdf_position) < 1e-7 and max(urdf_rotation) < 1e-7
        errors = np.asarray(residuals)
        norms = np.linalg.norm(errors, axis=1)
        legacy_norms = np.linalg.norm(np.asarray(legacy), axis=1)
        assert np.median(norms) < 1e-7  # rounded planner T_BP; exact run_IK T_WP gives ~1e-16
        self.reset(options={"randomize": False})
        self._update_belief()
        tau = float(self.planner.mu[2, 0])
        point = R.T @ (np.asarray(self.ball_state(tau)[:3]) - t)
        expected = np.r_[0., self.planner.mu[:2, 0]]
        crossing_error = float(np.linalg.norm(point - expected))
        assert crossing_error < 1e-7
        self.controller.command(self.goal_result(0))
        for _ in range(100):
            v = self.controller.tick(self.arm.positions)
            self.arm.command_callback(Float64MultiArray(data=v.tolist()))
            self.runtime.tick(self.arm.dt)
        controller_motion = float(np.linalg.norm(np.asarray(self.arm.positions) - q0))
        assert controller_motion > 0.01
        result = {"bridge_integration_pass": True, "bridge_timeout_pass": True,
                  "joint_delta_for_0_1_rad_s_over_0_1_s": delta,
                  "shield_center_motion_m": float(np.linalg.norm(center1 - center0)),
                  "controller_motion_over_0_4_s_rad": controller_motion,
                  "frame_roundtrip_max_error_m": roundtrip,
                  "planner_projectile_crossing_error_m": crossing_error,
                  "geometry_inside_distance_m": inside, "geometry_outside_distance_m": outside,
                  "geometry_analytic_fixture_count": len(geometry_cases),
                  "geometry_analytic_max_error_m": max(geometry_cases),
                  "goal_count": len(errors),
                  "fk_comparison_frame": "planner plane (stored task_position frame)",
                  "world_from_plane_vs_run_ik_max_error": plane_frame_error,
                  "urdf_shield_center_max_error_m": max(urdf_position),
                  "urdf_shield_rotation_max_element_error": max(urdf_rotation),
                  "urdf_vs_mujoco_shield_half_extents_m": [urdf.half_extents, self.evaluator.half_extents],
                  "stored_goal_fk_median_xyz_error_m": float(np.median(norms)),
                  "stored_goal_fk_median_yz_error_m": float(np.median(np.linalg.norm(errors[:, 1:], axis=1))),
                  "stored_goal_fk_p95_xyz_error_m": float(np.percentile(norms, 95)),
                  "stored_goal_fk_xyz_error_range_m": [float(np.min(norms)), float(np.max(norms))],
                  "stored_goal_fk_count_over_m": {str(x): int(np.sum(norms > x)) for x in (1e-3, 1e-2, 5e-2, 1e-1)},
                  "stored_goal_fk_residuals_xyz_m": errors,
                  "legacy_world_minus_plane_median_xyz_error_m": float(np.median(legacy_norms)),
                  "legacy_world_minus_plane_xyz_error_range_m": [float(np.min(legacy_norms)), float(np.max(legacy_norms))],
                  "physical_interception_validation": "see controlled_interception",
                  "geometry_sampling_interval_s": self.dt,
                  "continuous_collision_detection_verified": False,
                  "contact_dynamics_verified": False,
                  "home_shield_center_odom": center_base,
                  "model_actuator_count": int(self.evaluator.model.nu)}
        self.reset(options={"randomize": False})
        return result

    def swept_check(self, subdivisions=100, q_history=None):
        """Independent sub-tick check of this episode's (or a given) arm trajectory."""
        qs = self.q_history if q_history is None else q_history

        def ball_at(t):
            state = self.ball_state(t)
            return np.asarray(state[:3]), np.asarray(state[3:])
        result = self.swept.sweep(qs, ball_at, self.dt, int(subdivisions))
        # Upstream evaluator (frame corrected) vs MuJoCo collision at tick samples, outside only.
        disagreement = 0.0
        for k, q in enumerate(qs):
            ball = ball_at(k * self.dt)[0]
            sampled = self.evaluator.distance_at(ball, np.asarray(q))[0]
            if sampled > 0:
                disagreement = max(disagreement, abs(sampled - self.swept.distance(q, ball)[0]))
        if q_history is None:
            result["tick_sampled_min_surface_distance_m"] = self.min_surface
        result["evaluator_vs_mujoco_max_outside_disagreement_m"] = disagreement
        return result

    def controlled_interception(self, action, moving=True, speed=6.5, launch_angle_deg=5.0,
                                settle_margin_s=0.1, contact_time_s=None, subdivisions=100):
        """Aim an analytic throw, along the plane normal, at the shield centre.

        A probe episode with an unreachable projectile records the fixed-goal arm
        trajectory (independent of the projectile). The contact time is the last tick
        at >= half the peak shield speed (moving; keeps a realistic throw distance),
        settle time + margin (static), or `contact_time_s`. The real episode
        then reuses the same goal action; its trajectory must replay the probe.
        """
        config = {"substeps": 10, "max_ticks": 1000}
        far = {"x0": 100., "y0": 100., "z0": 1000., "speed": 1., "launch_angle_deg": 0., "side_angle_deg": 0.}
        self.reset(options={"randomize": False, "throw": far}, config=config)
        while not self.done:
            self.step(int(action))
        probe = np.asarray(self.q_history)
        R, t = self.planner.R, self.planner.t
        centers = np.array([R @ self.evaluator.shield_geometry(q)[0] + t for q in probe])
        shield_speed = np.linalg.norm(np.diff(centers, axis=0), axis=1) / self.dt
        moved = np.flatnonzero(np.max(np.abs(probe - probe[-1]), axis=1) > 1e-6)
        settle = int(moved[-1] + 1) if len(moved) else 0
        if contact_time_s is not None:
            tick = int(round(float(contact_time_s) / self.dt))
        elif moving:
            tick = int(np.flatnonzero(shield_speed[:settle] >= 0.5 * np.max(shield_speed[:settle]))[-1])
        else:
            tick = settle + int(round(settle_margin_s / self.dt))
        if not 0 <= tick < len(probe):
            raise ValueError("Contact time is outside the probe trajectory")
        contact_time = tick * self.dt
        aim = centers[tick]
        normal = R[:, 0]  # Plane +x in base; the default throw travels along it.
        side = np.arctan2(normal[1], -normal[0])
        launch = np.deg2rad(launch_angle_deg)
        velocity = np.array([-speed * np.cos(launch) * np.cos(side), speed * np.cos(launch) * np.sin(side),
                             speed * np.sin(launch)])
        start = aim - velocity * contact_time
        start[2] += 0.5 * 9.81 * contact_time ** 2
        if start[2] < 0:
            raise ValueError("Designed throw would start below ground")
        throw = {"x0": start[0], "y0": start[1], "z0": start[2], "speed": speed,
                 "launch_angle_deg": launch_angle_deg, "side_angle_deg": float(np.rad2deg(side))}
        self.reset(options={"randomize": False, "throw": throw}, config=config)
        while not self.done:
            response = self.step(int(action))
        episode = np.asarray(self.q_history)
        replay_error = float(np.max(np.abs(episode - probe[:len(episode)])))
        assert replay_error == 0.0
        aim_error = float(np.linalg.norm(np.asarray(self.ball_state(contact_time)[:3]) - aim))
        # Full encounter on the probe trajectory: the ball path does not depend on termination.
        horizon = min(len(probe), tick + int(round(0.1 / self.dt)) + 1)
        encounter = self.swept_check(subdivisions, probe[:horizon])
        episode_check = self.swept_check(subdivisions)
        k = min(tick, len(shield_speed) - 1)
        return {"action": int(action), "goal_node_index": int(self.planner.goal_nodes[action].idx),
                "goal_fk_error_m": self.goal_fk_error[action], "moving": bool(moving), "throw": throw,
                "designed_contact_time_s": contact_time, "aim_point_odom": aim, "aim_error_m": aim_error,
                "arm_settle_time_s": settle * self.dt,
                "shield_speed_at_designed_contact_m_s": float(shield_speed[k]),
                "probe_replay_max_joint_error_rad": replay_error,
                "termination_reason": response["info"]["termination_reason"],
                "sampled_termination_time_s": self.elapsed,
                "is_success": response["info"]["is_success"],
                "reward": response["reward"],
                "episode_swept": episode_check, "encounter_swept": encounter}

    def close(self):
        self.runtime.close()
        if self.controller:
            self.controller.close()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--outliers", action="store_true", help="write results/goal_outliers.{json,csv}")
    parser.add_argument("--reach-times", action="store_true", help="write results/goal_reach_times.json")
    args = parser.parse_args()
    backend = None
    output = sys.stdout
    try:
        with contextlib.redirect_stdout(sys.stderr):
            backend = Backend()
        if args.verify:
            with contextlib.redirect_stdout(sys.stderr):
                report = backend.verify()
            print(json.dumps(report, default=json_default, indent=2), file=output)
            return
        if args.reach_times:
            with contextlib.redirect_stdout(sys.stderr):
                times = backend.goal_reach_times()
            report = {"definition": "seconds from home, fixed goal action, until all joints stay within 0.01 rad",
                      "goal_node_indices": [int(n.idx) for n in backend.planner.goal_nodes],
                      "goal_fk_error_m": backend.goal_fk_error, "reach_time_s": times}
            (ROOT / "results/goal_reach_times.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"max_s": max(t for t in times if t is not None),
                              "unreached": sum(t is None for t in times)}), file=output)
            return
        if args.outliers:
            from .outliers import analyze, write
            summary, rows = analyze(backend)
            write(json.loads(json.dumps(summary, default=json_default)),
                  json.loads(json.dumps(rows, default=json_default)))
            print(json.dumps(summary, default=json_default, indent=2), file=output)
            return
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request.pop("op")
                if op not in {"metadata", "reset", "step", "baseline_action", "verify",
                              "swept_check", "controlled_interception", "episode_diagnostics",
                              "perception_trace"}:
                    raise ValueError(f"Unknown operation: {op}")
                with contextlib.redirect_stdout(sys.stderr):
                    result = getattr(backend, op)(**request)
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(response, default=json_default, allow_nan=False), file=output, flush=True)
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
