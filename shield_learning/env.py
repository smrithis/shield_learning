"""Gymnasium goal selection over the isolated, existing SHIELD worker."""
import json
import math
from pathlib import Path
import selectors
import subprocess
import tempfile

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]


class ShieldInterceptionEnv(gym.Env):
    """Actions index stored goals, never torques. Observations use SI units.

    Projectile mean/covariance are in odom_combined; joints are in model order.
    Reward uses sampled finite-box overlap, NOT validated physical contact.
    Native trajectory-generator internal state is hidden: this is a partially
    observed goal-selection interface, not a claim of a fully Markov state.

    Goals whose stored-configuration shield-centre FK misses the stored target by
    more than `goal_mask_fk_error_m` are flagged; with `apply_goal_mask` they are
    masked (`action_masks()`), rejected by `step()`, and hidden from both baselines.
    The action space and tree stay unchanged: indices always map to the same goals.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, substeps=10, max_ticks=500, success_reward=10.0,
                 motion_weight=0.01, switch_weight=0.01, timeout=120.0,
                 goal_mask_fk_error_m=0.05, apply_goal_mask=True):
        for name, value in (("substeps", substeps), ("max_ticks", max_ticks)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not all(math.isfinite(x) for x in (success_reward, motion_weight, switch_weight, timeout,
                                              goal_mask_fk_error_m)):
            raise ValueError("Configuration must be finite")
        if min(motion_weight, switch_weight) < 0 or timeout <= 0 or goal_mask_fk_error_m <= 0:
            raise ValueError("Costs must be nonnegative; timeout and mask threshold positive")
        self.config = dict(substeps=int(substeps), max_ticks=int(max_ticks),
                           success_reward=float(success_reward), motion_weight=float(motion_weight),
                           switch_weight=float(switch_weight),
                           goal_mask_fk_error_m=float(goal_mask_fk_error_m), apply_goal_mask=bool(apply_goal_mask))
        self.timeout = timeout
        self.render_mode = None
        self._done = True
        self._closed = False
        # A file avoids deadlocks from filling an unread stderr pipe.
        self._stderr = tempfile.TemporaryFile(mode="w+t", dir=ROOT / "results")
        self._process = subprocess.Popen(
            ["bash", str(ROOT / "scripts/run_backend.sh")], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1)
        try:
            self.backend_metadata = self._request("metadata")
            self.action_space = spaces.Discrete(self.backend_metadata["n_actions"])
            errors = np.asarray(self.backend_metadata["goal_fk_error_m"])
            self.goal_flagged = errors > goal_mask_fk_error_m
            self._mask = ~self.goal_flagged if apply_goal_mask else np.ones(len(errors), dtype=bool)
            self.observation_space = spaces.Dict({
                key: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float64)
                for key, shape in {"projectile_mean": (6,), "projectile_covariance": (6, 6),
                                   "joint_position": (6,), "joint_velocity": (6,)}.items()})
        except BaseException:
            self.close()
            raise

    def _request(self, op, **kwargs):
        if self._closed:
            raise RuntimeError("Environment is closed")
        try:
            self._process.stdin.write(json.dumps({"op": op, **kwargs}, allow_nan=False) + "\n")
            self._process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                if not selector.select(self.timeout):
                    raise TimeoutError(f"SHIELD backend timed out during {op}")
            line = self._process.stdout.readline()
            if not line:
                self._stderr.seek(0)
                raise RuntimeError("SHIELD backend exited: " + self._stderr.read()[-4000:])
            response = json.loads(line)
        except (TimeoutError, BrokenPipeError, json.JSONDecodeError):
            self.close()  # Never consume a stale response on a later request.
            raise
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def _observation(self, result):
        obs = {k: np.asarray(v, dtype=np.float64) for k, v in result["observation"].items()}
        if not self.observation_space.contains(obs) or not all(np.isfinite(v).all() for v in obs.values()):
            raise RuntimeError("Backend returned invalid observation")
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Consume Gym's generator also for reset(seed=None); explicit seeds replay.
        worker_seed = int(self.np_random.integers(0, 2**31))
        self._done = True
        result = self._request("reset", seed=worker_seed, options=options, config=self.config)
        obs = self._observation(result)
        self._done = False
        return obs, result["info"]

    def _step(self, **kwargs):
        if self._done:
            raise RuntimeError("Call reset() before stepping or after episode end")
        result = self._request("step", **kwargs)
        self._done = bool(result["terminated"] or result["truncated"])
        return (self._observation(result), float(result["reward"]),
                bool(result["terminated"]), bool(result["truncated"]), result["info"])

    def action_masks(self):
        """Boolean validity per action (True = allowed); MaskablePPO-compatible name."""
        return self._mask.copy()

    def step(self, action):
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError("Action must index an existing SHIELD goal")
        if not self._mask[int(action)]:
            raise ValueError(f"Action {int(action)} is masked (goal FK error above threshold)")
        return self._step(action=int(action))

    def step_hold(self):
        """Baseline helper outside the action space: advance without a new command."""
        return self._step(hold=True)

    def episode_diagnostics(self):
        """Ball plane crossing, selected goal, arm state and failure cause of the last episode."""
        return self._request("episode_diagnostics")

    def baseline_action(self):
        """Existing planner's direct-goal scoring within this action space."""
        if self._done:
            raise RuntimeError("Call reset() before selecting a baseline action")
        return int(self._request("baseline_action"))

    def step_planner(self):
        """Full original planner, including intermediate tree actions (not Gym actions)."""
        return self._step(planner=True)

    def verify(self):
        self._done = True
        return self._request("verify")

    def swept_check(self, subdivisions=100):
        """Independent sub-tick sphere/box check of the current or last episode."""
        return self._request("swept_check", subdivisions=int(subdivisions))

    def controlled_interception(self, action, moving=True, **kwargs):
        """Designed throw aimed at the shield following a fixed goal; ends the episode."""
        self._done = True
        return self._request("controlled_interception", action=int(action), moving=bool(moving), **kwargs)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._process.stdin.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process.stdout.close()
        self._stderr.close()
