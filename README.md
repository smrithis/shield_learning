# SHIELD learning environment

A Gymnasium goal-selection wrapper around the existing SHIELD simulator and native
Ruckig controller. No policy is trained. The original `smrithi_ws` is read only;
all new code, caches, logs, and results live here. The existing `.venv` and
`build/libshield_learning_controller.so` are reused.

## Run on this machine

```bash
cd /home/shield/code/shield_learning
bash scripts/run_baseline.sh --episodes 5
.venv/bin/python -m unittest discover -s tests -v
bash scripts/run_backend.sh --outliers   # results/goal_outliers.{json,csv}
bash scripts/run_backend.sh --reach-times   # results/goal_reach_times.json (~20 s)
bash scripts/run_evaluation.sh              # 105 throws x mask on/off x 2 baselines (~2.5 min)
bash scripts/run_evaluation.sh --oracle     # every allowed goal on each masked failure (~2 min)
bash scripts/plot_evaluation.sh             # results/evaluation_coverage.png
```

`run_evaluation.sh --summarize-only` recomputes the summary from the saved episodes.
Run `--reach-times` before the evaluation; the evaluation joins those times into
its failure diagnostics.

The run script works from any directory. It runs Gymnasium's checker, geometry and
motion verification, then five episodes per baseline: the original
`ExistingBeliefPlanner.plan` and direct-goal scoring using the same planner.
Episode 0 is the fixed default throw; episodes 1–4 use seeded randomized throws.
Each episode is re-checked with the sub-tick swept check, followed by two designed
(controlled) interception throws. Outputs are `results/baselines.json` and
`results/verification_current.json`. Pre-frame-fix outputs are kept in
`results/before_frame_fix/`, and the first session's `results/verification.json`
remains intact. For a backend-only check, run `bash scripts/run_backend.sh --verify`.

### Dependency status — partially blocked

The current executable fallback uses **Gymnasium 0.28.1**, NumPy 1.26.4, and
Python 3.10.12. NumPy, typing-extensions, setuptools and wheel were installed from
local pip cache archives. Gymnasium was copied unchanged from the locally
installed Isaac Sim distribution and repacked as a wheel in `.cache/wheels`.
It is the actual Gymnasium package, not a replacement API.

**Dependency resolution is not complete:** network/DNS access to package indexes
failed. `pyproject.toml` still requests Gymnasium >=1.0,<2; that version has not
been installed or validated. The fallback's declared dependencies `cloudpickle`,
`farama-notifications`, and `jax-jumpy` are absent. `pip check` therefore fails.
They are not exercised by the synchronous wrapper, checker, and baseline tests;
vector workers and optional wrappers have not been validated.

When package-index access is available, finish the intended installation:

```bash
bash scripts/install_dependencies.sh
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
bash scripts/run_baseline.sh --episodes 5
```

To reproduce only the currently tested fallback from this workspace's cached
wheels, use `bash scripts/install_dependencies.sh --offline-fallback`. Its final
`pip check` intentionally returns failure while the missing dependencies remain.
No virtual environment needs to be recreated. Do not run `build_native.sh` unless
the existing bridge library is actually missing or incompatible.

## Interface

```python
from shield_learning import ShieldInterceptionEnv

with ShieldInterceptionEnv(substeps=10, max_ticks=500,
                           goal_mask_fk_error_m=0.05, apply_goal_mask=True) as env:
    observation, info = env.reset(seed=7)
    while True:
        mask = env.action_masks()       # bool (163,), True = allowed
        action = env.baseline_action()  # existing planner; never returns a masked goal
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

- `Discrete(163)`: index into existing interception goals. The mapping to tree
  nodes and joint configurations is available in `env.backend_metadata`.
  `backend_metadata['goal_fk_error_m']` gives each goal's shield-centre FK error
  against its stored target (34 exceed 1 mm; see "Remaining goal outliers").
- **Goal mask (default on).** Goals whose FK error exceeds
  `goal_mask_fk_error_m` (default 0.05 m) are flagged. These are 7 actions:
  0, 16, 101, 117, 131, 132, 162 (nodes 1, 17, 102, 118, 132, 133, 163).
  With `apply_goal_mask=True`:
  - `action_masks()` marks them False (the MaskablePPO convention), and reset `info['action_mask']` repeats it.
  - `step()` raises `ValueError` for a masked action.
  - `baseline_action()` only scores allowed goals.
  - `step_planner()` hides them from both of the planner's scoring paths (direct
    goal values and the tree's terminal `time_to_goals`), through a subclass. The
    planner source and tree file are untouched.

  The action space stays `Discrete(163)` with unchanged indices. With
  `apply_goal_mask=False`, nothing is excluded, but flagged goals are still
  reported. `info` always carries `selected_goal_node_index`,
  `selected_goal_fk_error_m`, `selected_goal_flagged` and `goal_mask_applied`.
- Dict observations contain float64 `projectile_mean` (6,), covariance (6,6),
  joint position (6,), and joint velocity (6,). Projectile position/velocity and
  covariance use `odom_combined`, metres and seconds. Joints use model order,
  radians and radians/second. Unbounded Boxes avoid invented clipping limits;
  observations are checked for shape and finiteness. Checker warnings about
  unbounded spaces and matrix observations are expected. `FlattenObservation`
  can flatten these later, but policy training is not included here.
- Each step advances up to ten 4 ms native controller/bridge ticks, stopping
  early at sampled overlap, ground crossing, or the time limit. Overlap/ground
  set `terminated`; `max_ticks` sets `truncated`.
- Reward = 10 for sampled overlap − 0.01 times integrated squared joint
  velocity − 0.01 per goal switch. `info['reward_terms']` exposes each term.
  `is_success` means **sampled geometric overlap only**;
  `reward_is_physical_contact` is always false.
- `env.swept_check(subdivisions=100)` re-checks the current/last episode at
  4 ms/`subdivisions` with MuJoCo's own box–sphere distance, independently of
  the evaluator code and planner transform. `env.controlled_interception(action,
  moving=True)` designs a throw aimed at the shield following that goal (at the
  last tick with ≥50 % of peak shield speed, or after settling with
  `moving=False`), runs it, and returns both checks. It ends the current episode.
- Repeated explicit seeds reproduce trajectories. `reset(seed=None)` advances
  Gymnasium's RNG. Use `options={'randomize': False, 'throw': {...}}` for custom
  `x0,y0,z0,speed,launch_angle_deg,side_angle_deg`. Randomization overrides all
  throw parameters except x0, as in the original projectile publisher.
  `options={'launch_time_s': -0.2}` starts the episode 0.2 s after release (late
  tracking, shorter reaction time); allowed range [−1, 0]. A positive delay
  (ball held before release) would need a no-op action, so it is not offered.
- `env.episode_diagnostics()` returns:
  - the analytic plane crossing (time, Y/Z, speed, incidence);
  - the goal commanded before the crossing, and the arm's joint error at the crossing;
  - the gap to the nearest allowed goal;
  - a `failure_cause`: `outside_goal_coverage`, `goal_selection`, `goal_fk_error`,
    `arm_not_arrived`, `near_miss_geometry`, `no_goal_commanded` or `no_plane_crossing`.

  Gaps are plane Chebyshev distances compared with the shield half-width plus
  ball radius (0.208 m). This is attribution, not a second success test.
  `env.step_hold()` advances without a new command. Baselines use it only while
  the planner has no valid prediction.
- `step_planner()` runs the full original planner, including intermediate tree
  actions that are outside the Gym action space. The two baseline modes are
  deliberately recorded separately.
- The controller's internal acceleration/trajectory state is hidden. This is a
  partially observed goal-selection interface; a fully Markov state is not claimed.
  `close()` stops the worker and controller. There is no rendering mode.

The backend uses the existing `ros_new` Python environment in a subprocess;
Gymnasium never needs to import ROS/MuJoCo. Defaults are
`SHIELD_WS=/home/shield/code/smrithi_ws` and
`SHIELD_ROS_ENV=/home/shield/mambaforge/envs/ros_new`. Those variables can select
compatible existing installations. The native library also has build-time rpaths,
so relocating libraries is outside this run's validation.

## Frame correction

MuJoCo `geom_xpos` is in the MuJoCo **world** frame W. The stored goal
`task_position` is in the interception **plane** frame P, which `run_IK.py`
defined as `p_W = Rz(-45°) p_P + 0.8·[cos 135°, sin 135°, 0]`. The robot **base**
B is `odom_combined`, published as identity to `base_link`. The upstream
`ShieldProjectileEvaluator.shield_geometry()` returns raw world FK, but
`distance_at()` treats it as plane coordinates. The earlier FK check subtracted
the same mixed frames.

`shield_learning/frames.py` subclasses the existing evaluator and converts only
the shield centre and orientation from W to P:
`T_WP = T_WB · T_BP`, with T_WB taken from the MuJoCo `base` body and T_BP from the
planner's own `R, t`. `distance_at()`, the tree, the planner and `smrithi_ws` are
unchanged. The same evaluator class still contains the mix-up in `smrithi_ws`, and
so does `planned_intercept_visualizer.py`; neither was edited.

| Check | Before | After |
|---|---:|---:|
| Stored-goal FK median XYZ error (163 goals) | 0.800 m | 6.4×10⁻⁹ m |
| FK median YZ error | 0.566 m | 4.3×10⁻⁹ m |
| FK 95th percentile / max | 1.41 m / 1.50 m | 22.3 mm / 794 mm |
| Home shield centre in `odom_combined` | [1.4829, −0.2029, 0.9615] | [0.7921, 0.2122, 0.9615] |
| Shield centre vs independent URDF FK (home + 163 goals) | not checked | ≤ 9.3×10⁻⁹ m, rotation ≤ 9.6×10⁻⁹ |
| Derived T_WP vs `run_IK.py` literal | — | 5.7×10⁻⁹ |
| Baseline sampled interceptions (10 episodes) | 0/10 | 10/10 |
| Best baseline surface distance | +0.2318 m | −0.0173 m |

Residual ~10⁻⁸ errors come from the planner's 8-decimal `R` (|RᵀR − I| ≈ 9.7×10⁻⁹).
With `run_IK.py`'s exact transform they fall to ~10⁻¹⁶, as Codex found. The derived
transform is used deliberately, so the ball and shield go through one transform chain.

## What is verified

Thirteen integration tests pass (`results/tests.log`). They cover the Gym contract,
seeded replay and motion, invalid inputs, ground termination, time-limit
truncation, worker cleanup, and the initial-overlap fixture. The frame correction
is covered by these regressions:

- The legacy 0.800 m comparison and the corrected median < 10⁻⁷ m.
- URDF and `run_IK` agreement < 10⁻⁷, plus the corrected home centre.
- The exact outlier counts (34/18/7/4 over 1 mm/1 cm/5 cm/10 cm), which change only if the tree changes.
- A reset-time static check: a ball at the corrected home centre overlaps, one 10 cm in front does not, and the legacy centre is more than 0.5 m away.
- The sub-tick swept check agrees with tick sampling.
- Both controlled interceptions.

Three more tests cover the goal mask and diagnostics:

- The mask is exactly the 7 goals over 5 cm. Each masked action is rejected, and
  `baseline_action()` stays inside the mask.
- On a throw at node 1's stored target, both unmasked baselines pick node 1 and
  miss (`goal_fk_error`), while both masked baselines intercept.
- Late launch shifts the initial state analytically. Out-of-range launch times
  are rejected. A throw 1.3 m to the side is reported as `outside_goal_coverage`
  at its exact crossing.

Bridge integration, controller motion (1.499 rad over 0.4 s), command timeout,
planner/projectile crossing (5.9×10⁻¹⁶ m) and 20 analytic face/edge/corner
fixtures (2.1×10⁻⁹ m) still pass.

**Seeded baselines** (same seeds as before). All ten episodes now end in sampled
interception at 0.468–0.524 s. In each one, the independent 40 µs swept check finds
first contact 0.96–2.92 ms earlier, within the same 4 ms tick. Every first contact is
on the shield's front face, at most 81 mm from its centre (half-width 170 mm). The
shield is nearly settled at contact (≤ 0.35 m/s), so these are close to static
interceptions. The corrected evaluator and MuJoCo collision distance agree to
≤ 3.8×10⁻⁸ m outside the box. The maximum relative motion between swept samples is
0.39 mm.

**Controlled interceptions** use node 45 (action 44), with FK error 5.6×10⁻⁹ m. A
probe episode records the fixed-goal arm trajectory, and a 6.5 m/s throw along
the plane normal is aimed at the shield centre at a chosen time. The real episode
replays the probe exactly, with joint error 0.

| Case | Throw start (odom) | Contact design | Sampled | Swept first contact | Shield speed |
|---|---|---|---:|---:|---:|
| Moving | (2.93, 0.56, 0.84) m | 0.332 s, mid-motion | 0.328 s | 0.32504 s | 1.15 m/s centre, 1.46 m/s max point |
| Static | (4.74, 1.02, 2.00) m | settle + 0.1 s | 0.620 s | 0.61736 s | 0 |

On both throws, the original planner and the discrete baseline also select node 45
and intercept. Because the throw is designed with knowledge of the arm trajectory,
this verifies the geometry and timing pipeline, not planner skill.

## Remaining goal outliers (tree not modified)

`results/goal_outliers.{json,csv}` lists every goal over 1 mm, with a
cross-reference to the per-goal classification in
`smrithi_ws/stored_goal_outliers_2026-10-04`. All 18 goals from that study are
reproduced. Of the 34 goals over 1 mm, 28 have a joint exactly at a `run_IK.py`
clip bound, and none are in model contact.

- **Bottom two rows (Z = 0.700 and 0.814 m, 22 goals).** These are limited by
  `run_IK.py`'s q5 clip of ±1.0071 rad. The XML joint range is ±2.0071. Errors are
  1.6–14.8 mm, with 1.3–12.1° shield tilt. **All 10 baseline interceptions used
  goals from these rows** (nodes 8, 10, 11, 12, 28). The ≤ 15 mm offset is small
  next to the 170 mm half-width, so the hits are real, but the reward depends on them.
- **Gross outliers over 5 cm (7 goals: nodes 1, 17, 102, 118, 132, 133, 163).**
  These are 58–794 mm off. Nodes 1 and 17 reach Z ≈ 1.40 m instead of 0.70 m
  (q2 at its clip bound). None were selected in these episodes. They remain
  selectable Gym actions, and the planner scores them by their intended [Y, Z].
- The 4 non-goal tree nodes have no task position, so this check does not apply to them.

## Goal mask and diverse-throw evaluation

`shield_learning/scenarios.py` defines each throw by **where it crosses the
interception plane** (Y, Z), plus speed, elevation, side angle relative to the
plane normal, release distance and launch time. These are solved into the
unchanged publisher's parameters. The set has 105 throws:

- **Structured (45):**
  - low (3), below the grid (3), mid (5), high and lob (6), above the grid (3);
  - lateral (4), oblique ±20° (2), lateral beyond the grid (2);
  - speeds of 5/8/10/12 m/s (4);
  - late launches of −0.1/−0.2/−0.3 s (6);
  - one throw at each masked goal's stored target (7).
- **Seeded random (60, seed 2026):** Y ∈ [−1.2, 1.2], Z ∈ [0.5, 2.1],
  5–10 m/s, 0–25° elevation, ±25° side, 3–5 m release distance, launch time −0.3–0 s.

Coverage: Y −1.3–1.3 m, Z 0.5–2.2 m, 5–12 m/s, 0–30° elevation, side
−24–+25°, incidence 12–56°, reaction time 0.13–0.85 s. 87 of 105 crossings are
within capture distance of an allowed goal. 59 distinct goals were selected
across runs. Every episode was re-checked at 40 µs: in all 420 episodes, the
sampled and swept outcomes agree.

| 105 throws | Successes | In coverage | Failure causes | Flagged goal picked |
|---|---:|---:|---|---:|
| Planner, unmasked | 68 | 68/88 | 17 outside coverage, 16 arm not arrived, 4 goal FK error | 22 (2 hits) |
| Planner, masked | **77** | 77/87 | 18 outside coverage, 9 arm not arrived, 1 goal FK error | 0 |
| Discrete, unmasked | 69 | 69/88 | 17 outside coverage, 15 arm not arrived, 4 goal FK error | 22 (2 hits) |
| Discrete, masked | **78** | 78/87 | 18 outside coverage, 8 arm not arrived, 1 goal FK error | 0 |

These numbers come from `results/evaluation.json`, `results/evaluation_episodes.csv`
and `results/evaluation_coverage.png`.

- **The mask helps and never hurts here.** Per scenario, 9 unmasked misses
  become hits for each baseline, and no hit becomes a miss. Without the mask, the
  baselines choose a flagged goal in 22 episodes and hit only 2. The 7 throws at
  masked goals' stored targets go from 1/7 to 7/7.
- **Flagged goals are also the slowest goals.** `results/goal_reach_times.json`
  gives the median reach time from home as 0.42 s. Nodes 1/17/118/133 need about
  1.15 s and 102/132 need 0.83 s, while typical flight times are about 0.6 s.
  Choosing them fails twice over: the shield is in the wrong place and arrives late.
- **Remaining masked failures** fall into three groups:
  - **Outside coverage (18):** above Z ≈ 2.0 m, beyond |Y| ≈ 1.05 m, or in the
    sparse upper corners.
  - **Arm not arrived (8 discrete, 9 planner):** these are 0.18–0.51 s reaction
    times where the selected goal needs 0.40–0.70 s. In planner scenario
    `random_028`, the tree search switches back to intermediate node 166 after
    selecting goal 156; the discrete baseline intercepts that throw.
  - **Goal FK error (1):** `random_015` is a grazing case at the coverage edge,
    0.204 m against 0.208 m.
- **Fixed-goal oracle** (`results/evaluation_oracle.log`): every one of the 156
  allowed goals was simulated as a fixed action on each of the 28 masked-failure
  throws. **Only 1 throw is recoverable** (`random_028`, swept-confirmed). On this
  benchmark, the masked discrete baseline (78/105) is therefore at the ceiling for
  choosing one goal at t = 0.
- The original 5 seeded baselines are unchanged at 10/10, with none choosing a
  flagged goal. The controlled interceptions still pass.

## Readiness and remaining limitations

**The mechanics are ready for learning:**
- Frames are verified, and the goal mask is enforced consistently in the env,
  both baselines and evaluation.
- Episodes are reproducible.
- Each outcome has an attributed cause and an independent fine-resolution check.

**The current task is not yet useful for RL.** Projectile observations are the
exact analytic state with a constant covariance, so the existing planner's
belief is effectively perfect. On the evaluated throws, the masked baseline
already matches the fixed-goal oracle. A learned goal selector has almost no
headroom unless the task changes. Options include noisy or delayed observations
from the existing Kalman pipeline, a broader or different throw distribution, or
an action space that can benefit from mid-flight switching.

Other limitations:
- **Coverage gaps belong to the tree, not the policy.** The tree has no goals
  above Z ≈ 1.84 m or beyond |Y| ≈ 0.95 m, and the upper corners are sparse.
  Fixing that means regenerating the tree, which was not done.
- **Goals with errors under 5 cm stay allowed.** 27 goals miss by 1–50 mm,
  including the q5-clipped bottom rows that the seeded baselines use. The
  threshold is configurable.
- **Geometric overlap only.** There is no contact dynamics or deflection. The
  reward uses 4 ms sampling. The 40 µs swept check is post hoc and chord-bounded,
  not certified continuous collision detection. At 12 m/s it still agrees with
  sampling.
- **Shield size.** The evaluator box is 0.34 m and the URDF box is 0.35 m.
- **No parity with the original runtime's safety logic.** The bridge does not run
  the original controller's collision checks, replanning gates or stop/home
  sequencing. Joint and self-collision safety are unverified. The upstream
  `smrithi_ws` evaluator and visualizer still have the frame mix-up.
- **Launch time.** Only late tracking (≤ 0) is supported.
- **Dependencies.** Gymnasium ≥ 1.0 is still not installed; the 0.28.1 fallback
  is unchanged (see above).

**Exact next step before training:** make the task non-trivial for a learner.
For example, feed the observation through the existing Kalman filter with
measurement noise and latency. Then re-run `run_evaluation.sh` and `--oracle`
to confirm that the baseline falls below the oracle. After that, finish the
Gymnasium ≥ 1.0 installation.
