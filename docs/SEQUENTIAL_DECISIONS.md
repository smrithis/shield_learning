# Repeated decisions and replanning in the SHIELD Gym environment

Status: design notes for sequential learning; no RL is implemented. Original SHIELD
code (`smrithi_ws`) is unchanged; everything below is emulated in `shield_learning`.

## 1. How the original system decides (source + bag evidence)

Sources: `belief_planner/shield_replanning/src/controller.cpp`,
`existing_ruckig_controller.cpp`, `belief-space-planner/scripts/planner.py`,
`existing_belief_planner.py`, `shield_perception/src/track_ball_meancovariance_node.cpp`.

| Stage | Original behaviour |
|---|---|
| Perception | ZED at 100 Hz; each processed frame either adds an accepted detection to a ≤10-sample buffer or counts as unusable (5 in a row clear the buffer). With ≥4 samples the unchanged SlidingKalman re-runs on the window; the estimate is published on `/projectile` only if \|vx\| > 2 m/s. |
| Planner input | `projectile_callback` runs `update_belief` (propagation to the plane) on every message; only the latest belief is kept. `plan()` consumes it (`mu = None`), so at most one decision per published belief. |
| Request loop | Idle: `get_best_action` is requested continuously with `cur_time = 0`. Executing: `cur_time` = time since execution start; direct-goal results (`wait = False`) re-request continuously; tree results (`wait = True`) only when the first three joints are within 0.1 rad of the target. |
| Timing | `plan()` runs at the **end** of each request's round trip, on the belief available then. Replaying all recorded beliefs and requests of bags 015–021 through the unchanged planner reproduces **8,971/8,976** recorded SUCCEEDED/ABORTED outcomes and **58/59** returned targets exactly; planning at request time reproduces only 8,940 and 30/42. Round trips are ~4 ms, with ~11.7 % at ~100–120 ms in flight (an action-server queueing race). |
| Applying a result | Ruckig retargets online from the current (position, velocity, acceleration) via `pass_to_input`; `minimum_duration` is reset. The goal configuration is checked for contact and `link_6` z < 0.6 m (rejects 0/163 goals and 0/3 intermediate nodes). |
| End of execution | When Ruckig reports `Finished`, execution ends: zero velocity, 1 s pause, then the home trajectory. **No retargeting after arrival.** The remaining trajectory is collision-checked every 4 ms cycle and aborts on contact or `link_6` z < 0.6 m. |

## 2. What the environment now reproduces

`ShieldInterceptionEnv(original_runtime=True, substeps=1, perception=...)` with
`step_planner()`:

- The request loop is emulated per 4 ms tick. `cur_time` follows the original. Each
  request draws a queueing delay from the 1,178 in-flight round trips recorded in
  the bags, and `plan()` runs when that delay expires. The `wait` replan gate and
  the post-`Finished` gate are reproduced (`info['runtime']`).
- Ruckig receives the measured joint state only on execution cycle 0, every 15th cycle,
  and when a new goal is applied, as `controller.cpp` does. On the other cycles it runs
  open-loop (`shield_tick(ptr, NULL, ...)`). This is not cosmetic. Feeding the
  Euler-integrated simulated joints back every tick makes Ruckig recalculate each cycle,
  and stalls motion towards tree nodes that carry a nonzero target velocity. With
  every-tick feedback, arm B scores 254/525 instead of 319/525.
- Beliefs reach the planner at their receipt time (`available_s`), in order, through
  the same `update_belief` call.
- Perception modes keep **ground truth, measurements and beliefs separate**:
  - `oracle`: the exact state every tick. This is the previous behaviour and the default.
  - `timed_truth`: the exact state delivered only at the calibrated publication times.
  - `calibrated`: the calibrated measurement stream run through the unchanged filter.

  Truth is used for collision and reward only. It appears in
  `info['privileged_true_projectile_state']` and `perception_trace()`, never in the
  observation.
- Observations gain `belief_valid` and `belief_age_s`. Before the first published
  belief, the projectile entries are zero with `belief_valid = 0`; no state is invented.

The default (`oracle`, `original_runtime=False`) is unchanged and still passes all
earlier tests and results.

## 3. Not reproduced (fidelity limits)

- The per-cycle **trajectory** collision check, because Ruckig's internal trajectory
  is not exposed by the bridge. `episode_diagnostics` can only check the path that
  was actually executed.
- The home sequence that starts 1 s after `Finished` (`info['runtime']['home_sequence_due']`).
- The 2 s execution `break_time` abort, and the extra wait inside `updateState` for a
  fresh joint-state message.
- Projectile drag. The calibrated reference fits give 0.4–3.8 m/s² of deceleration
  along the throw, but the simulated projectile has none.
- A constant exposure-to-stamp delay. It cannot be identified from the bags and
  defaults to 0; the `exposure_offset_s` option is available for a sweep.

## 4. Recommended sequential-learning interface (to implement before RL)

1. **Decision points.** Make the decision rate configurable: either one agent step per
   delivered belief (the original's "one decision per belief"), or a fixed 4 ms step.
   The current fixed `substeps` stepping should remain for comparison.
2. **Explicit hold action.** Add `HOLD = 163`, which is always allowed. Re-selecting the
   current goal already means "continue", but there is no valid action before the
   first commitment or while waiting for a better belief. `step_hold()` is only a
   baseline helper outside the action space.
3. **Timing parity for the agent.** Agent actions should go through the same request
   queueing delay as the planner (an option, on for evaluation). Otherwise a learned
   policy gets an unfair ~11 % chance of acting 100 ms sooner than the baseline.
4. **Controller state in the observation.** Add the commanded goal index, a `finished`
   flag (retargeting is impossible after arrival), and the time since the last
   command. Without these the task is not Markov in the controller state. After
   `Finished`, mask every action except HOLD.
5. **Masks.** Combine the 5 cm FK mask with the post-`Finished` mask in `action_masks()`.
6. **Privileged teacher data.** `perception_trace()` and the privileged truth in `info`
   are enough for oracle labels (fixed-goal outcomes per throw) for teacher–student
   training, and keep the student's observation clean.
7. **Evaluation.** Always report paired arms A–D (see `paired_evaluation.py`) and the
   fixed-goal oracle on the same throws and seeds.
