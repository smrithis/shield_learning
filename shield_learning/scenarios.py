"""Reproducible projectile scenarios specified by their interception-plane crossing.

Each scenario fixes where the ball crosses the planner plane (Y, Z), its speed,
launch (elevation) angle, side angle relative to the plane normal, horizontal
release distance and launch time, then solves for the existing publisher's
`x0, y0, z0, speed, launch_angle_deg, side_angle_deg` (odom_combined). No new
projectile model: the throw is still the publisher's analytic ballistic arc.
"""
import numpy as np

GRAVITY = 9.81
MAX_RELEASE_HEIGHT = 3.0


def throw_for_crossing(R, t, y, z, speed, launch_deg, side_offset_deg, distance):
    """Publisher parameters for a throw crossing plane point (0, y, z); None if infeasible."""
    R, t = np.asarray(R), np.asarray(t)
    normal = R[:, 0]  # Horizontal plane normal; the default throw travels along it.
    side = np.arctan2(normal[1], -normal[0]) + np.deg2rad(side_offset_deg)
    launch = np.deg2rad(launch_deg)
    direction = np.array([-np.cos(side), np.sin(side)])
    horizontal = speed * np.cos(launch)
    # The release point is `distance` back along the horizontal path from the crossing.
    flight = distance / horizontal
    crossing = R @ np.array([0., y, z]) + t
    x0, y0 = crossing[:2] - direction * horizontal * flight
    z0 = crossing[2] - speed * np.sin(launch) * flight + 0.5 * GRAVITY * flight ** 2
    if not 0.0 <= z0 <= MAX_RELEASE_HEIGHT:
        return None, flight
    return {"x0": float(x0), "y0": float(y0), "z0": float(z0), "speed": float(speed),
            "launch_angle_deg": float(launch_deg), "side_angle_deg": float(np.rad2deg(side))}, flight


def _scenario(name, category, R, t, y, z, speed=6.5, launch_deg=5.0, side_offset_deg=0.0,
              distance=4.0, launch_time_s=0.0):
    throw, flight = throw_for_crossing(R, t, y, z, speed, launch_deg, side_offset_deg, distance)
    return {"name": name, "category": category, "plane_y_m": y, "plane_z_m": z, "speed_m_s": speed,
            "launch_angle_deg": launch_deg, "side_offset_deg": side_offset_deg,
            "release_distance_m": distance, "launch_time_s": launch_time_s,
            "flight_time_to_plane_s": flight, "reaction_time_s": flight + launch_time_s,
            "throw": throw, "feasible": throw is not None}


def structured(R, t, masked_targets=()):
    """Named grid: low, mid, high, lateral, speeds, lobs, late launches, masked-goal regions."""
    rows = []
    for y in (-0.6, 0.0, 0.6):
        rows.append(_scenario(f"low_z0.75_y{y:+.1f}", "low", R, t, y, 0.75))
        rows.append(_scenario(f"below_grid_z0.50_y{y:+.1f}", "below_grid", R, t, y, 0.50, launch_deg=0.0))
    for y in (-0.9, -0.45, 0.0, 0.45, 0.9):
        rows.append(_scenario(f"mid_z1.20_y{y:+.2f}", "mid", R, t, y, 1.20, launch_deg=10.0))
    for y in (-0.6, 0.0, 0.6):
        rows.append(_scenario(f"high_z1.65_y{y:+.1f}", "high", R, t, y, 1.65, launch_deg=15.0))
        rows.append(_scenario(f"high_lob_z1.80_y{y:+.1f}", "high", R, t, y, 1.80, speed=7.5, launch_deg=30.0))
        rows.append(_scenario(f"above_grid_z2.20_y{y:+.1f}", "above_grid", R, t, y, 2.20, speed=8.0, launch_deg=25.0))
    for y in (-0.95, 0.95):
        for z in (0.9, 1.4):
            rows.append(_scenario(f"lateral_y{y:+.2f}_z{z:.1f}", "lateral", R, t, y, z, launch_deg=8.0))
        rows.append(_scenario(f"oblique_y{y:+.2f}", "lateral_oblique", R, t, y, 1.1, launch_deg=8.0,
                              side_offset_deg=20.0 * np.sign(y)))
        rows.append(_scenario(f"outside_y{1.3 * np.sign(y):+.2f}", "lateral_beyond_grid", R, t, 1.3 * np.sign(y), 1.1,
                              launch_deg=8.0))
    for speed, launch in ((5.0, 20.0), (8.0, 8.0), (10.0, 5.0), (12.0, 3.0)):
        rows.append(_scenario(f"speed_{speed:.0f}", "speed", R, t, 0.2, 1.0, speed=speed, launch_deg=launch))
    for launch_time in (-0.1, -0.2, -0.3):
        rows.append(_scenario(f"late_launch_{launch_time:+.1f}", "launch_time", R, t, 0.2, 1.0,
                              launch_deg=8.0, launch_time_s=launch_time))
        rows.append(_scenario(f"late_fast_launch_{launch_time:+.1f}", "launch_time", R, t, -0.3, 1.2,
                              speed=8.0, launch_deg=8.0, launch_time_s=launch_time))
    for i, (y, z) in enumerate(masked_targets):
        rows.append(_scenario(f"masked_goal_region_{i}_y{y:+.2f}_z{z:.2f}", "masked_goal_region", R, t, y, z,
                              launch_deg=10.0))
    return rows


def randomized(R, t, count, seed):
    """Seeded broad distribution; infeasible release heights are resampled."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        row = _scenario(f"random_{len(rows):03d}", "random", R, t, float(rng.uniform(-1.2, 1.2)),
                        float(rng.uniform(0.5, 2.1)), speed=float(rng.uniform(5.0, 10.0)),
                        launch_deg=float(rng.uniform(0.0, 25.0)), side_offset_deg=float(rng.uniform(-25, 25)),
                        distance=float(rng.uniform(3.0, 5.0)), launch_time_s=float(rng.uniform(-0.3, 0.0)))
        if row["feasible"]:
            rows.append(row)
    return rows
