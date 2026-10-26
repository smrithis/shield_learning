"""Frame correction and independent geometry checks (backend environment only).

Frames: W = MuJoCo world (`geom_xpos`), B = robot base = odom_combined (the
launch publishes an identity odom_combined -> base_link), P = interception plane
holding stored `task_position`. run_IK.py solved IK for p_W = R_WP p_P + t_WP;
the planner maps p_P = R_BP^T (p_B - t_BP) with T_BP = T_WB^-1 T_WP. The upstream
evaluator treats raw MuJoCo world FK as plane coordinates; this module converts
shield FK W -> P at that boundary instead of changing the tree or evaluator.
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def base_pose(model):
    """T_WB of the fixed MuJoCo body named 'base' (independent of qpos)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if body < 0:
        raise RuntimeError("MuJoCo model has no body named 'base'")
    return data.xmat[body].reshape(3, 3).copy(), data.xpos[body].copy()


def world_from_plane(model, base_from_plane_rotation, base_from_plane_translation):
    """T_WP = T_WB T_BP, using the planner's own (rounded) T_BP."""
    R_wb, t_wb = base_pose(model)
    return R_wb @ base_from_plane_rotation, R_wb @ base_from_plane_translation + t_wb


def ik_world_from_plane():
    """run_IK.py's literal plane definition, kept only for cross-checking."""
    c, s = np.cos(np.deg2rad(-45)), np.sin(np.deg2rad(-45))
    return (np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]),
            0.8 * np.array([np.cos(np.deg2rad(135)), np.sin(np.deg2rad(135)), 0.]))


def plane_frame_evaluator(evaluator_class):
    """Existing evaluator whose shield_geometry() returns plane-frame pose.

    `distance_at()` is unchanged; it already assumes plane-frame shield pose and
    maps its base-frame ball and outputs with the planner transform.
    """
    class PlaneFrameEvaluator(evaluator_class):
        def __init__(self):
            super().__init__()
            self.world_from_plane_rotation, self.world_from_plane_translation = world_from_plane(
                self.model, self.base_from_plane_rotation, self.base_from_plane_translation)

        def world_shield_geometry(self, q):
            return super().shield_geometry(q)

        def shield_geometry(self, q):
            center, rotation = super().shield_geometry(q)
            R = self.world_from_plane_rotation
            return R.T @ (center - self.world_from_plane_translation), R.T @ rotation

    return PlaneFrameEvaluator


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _axis_angle(axis, angle):
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class UrdfShield:
    """Shield-box pose in base_link (= odom_combined) from the displayed URDF.

    Independent of MuJoCo root placement and of the planner transform.
    """

    def __init__(self, path, root="base_link", tip="link_6"):
        tree = ET.parse(path).getroot()
        joints = {j.find("child").get("link"): j for j in tree.findall("joint")}
        self.chain = []
        link = tip
        while link != root:
            joint = joints[link]
            origin = joint.find("origin")
            xyz = np.array(origin.get("xyz", "0 0 0").split(), dtype=float)
            rpy = np.array(origin.get("rpy", "0 0 0").split(), dtype=float)
            axis = joint.find("axis")
            self.chain.append((joint.get("name"), joint.get("type"), xyz, _rpy(*rpy),
                               None if axis is None else np.array(axis.get("xyz").split(), dtype=float)))
            link = joint.find("parent").get("link")
        self.chain.reverse()
        box = next(v for l in tree.findall("link") if l.get("name") == tip
                      for v in l.findall("collision") if v.find("geometry/box") is not None)
        self.offset = np.array(box.find("origin").get("xyz").split(), dtype=float)
        self.half_extents = np.array(box.find("geometry/box").get("size").split(), dtype=float) / 2

    def pose(self, q):
        q = dict(zip([f"joint_{i}" for i in range(1, 7)], q))
        R, t = np.eye(3), np.zeros(3)
        for name, kind, xyz, rot, axis in self.chain:
            t = t + R @ xyz
            R = R @ rot
            if kind == "revolute":
                R = R @ _axis_angle(axis, q[name])
        return t + R @ self.offset, R


class SweptChecker:
    """Sub-tick sphere/box check using MuJoCo's own collision distance.

    Independent of the upstream evaluator's distance code and of the planner
    transform: the ball is placed in MuJoCo world with the model's base pose.
    The bridge integrates constant joint velocity per tick, so q is linear in time
    between recorded ticks; the analytic projectile is evaluated exactly.
    """

    def __init__(self, model_path, ball_radius):
        spec = mujoco.MjSpec.from_file(str(model_path))
        body = spec.worldbody.add_body(name="swept_check_ball", mocap=True)
        body.add_geom(name="swept_check_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[ball_radius, 0, 0], contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.shield = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "shield")
        self.ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "swept_check_ball")
        self.half_extents = self.model.geom_size[self.shield].copy()
        self.R_wb, self.t_wb = base_pose(self.model)
        self.corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                                 for sz in (-1, 1)]) * self.half_extents
        self._fromto = np.zeros(6)

    def distance(self, q, ball_base):
        self.data.qpos[:6] = q
        self.data.mocap_pos[0] = self.R_wb @ ball_base + self.t_wb
        mujoco.mj_kinematics(self.model, self.data)
        d = mujoco.mj_geomDistance(self.model, self.data, self.shield, self.ball, 10.0, self._fromto)
        corners = self.data.geom_xpos[self.shield] + self.corners @ self.data.geom_xmat[self.shield].reshape(3, 3).T
        return float(d), corners

    def sweep(self, qs, ball_at, dt, subdivisions):
        """qs[k] is q at time k*dt; ball_at(t) gives (position, velocity) in base frame."""
        qs = np.asarray(qs)
        best = (np.inf, 0.0)
        first_contact = contact_local = contact_speed = None
        max_step = 0.0
        previous = None
        count = 0
        for k in range(len(qs) - 1):
            for s in range(subdivisions + (k == len(qs) - 2)):
                alpha = s / subdivisions
                t = (k + alpha) * dt
                position, _ = ball_at(t)
                d, corners = self.distance(qs[k] + alpha * (qs[k + 1] - qs[k]), np.asarray(position))
                count += 1
                corner_step = 0.0
                if previous is not None:
                    # Max displacement of a rigid convex box is at a corner; add ball motion.
                    ball_step = np.linalg.norm(self.R_wb @ (np.asarray(position) - previous[1]))
                    corner_step = float(np.max(np.linalg.norm(corners - previous[0], axis=1)))
                    max_step = max(max_step, corner_step + ball_step)
                previous = (corners, np.asarray(position))
                if d < best[0]:
                    best = (d, t)
                if d <= 0 and first_contact is None:
                    first_contact = t
                    # Ball centre in shield-box coordinates: x normal, |y|,|z| <= half extents on the face.
                    rotation = self.data.geom_xmat[self.shield].reshape(3, 3)
                    contact_local = rotation.T @ (self.data.mocap_pos[0] - self.data.geom_xpos[self.shield])
                    contact_speed = corner_step * subdivisions / dt
        return {"swept_min_surface_distance_m": best[0], "swept_min_time_s": best[1],
                "swept_first_contact_time_s": first_contact, "swept_sample_interval_s": dt / subdivisions,
                "swept_first_contact_ball_in_shield_m": contact_local,
                "swept_first_contact_max_shield_point_speed_m_s": contact_speed,
                "swept_sample_count": count,
                # Chord bound between samples; distance can dip below samples by at most ~this.
                "swept_max_relative_motion_between_samples_m": max_step}
