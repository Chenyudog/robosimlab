"""Scripted top-down pick-and-place expert for the xArm7 lab room.

The expert plans straight-line Cartesian moves for ``link_tcp``, converts each
20 Hz control step to joint targets with damped least-squares IK and sends
them to the position actuators. It is the demonstration source for the VLA
dataset: functions in ``ArmController.hooks`` run after every control step.

    MUJOCO_GL=egl python expert.py --object cola_can --bin right_bin --snapshots /tmp/snap
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from scene_state import load_scene

TABLE_Z = 0.620787  # tabletop surface, world z
CONTROL_HZ = 20
GRIP_OPEN, GRIP_CLOSED = 0.0, 255.0  # ctrl 0: 88.8 mm finger gap, 255: fully closed
TRANSIT_Z = TABLE_Z + 0.30  # TCP height for horizontal moves
FINGERTIP_CLEARANCE = 0.013  # min TCP height above table; closed tips sit ~3 mm below TCP
PALM_ABOVE_TCP = 0.068  # lowest point of the gripper base (palm) above link_tcp
PALM_CLEARANCE = 0.010  # keep the palm this far above the object's top
IK_TOL = 2e-3  # max accepted IK residual (m / rad)
OBJECTS = ("cola_can", "glass_bottle_1", "cereal_box", "milk_carton", "bread", "fruit_lemon_1")
BINS = ("left_bin", "right_bin")


def _world_points(model, data, body_name):
    """World-frame vertices of a body's collision geoms (meshes and boxes)."""
    body = model.body(body_name).id
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            local = corners * model.geom_size[g]
        else:
            raise ValueError(f"unsupported geom type in {body_name}")
        pts.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.vstack(pts)


def world_aabb(model, data, body_name):
    """Axis-aligned world bounds (lo, hi) of a body's collision geometry."""
    pts = _world_points(model, data, body_name)
    return pts.min(0), pts.max(0)


def grasp_yaw(model, data, body_name):
    """TCP yaw that squeezes the object across its narrower horizontal side.

    With yaw psi the fingers close along world (-sin psi, cos psi). The result
    is wrapped to the smallest rotation away from the home yaw (0).
    """
    pts = _world_points(model, data, body_name)
    rot = data.xmat[model.body(body_name).id].reshape(3, 3)
    local = (pts - data.xpos[model.body(body_name).id]) @ rot
    ext = local.max(0) - local.min(0)
    axis = rot[:, 0] if ext[0] <= ext[1] else rot[:, 1]  # narrower horizontal body axis
    psi = np.arctan2(axis[1], axis[0]) - np.pi / 2
    period = np.pi / 2 if abs(ext[0] - ext[1]) < 0.005 else np.pi  # round/square vs. oblong
    return (psi + period / 2) % period - period / 2


def top_down_rotation(yaw):
    """TCP orientation with the tool z-axis pointing down, rotated by ``yaw``."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ np.diag([1.0, -1.0, -1.0])


def _quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(9))
    return q


def _slerp_mat(r0, r1, t):
    q0, q1 = _quat(r0), _quat(r1)
    if q0 @ q1 < 0:
        q1 = -q1
    ang = np.arccos(np.clip(q0 @ q1, -1.0, 1.0))
    q = q0 if ang < 1e-6 else (np.sin((1 - t) * ang) * q0 + np.sin(t * ang) * q1) / np.sin(ang)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q / np.linalg.norm(q))
    return out.reshape(3, 3)


def _min_jerk(t):
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


class ArmController:
    """Cartesian TCP control of the simulated xArm7 at ``CONTROL_HZ``."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.ik_data = mujoco.MjData(model)
        self.tcp = model.site("link_tcp").id
        joints = [model.joint(f"joint{i}").id for i in range(1, 8)]
        self.qadr = model.jnt_qposadr[joints]
        self.dadr = model.jnt_dofadr[joints]
        self.lo, self.hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]
        self.substeps = round(1.0 / (CONTROL_HZ * model.opt.timestep))
        self.q_cmd = data.ctrl[:7].copy()
        self.grip_cmd = float(data.ctrl[7])
        self.hooks = []  # callables(controller) run after every control step
        robot_root = model.body("link_base").id
        self.robot_bodies = {b for b in range(model.nbody) if model.body_rootid[b] == robot_root}
        self.gripper_bodies = {model.body(n).id for n in (
            "xarm_gripper_base_link", "left_outer_knuckle", "left_finger", "left_inner_knuckle",
            "right_outer_knuckle", "right_finger", "right_inner_knuckle")}
        self.allowed_body = None  # object the gripper may touch
        self.contacts = set()  # unexpected robot contacts seen so far

    def fk(self, q):
        """TCP position and rotation for arm joints ``q`` (other qpos from sim)."""
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.tcp].copy(), d.site_xmat[self.tcp].reshape(3, 3).copy()

    def ik(self, pos, rot, q_seed, iters=100):
        """Damped least-squares IK; returns (q, residual)."""
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q_seed
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cur = d.site_xmat[self.tcp].reshape(3, 3)
            err = np.hstack([pos - d.site_xpos[self.tcp],
                             0.5 * sum(np.cross(cur[:, i], rot[:, i]) for i in range(3))])
            if np.abs(err).max() < 1e-5:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            jac = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        return d.qpos[self.qadr].copy(), float(np.abs(err).max())

    def step(self, q_target, grip_target):
        """One control step: ramp joint/gripper targets linearly over the substeps."""
        m, d = self.model, self.data
        q0, g0 = self.q_cmd.copy(), self.grip_cmd
        for k in range(1, self.substeps + 1):
            a = k / self.substeps
            d.ctrl[:7] = q0 + a * (q_target - q0)
            d.ctrl[7] = g0 + a * (grip_target - g0)
            mujoco.mj_step(m, d)
        self.q_cmd, self.grip_cmd = np.asarray(q_target, float).copy(), float(grip_target)
        self._log_contacts()
        for hook in self.hooks:
            hook(self)

    def _log_contacts(self):
        m, d = self.model, self.data
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for rb, other in ((b1, b2), (b2, b1)):
                if rb in self.robot_bodies and other not in self.robot_bodies:
                    if rb in self.gripper_bodies and other == self.allowed_body:
                        continue
                    self.contacts.add((m.body(rb).name, m.body(other).name))

    def tcp_pose(self):
        return self.data.site_xpos[self.tcp].copy(), self.data.site_xmat[self.tcp].reshape(3, 3).copy()

    def move_to(self, pos, yaw, speed=0.15):
        """Min-jerk straight-line TCP move from the commanded pose to ``pos``/``yaw``."""
        p0, r0 = self.fk(self.q_cmd)
        r1 = top_down_rotation(yaw)
        ang = np.arccos(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1.0, 1.0))
        duration = max(np.linalg.norm(pos - p0) / speed, ang / 1.0, 0.5)
        n = max(1, int(np.ceil(duration * CONTROL_HZ)))
        q = self.q_cmd.copy()
        for k in range(1, n + 1):
            s = _min_jerk(k / n)
            q, res = self.ik(p0 + s * (pos - p0), _slerp_mat(r0, r1, s), q)
            if res > IK_TOL:
                raise RuntimeError(f"IK failed (residual {res:.4f}) moving to {np.round(pos, 3)}")
            self.step(q, self.grip_cmd)

    def set_gripper(self, value, duration=0.8):
        n = int(round(duration * CONTROL_HZ))
        g0 = self.grip_cmd
        for k in range(1, n + 1):
            self.step(self.q_cmd, g0 + min(1.0, 2 * k / n) * (value - g0))

    def hold(self, duration):
        for _ in range(int(round(duration * CONTROL_HZ))):
            self.step(self.q_cmd, self.grip_cmd)


def in_bin(model, data, obj, bin_name):
    """Object centre inside the bin walls and its bottom below the rim."""
    lo, hi = world_aabb(model, data, bin_name)
    wall = 0.012  # wall thickness (2 x 0.006 half-size)
    olo, ohi = world_aabb(model, data, obj)
    c = (olo + ohi) / 2
    inside = np.all(c[:2] > lo[:2] + wall) and np.all(c[:2] < hi[:2] - wall)
    return bool(inside and olo[2] < hi[2])


def pick_and_place(ctrl, obj, bin_name, snapshot=None):
    """Run one pick-and-place episode; returns a result dict.

    ``snapshot(label)`` is called after each phase (e.g. to save images).
    """
    m, d = ctrl.model, ctrl.data
    snap = snapshot or (lambda label: None)
    ctrl.allowed_body = m.body(obj).id
    lo, hi = world_aabb(m, d, obj)
    centre = (lo + hi) / 2
    yaw = grasp_yaw(m, d, obj)
    # Pad centres sit ~2.5 cm above link_tcp: aim them at the object's middle,
    # but keep fingertips off the table and the palm above the object's top.
    grasp_z = max(centre[2] - 0.02, TABLE_Z + FINGERTIP_CLEARANCE, hi[2] + PALM_CLEARANCE - PALM_ABOVE_TCP)
    grasp_z = min(grasp_z, hi[2] - 0.015)
    start_z = d.body(obj).xpos[2]

    ctrl.set_gripper(GRIP_OPEN, 0.4)
    ctrl.move_to(np.array([centre[0], centre[1], TRANSIT_Z]), yaw)
    snap("1_above")
    ctrl.move_to(np.array([centre[0], centre[1], grasp_z]), yaw, speed=0.08)
    snap("2_descend")
    ctrl.set_gripper(GRIP_CLOSED)
    ctrl.hold(0.3)
    snap("3_grasp")
    ctrl.move_to(np.array([centre[0], centre[1], TRANSIT_Z]), yaw, speed=0.10)
    lifted = d.body(obj).xpos[2] - start_z
    tcp_p, tcp_r = ctrl.tcp_pose()
    rel_lift = tcp_r.T @ (d.body(obj).xpos - tcp_p)
    snap("4_lift")
    if lifted < 0.10:
        ctrl.allowed_body = None
        return {"success": False, "failure": "not_lifted", "lifted": float(lifted),
                "grasp_z": float(grasp_z), "yaw": float(yaw), "contacts": sorted(ctrl.contacts)}

    bin_body = d.body(bin_name)
    base = m.geom(f"{bin_name}_base_collision").id
    floor_z = d.geom_xpos[base][2] + m.geom_size[base][2]
    hang = world_aabb(m, d, obj)[0][2] - tcp_p[2]  # object bottom relative to TCP
    release_z = max(floor_z + 0.025 - hang, TABLE_Z + 0.12)
    ctrl.move_to(np.array([bin_body.xpos[0], bin_body.xpos[1], TRANSIT_Z]), yaw)
    tcp_p, tcp_r = ctrl.tcp_pose()
    slip = float(np.linalg.norm(tcp_r.T @ (d.body(obj).xpos - tcp_p) - rel_lift))
    snap("5_over_bin")
    ctrl.move_to(np.array([bin_body.xpos[0], bin_body.xpos[1], release_z]), yaw, speed=0.08)
    ctrl.set_gripper(GRIP_OPEN)
    snap("6_release")
    ctrl.move_to(np.array([bin_body.xpos[0], bin_body.xpos[1], TRANSIT_Z]), yaw)
    p_before = d.body(obj).xpos.copy()
    ctrl.hold(1.0)
    settle = float(np.linalg.norm(d.body(obj).xpos - p_before))
    snap("7_done")
    ctrl.allowed_body = None
    success = in_bin(m, d, obj, bin_name) and settle < 0.005
    return {"success": success, "failure": None if success else "not_in_bin",
            "lifted": float(lifted), "slip": slip, "settle": settle,
            "grasp_z": float(grasp_z), "yaw": float(yaw), "contacts": sorted(ctrl.contacts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object", default="cola_can", choices=OBJECTS)
    parser.add_argument("--bin", default="right_bin", choices=BINS)
    parser.add_argument("--snapshots", type=Path, help="directory for per-phase fixed/wrist images")
    args = parser.parse_args()

    model, data = load_scene()
    ctrl = ArmController(model, data)
    snapshot = None
    if args.snapshots:
        from PIL import Image
        args.snapshots.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=360, width=640)

        def snapshot(label):
            for cam in ("fixed_camera", "wrist_camera"):
                renderer.update_scene(data, camera=cam)
                Image.fromarray(renderer.render()).save(args.snapshots / f"{label}_{cam}.png")

    result = pick_and_place(ctrl, args.object, args.bin, snapshot)
    print(f"{args.object} -> {args.bin}: " + ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in result.items()))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())