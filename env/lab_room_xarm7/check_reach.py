"""Check top-down reachability of every object and bin from the settled home state.

For each target the TCP (site ``link_tcp``) must reach the position with the
gripper z-axis pointing straight down. Objects are tested at the grasp point,
a pre-grasp point above it, and with a horizontal offset so that randomized
object placements still stay inside the workspace. Exit code 1 on failure.
"""

import sys

import mujoco
import numpy as np

from scene_state import load_scene

OBJECTS = ("cola_can", "glass_bottle_1", "cereal_box", "milk_carton", "bread", "fruit_lemon_1")
BINS = ("left_bin", "right_bin")
PLACE_TARGETS = ("robosuite_plate",)
PRE_GRASP_HEIGHT = 0.15  # m above the grasp point
RELEASE_HEIGHT = 0.15  # m above the bin / plate body origin
MARGIN = 0.05  # m horizontal offset for randomized placements
POS_TOL = 0.003  # m
TILT_TOL = 3.0  # deg


class TopDownIK:
    """Damped least-squares IK on the seven arm joints, kinematics only."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.q_home = data.qpos.copy()
        self.tcp = model.site("link_tcp").id
        joints = [model.joint(f"joint{i}").id for i in range(1, 8)]
        self.qadr = model.jnt_qposadr[joints]
        self.dadr = model.jnt_dofadr[joints]
        self.lo, self.hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]

    def solve(self, goal, seeds: int = 6, iters: int = 200):
        """Return (position error m, tilt deg, arm qpos) of the best seed."""
        m, d = self.model, self.data
        rng = np.random.default_rng(0)
        best = (np.inf, np.inf, None)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for seed in range(seeds):
            d.qpos[:] = self.q_home
            if seed:
                d.qpos[self.qadr] = rng.uniform(self.lo, self.hi)
            for _ in range(iters):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                rot = d.site_xmat[self.tcp].reshape(3, 3)
                err = np.hstack([goal - d.site_xpos[self.tcp], np.cross(rot[:, 2], [0.0, 0.0, -1.0])])
                mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
                jac = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
                d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + 0.5 * dq, self.lo, self.hi)
            mujoco.mj_kinematics(m, d)
            rot = d.site_xmat[self.tcp].reshape(3, 3)
            pos_err = np.linalg.norm(goal - d.site_xpos[self.tcp])
            tilt = np.degrees(np.arccos(np.clip(-rot[2, 2], -1.0, 1.0)))
            if pos_err + tilt / 100 < best[0] + best[1] / 100:
                best = (pos_err, tilt, d.qpos[self.qadr].copy())
            if pos_err < POS_TOL and tilt < TILT_TOL:
                break
        d.qpos[:] = self.q_home
        mujoco.mj_forward(m, d)
        return best


def reach_targets(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Named world-frame TCP goals derived from the current object/bin poses."""
    base_xy = data.body("link_base").xpos[:2]
    goals = {}
    for name in OBJECTS:
        p = data.body(name).xpos.copy()
        radial = p[:2] - base_xy
        radial /= np.linalg.norm(radial)
        tangent = np.array([-radial[1], radial[0]])
        goals[f"{name}/grasp"] = p
        goals[f"{name}/pre"] = p + [0, 0, PRE_GRASP_HEIGHT]
        for label, offset in (("far", radial), ("side+", tangent), ("side-", -tangent)):
            goals[f"{name}/{label}"] = p + np.hstack([MARGIN * offset, 0.0])
    for name in BINS + PLACE_TARGETS:
        goals[f"{name}/release"] = data.body(name).xpos.copy() + [0, 0, RELEASE_HEIGHT]
    return goals


def check(model: mujoco.MjModel, data: mujoco.MjData, verbose: bool = True) -> list:
    """Return names of unreachable goals."""
    ik = TopDownIK(model, data)
    failed = []
    for name, goal in reach_targets(model, data).items():
        pos_err, tilt, _ = ik.solve(goal)
        ok = pos_err < POS_TOL and tilt < TILT_TOL
        if not ok:
            failed.append(name)
        if verbose:
            print(f"{name:26s} err={pos_err * 1000:6.1f} mm  tilt={tilt:4.1f} deg  {'OK' if ok else 'FAIL'}")
    return failed


if __name__ == "__main__":
    model, data = load_scene()
    failed = check(model, data)
    print(f"\n{len(failed)} unreachable" + (f": {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
