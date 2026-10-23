"""Headless physics smoke check for the existing SO-101 MuJoCo scene."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_SCENE = Path(__file__).resolve().parent.parent / "mujoco_ws/so_arm101/grasp_scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--check-camera", action="store_true", help="Render one wrist-camera frame")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if camera < 0 or cube < 0 or model.nu != 6:
        raise RuntimeError("Expected SO-101 scene with wrist_cam, cube, and six actuators")

    # Position actuators hold their corresponding initial joint positions.
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i, 0])
        qpos_id = int(model.jnt_qposadr[joint_id])
        data.ctrl[i] = np.clip(data.qpos[qpos_id], *model.actuator_ctrlrange[i])

    mujoco.mj_forward(model, data)
    initial_cube_z = float(data.xpos[cube, 2])
    for _ in range(args.steps):
        mujoco.mj_step(model, data)
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        raise RuntimeError("Simulation produced non-finite state")

    camera_shape = None
    if args.check_camera:
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            renderer.update_scene(data, camera="wrist_cam")
            camera_shape = list(renderer.render().shape)

    print(json.dumps({
        "status": "ok",
        "scene": str(args.scene.resolve()),
        "mujoco_version": mujoco.__version__,
        "steps": args.steps,
        "sim_time_s": round(float(data.time), 4),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "wrist_camera": True,
        "rendered_image_shape": camera_shape,
        "cube_initial_z_m": round(initial_cube_z, 4),
        "cube_final_z_m": round(float(data.xpos[cube, 2]), 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
