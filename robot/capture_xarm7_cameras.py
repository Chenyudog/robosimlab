"""Render appearance and wrist camera previews for the local xArm7 scene."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "ufactory_xarm7"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    if wrist_id < 0:
        raise RuntimeError("missing wrist_camera")

    joint7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    wrist_pose_before = (data.cam_xpos[wrist_id].copy(), data.cam_xmat[wrist_id].copy())
    data.qpos[model.jnt_qposadr[joint7_id]] += 0.4
    mujoco.mj_forward(model, data)
    if np.allclose(data.cam_xmat[wrist_id], wrist_pose_before[1]):
        raise RuntimeError("wrist camera did not follow the wrist joint")

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    appearance = mujoco.MjvCamera()
    appearance.type = mujoco.mjtCamera.mjCAMERA_FREE
    appearance.lookat[:] = (0.25, 0.0, 0.55)
    appearance.distance = 1.9
    appearance.azimuth = 135
    appearance.elevation = -20

    for camera, filename in (
        (appearance, "xarm7_render.png"),
        ("wrist_camera", "wrist_camera.png"),
    ):
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        Image.fromarray(rgb).save(FOLDER / filename)
        print(f"saved {FOLDER / filename} ({rgb.shape[1]}x{rgb.shape[0]})")

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_camera")
    depth = renderer.render()
    if depth.shape != (720, 1280) or not np.isfinite(depth).all():
        raise RuntimeError("invalid wrist_camera depth image")
    print(f"wrist_camera depth render OK, range {depth.min():.3f}..{depth.max():.3f} m")
    renderer.close()


if __name__ == "__main__":
    main()
