"""Render appearance and camera previews for the local Franka Panda scene."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "franka_emika_panda"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    expected = {"wrist_camera"}
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        for camera_id in range(model.ncam)
    }
    if not expected.issubset(actual):
        raise RuntimeError(f"missing cameras: {expected - actual}")

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
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
    appearance.lookat[:] = (0.45, 0.0, 0.55)
    appearance.distance = 1.7
    appearance.azimuth = 135
    appearance.elevation = -22

    outputs = (
        (appearance, "franka.png"),
        ("wrist_camera", "wrist_camera.png"),
    )
    for camera, filename in outputs:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        Image.fromarray(rgb).save(FOLDER / filename)
        print(f"saved {FOLDER / filename} ({rgb.shape[1]}x{rgb.shape[0]})")

    renderer.enable_depth_rendering()
    for camera in sorted(expected):
        renderer.update_scene(data, camera=camera)
        depth = renderer.render()
        if depth.shape != (720, 1280) or not np.isfinite(depth).all():
            raise RuntimeError(f"invalid {camera} depth image")
        print(f"{camera} depth render OK, range {depth.min():.3f}..{depth.max():.3f} m")
    renderer.close()


if __name__ == "__main__":
    main()
