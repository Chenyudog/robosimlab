"""Render appearance and camera previews for the local UR5e/UR10e scenes.

Run with MUJOCO_GL=egl or MUJOCO_GL=osmesa. The wrist and fixed cameras
can also produce depth arrays via MuJoCo's Renderer.enable_depth_rendering().
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ROBOTS = {
    "ur5e": (ROOT / "universal_robots_ur5e", 0.45, 2.2),
    "ur10e": (ROOT / "universal_robots_ur10e", 0.65, 3.0),
}


def capture(name: str, folder: Path, lookat_height: float, distance: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    expected = {"wrist_camera", "fixed_camera"}
    actual = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        for camera_id in range(model.ncam)
    }
    if not expected.issubset(actual):
        raise RuntimeError(f"{name}: missing cameras: {expected - actual}")

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    fixed_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
    wrist_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_3_joint")
    wrist_rotation_before = data.cam_xmat[wrist_id].copy()
    fixed_pose_before = (data.cam_xpos[fixed_id].copy(), data.cam_xmat[fixed_id].copy())
    data.qpos[model.jnt_qposadr[wrist_joint_id]] += 0.4
    mujoco.mj_forward(model, data)
    if np.allclose(data.cam_xmat[wrist_id], wrist_rotation_before):
        raise RuntimeError(f"{name}: wrist camera did not follow the wrist joint")
    if not np.allclose(data.cam_xpos[fixed_id], fixed_pose_before[0]) or not np.allclose(
        data.cam_xmat[fixed_id], fixed_pose_before[1]
    ):
        raise RuntimeError(f"{name}: fixed camera moved with the wrist joint")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=720, width=1280)
    appearance = mujoco.MjvCamera()
    appearance.type = mujoco.mjtCamera.mjCAMERA_FREE
    appearance.lookat[:] = (0.0, 0.0, lookat_height)
    appearance.distance = distance
    appearance.azimuth = 135
    appearance.elevation = -20

    for camera, filename in (
        (appearance, f"{name}.png"),
        ("wrist_camera", "wrist_camera.png"),
        ("fixed_camera", "fixed_camera.png"),
    ):
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        Image.fromarray(rgb).save(folder / filename)
        print(f"saved {folder / filename} ({rgb.shape[1]}x{rgb.shape[0]})")

    renderer.enable_depth_rendering()
    for camera in sorted(expected):
        renderer.update_scene(data, camera=camera)
        depth = renderer.render()
        if depth.shape != (720, 1280) or not np.isfinite(depth).all():
            raise RuntimeError(f"{name}: invalid {camera} depth image")
        print(f"{name}: {camera} depth render OK, range {depth.min():.3f}..{depth.max():.3f} m")
    renderer.close()


if __name__ == "__main__":
    for robot_name, (robot_folder, height, camera_distance) in ROBOTS.items():
        capture(robot_name, robot_folder, height, camera_distance)
