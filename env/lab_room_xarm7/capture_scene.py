"""Render previews for the xArm7 enclosed lab-room scene."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    expected = ("room_overview", "door_view", "table_view", "wrist_camera")
    cameras = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        for camera_id in range(model.ncam)
    }
    for camera in expected:
        if camera not in cameras:
            raise RuntimeError(f"missing camera: {camera}")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    for camera in expected:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        out = ROOT / f"{camera}.png"
        Image.fromarray(rgb).save(out)
        print(f"saved {out} ({rgb.shape[1]}x{rgb.shape[0]})")

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_camera")
    depth = renderer.render()
    if depth.shape != (720, 1280) or not np.isfinite(depth).all():
        raise RuntimeError("invalid wrist_camera depth image")
    print(f"wrist_camera depth render OK, range {depth.min():.3f}..{depth.max():.3f} m")
    renderer.close()


if __name__ == "__main__":
    main()
