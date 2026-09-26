"""Render previews for the xArm7 pick-and-place scene."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scene.xml"


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def save_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str, filename: str) -> None:
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()
    Image.fromarray(rgb).save(ROOT / filename)
    print(f"saved {ROOT / filename} ({rgb.shape[1]}x{rgb.shape[0]})")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    reset_home(model, data)

    cameras = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
        for camera_id in range(model.ncam)
    }
    for camera in ("overview_camera", "wrist_camera"):
        if camera not in cameras:
            raise RuntimeError(f"missing camera: {camera}")

    renderer = mujoco.Renderer(model, height=720, width=1280)
    save_rgb(renderer, data, "overview_camera", "overview_camera.png")
    save_rgb(renderer, data, "wrist_camera", "wrist_camera.png")

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="wrist_camera")
    depth = renderer.render()
    if depth.shape != (720, 1280) or not np.isfinite(depth).all():
        raise RuntimeError("invalid wrist_camera depth image")
    print(f"wrist_camera depth render OK, range {depth.min():.3f}..{depth.max():.3f} m")
    renderer.close()


if __name__ == "__main__":
    main()
