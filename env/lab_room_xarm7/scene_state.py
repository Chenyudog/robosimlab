"""Initialize the lab room with objects resting on the tabletop."""

from pathlib import Path

import mujoco


def load_scene(settle_seconds: float = 1.0):
    scene = Path(__file__).resolve().parent / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    # Let contact forces and small orientation changes settle before the
    # first image or control action. All free joints remain dynamic.
    for _ in range(round(settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.time = 0.0
    mujoco.mj_forward(model, data)
    return model, data
