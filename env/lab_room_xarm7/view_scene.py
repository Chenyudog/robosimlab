"""Open the lab room with the complete robot and object home state."""

import mujoco.viewer

from scene_state import load_scene


if __name__ == "__main__":
    model, data = load_scene()
    mujoco.viewer.launch(model, data)
