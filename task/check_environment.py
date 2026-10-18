"""Check the local LeRobot, Pi0 configuration, and Franka simulation setup."""

from pathlib import Path

import lerobot
import mujoco
import numpy as np
import torch
from lerobot.policies.pi0 import PI0Config, PI0Policy


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Path(lerobot.__file__).resolve()
    assert source.is_relative_to(ROOT / "lerobot"), source
    print(f"LeRobot source: {source}")
    print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    config = PI0Config.from_pretrained(str(ROOT / "model/pi0"))
    print(f"Pi0 class: {PI0Policy.__name__}; action shape: {config.output_features['action'].shape}")
    weights = ROOT / "model/pi0/model.safetensors"
    assert weights.is_file(), weights
    print(f"Local weights: {weights.stat().st_size / 1e9:.2f} GB (not loaded)")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot/franka_emika_panda/scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    print(f"Franka simulation OK: {model.nu} actuators, {model.ncam} cameras, time={data.time:.3f}s")


if __name__ == "__main__":
    main()
