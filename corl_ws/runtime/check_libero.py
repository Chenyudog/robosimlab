"""Start one real LIBERO-Spatial task, render observations, and step physics."""

import argparse
import json
import os

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot", action="store_true", help="Use LeRobot's Gymnasium LIBERO wrapper")
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    if args.lerobot:
        from lerobot.envs.libero import LiberoEnv

        env = LiberoEnv(suite, 0, "libero_spatial", observation_width=64,
                        observation_height=64, num_steps_wait=1)
        try:
            observation, _ = env.reset(seed=0)
            next_observation, reward, terminated, truncated, _ = env.step(
                np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            images = {name: list(value.shape) for name, value in observation["pixels"].items()}
            if not images or any(shape != [64, 64, 3] for shape in images.values()):
                raise RuntimeError("LeRobot wrapper returned invalid camera observations")
            print(json.dumps({"status": "ok", "wrapper": "lerobot", "suite": "libero_spatial",
                              "task": task.language, "cameras": images, "steps": 1,
                              "reward": float(reward), "terminated": bool(terminated),
                              "truncated": bool(truncated), "observation_keys": len(next_observation)},
                             ensure_ascii=False, indent=2))
        finally:
            env.close()
        return

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
    try:
        env.seed(0)
        observation = env.reset()
        next_observation, reward, done, _ = env.step([0.0] * 7)
        images = {name: list(value.shape) for name, value in observation.items()
                  if name.endswith("_image") and isinstance(value, np.ndarray)}
        if not images or not all(value.shape[-1] == 3 for name, value in observation.items()
                                 if name in images):
            raise RuntimeError("LIBERO returned no RGB camera observation")
        print(json.dumps({"status": "ok", "suite": "libero_spatial", "task": task.language,
                          "cameras": images, "steps": 1, "reward": float(reward),
                          "done": bool(done), "observation_keys": len(next_observation)},
                         ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
