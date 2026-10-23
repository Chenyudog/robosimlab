"""Offline LIBERO action-expert features from a local LeRobot pi0 checkpoint."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/cyd/lerobot/src")

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import sentencepiece
import torch
from accelerate import init_empty_weights
from safetensors import safe_open

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
from lerobot.policies.common.vla_utils import resize_with_pad_torch

ROOT = Path(__file__).resolve().parent
# Match official LIBERO suite by language, rather than assuming task-index order.
GOAL_TASKS = [
    "open the middle drawer of the cabinet", "put the bowl on the stove",
    "put the wine bottle on top of the cabinet", "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet", "push the plate to the front of the stove",
    "put the cream cheese in the bowl", "turn on the stove",
    "put the bowl on the plate", "put the wine bottle on the rack",
]


def load_model(weights, device):
    """Stream weights; fail if any inference parameter is unaccounted for.

    Vocabulary prediction heads are unused by flow matching. The PaliGemma
    lm_head tensor supplies the tied language input embedding in this file.
    """
    config = PI0Config(dtype="bfloat16", device=device)
    with init_empty_weights():
        model = PI0Pytorch(config)
    model.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Identity()
    model.paligemma_with_expert.paligemma.lm_head = torch.nn.Identity()
    expected = dict(model.named_parameters())
    used = set()
    alias = {
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight":
        "paligemma_with_expert.paligemma.lm_head.weight",
    }
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for index, (name, param) in enumerate(expected.items()):
            key = name if name in keys else alias.get(name, name)
            if key not in keys:
                raise RuntimeError(f"Missing checkpoint parameter: {name}")
            value = f.get_tensor(key)
            if tuple(value.shape) != tuple(param.shape):
                raise RuntimeError(f"Shape mismatch: {name}")
            # Keep normalization arithmetic in FP32; use autocast for matmuls.
            dtype = torch.float32 if any(x in name for x in (
                "input_layernorm", "post_attention_layernorm", ".norm.")) else torch.bfloat16
            value = value.to(device=device, dtype=dtype)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, torch.nn.Parameter(value, requires_grad=False))
            used.add(key)
            if index % 150 == 0:
                print(f"weights {index}/{len(expected)}", flush=True)
    allowed_unused = {"paligemma_with_expert.gemma_expert.lm_head.weight"}
    if keys - used - allowed_unused:
        raise RuntimeError(f"Unconsumed weights: {keys-used-allowed_unused}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Unloaded meta parameters")
    model.to(device).eval()
    return model


def image_tensor(cell, device):
    image = np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"), copy=True)
    x = torch.tensor(image, device=device, dtype=torch.float32)[None] / 255
    x = resize_with_pad_torch(x, 224, 224)
    return (x * 2 - 1).permute(0, 3, 1, 2)


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("/home/cyd/datasets/libero_pi"))
    ap.add_argument("--weights", type=Path, default=Path("/home/cyd/robosimlab/model/pi0/model.safetensors"))
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/pretrained_goal")
    ap.add_argument("--episodes-per-task", type=int, default=10, help="0 selects all episodes")
    ap.add_argument("--frames-per-episode", type=int, default=8, help="0 selects all frames")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layer-index", type=int, default=10, choices=range(18))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(s) for s in (args.dataset / "meta/tasks.jsonl").read_text().splitlines()]
    goal_ids = {t["task_index"]: t["task"] for t in tasks if t["task"] in GOAL_TASKS}
    if len(goal_ids) != 10:
        raise RuntimeError(f"Expected ten Goal tasks, got {goal_ids}")
    episodes = [json.loads(s) for s in (args.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    selected = []
    rng = np.random.default_rng(args.seed)
    for tid, task in sorted(goal_ids.items()):
        candidates = [e for e in episodes if e["tasks"] == [task]]
        n = args.episodes_per_task or len(candidates)
        chosen = sorted(rng.choice(len(candidates), min(n, len(candidates)), replace=False))
        selected.extend([{**candidates[i], "task_index": tid} for i in chosen])
    if args.limit:
        selected = selected[:args.limit]
    manifest = {
        "stage": "pretrained", "weights": str(args.weights), "dataset": str(args.dataset),
        "dataset_revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
        "suite": "libero_goal", "tasks": goal_ids, "episodes": selected,
        "layer_index_zero_based": args.layer_index,
        "feature": "post-block action tokens, mean over 50 tokens, then sampled frames",
        "feature_mode": "offline observations, 10-step model denoising, capture before update at t=0.1",
        "seed": args.seed, "frames_per_episode": args.frames_per_episode,
        "dtype": "BF16 with FP32 normalization and autocast",
        "normalization": "OpenPI pi0_libero quantiles, assets/norm_stats.json",
        "sampling": "uniform frame positions; seeded random episodes per task",
        "precision_note": "LeRobot port; sampled-frame reproduction, not exact paper checkpoint/protocol",
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise RuntimeError("Output manifest differs; choose a fresh output directory")
    atomic_json(manifest_path, manifest)
    pending = [e for e in selected if not (args.output / f"episode_{e['episode_index']:06d}.npz").exists()]
    print(f"Selected {len(selected)} trajectories, {len(pending)} pending. Goal task IDs: {list(goal_ids)}", flush=True)
    if not pending:
        return
    sp = sentencepiece.SentencePieceProcessor(model_file=str(ROOT / "assets/paligemma_tokenizer.model"))
    ns = json.loads((ROOT / "assets/norm_stats.json").read_text())["norm_stats"]["state"]
    lo, hi = np.array(ns["q01"], dtype=np.float32)[:8], np.array(ns["q99"], dtype=np.float32)[:8]
    model = load_model(args.weights, "cuda")
    captured = []
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -50:, :].float().mean(1).cpu().numpy())
    handle = model.paligemma_with_expert.gemma_expert.model.layers[args.layer_index].register_forward_hook(hook)
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for count, episode in enumerate(pending, 1):
            eid = episode["episode_index"]
            path = args.dataset / f"data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet"
            table = pq.read_table(path, columns=["image", "wrist_image", "state", "task_index", "frame_index"])
            if set(table["task_index"].to_pylist()) != {episode["task_index"]}:
                raise RuntimeError(f"Task/episode mismatch {eid}")
            n = len(table)
            indices = np.unique(np.linspace(0, n-1, min(args.frames_per_episode or n, n), dtype=int))
            text = goal_ids[episode["task_index"]]
            tokens = sp.encode(text.strip(), add_bos=True) + sp.encode("\n")
            tokens = tokens[:48]
            mask = torch.tensor([[True]*len(tokens)+[False]*(48-len(tokens))], device="cuda")
            tokens = torch.tensor([tokens+[0]*(48-len(tokens))], device="cuda")
            features = []
            for fi in indices:
                frame = table.slice(int(fi), 1).to_pylist()[0]
                state = np.array(frame["state"], dtype=np.float32)
                state = 2*(state-lo)/(hi-lo+1e-6)-1
                state = torch.tensor(np.pad(state, (0,24))[None], device="cuda")
                images = [image_tensor(frame[key], "cuda") for key in ("image", "wrist_image")]
                masks = [torch.ones(1, dtype=torch.bool, device="cuda") for _ in images]
                gen = torch.Generator(device="cuda").manual_seed(args.seed + eid*100000 + int(fi))
                noise = torch.randn((1,50,32), generator=gen, device="cuda")
                captured.clear()
                actions = model.sample_actions(images, masks, tokens, mask, state, noise=noise, num_steps=10)
                if len(captured) != 10 or not np.isfinite(captured[-1]).all() or not torch.isfinite(actions).all():
                    raise RuntimeError(f"Invalid inference for episode {eid} frame {fi}")
                features.append(captured[-1][0])
            frames = np.stack(features)
            target = args.output / f"episode_{eid:06d}.npz"
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, feature=frames.mean(0), frame_features=frames,
                                frame_indices=indices, task_index=episode["task_index"], episode_index=eid)
            temporary.replace(target)
            elapsed = time.monotonic()-start
            print(f"[{count}/{len(pending)}] episode={eid} task={episode['task_index']} frames={len(indices)} "
                  f"elapsed={elapsed:.1f}s peak_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    handle.remove()
    print("Extraction complete", flush=True)


if __name__ == "__main__":
    main()
