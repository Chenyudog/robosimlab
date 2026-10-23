"""Plot actual 1024D trajectory features; no label-dependent coordinate changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    paths = [args.directory/f"episode_{e['episode_index']:06d}.npz" for e in manifest["episodes"]]
    if not all(p.exists() for p in paths):
        raise RuntimeError("Extraction incomplete; refusing to plot a partial task selection")
    data = [np.load(p) for p in paths]
    x = np.stack([d["feature"] for d in data])
    y = np.array([int(d["task_index"]) for d in data])
    if not np.isfinite(x).all() or len(np.unique(y)) < 2 or len(x) <= len(np.unique(y)):
        raise RuntimeError("Need finite features and multiple trajectories per task")
    sil = float(silhouette_score(x, y, metric="cosine"))
    perplexity = min(30, max(5, len(x)//15), len(x)-1)
    z = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
             random_state=0, max_iter=2000).fit_transform(x)
    colors = ["#5E7A91","#C0764A","#7E9B77","#A86A95","#C9A85A",
              "#6F7CB2","#B85C5C","#7DAFAE","#8E7BA8","#D4A76A"]
    fig, ax = plt.subplots(figsize=(8,6))
    for tid, color in zip(sorted(np.unique(y)), colors):
        m = y == tid
        ax.scatter(z[m,0], z[m,1], s=45, c=color, edgecolors="white", linewidths=.5,
                   alpha=.85, label=f"{tid}: {manifest['tasks'][str(tid)]}")
    ax.set(xticks=[], yticks=[], title="Pretrained pi0 - LIBERO-Goal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.5,-.035,f"Silhouette (1024D cosine) = {sil:.3f}\n"
            f"{len(x)} trajectories; {manifest['frames_per_episode']} sampled frames/trajectory",
            transform=ax.transAxes,ha="center",va="top",fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.15), frameon=False, fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(args.directory/"pretrained_goal_tsne.png",dpi=220,bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,6))
    for tid, color in zip(sorted(np.unique(y)), colors):
        m = y == tid
        ax.scatter(z[m,0], z[m,1], s=42, c=color, edgecolors="white", linewidths=.5, alpha=.85)
    ax.set(xticks=[], yticks=[])
    center = (z.max(axis=0)+z.min(axis=0))/2
    half = np.ptp(z,axis=0).max()*.56
    ax.set_xlim(center[0]-half,center[0]+half)
    ax.set_ylim(center[1]-half,center[1]+half)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(.98,.08,f"Silhouette = {sil:.3f}",transform=ax.transAxes,ha="right",fontsize=12)
    ax.text(.98,.03,"Pretrained",transform=ax.transAxes,ha="right",fontsize=18,fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.directory/"pretrained_goal_tsne_compact.png",dpi=220,bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(args.directory/"features_and_tsne.npz",features=x,task_indices=y,
                        embedding=z,episode_indices=[int(d['episode_index']) for d in data])
    metrics={"n_trajectories":len(x),"n_tasks":len(np.unique(y)),"silhouette_cosine_1024d":sil,
             "tsne":{"metric":"cosine","perplexity":perplexity,"init":"pca","random_state":0,"max_iter":2000}}
    (args.directory/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
