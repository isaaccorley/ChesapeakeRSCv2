"""Post-hoc distance-weighted recall (DWR) evaluation.

Weight recall on canopy-covered road pixels by their L1 distance to visible road.
Correct predictions farther from visible road receive more weight.
DWR = sum(distance_i * [pred_i is road]) / sum(distance_i), over all occluded pixels i.

Repeated four-neighbor dilation computes the distance on the model's device.

Report all test patches and the leaf-on subset separately. Roads labeled as
canopy-covered can be visible in leaf-off or transitional imagery.
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from chesapeakersc.datasets import ChesapeakeRSCv2
from chesapeakersc.modules import ChesapeakeRSCv2Task

ROAD, OCCLUDED, IGNORE = 1, 2, 255


@torch.no_grad()
def l1_distance_to_road(mask: torch.Tensor) -> torch.Tensor:
    """Batched, exact L1 distance transform to the nearest ROAD (class 1) pixel.

    mask: (B, H, W) long tensor. Returns a (B, H, W) float tensor of the same shape.
    Four-neighbor dilation advances one pixel per step. The first step reaching a
    pixel gives its L1 distance. Patches without visible roads retain zero distance.
    """
    road = (mask == ROAD).float()
    reached = road.clone()
    dist = torch.zeros_like(reached)
    frontier = reached
    step = 0
    while True:
        step += 1
        shifted = torch.zeros_like(frontier)
        shifted[:, 1:, :] = torch.maximum(shifted[:, 1:, :], frontier[:, :-1, :])
        shifted[:, :-1, :] = torch.maximum(shifted[:, :-1, :], frontier[:, 1:, :])
        shifted[:, :, 1:] = torch.maximum(shifted[:, :, 1:], frontier[:, :, :-1])
        shifted[:, :, :-1] = torch.maximum(shifted[:, :, :-1], frontier[:, :, 1:])
        newly = (shifted > 0) & (reached == 0)
        if not newly.any():
            break
        dist = torch.where(newly, torch.full_like(dist, float(step)), dist)
        reached = torch.maximum(reached, shifted)
        frontier = shifted * newly.float()
    return dist


@torch.no_grad()
def dwr(
    checkpoint,
    manifest,
    dataset,
    split,
    bands,
    batch_size=16,
    num_workers=8,
    leaf_state=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = ChesapeakeRSCv2Task.load_from_checkpoint(checkpoint, map_location=device)
    model = task.model.eval().to(device)
    in_ch = 3 if bands == "rgb" else 4

    # Keep visible and canopy-covered roads separate for DWR evaluation.
    ds = ChesapeakeRSCv2(
        manifest=manifest,
        split=split,
        dataset=dataset,
        bands=bands,
        differentiate_tree_canopy_over_roads=True,
        leaf_state=leaf_state,
    )
    if len(ds) == 0:
        return dict(n_patches=0, n_occluded_px=0, dwr=float("nan"))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    counts = {
        k: torch.zeros((), device=device, dtype=torch.long)
        for k in ("tp", "fp", "fn", "vis_tp", "vis", "occ_tp", "occ")
    }
    t0 = time.perf_counter()
    for batch in dl:
        x = (batch["image"][:, :in_ch].to(device)) / 255.0
        y = batch["mask"].to(device)
        preds = model(x).argmax(dim=1)
        distance = l1_distance_to_road(y)
        valid = y != IGNORE
        gt_road = (y == ROAD) | (y == OCCLUDED)
        pred_road = (preds == ROAD) & valid
        vis = y == ROAD
        occ = y == OCCLUDED
        counts["tp"] += (pred_road & gt_road).sum()
        counts["fp"] += (pred_road & ~gt_road).sum()
        counts["fn"] += (~pred_road & gt_road).sum()
        counts["vis_tp"] += (pred_road & vis).sum()
        counts["vis"] += vis.sum()
        counts["occ_tp"] += (pred_road & occ).sum()
        counts["occ"] += occ.sum()
        if occ.any():
            numerator += (distance[occ] * (preds[occ] == ROAD).float()).sum()
            denominator += distance[occ].sum()
    dt = time.perf_counter() - t0
    c = {k: v.item() for k, v in counts.items()}
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    return dict(
        n_patches=len(ds),
        n_occluded_px=c["occ"],
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall)
        if precision + recall
        else float("nan"),
        visible_recall=c["vis_tp"] / c["vis"] if c["vis"] else float("nan"),
        occluded_recall=c["occ_tp"] / c["occ"] if c["occ"] else float("nan"),
        dwr=(numerator / denominator).item() if denominator > 0 else float("nan"),
        seconds=dt,
    )


def main(a):
    print(f"checkpoint: {a.checkpoint}", flush=True)
    subsets = [(None, "all"), ("leaf_on", "leaf_on_only")]
    if a.all_only:
        subsets = subsets[:1]
    for leaf_state, label in subsets:
        r = dwr(
            a.checkpoint,
            a.manifest,
            a.dataset,
            a.split,
            a.bands,
            a.batch_size,
            a.num_workers,
            leaf_state,
        )
        if r["n_patches"] == 0:
            print(
                f"  [{label}] no patches match (leaf_state not tracked for this dataset/split?)",
                flush=True,
            )
            continue
        print(
            f"  [{a.split}/{label}] n_patches={r['n_patches']} n_occluded_px={r['n_occluded_px']} "
            f"precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f} "
            f"visible_recall={r['visible_recall']:.4f} occluded_recall={r['occluded_recall']:.4f} "
            f"DWR={r['dwr']:.4f} ({r['seconds']:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset", choices=["core", "legacy", "both"], default="legacy")
    p.add_argument("--split", default="test")
    p.add_argument("--bands", choices=["rgb", "rgbn"], required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--all_only", action="store_true", help="Skip the leaf-on-only pass."
    )
    main(p.parse_args())
