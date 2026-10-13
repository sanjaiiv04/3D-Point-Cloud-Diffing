import os
# To avoid the OpenMP runtime conflict caused by multiple libraries linking their own independent copies of the OpenMP library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from rich.console import Console

from geometry import load_cams, project
from label import label_cluster

console = Console()

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
OUT = ROOT / "output"

VOXEL_FRAC = 0.006        # point spacing used for the diff
TAU_BASE_FRAC = 0.004     # constant part of depth tolerance
TAU_SIGMA_MULT = 3.0      # tolerance = 3 x calibrated depth noise
TAU_REL_MIN, TAU_REL_MAX = 0.02, 0.12
EDGE_REL = 0.05           # depth discontinuity
BORDER_FRAC = 0.03        # ignore image borders
MIN_SEEN_SELF = 3         # own-video cameras that must agree the point is a surface
MIN_VOTES = 3             # other-video cameras that must see "through" the point
MIN_THROUGH_RATIO = 0.75  # per point: through / (through + same)
CLUSTER_MIN_RATIO = 0.95  # per cluster: mean of that ratio (ghost points from depth noise score lower)
MAX_CHANGED_FRAC = 0.10   # if more of the observed scene is flagged, widen tolerance
TAU_ESCALATION = [1.0, 1.3, 1.7, 2.2, 3.0]
MIN_CLUSTER_PTS = 15
MIN_SIZE_FRAC, MAX_SIZE_FRAC = 0.03, 0.50
TOP_K = 10                # at most this many strongest changes per direction
MOVE_COST_MAX = 0.8
MAX_POINTS = 400_000

def voxel_downsample(points, colors, voxel):
    key = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return points[first], colors[first]


def prepare_cams(cams):
    """Adds a per-camera 'edge' mask: depth discontinuities and image borders (unreliable evidence)."""
    for cam in cams:
        D = cam["D"]
        h, w = D.shape
        gy, gx = np.gradient(D)
        rel = np.hypot(gx, gy) / np.maximum(D, 1e-6)
        edge = cv2.dilate((rel > EDGE_REL).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mh, mw = max(2, int(h * BORDER_FRAC)), max(2, int(w * BORDER_FRAC))
        edge[:mh], edge[-mh:], edge[:, :mw], edge[:, -mw:] = True, True, True, True
        cam["edge"] = edge
        # Neighborhood-MIN depth: "through" requires even the NEAREST nearby surface to be behind the point.
        # This stops silhouette points of static objects from being mistaken for "seen through".
        k = max(5, int(round(0.015 * w)) | 1)
        cam["Dmin"] = cv2.erode(D, np.ones((k, k), np.uint8))
    return cams


def evidence(points, cams, tau_base, tau_rel):
    n = len(points)
    same, through, occluded = (np.zeros(n, np.int32) for _ in range(3))
    for cam in cams:
        u, v, z, ok = project(points, cam)
        ok &= ~cam["edge"][v, u]
        d = np.where(ok, cam["D"][v, u], 0.0)
        valid = ok & (d > 0)
        tau = tau_base + tau_rel * z
        diff = d - z
        diff_near = np.where(ok, cam["Dmin"][v, u], 0.0) - z
        same += valid & (np.abs(diff) < tau)
        through += valid & (diff_near > tau) & (diff > tau)
        occluded += valid & (diff < -tau)
    return same, through, occluded


def calibrate_tau_rel(points, cams, seed=0):
    """Robust relative depth disagreement between this scene's points and another video's depth maps."""
    rng = np.random.default_rng(seed)
    p = points[rng.choice(len(points), min(20000, len(points)), replace=False)]
    chosen = rng.choice(len(cams), min(20, len(cams)), replace=False)
    rel = []
    for i in chosen:
        u, v, z, ok = project(p, cams[i])
        ok &= ~cams[i]["edge"][v, u]
        r = (cams[i]["D"][v, u] - z) / np.maximum(z, 1e-6)
        keep = ok & (np.abs(r) < 0.2)#static surfaces only; real changes are far larger
        rel.append(r[keep])
    rel = np.concatenate(rel) if rel else np.zeros(1)
    bias = float(np.median(rel))
    sigma = float(1.4826 * np.median(np.abs(rel - bias)))
    tau_rel = float(np.clip(TAU_SIGMA_MULT * sigma, TAU_REL_MIN, TAU_REL_MAX))
    return tau_rel, sigma, bias


def one_direction(pts, own_cams, other_cams, tau_base, tau_rel):
    same_self, _, _ = evidence(pts, own_cams, tau_base, tau_rel)
    same, through, occ = evidence(pts, other_cams, tau_base, tau_rel)
    solid = same_self >= MIN_SEEN_SELF
    observed = solid & ((same + through + occ) > 0)
    ratio = through / np.maximum(through + same, 1)
    changed = solid & (through >= MIN_VOTES) & (ratio >= MIN_THROUGH_RATIO)
    return changed, ratio, observed, solid, through


def detect_pair(ptsA, ptsB, camsA, camsB, tau_base, tau_rel0):
    """Runs both directions, widening the tolerance until the flagged fraction is plausible."""
    for mult in TAU_ESCALATION:
        tau_rel = tau_rel0 * mult
        chA, rA, obsA, solA, thA = one_direction(ptsA, camsA, camsB, tau_base, tau_rel)
        chB, rB, obsB, solB, thB = one_direction(ptsB, camsB, camsA, tau_base, tau_rel)
        fA = chA.sum() / max(obsA.sum(), 1)
        fB = chB.sum() / max(obsB.sum(), 1)
        console.print(f"  tol x{mult:.1f} (tau_rel={tau_rel:.3f}): flagged A→B {fA:.1%}, B→A {fB:.1%} "
                      f"of observed solid points")
        if max(fA, fB) <= MAX_CHANGED_FRAC:
            break
    else:
        console.print("[yellow]  Still many flagged points at max tolerance: alignment/scale between the two "
                      "videos is likely poor (check the overlay panel in Rerun).[/yellow]")
    console.print(f"  observed solid points: A {obsA.sum():,}/{len(ptsA):,}  B {obsB.sum():,}/{len(ptsB):,}")
    return (chA, rA, thA), (chB, rB, thB), tau_rel


def cluster_changes(points, colors, ratio, voxel, diag, votes):
    if len(points) < MIN_CLUSTER_PTS:
        return []
    pairs = np.array(list(cKDTree(points).query_pairs(2.5 * voxel)))
    n = len(points)
    if len(pairs) == 0:
        return []
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    clusters = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < MIN_CLUSTER_PTS:
            continue
        p = points[idx]
        ext = p.max(0) - p.min(0)
        if ext.max() < MIN_SIZE_FRAC * diag or ext.max() > MAX_SIZE_FRAC * diag:
            continue                                
        if ratio[idx].mean() < CLUSTER_MIN_RATIO:
            continue                              
        clusters.append(dict(points=p, center=(p.max(0) + p.min(0)) / 2, extent=ext,
                             color=colors[idx].mean(0), score=float(len(idx) * ratio[idx].mean()),
                             votes=float(np.median(votes[idx]))))
    clusters.sort(key=lambda c: -c["score"])
    return clusters[:TOP_K]


def match_moved(rem, add):
    if not rem or not add:
        return []
    cost = np.zeros((len(rem), len(add)))
    for i, r in enumerate(rem):
        for j, a in enumerate(add):
            size = np.mean(np.abs(np.log((np.sort(r["extent"]) + 1e-9) / (np.sort(a["extent"]) + 1e-9))))
            color = np.linalg.norm(r["color"] - a["color"])
            sem = 0.0 if r["label"] == a["label"] else 0.4
            cost[i, j] = size + color + sem
    ri, ci = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(ri, ci) if cost[i, j] < MOVE_COST_MAX]


def dims_cm(extent, upm):
    d = np.sort(extent)[::-1] / upm * 100
    return f"{d[0]:.0f}×{d[1]:.0f}×{d[2]:.0f} cm"


def load_cloud(path, voxel):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    pts, cols = voxel_downsample(pts, cols, voxel)
    if len(pts) > MAX_POINTS:
        sel = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[sel], cols[sel]
    return pts, cols, np.asarray(pcd.get_center())

def main():
    with open(OUT / "cameras.pkl", "rb") as f:
        meta = pickle.load(f)
    diag, upm = meta["diag"], meta["units_per_meter"]
    voxel, tau_base = VOXEL_FRAC * diag, TAU_BASE_FRAC * diag

    ptsA, colA, center = load_cloud(OUT / "scene_A.ply", voxel)
    ptsB, colB, _ = load_cloud(OUT / "scene_B.ply", voxel)
    camsA = prepare_cams(load_cams(meta["cams"], "A"))
    camsB = prepare_cams(load_cams(meta["cams"], "B"))

    console.print("[bold cyan]─── Calibrating depth tolerance from static surfaces ───[/bold cyan]")
    tA, sA, bA = calibrate_tau_rel(ptsA, camsB)
    tB, sB, bB = calibrate_tau_rel(ptsB, camsA)
    tau_rel0 = max(tA, tB)
    console.print(f"  A-vs-B depth noise σ≈{max(sA, sB):.3f} (relative), bias {bA:+.3f}/{bB:+.3f}, "
                  f"base tolerance {tau_rel0:.3f}")
    if max(sA, sB) > 0.08:
        console.print("[yellow]  Depth scale is inconsistent between frames/videos (high σ): expect missed or "
                      "noisy detections. Check the Alignment panel; re-record with slower, steadier motion.[/yellow]")
    if max(abs(bA), abs(bB)) > 0.08:
        console.print("[yellow]  Large systematic depth offset between videos: scene scale/alignment is off.[/yellow]")

    console.print("[bold cyan]─── Occlusion-aware diff ───[/bold cyan]")
    (chA, rA, thA), (chB, rB, thB), tau_rel = detect_pair(ptsA, ptsB, camsA, camsB, tau_base, tau_rel0)

    rem = cluster_changes(ptsA[chA], colA[chA], rA[chA], voxel, diag, thA[chA])
    add = cluster_changes(ptsB[chB], colB[chB], rB[chB], voxel, diag, thB[chB])
    console.print(f"  flagged points -> clusters kept: removed {chA.sum():,}→{len(rem)}, added {chB.sum():,}→{len(add)}")

    console.print("[bold cyan]─── Semantic labeling ───[/bold cyan]")
    for i, c in enumerate(rem):
        c["label"] = label_cluster(c["points"], camsA, tau_base, tau_rel, f"removed_{i}")
    for j, c in enumerate(add):
        c["label"] = label_cluster(c["points"], camsB, tau_base, tau_rel, f"added_{j}")

    pairs = match_moved(rem, add)
    moved_r, moved_a = {i for i, _ in pairs}, {j for _, j in pairs}

    records, arrays = [], {}

    def box(c, scene, key):
        arrays[key] = c["points"].astype(np.float32)
        return dict(scene=scene, center=c["center"].tolist(), extent=c["extent"].tolist(), points_key=key)

    def where(c):
        r = (c["center"] - center) / upm
        return f"({r[0]:+.2f}, {r[1]:+.2f}, {r[2]:+.2f}) m from scene center"

    for i, j in pairs:
        r, a = rem[i], add[j]
        dist_cm = np.linalg.norm(a["center"] - r["center"]) / upm * 100
        records.append(dict(
            type="moved", label=a["label"],
            description=f"{a['label'].capitalize()} (~{dims_cm(a['extent'], upm)}) was moved ~{dist_cm:.0f} cm, "
                        f"from {where(r)} to {where(a)}.",
            boxes=[box(r, "A", f"rem{i}"), box(a, "B", f"add{j}")]))
    for i, c in enumerate(rem):
        if i not in moved_r:
            records.append(dict(
                type="removed", label=c["label"],
                description=f"{c['label'].capitalize()} (~{dims_cm(c['extent'], upm)}) was removed from {where(c)}.",
                boxes=[box(c, "A", f"rem{i}")]))
    for j, c in enumerate(add):
        if j not in moved_a:
            records.append(dict(
                type="added", label=c["label"],
                description=f"{c['label'].capitalize()} (~{dims_cm(c['extent'], upm)}) was added at {where(c)}.",
                boxes=[box(c, "B", f"add{j}")]))

    with open(OUT / "changes.json", "w") as f:
        json.dump(records, f, indent=2)
    np.savez_compressed(OUT / "changes.npz", **arrays)

    console.print("\n[bold cyan]─── Detected changes ───[/bold cyan]")
    for r in records:
        color = {"removed": "red", "added": "green", "moved": "yellow"}[r["type"]]
        console.print(f"[{color}]• {r['type'].upper()}[/{color}] {r['description']}")
    if not records:
        console.print("[yellow]No changes found. If you know there are some, check the Alignment panel in Rerun first.[/yellow]")


if __name__ == "__main__":
    main()
