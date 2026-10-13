import sys
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import open3d as o3d
import pycolmap
from PIL import Image
from rich.console import Console
from rich.progress import track

from load_model import download_and_cache_models, DEVICE

console = Console()

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

# Define all paths relative to the project root directory
DATA = ROOT / "data"
OUT = ROOT / "output"

FRAMES, DEPTH, COLMAP = DATA / "frames_all", DATA / "depth", DATA / "colmap"
MAX_SIDE = 960
FRAMES_PER_VIDEO = 80

def extract_keyframes(video_path, prefix, out_dir=FRAMES, max_frames=FRAMES_PER_VIDEO):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Unable to read video: {video_path}")

    wanted = set(np.linspace(0, total - 1, min(max_frames, total), dtype=int).tolist())
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            h, w = frame.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1:
                frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"{prefix}_{saved:03d}.png"), frame)
            saved += 1
        idx += 1
    cap.release()
    console.print(f"[green]✔[/green] {prefix}: extracted {saved} frames")


def run_sfm(image_dir=FRAMES, work=COLMAP):
    work.mkdir(parents=True, exist_ok=True)
    (work / "sparse").mkdir(exist_ok=True)
    db = work / "db.db"

    with console.status("[yellow]COLMAP: extracting features (SIFT)..."):
        # Assuming the videos are frame the same source, so the intrinsics should be same. 
        pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE)
    with console.status("[yellow]COLMAP: matching all frame pairs..."):
        pycolmap.match_exhaustive(db)
    with console.status("[yellow]COLMAP: incremental mapping..."):
        maps = pycolmap.incremental_mapping(db, image_dir, work / "sparse")

    if not maps:
        raise RuntimeError("COLMAP produced no model. Record with more texture/overlap.")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())

    names = [im.name for im in rec.images.values()]
    nA = sum(n.startswith("A_") for n in names)
    nB = sum(n.startswith("B_") for n in names)
    console.print(f"[bold]Registered frames -> A: {nA}, B: {nB}[/bold]")
    if nA < 10 or nB < 10:
        raise RuntimeError(
            "The two videos did NOT merge into one model. Re-record with more shared background "
            "and overlap (see README capture tips)."
        )
    return rec


def cam_from_world_4x4(image):
    cfw = image.cam_from_world
    if callable(cfw):          # pycolmap API differs between versions
        cfw = cfw()
    T = np.eye(4)
    T[:3, :4] = cfw.matrix()
    return T


def scene_diag(rec):
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    return float(np.linalg.norm(hi - lo))

def fit_scale(pred, rec, image, T):
    """COLMAP units per meter for this frame, from sparse points (robust median)."""
    pts = [(p.xy, p.point3D_id) for p in image.points2D if p.has_point3D()]
    if len(pts) < 30:
        return None
    xy = np.array([p[0] for p in pts])
    X = np.array([rec.points3D[p[1]].xyz for p in pts])
    z = (T[:3, :3] @ X.T + T[:3, 3:4])[2]
    u, v = xy[:, 0].astype(int), xy[:, 1].astype(int)
    h, w = pred.shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    d = pred[v[ok], u[ok]]
    good = d > 1e-3
    ratios = z[ok][good] / d[good]
    if len(ratios) < 30:
        return None
    return float(np.median(ratios))


def generate_depth(rec, processor, model):
    DEPTH.mkdir(parents=True, exist_ok=True)
    model = model.to(DEVICE).eval()
    cams, scales = [], []

    for image in track(list(rec.images.values()), description="Depth + scale fit"):
        img_path = FRAMES / image.name
        pil = Image.open(img_path).convert("RGB")      # RGB (cv2 would give BGR)
        W, H = pil.size

        inputs = processor(images=pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            pred = model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

        T = cam_from_world_4x4(image)
        s = fit_scale(pred, rec, image, T)
        if s is None:
            continue                             
        scales.append(s)

        depth_path = DEPTH / (Path(image.name).stem + ".npy")
        np.save(depth_path, (pred * s).astype(np.float16))

        cams.append(dict(
            name=image.name, video=image.name[0],
            K=rec.cameras[image.camera_id].calibration_matrix(), T=T,
            depth_path=str(depth_path), image_path=str(img_path),
        ))

    units_per_meter = float(np.median(scales))
    console.print(f"[green]✔[/green] {len(cams)} depth maps | ~{units_per_meter:.3f} COLMAP units per meter")
    return cams, units_per_meter

def fuse_video(cams, diag, stride=1):  #Changed stride from 3 to 1 for 9x higher point density
    pts_all, col_all = [], []
    for c in track(cams, description="Fusing depth"):
        D = np.load(c["depth_path"]).astype(np.float32)
        img = cv2.cvtColor(cv2.imread(c["image_path"]), cv2.COLOR_BGR2RGB)
        h, w = D.shape

        gy, gx = np.gradient(D)
        rel = np.hypot(gx, gy) / np.maximum(D, 1e-6)

        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        z = D[vs, us]
        ok = (z > 0) & (z < 0.8 * diag) & (rel[vs, us] < 0.08) # Slightly relaxed edge noise filter
        u, v, z = us[ok].astype(np.float32), vs[ok].astype(np.float32), z[ok]

        K = c["K"]
        Xc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
        R, t = c["T"][:3, :3], c["T"][:3, 3]
        pts_all.append((Xc - t) @ R)
        col_all.append(img[vs, us][ok] / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts_all).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(col_all).astype(np.float64))
    
    # 0.0015 * diag keeps high-definition surface details
    pcd = pcd.voxel_down_sample(0.0015 * diag)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    return pcd


if __name__ == "__main__":

    video_A = DATA / "video_A.mp4"
    video_B = DATA / "video_B.mp4"

    if not video_A.exists() or not video_B.exists():
        console.print(f"[red]Missing input videos:[/red]\n - {video_A}\n - {video_B}")
        sys.exit(1)

    for d in (FRAMES, DEPTH, COLMAP):
        shutil.rmtree(d, ignore_errors=True)
    OUT.mkdir(exist_ok=True)

    console.print(f"[bold cyan]─── Reconstruction (device: {DEVICE}) ───[/bold cyan]")
    extract_keyframes(video_A, "A")
    extract_keyframes(video_B, "B")

    rec = run_sfm()
    diag = scene_diag(rec)

    processor, model = download_and_cache_models()
    cams, units_per_meter = generate_depth(rec, processor, model)

    for v in ("A", "B"):
        pcd = fuse_video([c for c in cams if c["video"] == v], diag)
        o3d.io.write_point_cloud(str(OUT / f"scene_{v}.ply"), pcd)
        console.print(f"[green]✔[/green] scene_{v}.ply: {len(pcd.points):,} points")

    with open(OUT / "cameras.pkl", "wb") as f:
        pickle.dump(dict(cams=cams, diag=diag, units_per_meter=units_per_meter), f)
    console.print("[bold green]✔ Reconstruction complete (both scenes share one coordinate frame).[/bold green]")
