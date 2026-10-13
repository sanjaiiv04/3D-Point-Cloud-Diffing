import json
from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb

HERE = Path(__file__).parent.resolve()
OUT = HERE.parent / "output"
FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float32)   # COLMAP is y-down and Rerun y-up. So we flip it. 
COLORS = {"removed": [255, 0, 0], "added": [0, 255, 0], "moved": [255, 210, 0]}


def load_display(path, voxel):
    pcd = o3d.io.read_point_cloud(str(path))
    diag = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    pcd = pcd.voxel_down_sample(voxel * diag)
    return np.asarray(pcd.points, np.float32) * FLIP, np.asarray(pcd.colors, np.float32), diag


def build_recording(rrd_path=OUT / "scene_changes.rrd"):
    pts_A, col_A, diag = load_display(OUT / "scene_A.ply", 0.004)
    pts_B, col_B, _ = load_display(OUT / "scene_B.ply", 0.004)
    radius = 0.002 * diag

    rr.init("3D_Room_Change_Detection")
    rr.save(str(rrd_path))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(name="Scene A (Before)", origin="scene_A", contents=["scene_A/**"]),
        rrb.Spatial3DView(name="Scene B (After)", origin="scene_B", contents=["scene_B/**"]),
        rrb.Vertical(
            rrb.Spatial3DView(name="Alignment check (A=cyan, B=magenta)", origin="overlay", contents=["overlay/**"]),
            rrb.TextDocumentView(name="Detected changes", origin="report"),
        ),
    )))

    rr.log("scene_A/points", rr.Points3D(pts_A, colors=col_A, radii=radius))
    rr.log("scene_B/points", rr.Points3D(pts_B, colors=col_B, radii=radius))
    rr.log("overlay/A", rr.Points3D(pts_A, colors=[0, 204, 255], radii=radius))
    rr.log("overlay/B", rr.Points3D(pts_B, colors=[255, 0, 204], radii=radius))

    changes = json.load(open(OUT / "changes.json"))
    arrays = np.load(OUT / "changes.npz")
    lines = ["# Detected changes\n"]

    for n, ch in enumerate(changes):
        color = COLORS[ch["type"]]
        lines.append(f"- **{ch['type'].upper()}**: {ch['description']}")
        for b in ch["boxes"]:
            root = f"scene_{b['scene']}"
            pts = arrays[b["points_key"]] * FLIP
            rr.log(f"{root}/changes/{n}_pts", rr.Points3D(pts, colors=color, radii=radius * 2))
            rr.log(f"{root}/changes/{n}_box", rr.Boxes3D(
                centers=[np.asarray(b["center"], np.float32) * FLIP],
                half_sizes=[np.asarray(b["extent"], np.float32) / 2],
                colors=[color],
                labels=[f"{ch['type']}: {ch['label']}"]))

    if len(lines) == 1:
        lines.append("_No changes detected._")
    rr.log("report", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
    print(f"[+] Saved recording to '{rrd_path}'")


if __name__ == "__main__":
    build_recording()
