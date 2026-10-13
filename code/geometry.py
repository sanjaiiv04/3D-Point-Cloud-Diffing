"""Shared camera helpers used by detect.py and label.py."""
import numpy as np


def load_cams(meta_cams, video):
    """Loads cameras (K, T_cam_from_world, depth map in COLMAP units) for the video."""
    cams = []
    for c in meta_cams:
        if c["video"] != video:
            continue
        cams.append(dict(
            name=c["name"], K=c["K"], T=c["T"],
            D=np.load(c["depth_path"]).astype(np.float32),
            image_path=c["image_path"],
        ))
    return cams


def project(points, cam):
    """Projects Nx3 world points into a camera. Returns u, v (int), z (depth), ok (inside image & in front)."""
    T, K, D = cam["T"], cam["K"], cam["D"]
    h, w = D.shape
    Xc = points @ T[:3, :3].T + T[:3, 3]
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = np.floor(K[0, 0] * Xc[:, 0] / zs + K[0, 2]).astype(int)
    v = np.floor(K[1, 1] * Xc[:, 1] / zs + K[1, 2]).astype(int)
    ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.clip(u, 0, w - 1), np.clip(v, 0, h - 1), z, ok
