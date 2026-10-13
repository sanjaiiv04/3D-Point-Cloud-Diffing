import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch/open3d/pycolmap each bundle libomp on macOS
import os
import base64
from pathlib import Path

import cv2
import numpy as np

from geometry import project

CROP_DIR = Path(__file__).resolve().parent.parent / "output" / "crops"
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
PROMPT = (
    "This crop shows an object in a room that was added, removed, or moved. "
    "Name the main object in 1-3 words, lowercase, no punctuation."
)


def best_view(points, cams, tau_base, tau_rel):
    """Camera where most cluster points are actually visible (depth-consistent, not occluded)."""
    best = None
    for cam in cams:
        u, v, z, ok = project(points, cam)
        vis = ok & (np.abs(cam["D"][v, u] - z) < tau_base + tau_rel * z)
        n = int(vis.sum())
        if best is None or n > best[0]:
            best = (n, cam, u[vis], v[vis])
    return best


def crop_cluster(points, cams, tau_base, tau_rel, tag):
    best = best_view(points, cams, tau_base, tau_rel)
    if best is None or best[0] < 10:
        return None
    _, cam, u, v = best
    img = cv2.imread(cam["image_path"])
    h, w = img.shape[:2]
    x0, x1 = np.percentile(u, [2, 98])
    y0, y1 = np.percentile(v, [2, 98])
    pad = 0.25 * max(x1 - x0, y1 - y0, 48)
    x0, x1 = int(max(0, x0 - pad)), int(min(w, x1 + pad))
    y0, y1 = int(max(0, y0 - pad)), int(min(h, y1 + pad))
    crop = img[y0:y1, x0:x1]
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CROP_DIR / f"{tag}.jpg"), crop)
    return crop


def ask_claude(crop):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "object"
    try:
        import anthropic
        ok, buf = cv2.imencode(".jpg", crop)
        data = base64.standard_b64encode(buf.tobytes()).decode()
        resp = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=20,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": PROMPT},
            ]}],
        )
        return resp.content[0].text.strip().lower().strip(".")
    except Exception as e:
        print(f"[!] Labeling failed ({e}); using generic label")
        return "object"


def label_cluster(points, cams, tau_base, tau_rel, tag):
    crop = crop_cluster(points, cams, tau_base, tau_rel, tag)
    return "object" if crop is None else ask_claude(crop)
