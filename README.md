# What Changed in the Room?

Compares two phone recordings of the same indoor area and reports what was **added, removed, or moved**, in 3D, with a semantic label and a plain-language description for each change. Everything runs locally (no API keys). However, if this is to be extended, we can add **Anthropic**'s API key to allow for understanding 'what' changed in the room!!

<!-- TODO: paste your Rerun screenshot here, e.g. ![result](docs/result.png) -->
<img width="1280" height="832" alt="Screenshot 2026-10-12 at 10 32 02 PM" src="https://github.com/user-attachments/assets/6feddda7-2902-4c3c-8263-26aeb3fcfe23" />

## How it works

1. **Joint reconstruction.** Frames from *both* videos go into a single COLMAP (pycolmap) structure-from-motion run, so all camera poses land in one shared world frame. No point-cloud registration afterwards.
2. **Metric depth.** Depth Anything V2 (metric indoor) predicts depth per frame, scaled per frame to the COLMAP geometry using its sparse points.
3. **Per-video fusion.** Each video's depth maps are fused into its own point cloud (`scene_A.ply`, `scene_B.ply`), already aligned with each other.
4. **Occlusion-aware diff.** Each point of one scene is projected into the *other* video's cameras. If that video saw a surface behind the point, the point is empty there (**changed**). If something is in front of it, or no camera saw it, it is **ignored** (occlusion / unknown), so missing observations are not reported as changes.
5. **Clustering and matching.** Changed points are clustered into objects. A removed object and an added object of similar size, color and label are paired as **moved**.
6. **Semantics.** The best camera view of each object is cropped and labeled with CLIP (zero-shot, local), then turned into a description such as *"Bottle (~12×6×6 cm) was removed from …"*.
7. **Visualization.** A Rerun recording with Scene A, Scene B, an alignment-check overlay, 3D boxes for each change (red removed, green added, yellow moved) and a text report.

## Setup

Requires Python 3.12.12. Tested on Apple Silicon (MPS); CPU also works but is slower. 
Install _uv_ for downloading dependencies much faster than pip/conda. 

```bash
brew install uv
```
After installing _uv_, make sure to create a virtual environment. You can create a ._venv_ with _uv_ in different ways:

```bash
uv sync
```
or
```bash
uv venv --python 3.12.12 .venv
```
Once you create the virtual environment, make sure to activate it:

```bash
source .venv/bin/activate
```
Once you are inside the virtual environment, it is time to install the necessary libraries. To make it easier for you, I have added a _**requirements.txt**_ that you can simply load to the venv:
```bash
uv pip install -r requirements.txt
```
This should install all necessary libraries that you would need to run scripts. 
## Run

1. Put your two recordings at `data/video_A.mp4` (before) and `data/video_B.mp4` (after).
2. Run the whole pipeline:

```bash
uv run code/run.py
```

This runs model download → reconstruction → change detection → visualization, then opens Rerun. To reopen the result later:

```bash
uv run python -m rerun output/scene_changes.rrd
```

Outputs are written to `output/`:

| File | Content |
|---|---|
| `scene_A.ply`, `scene_B.ply` | Reconstructed point clouds in one shared frame |
| `changes.json` | Detected changes: type, label, description, 3D box |
| `scene_changes.rrd` | Rerun recording (open this to explore in 3D) |
| `crops/` | The image crops used for labeling |

## Capture tips

- Record the same area twice with a **different path** (different viewpoints) but **plenty of shared background** (walls, floor, large furniture visible in both).
- Move slowly, avoid pure rotation in place, keep lighting similar, avoid motion blur.
- Change only a few objects between recordings.
- If the two videos do not merge into one model, reconstruction stops with an error. Re-record with more overlap.

## Project layout

```
code/
  run.py            # single entry point
  load_model.py     # downloads Depth Anything V2 (metric indoor)
  reconstruct.py    # frames -> joint SfM -> metric depth -> fused point clouds
  detect.py         # occlusion-aware diff, clustering, moved matching, descriptions
  label.py          # local CLIP object labeling
  geometry.py       # shared camera projection helpers
  visualize.py      # Rerun recording
data/               # input videos 
output/             # results: extracted frames, depth maps for each frame, point cloud after reconstruction and final output to be rendered in Rerun.
```

## Design choices and tradeoffs
- **Switching from Dust3R to Depth-Anything**: At first, I figured out this popular reconstruction model called Dust3R that worked extremely well in reconstruction. However, due to its enormous size of reconstruction, I had difficulties handling the data, often leading to a terminal crash or out of memory.
- **Joint SfM instead of aligning two clouds.** Monocular depth has an arbitrary scale per video, which makes ICP-style alignment unreliable. Reconstructing both videos together puts them in one frame by construction.
- **Free-space reasoning instead of nearest-neighbor distance.** A plain distance test calls anything the second video did not observe "removed". Checking what the other cameras actually saw lets the system tell *changed* from *occluded* or *not observed*.
- **No hand-tuned meter thresholds.** The depth tolerance is calibrated from static surfaces in the data, and size limits are fractions of the scene size, so settings carry over between scenes.
- **Local models only.** Depth Anything V2 and CLIP run on-device, so the project needs no keys or network after the first model download.
- **Tradeoff:** monocular depth is noisy, so very small, thin or flat objects (a sheet of paper) can be missed, and sizes in centimeters are approximate.

## Limitations

- Needs enough texture and overlap for COLMAP to merge both videos.
- Labels come from a fixed list in `label.py` (edit `LABELS` for your scene). Low-confidence objects are reported as "object".
- Moved-object matching relies on similar size, color and label, so a heavily rotated or re-colored object may show up as one removal plus one addition.
- Objects not observed in the other video are never reported as changed.

## Remarks
Overall, I had a fun few days trying to build this thing from ground up and had sleepless nights trying to figure out why my detection was not absolutely perfect. However, I came to the conclusion that trade-offs are there for a reason and with the level of restrictions I worked with, I am extremely proud of what I have built. 
I hope the team finds the code clean and able to run it in a single-go ( I built with the team in mind!!!).
