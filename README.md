# Virtual Try-On — Campus Project

360°-style virtual clothing try-on using real-time body pose detection and multi-angle 2D garment overlay.

## Current Milestone

**Milestone 1: QR + Mobile UI + Camera + Pose Detection** (in progress)

## Project Structure

```
Virtual-Try-On/
    frontend/     # Mobile web interface (QR scan, garment selection)
    backend/      # Flask/FastAPI server, pose detection, garment logic
    models/       # ML models (pose, segmentation) - not committed, downloaded via setup
    garments/     # Garment image assets (multi-angle PNGs + anchor point JSONs)
    dataset/      # Training/reference datasets (VITON-HD, DressCode) - not committed
    docs/         # SRS, UML diagrams, reports
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

2. Run the pose detection smoke test to confirm your webcam + MediaPipe are working:
   ```bash
   python backend/pose_test.py
   ```
   Press `q` to quit. You should see 33 body keypoints overlaid on your webcam feed.

3. Set up the dataset (see `dataset/README.md` for full instructions):
   - **VITON-HD** — no approval needed, download directly.
   - **DressCode** — requires an institutional-email request form; apply early, approval can take up to a week.

## Team Roles (reference)

| Member | Responsibility |
|---|---|
| Member 1 | Mobile UI, QR Module, Flask/API |
| Member 2 | Pose Detection & Body Segmentation |
| Member 3 | Cloth Warping, Garment Mapping, Rendering |
| Member 4 | Recommendation System, Dataset, Testing, Documentation |

## Roadmap

See `docs/roadmap.md` for the full 15-phase plan and milestone breakdown.
