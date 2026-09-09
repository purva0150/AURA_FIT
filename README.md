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
   source venv/bin/activate   # Windows PowerShell: .\venv\Scripts\Activate.ps1
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

## Cloth-wrapping test

Prepare a front-facing catalogue image (the tool removes a border-connected
background without erasing white fabric):

```bash
python backend/tools/prepare_garment.py "path/to/shirt.png" \
  --id my_shirt --name "My shirt"
```

Test it on one image before opening the webcam:

```bash
python backend/render_tryon_image.py "path/to/person.png" \
  --garment my_shirt --output tryon_preview.png
```

Then run the live mirror:

```bash
python backend/tryon_live.py --garment my_shirt
```

## Complete desktop + phone experience

Start the paired fitting room from the repository root:

```bash
python backend/app.py
```

Then open `http://127.0.0.1:5000` on the laptop. The laptop immediately starts
the webcam and body skeleton. Scan the on-screen QR code with a phone connected
to the same Wi-Fi network. The phone shows the processed live preview and
controls the garment, fitted/regular/relaxed sizing, skeleton, and
Auto/Front/Back viewing mode. Both screens share one camera-processing loop.

For Back mode, turn your back toward the laptop camera. A real back photograph
and anchor JSON gives the best result. The included white T-shirt uses its
plain front image as an estimated back fallback because no separate back photo
was supplied; replace that asset for an accurate rear neckline and print.

On Windows, allow Python through the Private network when the Firewall prompt
appears. If the computer has more than one camera, select it before launch:

```powershell
$env:VTO_CAMERA="1"
python backend\app.py
```

The pairing token is regenerated whenever the server restarts, so an old QR
link cannot control a later session.

## Team Roles (reference)

| Member | Responsibility |
|---|---|
| Member 1 | Mobile UI, QR Module, Flask/API |
| Member 2 | Pose Detection & Body Segmentation |
| Member 3 | Cloth Warping, Garment Mapping, Rendering |
| Member 4 | Recommendation System, Dataset, Testing, Documentation |

## Roadmap

See `docs/roadmap.md` for the full 15-phase plan and milestone breakdown.
