"""Paired desktop/mobile virtual try-on. Run: python backend/app.py"""
from __future__ import annotations

import io, os, secrets, socket, sys, threading, time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
import cv2, numpy as np, qrcode
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
GARMENTS_DIR = REPO_ROOT / "garments"
sys.path.insert(0, str(BACKEND_DIR))
from tryon.garment import GarmentSet, discover_garments  # noqa: E402
from tryon.pose import PoseEstimator, draw_skeleton  # noqa: E402
from tryon_live import LiveTryOn  # noqa: E402

app = Flask(__name__, template_folder=str(FRONTEND_DIR / "templates"), static_folder=str(FRONTEND_DIR / "static"))
SESSION_TOKEN = secrets.token_urlsafe(18)
STATE_LOCK = threading.RLock()
STATE = {"selected_garment": None, "phone_name": None, "last_phone_seen": 0.0,
         "show_skeleton": True, "view_mode": "auto", "fit_mode": "regular"}
FIT_SCALES = {"fitted": 0.94, "regular": 1.0, "relaxed": 1.06}

def catalogue() -> list[GarmentSet]:
    return discover_garments(str(GARMENTS_DIR))

def catalogue_json() -> list[dict]:
    return [{"id": g.garment_id, "name": g.name, "color": g.dominant_color,
             "thumbnail": f"/api/garments/{g.garment_id}/thumbnail"} for g in catalogue()]

def local_ip() -> str:
    if os.environ.get("VTO_PUBLIC_HOST"): return os.environ["VTO_PUBLIC_HOST"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80)); return sock.getsockname()[0]
    except OSError:
        try: return socket.gethostbyname(socket.gethostname())
        except OSError: return "127.0.0.1"
    finally: sock.close()

def valid_token(value: Optional[str]) -> bool:
    return bool(value) and secrets.compare_digest(value, SESSION_TOKEN)

def state_json() -> dict:
    with STATE_LOCK:
        connected = time.time() - float(STATE["last_phone_seen"]) < 8.0
        return {"connected": connected, "phone_name": STATE["phone_name"] if connected else None,
                "selected_garment": STATE["selected_garment"], "show_skeleton": STATE["show_skeleton"],
                "view_mode": STATE["view_mode"], "fit_mode": STATE["fit_mode"]}

class CameraEngine:
    def __init__(self) -> None:
        self.garments = catalogue()
        self.tryon = LiveTryOn(self.garments, use_tps=True) if self.garments else None
        self.condition = threading.Condition()
        self.latest_jpeg: Optional[bytes] = None
        self.sequence = 0
        self.started = False

    def select(self, garment_id: Optional[str]) -> bool:
        if not garment_id or self.tryon is None: return False
        for index, garment in enumerate(self.garments):
            if garment.garment_id == garment_id:
                if self.tryon.index != index:
                    self.tryon.index = index; self.tryon.smoother.reset()
                return True
        return False

    def start(self) -> None:
        with self.condition:
            if self.started: return
            self.started = True
            threading.Thread(target=self._capture_loop, name="vto-camera", daemon=True).start()

    def _publish(self, frame: "cv2.typing.MatLike") -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok: return
        with self.condition:
            self.latest_jpeg = encoded.tobytes(); self.sequence += 1; self.condition.notify_all()

    def _capture_loop(self) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(os.environ.get("VTO_CAMERA", "0")), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            error = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(error, "Camera unavailable", (420, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240,240,240), 2)
            self._publish(error); return
        try:
            with PoseEstimator(segmentation=True, model_complexity=1) as estimator:
                while True:
                    ok, frame = cap.read()
                    if not ok: time.sleep(.02); continue
                    frame = cv2.flip(frame, 1); pose = estimator.process(frame)
                    with STATE_LOCK:
                        garment_id, skeleton = STATE["selected_garment"], bool(STATE["show_skeleton"])
                        view_mode, fit_mode = STATE["view_mode"], STATE["fit_mode"]
                    has_garment = self.select(garment_id)
                    output = self.tryon.process(frame, pose, view_mode=view_mode,
                                                fit_scale=FIT_SCALES[fit_mode]) if pose is not None and has_garment and self.tryon else frame.copy()
                    if pose is not None and (skeleton or not has_garment): draw_skeleton(output, pose, color=(63, 238, 171))
                    self._publish(output)
        finally:
            cap.release()

    def frames(self):
        self.start(); seen = -1
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.sequence != seen, timeout=2.0)
                if self.latest_jpeg is None: continue
                seen, payload = self.sequence, self.latest_jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + payload + b"\r\n"

CAMERA = CameraEngine()

@app.get("/")
def desktop_home(): return render_template("desktop.html", token=SESSION_TOKEN, garments=catalogue_json())

@app.get("/mobile")
def mobile_home():
    token = request.args.get("session", "")
    if not valid_token(token): return render_template("invalid_session.html"), 403
    return render_template("mobile.html", token=token, garments=catalogue_json())

@app.get("/qr")
def qr_code():
    url = f"http://{local_ip()}:5000/mobile?{urlencode({'session': SESSION_TOKEN})}"
    qr = qrcode.QRCode(box_size=9, border=2); qr.add_data(url); qr.make(fit=True)
    image = qr.make_image(fill_color="#101114", back_color="#ffffff")
    buffer = io.BytesIO(); image.save(buffer, format="PNG"); buffer.seek(0)
    return send_file(buffer, mimetype="image/png", max_age=0)

@app.get("/video-feed")
def video_feed(): return Response(CAMERA.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/garments/<garment_id>/thumbnail")
def garment_thumbnail(garment_id: str):
    garment = next((g for g in catalogue() if g.garment_id == garment_id), None)
    if garment is None: abort(404)
    angle = garment.nearest(0)
    if not angle.image_path: abort(404)
    return send_file(angle.image_path, mimetype="image/png", max_age=3600)

@app.get("/api/state")
def get_state(): return jsonify(state_json())

@app.post("/api/connect")
def connect_phone():
    data = request.get_json(silent=True) or {}
    if not valid_token(data.get("session")): abort(403)
    with STATE_LOCK:
        STATE["phone_name"] = str(data.get("device") or "Mobile controller")[:50]; STATE["last_phone_seen"] = time.time()
    return jsonify({"ok": True, **state_json()})

@app.post("/api/select")
def select_garment():
    data = request.get_json(silent=True) or {}
    if not valid_token(data.get("session")): abort(403)
    garment_id = data.get("garment_id")
    if garment_id is not None and garment_id not in {g["id"] for g in catalogue_json()}:
        return jsonify({"ok": False, "error": "Unknown garment"}), 400
    with STATE_LOCK:
        STATE["selected_garment"] = garment_id
        if data.get("source") == "mobile": STATE["last_phone_seen"] = time.time()
    return jsonify({"ok": True, **state_json()})

@app.post("/api/skeleton")
def toggle_skeleton():
    data = request.get_json(silent=True) or {}
    if not valid_token(data.get("session")): abort(403)
    with STATE_LOCK:
        STATE["show_skeleton"] = bool(data.get("enabled"))
        if data.get("source") == "mobile": STATE["last_phone_seen"] = time.time()
    return jsonify({"ok": True, **state_json()})

@app.post("/api/view")
def select_view():
    data = request.get_json(silent=True) or {}
    if not valid_token(data.get("session")): abort(403)
    mode = str(data.get("mode", "auto"))
    if mode not in {"auto", "front", "back"}: return jsonify({"ok": False, "error": "Invalid view"}), 400
    with STATE_LOCK:
        STATE["view_mode"] = mode
        if data.get("source") == "mobile": STATE["last_phone_seen"] = time.time()
    return jsonify({"ok": True, **state_json()})

@app.post("/api/fit")
def select_fit():
    data = request.get_json(silent=True) or {}
    if not valid_token(data.get("session")): abort(403)
    mode = str(data.get("mode", "regular"))
    if mode not in FIT_SCALES: return jsonify({"ok": False, "error": "Invalid fit"}), 400
    with STATE_LOCK:
        STATE["fit_mode"] = mode
        if data.get("source") == "mobile": STATE["last_phone_seen"] = time.time()
    return jsonify({"ok": True, **state_json()})

if __name__ == "__main__":
    print("\nVirtual Try-On\nLaptop: http://127.0.0.1:5000")
    print(f"Phone network: http://{local_ip()}:5000\nKeep both devices on the same Wi-Fi.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
