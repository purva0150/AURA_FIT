"""AURA FIT — FastAPI backend for virtual try-on with QR-paired mobile remote."""
from __future__ import annotations

import io
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional
from urllib.parse import quote

import qrcode
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, BeforeValidator, Field

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
sessions = db["sessions"]

app = FastAPI(title="AURA FIT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve garment PNGs directly (transparent-bg for canvas overlay)
app.mount("/api/static", StaticFiles(directory=str(ROOT / "static_garments")), name="static_garments")


# ── Pydantic helpers ─────────────────────────────────────────────
def _oid_to_str(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)


PyObjectId = Annotated[str, BeforeValidator(_oid_to_str)]


class SessionState(BaseModel):
    id: PyObjectId = Field(alias="_id", default=None)
    token: str
    selected_garment: Optional[str] = None
    fit_mode: str = "regular"          # fitted | regular | relaxed
    view_mode: str = "auto"            # auto | front | back
    show_skeleton: bool = True
    phone_name: Optional[str] = None
    phone_connected: bool = False
    last_phone_seen: float = 0.0
    version: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionUpdate(BaseModel):
    selected_garment: Optional[str] = None
    fit_mode: Optional[str] = None
    view_mode: Optional[str] = None
    show_skeleton: Optional[bool] = None
    phone_name: Optional[str] = None
    source: Optional[str] = "phone"    # phone | mirror
    clear_garment: bool = False


# ── Garment catalog: real 3D GLB t-shirt with per-garment tint ─
# The frontend loads the GLB with Three.js and applies MediaPipe pose
# world-landmarks to translate / scale / rotate the mesh onto the body.
GARMENTS: list[dict[str, Any]] = [
    {"id": "white_tee",     "name": "AURA Oversized Heavy Tee", "category": "T-Shirts", "color": "Essential White", "tint": "#F2F0EA", "model": "/api/static/shirt.glb"},
    {"id": "midnight_tee",  "name": "Midnight Obsidian Tee",    "category": "T-Shirts", "color": "Obsidian Black", "tint": "#111318", "model": "/api/static/shirt.glb"},
    {"id": "acid_tee",      "name": "Acid Lime Statement Tee",  "category": "T-Shirts", "color": "Acid Lime",      "tint": "#E2F13B", "model": "/api/static/shirt.glb"},
    {"id": "cobalt_tee",    "name": "Cobalt Rush Tee",          "category": "T-Shirts", "color": "Cobalt Blue",    "tint": "#2A4CFF", "model": "/api/static/shirt.glb"},
    {"id": "terracotta_tee","name": "Terracotta Studio Tee",    "category": "T-Shirts", "color": "Terracotta",     "tint": "#C1613A", "model": "/api/static/shirt.glb"},
    {"id": "forest_tee",    "name": "Deep Forest Tee",          "category": "T-Shirts", "color": "Forest Green",   "tint": "#2E5A3A", "model": "/api/static/shirt.glb"},
    {"id": "rose_tee",      "name": "Blush Rose Tee",           "category": "T-Shirts", "color": "Blush Rose",     "tint": "#D97A8A", "model": "/api/static/shirt.glb"},
    {"id": "cream_tee",     "name": "Cream Editorial Tee",      "category": "T-Shirts", "color": "Warm Cream",     "tint": "#EFE4CE", "model": "/api/static/shirt.glb"},
]


# ── Helpers ─────────────────────────────────────────────────────
def _new_token() -> str:
    return secrets.token_urlsafe(9)


def _serialize(doc: dict) -> dict:
    if not doc:
        return doc
    d = dict(doc)
    d["_id"] = str(d.pop("_id"))
    d["phone_connected"] = (time.time() - float(d.get("last_phone_seen", 0))) < 10.0
    return d


def _origin_from_request(req: Request) -> str:
    # Prefer explicit APP_URL / origin so QR works across networks (not just LAN)
    env_url = os.environ.get("APP_URL")
    if env_url:
        return env_url.rstrip("/")
    host = req.headers.get("x-forwarded-host") or req.headers.get("host") or "localhost:3000"
    scheme = req.headers.get("x-forwarded-proto") or ("https" if "emergentagent" in host else "http")
    return f"{scheme}://{host}"


# ── Routes ──────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "service": "aura-fit"}


@app.get("/api/garments")
async def list_garments() -> dict:
    return {"garments": GARMENTS}


@app.post("/api/sessions")
async def create_session(request: Request) -> dict:
    token = _new_token()
    doc = SessionState(token=token).model_dump(by_alias=True, exclude={"id"})
    result = await sessions.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    origin = _origin_from_request(request)
    return {
        "session": _serialize(doc),
        "mobile_url": f"{origin}/remote?s={quote(token)}",
        "origin": origin,
    }


@app.get("/api/sessions/{token}")
async def get_session(token: str) -> dict:
    doc = await sessions.find_one({"token": token})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": _serialize(doc)}


@app.post("/api/sessions/{token}")
async def update_session(token: str, payload: SessionUpdate) -> dict:
    doc = await sessions.find_one({"token": token})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    updates: dict[str, Any] = {}
    if payload.clear_garment:
        updates["selected_garment"] = None
    elif payload.selected_garment is not None:
        if payload.selected_garment not in {g["id"] for g in GARMENTS}:
            raise HTTPException(status_code=400, detail="Unknown garment")
        updates["selected_garment"] = payload.selected_garment
    if payload.fit_mode is not None:
        if payload.fit_mode not in {"fitted", "regular", "relaxed"}:
            raise HTTPException(status_code=400, detail="Invalid fit")
        updates["fit_mode"] = payload.fit_mode
    if payload.view_mode is not None:
        if payload.view_mode not in {"auto", "front", "back"}:
            raise HTTPException(status_code=400, detail="Invalid view")
        updates["view_mode"] = payload.view_mode
    if payload.show_skeleton is not None:
        updates["show_skeleton"] = bool(payload.show_skeleton)
    if payload.source == "phone":
        updates["last_phone_seen"] = time.time()
        updates["phone_connected"] = True
        if payload.phone_name:
            updates["phone_name"] = payload.phone_name[:60]

    updates["version"] = int(doc.get("version", 0)) + 1

    await sessions.update_one({"_id": doc["_id"]}, {"$set": updates})
    doc = await sessions.find_one({"_id": doc["_id"]})
    return {"session": _serialize(doc)}


@app.post("/api/sessions/{token}/heartbeat")
async def heartbeat(token: str, payload: dict) -> dict:
    doc = await sessions.find_one({"token": token})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    updates = {"last_phone_seen": time.time(), "phone_connected": True}
    if payload.get("phone_name"):
        updates["phone_name"] = str(payload["phone_name"])[:60]
    await sessions.update_one({"_id": doc["_id"]}, {"$set": updates})
    doc = await sessions.find_one({"_id": doc["_id"]})
    return {"session": _serialize(doc)}


@app.get("/api/sessions/{token}/qr")
async def qr_for_session(token: str, request: Request) -> Response:
    origin = _origin_from_request(request)
    url = f"{origin}/remote?s={quote(token)}"
    qr = qrcode.QRCode(box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#E2F13B", back_color="#0B0D14")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
