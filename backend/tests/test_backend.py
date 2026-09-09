"""AURA FIT backend API tests for 3D contract and session sync behavior."""
import os
from pathlib import Path

import pytest
import requests


def _base_url() -> str:
    # Read frontend public URL from env file (no hardcoded fallback)
    from dotenv import dotenv_values

    env_path = Path(__file__).resolve().parents[2] / "frontend" / ".env"
    values = dotenv_values(env_path)
    url = os.environ.get("REACT_APP_BACKEND_URL") or values.get("REACT_APP_BACKEND_URL")
    if not url:
        pytest.skip("REACT_APP_BACKEND_URL is missing")
    return str(url).rstrip("/")


BASE = _base_url()


@pytest.fixture(scope="module")
def session_token():
    r = requests.post(f"{BASE}/api/sessions", timeout=20)
    assert r.status_code == 200
    data = r.json()
    token = data["session"]["token"]
    assert isinstance(token, str) and len(token) > 3
    return token


# Core service health endpoint
def test_health():
    r = requests.get(f"{BASE}/api/health", timeout=20)
    assert r.status_code == 200
    payload = r.json()
    assert payload["ok"] is True
    assert payload["service"] == "aura-fit"


# Garments endpoint should expose 3D GLB contract
def test_garments_3d_contract():
    r = requests.get(f"{BASE}/api/garments", timeout=20)
    assert r.status_code == 200
    garments = r.json()["garments"]
    assert len(garments) >= 8
    assert all(g["model"] == "/api/static/shirt.glb" for g in garments)
    first = garments[0]
    assert first["id"] == "white_tee"
    assert first["tint"].startswith("#")


# Static 3D asset should be served as GLB
def test_static_shirt_glb_magic_and_length():
    r = requests.get(f"{BASE}/api/static/shirt.glb", timeout=30)
    assert r.status_code == 200
    assert "model/gltf-binary" in r.headers.get("content-type", "")
    data = r.content
    assert len(data) > 1_000_000
    assert data[:4] == b"glTF"
    version = int.from_bytes(data[4:8], "little")
    declared_length = int.from_bytes(data[8:12], "little")
    assert version == 2
    assert declared_length == len(data)


# Session lifecycle create/get/update/clear
def test_session_defaults_on_create_and_get(session_token):
    r = requests.get(f"{BASE}/api/sessions/{session_token}", timeout=20)
    assert r.status_code == 200
    session = r.json()["session"]
    assert session["token"] == session_token
    assert session["selected_garment"] is None
    assert session["fit_mode"] == "regular"
    assert session["view_mode"] == "auto"
    assert session["show_skeleton"] is True
    assert session["cloth_enabled"] is True


def test_invalid_session_token_routes(session_token):
    r_get = requests.get(f"{BASE}/api/sessions/DOES_NOT_EXIST", timeout=20)
    r_post = requests.post(f"{BASE}/api/sessions/DOES_NOT_EXIST", json={"fit_mode": "regular"}, timeout=20)
    r_beat = requests.post(f"{BASE}/api/sessions/DOES_NOT_EXIST/heartbeat", json={"phone_name": "QA"}, timeout=20)
    assert r_get.status_code == 404
    assert r_post.status_code == 404
    assert r_beat.status_code == 404


# Session update route behavior (POST only, strict validation)
def test_patch_not_supported_for_session_update(session_token):
    r = requests.patch(f"{BASE}/api/sessions/{session_token}", json={"fit_mode": "fitted"}, timeout=20)
    assert r.status_code in (404, 405)


def test_post_updates_and_persistence(session_token):
    update_payload = {
        "selected_garment": "acid_tee",
        "fit_mode": "relaxed",
        "view_mode": "back",
        "show_skeleton": False,
        "cloth_enabled": False,
        "source": "phone",
        "phone_name": "QA Pixel",
    }
    post = requests.post(f"{BASE}/api/sessions/{session_token}", json=update_payload, timeout=20)
    assert post.status_code == 200
    posted = post.json()["session"]
    assert posted["selected_garment"] == "acid_tee"
    assert posted["fit_mode"] == "relaxed"
    assert posted["view_mode"] == "back"
    assert posted["show_skeleton"] is False
    assert posted["cloth_enabled"] is False
    assert posted["phone_name"] == "QA Pixel"

    fetched = requests.get(f"{BASE}/api/sessions/{session_token}", timeout=20)
    assert fetched.status_code == 200
    persisted = fetched.json()["session"]
    assert persisted["selected_garment"] == "acid_tee"
    assert persisted["fit_mode"] == "relaxed"
    assert persisted["view_mode"] == "back"
    assert persisted["show_skeleton"] is False
    assert persisted["cloth_enabled"] is False


def test_cloth_enabled_toggle_persistence_true_false(session_token):
    r1 = requests.post(f"{BASE}/api/sessions/{session_token}", json={"cloth_enabled": True, "source": "mirror"}, timeout=20)
    assert r1.status_code == 200
    assert r1.json()["session"]["cloth_enabled"] is True

    r2 = requests.post(f"{BASE}/api/sessions/{session_token}", json={"cloth_enabled": False, "source": "mirror"}, timeout=20)
    assert r2.status_code == 200
    assert r2.json()["session"]["cloth_enabled"] is False

    r3 = requests.get(f"{BASE}/api/sessions/{session_token}", timeout=20)
    assert r3.status_code == 200
    assert r3.json()["session"]["cloth_enabled"] is False


def test_cloth_enabled_strict_bool_validation(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"cloth_enabled": "false"}, timeout=20)
    assert r.status_code == 422


def test_fit_view_validation_errors(session_token):
    bad_fit = requests.post(f"{BASE}/api/sessions/{session_token}", json={"fit_mode": "baggy"}, timeout=20)
    bad_view = requests.post(f"{BASE}/api/sessions/{session_token}", json={"view_mode": "left"}, timeout=20)
    bad_garment = requests.post(f"{BASE}/api/sessions/{session_token}", json={"selected_garment": "ghost"}, timeout=20)
    assert bad_fit.status_code == 400
    assert bad_view.status_code == 400
    assert bad_garment.status_code == 400


def test_clear_garment_persistence(session_token):
    set_garment = requests.post(f"{BASE}/api/sessions/{session_token}", json={"selected_garment": "white_tee"}, timeout=20)
    assert set_garment.status_code == 200
    assert set_garment.json()["session"]["selected_garment"] == "white_tee"

    clear = requests.post(f"{BASE}/api/sessions/{session_token}", json={"clear_garment": True}, timeout=20)
    assert clear.status_code == 200
    assert clear.json()["session"]["selected_garment"] is None

    get_after = requests.get(f"{BASE}/api/sessions/{session_token}", timeout=20)
    assert get_after.status_code == 200
    assert get_after.json()["session"]["selected_garment"] is None


# Phone sync helpers
def test_heartbeat_updates_phone_state(session_token):
    beat = requests.post(
        f"{BASE}/api/sessions/{session_token}/heartbeat",
        json={"phone_name": "Android QA"},
        timeout=20,
    )
    assert beat.status_code == 200
    session = beat.json()["session"]
    assert session["phone_connected"] is True
    assert session["phone_name"] == "Android QA"


def test_qr_png(session_token):
    r = requests.get(f"{BASE}/api/sessions/{session_token}/qr", timeout=20)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
