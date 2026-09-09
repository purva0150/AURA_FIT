"""AURA FIT backend API tests."""
import os
import time
import pytest
import requests

BASE = os.environ.get("REACT_APP_BACKEND_URL", "https://ac6ee5bb-e30b-4d80-b303-523174d822e2.preview.emergentagent.com").rstrip("/")


@pytest.fixture(scope="module")
def session_token():
    r = requests.post(f"{BASE}/api/sessions")
    assert r.status_code == 200
    data = r.json()
    return data["session"]["token"]


# ── health ─────────────────────────────────────────────────────
def test_health():
    r = requests.get(f"{BASE}/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "service": "aura-fit"}


# ── garments catalog ───────────────────────────────────────────
def test_garments_catalog():
    r = requests.get(f"{BASE}/api/garments")
    assert r.status_code == 200
    g = r.json()["garments"]
    assert len(g) == 6
    assert g[0]["id"] == "white_tee"
    assert g[1]["id"] == "midnight_tee"
    assert g[1]["tint"] == "#111318"
    for item in g:
        for k in ("id", "name", "color", "image", "width", "height", "anchors"):
            assert k in item, f"missing {k} in {item['id']}"
        for a in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            assert a in item["anchors"]


# ── static PNG ─────────────────────────────────────────────────
def test_static_png():
    r = requests.get(f"{BASE}/api/static/tee_front.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


# ── session create ─────────────────────────────────────────────
def test_create_session():
    r = requests.post(f"{BASE}/api/sessions")
    assert r.status_code == 200
    d = r.json()
    s = d["session"]
    assert s["token"]
    assert s["selected_garment"] is None
    assert s["fit_mode"] == "regular"
    assert s["view_mode"] == "auto"
    assert s["show_skeleton"] is True
    assert s["phone_connected"] is False
    assert s["version"] == 0
    assert "/remote?s=" in d["mobile_url"]
    assert d["origin"]


def test_get_session(session_token):
    r = requests.get(f"{BASE}/api/sessions/{session_token}")
    assert r.status_code == 200
    assert r.json()["session"]["token"] == session_token


def test_get_session_bad_token():
    r = requests.get(f"{BASE}/api/sessions/nonexistent_zzz")
    assert r.status_code == 404


# ── session updates ────────────────────────────────────────────
def test_update_selected_garment_from_phone(session_token):
    r = requests.post(
        f"{BASE}/api/sessions/{session_token}",
        json={"selected_garment": "midnight_tee", "source": "phone", "phone_name": "Sofia iPhone"},
    )
    assert r.status_code == 200
    s = r.json()["session"]
    assert s["selected_garment"] == "midnight_tee"
    assert s["phone_connected"] is True
    assert s["phone_name"] == "Sofia iPhone"
    assert s["version"] >= 1


def test_update_invalid_garment(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"selected_garment": "not_a_real_id"})
    assert r.status_code == 400


def test_update_fit_mode(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"fit_mode": "relaxed"})
    assert r.status_code == 200
    assert r.json()["session"]["fit_mode"] == "relaxed"
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"fit_mode": "bogus"})
    assert r.status_code == 400


def test_update_view_mode(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"view_mode": "back"})
    assert r.status_code == 200
    assert r.json()["session"]["view_mode"] == "back"
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"view_mode": "sideways"})
    assert r.status_code == 400


def test_toggle_skeleton(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"show_skeleton": False})
    assert r.status_code == 200
    assert r.json()["session"]["show_skeleton"] is False


def test_clear_garment(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}", json={"clear_garment": True})
    assert r.status_code == 200
    assert r.json()["session"]["selected_garment"] is None


def test_heartbeat(session_token):
    r = requests.post(f"{BASE}/api/sessions/{session_token}/heartbeat", json={"phone_name": "Android phone"})
    assert r.status_code == 200
    s = r.json()["session"]
    assert s["phone_connected"] is True
    assert s["phone_name"] == "Android phone"


def test_qr_png(session_token):
    r = requests.get(f"{BASE}/api/sessions/{session_token}/qr")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
