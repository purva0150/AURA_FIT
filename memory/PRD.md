# AURA FIT — Virtual AR Fitting Room

## Original Problem Statement
Web application / PWA for virtual try-on of clothes with real-time body tracking. Clothes should fit onto the body via body-tracked overlay. Laptop shows the mirror, phone controls the wardrobe via QR pairing. Use pre-made online clothing assets. Public demo (no auth).

## User Choices
- Body tracking: MediaPipe Pose (browser, GPU delegate) via @mediapipe/tasks-vision
- Overlay: Canvas 2D — affine warp of transparent-bg PNG garments using shoulder + hip landmarks
- Assets: existing repo PNGs + 5 tint variants (white / midnight / acid / cobalt / terracotta / studio neutral)
- Pairing: Laptop = mirror + camera; Phone = remote (swipe, fit, view, skeleton, clear)
- Auth: none — public demo

## Architecture
- **Backend** (FastAPI @ 8001) — /api/health, /api/garments, /api/sessions CRUD, /api/sessions/{token}/heartbeat, /api/sessions/{token}/qr (PNG), /api/static/* garment PNGs. MongoDB stores sessions.
- **Frontend** (React 18 + Tailwind + Framer Motion @ 3000)
  - `/` Home — hero, features, wardrobe preview, how-it-works
  - `/mirror?s={token}` — getUserMedia webcam → MediaPipe pose (33 landmarks) → canvas mirror flip + affine-warped garment overlay + skeleton + snapshot capture + QR panel
  - `/remote?s={token}` — mobile PWA controller: hero swipe card, wardrobe grid, fit / view / skeleton toggles, remove garment, heartbeat every 3s

## What's Been Implemented (2026-01-09)
- FastAPI backend with 6 endpoints, MongoDB-backed session state, QR generation, garment catalog, static PNG serving.
- Home page (editorial dark theme, Syne + JetBrains Mono + Plus Jakarta), Launch Mirror flow.
- Mirror page with live browser-side pose detection, affine cloth warp using shoulder/hip anchors, fit-scale drape (fitted/regular/relaxed), skeleton overlay, snapshot download, QR pairing panel, quick garment swap bar.
- Remote page with card-swipe, wardrobe grid, fit / view / skeleton controls, apply-to-mirror sync, phone-name heartbeat.
- Full data-testid coverage; PWA manifest; distinctive anti-AI-slop design (obsidian + acid-lime, Syne italic display).
- Backend pytest suite (14 cases, 100% pass).

## Prioritized Backlog
- **P1**: Persist snapshots to /gallery, share-sheet, install prompt banner, install PWA on phone
- **P2**: WebRTC direct stream (phone camera → laptop) so phone can be the camera, upload custom garment PNG
- **P2**: TPS warp (existing Python engine) as optional server-side "premium" mode
- **P3**: Multi-user room mode, saved outfit collections

## Known Limits
- Overlay is affine (shoulder + hip corners), not TPS — good enough for demo, slight drift on extreme poses.
- MediaPipe model fetched from Google CDN; requires internet.
