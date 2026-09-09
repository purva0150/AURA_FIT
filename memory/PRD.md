# AURA FIT — Virtual AR Fitting Room (3D)

## Original Problem Statement
Web application / PWA for virtual try-on of clothes with real-time body tracking. Clothes should fit/drape on the body via body-tracked overlay. Laptop shows the mirror, phone controls the wardrobe via QR pairing. Public demo (no auth).

## Iteration 2 — Real 3D GLB (2026-01-09)
User feedback: "flat PNG doesn't sell it, wrap us in actual 3D clothing".
- Sourced `shirt_baked.glb` (open t-shirt mesh, glTF 2.0, 1MB) and served from `/api/static/shirt.glb`.
- Built `TryOnScene` (`/app/frontend/src/lib/tryOnScene.js`): Three.js WebGL renderer over the mirrored webcam video with orthographic NDC camera, PBR-lit shirt mesh, acid rim light.
- Bind MediaPipe world-landmarks: shoulder + hip midpoints → position; shoulder-hip distance → scale; shoulder line 2D → roll Z; shoulder Z-diff (world) → yaw Y; shoulder/hip Y & Z → pitch X. All smoothed with EMA (α=0.35).
- Mirror.js rewritten: `<video>` (CSS-mirrored) + Three.js `<canvas>` + skeleton overlay canvas stacked. Snapshot composites all three.
- Wardrobe expanded to 8 tinted variants of the same GLB (white, midnight, acid lime, cobalt, terracotta, forest, rose, cream). MeshStandardMaterial `.color` is swapped per garment — no reload needed to change tint.
- Home & Remote refreshed with 3D-preview cards (SVG t-shirt silhouettes with tinted radial gradients + "GLB · 3D" badges).

## Iteration 1 — MVP (superseded)
- FastAPI backend + MongoDB sessions + QR + phone remote sync (still in place).
- Original PNG affine warp replaced by 3D approach above.

## Architecture
- **Backend** (FastAPI @ 8001): /api/health, /api/garments, /api/sessions CRUD, /api/sessions/{token}/heartbeat, /api/sessions/{token}/qr, /api/static/shirt.glb.
- **Frontend** (React 18 + Tailwind + Framer Motion + Three.js 0.186 + @mediapipe/tasks-vision 0.10.14 @ 3000)
  - `/` — Home
  - `/mirror?s={token}` — WebGL 3D mirror with pose-driven shirt
  - `/remote?s={token}` — mobile controller

## What's Ready
- Real 3D shirt draping on the torso (rotates and scales with your movement).
- QR-paired phone remote (swipe / fit / view / skeleton / clear / apply).
- MongoDB-backed session state with heartbeat + version.
- Snapshot download (video + 3D shirt + skeleton composite PNG).
- Editorial dark theme (Syne / JetBrains Mono / Plus Jakarta Sans, acid-lime accent).
- 14/14 backend pytest cases.

## Backlog
- P1 outfit gallery + share links
- P1 custom GLB upload (drag-drop your own shirt)
- P2 phone-as-camera via WebRTC
- P2 more garment shapes (hoodie/jacket GLBs)
- P3 cloth simulation (soft-body draping)
