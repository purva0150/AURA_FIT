# AURA FIT — Virtual AR Fitting Room

## Original problem statement
Public web app/PWA for virtual try-on with real-time body tracking. Laptop shows
a webcam mirror with real 3D GLB clothing; phone pairs by QR and controls garments,
fit and viewing options. Clothes should follow body perspective and drape rather
than behave as flat PNG overlays. No authentication.

Latest request: "Cloth Simulation: Turn on soft-body drape so the shirt bounces
and folds with your motion in real time. and make it for the straight poses,
normal ones".

## Status — 2026-09-09
Soft-body cloth and ordinary upright/relaxed-arm fitting implemented and tested.
Real-camera fit and target-device performance require user verification.
No mocked application APIs or production pose fixtures.

## Architecture
- FastAPI on 8001; MongoDB sessions via protected MONGO_URL / DB_NAME.
- React18/CRA5 on 3000; Three0.186, cannon-es0.20, MediaPipe Tasks Vision0.10.14.
- `/`: existing home. `/mirror?s={token}`: laptop mirror.
  `/remote?s={token}`: phone controller.
- `GET /api/health`, `GET /api/garments`, `POST /api/sessions`,
  `GET /api/sessions/{token}`, **POST** `/api/sessions/{token}` for updates,
  `POST /api/sessions/{token}/heartbeat`, `GET /api/sessions/{token}/qr`.
  Session updates use POST, not PATCH (older handoff was inaccurate).
- Asset: `/app/backend/static_garments/shirt.glb` served at
  `/api/static/shirt.glb`; 10,513 vertices / 19,517 triangles.
- Session data: token, selected_garment, fit_mode, view_mode, show_skeleton,
  cloth_enabled, phone_name, last_phone_seen, version, created_at.
- Mirror polls every900ms; remote heartbeat every3s. Phone-online computed on read.

## Implemented
### Prior iterations
- QR laptop/phone pairing, persisted session controls, eight shirt color variants.
- Replaced original 2D image warp with real GLB/Three.js rendering.
- Existing editorial dark/acid visual language preserved.

### Soft-body iteration — 2026-09-09
- Real cannon-es particle/distance-constraint simulation on a reduced, welded cage
  of the actual GLB surface, interpolated onto the high-resolution mesh.
- Shoulder/collar anchors, gravity, damping, torso support springs, coarse sphere
  collision proxy, motion inertia, bounded stretching, and reset on tracking gaps.
- Lower cloth reacts more than shoulders and settles after movement; no perpetual
  procedural sine-wave animation. Real-time approximate cloth, not exact physical sizing.
- `cloth_enabled` true by default (including older sessions); strict boolean update
  validation; accessible shared drape switch on mirror and phone.
- Stable torso-up alignment fixes flipped ordinary poses; correct MediaPipe Y-down
  pitch sign; relaxed-arm fallback and elbow-responsive sleeve rest shape.
- GLB centering/scaling baked correctly into geometry with shared normalization.
- Shared object-cover coordinate mapping aligns video, cloth, skeleton and snapshots.
- Pose results cached between camera frames; empty detections hide immediately.
  A1s stalled-worker grace window prevents slow-frame flicker.
- CPU MediaPipe inference runs in a dedicated worker with a single downscaled640px
  frame in flight, max20 submissions/sec; no synchronous inference on UI thread.
- WASM URL from frontend/.env REACT_APP_POSE_WASM_URL, version0.10.14.
  Model served locally at `/models/pose_landmarker_lite.float16.task`.
- Cancellable GLB fetch; loader skips unused printed-logo materials/textures;
  disposed/stale loads cannot reattach. Color swaps reuse the mesh/simulation.
- Explicit canvas stacking; capture queued immediately after a normal render and
  skeleton draw (no duplicate WebGL render); camera/worker/render resources cleaned up.
- Clear selection no longer reappears on next tracked frame; front/back/auto options
  now applied; remote selected preview stays synchronized.
- Responsive mirror toolbar, unique garment test IDs, control error messages,
  less frequent React pose-counter updates, invalid remote-session heartbeat cleanup.

## Key files
- `frontend/src/lib/clothCage.js`: surface reduction, mesh binding, sleeve posing.
- `frontend/src/lib/clothSimulation.js`: cannon-es physics and vertex updates.
- `frontend/src/lib/poseFit.js`: body transforms and video-cover coordinates.
- `frontend/src/lib/garmentGeometry.js`: shared normalization.
- `frontend/src/lib/garmentLoader.js`: cancellable, untextured GLB loading.
- `frontend/src/lib/tryOnScene.js`: render scene / pose / physics orchestration.
- `frontend/src/lib/poseWorker.js`, `poseTracker.js`: asynchronous CPU tracking.
- `frontend/src/components/ClothToggle.js`: shared accessible switch.
- `frontend/src/pages/Mirror.js`, `Remote.js`; `backend/server.py`.
- Legacy `frontend/src/lib/pose.js` is no longer imported by Mirror.

## Verification
- Backend:13/13 pytest cases PASS (real HTTP calls to current external preview URL).
- `cd frontend && bash tests/run-physics.sh`: real normalized GLB native harness
  PASS +6/6 Jest cases PASS. Dedicated Jest config transforms Three's new ESM-backed
  CJS shim; standard CRA defaults alone do not support this dependency.
- Physics metrics: motion delta0.014389; lower deformation0.06623 vs upper0.01529;
  settle delta0.000773; pinned drift0; final maximum particle offset0.02181.
- Tester verified mirror/remote controls in both directions, eight colors,
  fit/view/skeleton, clear/reapply, persistence, QR, snapshot download, and no
  horizontal overflow at widths320/768/1024/1440.
- Final main-agent follow-up: two real CPU-worker+GLB startup/navigation cycles PASS;
  three consecutive captured images each contain102,197 qualifying white garment
  pixels under a deterministic test pose; clear -> DRAPE WAITING PASS.
  No GLB/blob texture errors in final logs. No product source is mocked.
- Production build PASS; existing MediaPipe missing-source-map and dynamic import
  warnings remain. ReadPixels warnings persist on software-rendered camera input;
  worker isolation prevents synchronous UI blocking. They are not proof of a
  broken API or a blank garment. No fixed real-device FPS claim is made.
- Final completion-check compatibility: repository `eslint.config.js` supports
  ESLint9 flat configuration while CRA retains its ESLint8 setup. Global
  `eslint frontend eslint.config.js` and local modified-file ESLint both PASS.
- Test browser camera is a synthetic green feed with no human. Deterministic
  worker pose injection is **MOCKED, TEST-ONLY**, used for controlled geometry/pixel
  evidence; actual worker initialization is tested separately.
- Reports: iteration_2/3 preserve initial findings; final follow-up in
  `test_reports/cloth_final_verification.json` supersedes resolved findings.
- Test browser scripts in `/app/tests/` are Playwright SCRIPT snippets, not
  standalone `python file.py` runners. Credential/access notes: test_credentials.md.

## Important lessons / limits
- Test fixtures MUST normalize the GLB identically to production; raw geometry
  creates zero shoulder anchors and invalid collision scaling (false failures).
- Isolate torso inertia from sleeve rest-pose changes in physics comparisons.
- Verify actual rendered/captured pixels, not only READY/DRAPE LIVE status labels.
  Final blank screenshot after pressing Clear is expected, not a rendering failure.
- Eight catalog entries are tints of one shirt, not eight different GLB shapes.
- Approximate body collision only; no fabric self-collision, body segmentation/
  hand occlusion, pattern sewing, or guaranteed real-world garment size estimation.
- No gallery/upload persistence, phone rear-camera streaming, or phone capture
  command has been added in this iteration.

## Prioritized next actions
- P0: User verifies natural standing with arms relaxed, gentle sway/bounce, and
  shoulder-to-hip framing on a real laptop webcam; tune fit if needed.
- P1: Additional GLB garment shapes (jackets/hoodies/pants), per-asset anchors.
- P1: Profile real-device performance; adaptive inference/render quality if needed.
- P2: Gallery/share links; phone-as-camera WebRTC; custom GLB upload with storage.
- P2: Better body/hand occlusion, improved torso collision and optional self-collision.
- P2: Session TTL cleanup and lower-latency sync if polling becomes a bottleneck.
- Enhancement suggestion: fabric-weight presets (light jersey vs heavy cotton).