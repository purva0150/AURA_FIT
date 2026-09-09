# Cloth and pose regression tests

From `frontend/`, run `bash tests/run-physics.sh` (Node 20+).
This runs a native Node GLB/physics harness and the Jest pose/physics suite.
Three 0.186's CJS entry delegates to ESM, so the dedicated Jest configuration
transforms Three instead of mocking it. Cannon physics is real in these tests.

The GLB fixture uses the same centering, unit-height scaling, and front rotation
as `TryOnScene`. Raw/unscaled geometry is not a valid input to the cloth solver.
The inertia comparison keeps arm angles constant and samples during the motion
impulse; changing sleeve rest pose would mix fitting motion with cloth motion.

Browser checks use real APIs and MediaPipe startup. Deterministic pose fixtures
are TEST-ONLY for controlled WebGL validation, not a replacement for checking
fit with a real webcam. There is no fake-pose mode in the shipped application.