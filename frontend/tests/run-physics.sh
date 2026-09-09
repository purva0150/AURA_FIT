#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
node --experimental-default-type=module tests/physics_pose_harness.mjs
NODE_ENV=test BABEL_ENV=test node node_modules/jest/bin/jest.js --config jest.physics.config.cjs --runInBand