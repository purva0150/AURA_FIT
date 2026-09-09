import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import { createClothCage } from "../src/lib/clothCage.js";
import { fitFromPose, videoCover, videoPoint } from "../src/lib/poseFit.js";
import { normalizeGarmentGeometry } from "../src/lib/garmentGeometry.js";
import { ClothSimulation } from "../src/lib/clothSimulation.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function parseGlbMesh(glbPath) {
  const bytes = fs.readFileSync(glbPath);
  assert.equal(bytes.slice(0, 4).toString("ascii"), "glTF");
  assert.equal(bytes.readUInt32LE(8), bytes.length);
  let offset = 12;
  let jsonChunk;
  let binChunk;
  while (offset < bytes.length) {
    const chunkLength = bytes.readUInt32LE(offset);
    const chunkType = bytes.readUInt32LE(offset + 4);
    const chunkData = bytes.slice(offset + 8, offset + 8 + chunkLength);
    if (chunkType === 0x4e4f534a) jsonChunk = chunkData;
    if (chunkType === 0x004e4942) binChunk = chunkData;
    offset += 8 + chunkLength;
  }
  const gltf = JSON.parse(jsonChunk.toString("utf8"));
  const mesh = gltf.meshes[0].primitives[0];
  const posAccessor = gltf.accessors[mesh.attributes.POSITION];
  const posView = gltf.bufferViews[posAccessor.bufferView];
  const posOffset = (posView.byteOffset || 0) + (posAccessor.byteOffset || 0);
  const positions = new Float32Array(binChunk.buffer, binChunk.byteOffset + posOffset, posAccessor.count * 3).slice();
  const idxAccessor = gltf.accessors[mesh.indices];
  const idxView = gltf.bufferViews[idxAccessor.bufferView];
  const idxOffset = (idxView.byteOffset || 0) + (idxAccessor.byteOffset || 0);
  const indices = idxAccessor.componentType === 5125
    ? new Uint32Array(binChunk.buffer, binChunk.byteOffset + idxOffset, idxAccessor.count).slice()
    : new Uint16Array(binChunk.buffer, binChunk.byteOffset + idxOffset, idxAccessor.count).slice();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeBoundingBox();
  normalizeGarmentGeometry(geometry, geometry.boundingBox.getCenter(new THREE.Vector3()), geometry.boundingBox.getSize(new THREE.Vector3()).y);
  return { geometry, vertexCount: posAccessor.count, triCount: idxAccessor.count / 3 };
}

function avgAbsDelta(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

function assertFinite(arr, label) {
  for (let i = 0; i < arr.length; i++) {
    assert.ok(Number.isFinite(arr[i]), `${label} non-finite @ ${i}`);
  }
}

function run() {
  const failures = [];
  const check = (cond, message) => {
    if (!cond) failures.push(message);
  };
  const glbPath = path.resolve(__dirname, "../../backend/static_garments/shirt.glb");
  const { geometry, vertexCount, triCount } = parseGlbMesh(glbPath);
  console.log("GLB", { vertexCount, triCount });
  check(vertexCount >= 10000, `Unexpected vertex count: ${vertexCount}`);
  check(triCount >= 19000, `Unexpected tri count: ${triCount}`);
  assertFinite(geometry.attributes.position.array, "base-geometry");

  const cage = createClothCage(geometry, 0.085);
  check(cage.nodes.length > 100, `Cage nodes too low: ${cage.nodes.length}`);
  check(cage.edges.length > 100, `Cage edges too low: ${cage.edges.length}`);

  const sim = new ClothSimulation(geometry);
  const identity = new THREE.Matrix4().identity();
  const moved = new THREE.Matrix4().makeRotationZ(0.22).multiply(new THREE.Matrix4().makeTranslation(0.08, 0.06, 0));
  const rest = geometry.attributes.position.array.slice();

  for (let i = 0; i < 45; i++) sim.step(1 / 60, identity, -1.2, -1.2, true);
  const stable = geometry.attributes.position.array.slice();

  // Isolate torso inertia: do not change the sleeve rest pose in this comparison.
  // Sample during the impulse, not one second later after fabric has settled.
  for (let i = 0; i < 6; i++) sim.step(1 / 60, moved, -1.2, -1.2, true);
  const movedFrame = geometry.attributes.position.array.slice();
  const motionDelta = avgAbsDelta(stable, movedFrame);
  console.log("motionDelta", motionDelta);
  check(motionDelta > 0.0002, `No visible deformation under motion: ${motionDelta}`);

  // Movement should deform lower cloth more than shoulder/pin zone
  const pos = geometry.attributes.position;
  let lowerDelta = 0;
  let lowerCount = 0;
  let upperDelta = 0;
  let upperCount = 0;
  for (let i = 0; i < pos.count; i++) {
    const ry = rest[i * 3 + 1];
    const d = Math.abs(movedFrame[i * 3] - stable[i * 3]) + Math.abs(movedFrame[i * 3 + 1] - stable[i * 3 + 1]) + Math.abs(movedFrame[i * 3 + 2] - stable[i * 3 + 2]);
    if (ry < -0.05) {
      lowerDelta += d;
      lowerCount += 1;
    }
    if (ry > 0.22) {
      upperDelta += d;
      upperCount += 1;
    }
  }
  const lowerAvg = lowerDelta / Math.max(1, lowerCount);
  const upperAvg = upperDelta / Math.max(1, upperCount);
  console.log("lowerVsUpperDelta", { lowerAvg, upperAvg, lowerCount, upperCount });
  check(lowerAvg > upperAvg, `Lower cloth not deforming more than shoulder zone: lower=${lowerAvg} upper=${upperAvg}`);

  for (let i = 0; i < 220; i++) sim.step(1 / 60, identity, -1.2, -1.2, true);
  const settled = geometry.attributes.position.array.slice();
  const settleDelta = avgAbsDelta(stable, settled);
  console.log("settleDelta", settleDelta);
  check(settleDelta < 0.01, `Stationary input did not settle: ${settleDelta}`);

  sim.step(1 / 60, identity, -1.2, -1.2, false);
  sim.step(1 / 60, identity, -1.2, -1.2, false);
  const disabled = geometry.attributes.position.array.slice();
  const disabledDelta = avgAbsDelta(rest, disabled);
  console.log("disabledDelta", disabledDelta);
  check(disabledDelta < 0.02, `Off mode did not restore posed geometry: ${disabledDelta}`);

  sim.step(0.35, identity, -1.2, -1.2, true);
  for (let i = 0; i < 120; i++) sim.step(1 / 60, identity, -1.2, -1.2, true);
  assertFinite(geometry.attributes.position.array, "post-large-dt");

  const maxOffset = Math.max(...sim.particles.map((p, i) => {
    const t = sim.targets[i];
    return new THREE.Vector3(p.position.x, p.position.y, p.position.z).distanceTo(t);
  }));
  console.log("maxOffset", maxOffset);
  check(maxOffset <= 0.12, `Stretch bound exceeded ${maxOffset}`);

  const pinned = sim.particles.filter((p) => p.mass === 0);
  check(pinned.length > 0, "Normalized shirt must have shoulder anchors");
  const pinDrift = pinned.reduce((sum, p, i) => {
    const t = sim.targets[sim.particles.indexOf(p)];
    return sum + new THREE.Vector3(p.position.x, p.position.y, p.position.z).distanceTo(t);
  }, 0) / Math.max(1, pinned.length);
  console.log("pinDrift", pinDrift);
  check(pinDrift < 0.001, `Pinned nodes drift too much: ${pinDrift}`);

  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0 }));
  lm[11] = { x: 0.42, y: 0.34, visibility: 0.99 };
  lm[12] = { x: 0.58, y: 0.34, visibility: 0.99 };
  lm[23] = { x: 0.45, y: 0.58, visibility: 0.99 };
  lm[24] = { x: 0.55, y: 0.58, visibility: 0.99 };
  lm[13] = { x: 0.38, y: 0.47, visibility: 0.99 };
  lm[14] = { x: 0.62, y: 0.47, visibility: 0.99 };

  const world = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
  world[11] = { x: -0.2, y: -0.2, z: 0 };
  world[12] = { x: 0.2, y: -0.2, z: 0 };
  world[23] = { x: -0.16, y: 0.2, z: 0 };
  world[24] = { x: 0.16, y: 0.2, z: 0 };
  const fit = fitFromPose(lm, world, 1280, 720, 1024, 768, 1, "auto");
  check(!!fit, "fitFromPose returned null for upright pose");
  check(Math.abs(fit.rx) < 0.05, `pitch drift ${fit.rx}`);
  check(Math.abs(fit.rz) < 0.05, `roll drift ${fit.rz}`);

  [320, 768, 1024, 1440].forEach((w) => {
    const h = Math.round((w * 9) / 16);
    const cover = videoCover(1280, 720, w, h);
    const center = videoPoint({ x: 0.5, y: 0.5 }, 1280, 720, w, h);
    assert.ok(Math.abs(center[0] - (cover.x + cover.width / 2)) < 0.01);
    assert.ok(Math.abs(center[1] - (cover.y + cover.height / 2)) < 0.01);
  });

  const nullFit = fitFromPose(Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0 })), null, 1280, 720, 1024, 768, 1, "auto");
  check(nullFit === null, "missing landmarks must return null");

  sim.dispose();
  if (failures.length) {
    console.error("physics_pose_harness FAILURES:\n- " + failures.join("\n- "));
    throw new Error(`physics_pose_harness failed (${failures.length})`);
  }
  console.log("physics_pose_harness PASS");
}

run();