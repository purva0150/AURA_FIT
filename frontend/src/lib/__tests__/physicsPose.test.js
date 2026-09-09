import fs from "fs";
import path from "path";
import * as THREE from "three";
import { createClothCage } from "../clothCage";
import { ClothSimulation } from "../clothSimulation";
import { fitFromPose, videoCover, videoPoint } from "../poseFit";
import { normalizeGarmentGeometry } from "../garmentGeometry";

function parseGlbMesh(glbPath) {
  const bytes = fs.readFileSync(glbPath);
  expect(bytes.slice(0, 4).toString("ascii")).toBe("glTF");
  const declaredLength = bytes.readUInt32LE(8);
  expect(declaredLength).toBe(bytes.length);

  let offset = 12;
  const chunks = [];
  while (offset < bytes.length) {
    const chunkLength = bytes.readUInt32LE(offset);
    const chunkType = bytes.readUInt32LE(offset + 4);
    const chunkData = bytes.slice(offset + 8, offset + 8 + chunkLength);
    chunks.push({ chunkLength, chunkType, chunkData });
    offset += 8 + chunkLength;
  }

  const jsonChunk = chunks.find((c) => c.chunkType === 0x4e4f534a);
  const binChunk = chunks.find((c) => c.chunkType === 0x004e4942);
  const gltf = JSON.parse(jsonChunk.chunkData.toString("utf8"));
  const bin = binChunk.chunkData;

  const mesh = gltf.meshes[0].primitives[0];
  const posAccessor = gltf.accessors[mesh.attributes.POSITION];
  const posView = gltf.bufferViews[posAccessor.bufferView];
  const posOffset = (posView.byteOffset || 0) + (posAccessor.byteOffset || 0);
  const position = new Float32Array(
    bin.buffer,
    bin.byteOffset + posOffset,
    posAccessor.count * 3,
  ).slice();

  const indexAccessor = gltf.accessors[mesh.indices];
  const indexView = gltf.bufferViews[indexAccessor.bufferView];
  const indexOffset = (indexView.byteOffset || 0) + (indexAccessor.byteOffset || 0);
  let index;
  if (indexAccessor.componentType === 5125) {
    index = new Uint32Array(bin.buffer, bin.byteOffset + indexOffset, indexAccessor.count).slice();
  } else if (indexAccessor.componentType === 5123) {
    index = new Uint16Array(bin.buffer, bin.byteOffset + indexOffset, indexAccessor.count).slice();
  } else {
    throw new Error(`Unsupported index component type: ${indexAccessor.componentType}`);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeBoundingBox();
  normalizeGarmentGeometry(geometry, geometry.boundingBox.getCenter(new THREE.Vector3()), geometry.boundingBox.getSize(new THREE.Vector3()).y);
  return { geometry, vertexCount: posAccessor.count, triCount: indexAccessor.count / 3 };
}

function finitePositions(arr) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return false;
  }
  return true;
}

function avgDistance(a, b) {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sum += Math.abs(a[i] - b[i]);
  return sum / n;
}

describe("pose/cloth physics harness using real shirt.glb", () => {
  const glbPath = path.resolve(__dirname, "../../../../backend/static_garments/shirt.glb");

  test("GLB decodes to expected finite geometry", () => {
    const { geometry, vertexCount, triCount } = parseGlbMesh(glbPath);
    expect(vertexCount).toBeGreaterThan(10000);
    expect(triCount).toBeGreaterThan(19000);
    expect(finitePositions(geometry.attributes.position.array)).toBe(true);
  });

  test("cloth cage builds finite nodes/edges and constraints", () => {
    const { geometry } = parseGlbMesh(glbPath);
    const cage = createClothCage(geometry, 0.085);
    expect(cage.nodes.length).toBeGreaterThan(100);
    expect(cage.edges.length).toBeGreaterThan(100);
    expect(cage.bindings.length).toBe(geometry.attributes.position.count * 4);
    const finiteNodes = cage.nodes.every((n) => Number.isFinite(n.x) && Number.isFinite(n.y) && Number.isFinite(n.z));
    expect(finiteNodes).toBe(true);
  });

  test("simulation stays bounded, settles, and survives toggles/large dt", () => {
    const { geometry } = parseGlbMesh(glbPath);
    const sim = new ClothSimulation(geometry);
    const rest = geometry.attributes.position.array.slice();
    const identity = new THREE.Matrix4().identity();

    // Warm-up steady pose
    for (let i = 0; i < 40; i++) sim.step(1 / 60, identity, -1.2, -1.2, true);
    const stableA = geometry.attributes.position.array.slice();

    // Sudden motion
    const moved = new THREE.Matrix4().makeRotationZ(0.24).multiply(new THREE.Matrix4().makeTranslation(0.08, 0.06, 0));
    for (let i = 0; i < 50; i++) sim.step(1 / 60, moved, -0.9, -1.1, true);
    const movedFrame = geometry.attributes.position.array.slice();
    expect(avgDistance(stableA, movedFrame)).toBeGreaterThan(0.0002);

    // Back to steady; should settle (not perpetual wobble)
    for (let i = 0; i < 220; i++) sim.step(1 / 60, identity, -1.2, -1.2, true);
    const settled = geometry.attributes.position.array.slice();
    const driftAfterSettle = avgDistance(stableA, settled);
    expect(driftAfterSettle).toBeLessThan(0.01);

    // Disable drape should restore posed rest geometry
    sim.step(1 / 60, identity, -1.2, -1.2, false);
    sim.step(1 / 60, identity, -1.2, -1.2, false);
    const disabled = geometry.attributes.position.array.slice();
    expect(avgDistance(rest, disabled)).toBeLessThan(0.02);

    // Re-enable with large dt and validate finite bounded output
    sim.step(0.35, identity, -1.1, -1.1, true);
    for (let i = 0; i < 120; i++) sim.step(1 / 60, identity, -1.1, -1.1, true);
    const finalPos = geometry.attributes.position.array;
    expect(finitePositions(finalPos)).toBe(true);

    const maxOffset = Math.max(...sim.particles.map((p, i) => {
      const t = sim.targets[i];
      return new THREE.Vector3(p.position.x, p.position.y, p.position.z).distanceTo(t);
    }));
    expect(maxOffset).toBeLessThan(0.12);

    sim.dispose();
  });

  test("fitFromPose upright world coords keep pitch/roll near zero for straight pose", () => {
    const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0 }));
    lm[11] = { x: 0.42, y: 0.34, visibility: 0.99 };
    lm[12] = { x: 0.58, y: 0.34, visibility: 0.99 };
    lm[23] = { x: 0.45, y: 0.58, visibility: 0.99 };
    lm[24] = { x: 0.55, y: 0.58, visibility: 0.99 };
    lm[13] = { x: 0.38, y: 0.47, visibility: 0.99 };
    lm[14] = { x: 0.62, y: 0.47, visibility: 0.99 };

    const world = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
    world[11] = { x: -0.2, y: -0.2, z: 0.0 };
    world[12] = { x: 0.2, y: -0.2, z: 0.0 };
    world[23] = { x: -0.16, y: 0.2, z: 0.0 };
    world[24] = { x: 0.16, y: 0.2, z: 0.0 };

    const fit = fitFromPose(lm, world, 1280, 720, 1024, 768, 1, "auto");
    expect(fit).toBeTruthy();
    expect(Math.abs(fit.rx)).toBeLessThan(0.05);
    expect(Math.abs(fit.rz)).toBeLessThan(0.05);
  });

  test("videoPoint follows object-cover mapping across widths", () => {
    const widths = [320, 768, 1024, 1440];
    widths.forEach((w) => {
      const h = Math.round(w * 9 / 16);
      const cover = videoCover(1280, 720, w, h);
      const center = videoPoint({ x: 0.5, y: 0.5 }, 1280, 720, w, h);
      expect(Math.abs(center[0] - (cover.x + cover.width / 2))).toBeLessThan(0.01);
      expect(Math.abs(center[1] - (cover.y + cover.height / 2))).toBeLessThan(0.01);
      const leftTop = videoPoint({ x: 1, y: 0 }, 1280, 720, w, h);
      expect(leftTop[0]).toBeCloseTo(cover.x, 4);
      expect(leftTop[1]).toBeCloseTo(cover.y, 4);
    });
  });

  test("missing core landmarks returns null fit", () => {
    const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0 }));
    const fit = fitFromPose(lm, null, 1280, 720, 1024, 768, 1, "auto");
    expect(fit).toBeNull();
  });
});