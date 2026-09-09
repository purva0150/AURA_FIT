import * as THREE from "three";

// Reduce the actual GLB surface to a welded simulation cage, preserving front/back.
// Four-neighbour interpolation transfers displacement, not absolute positions, so
// the original high-resolution seams and folds remain intact.
export function createClothCage(geometry, cellSize = 0.085) {
  const rest = geometry.attributes.position.array.slice();
  const clusters = new Map(), vertexNodes = [];
  for (let i = 0; i < rest.length; i += 3) {
    const key = [0, 1, 2].map((a) => Math.round(rest[i + a] / cellSize)).join(":");
    if (!clusters.has(key)) clusters.set(key, { id: clusters.size, p: new THREE.Vector3(), count: 0 });
    const c = clusters.get(key);
    c.p.add(new THREE.Vector3(rest[i], rest[i + 1], rest[i + 2])); c.count++;
    vertexNodes.push(c.id);
  }
  const nodes = [...clusters.values()].map((c) => c.p.divideScalar(c.count));
  const edges = new Map();
  const index = geometry.index?.array || Array.from({ length: vertexNodes.length }, (_, i) => i);
  for (let i = 0; i < index.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const x = vertexNodes[index[i + a]], y = vertexNodes[index[i + (a + 1) % 3]];
      if (x !== y) edges.set(`${Math.min(x, y)}:${Math.max(x, y)}`, [x, y]);
    }
  }
  const bindings = new Uint16Array(vertexNodes.length * 4), weights = new Float32Array(bindings.length);
  const v = new THREE.Vector3();
  for (let i = 0; i < vertexNodes.length; i++) {
    v.fromArray(rest, i * 3);
    const nearest = [];
    nodes.forEach((p, id) => {
      const d = v.distanceToSquared(p);
      if (nearest.length < 4 || d < nearest[3].d) {
        nearest.push({ id, d }); nearest.sort((a, b) => a.d - b.d); nearest.length = Math.min(4, nearest.length);
      }
    });
    const total = nearest.reduce((sum, n) => sum + 1 / (n.d + 0.00001), 0);
    nearest.forEach((n, j) => { bindings[i * 4 + j] = n.id; weights[i * 4 + j] = 1 / (n.d + 0.00001) / total; });
  }
  return { rest, nodes, edges: [...edges.values()], bindings, weights };
}

export function sleeveTarget(x, y, z, leftArm, rightArm, out) {
  const side = x < 0 ? -1 : 1;
  const blend = THREE.MathUtils.smoothstep(Math.abs(x), 0.23, 0.41) * THREE.MathUtils.smoothstep(y, -0.03, 0.12);
  const angle = ((side < 0 ? leftArm : rightArm) + 0.92) * side * blend;
  const dx = x - side * 0.235, dy = y - 0.37;
  return out.set(side * 0.235 + dx * Math.cos(angle) - dy * Math.sin(angle),
    0.37 + dx * Math.sin(angle) + dy * Math.cos(angle), z);
}