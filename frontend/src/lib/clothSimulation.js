import * as THREE from "three";
import { World, Body, Particle, Sphere, Vec3, DistanceConstraint, Spring, SAPBroadphase } from "cannon-es";
import { createClothCage, sleeveTarget } from "./clothCage.js";

// Cannon solves a low-resolution mass/constraint surface. No looping wave animation:
// fabric inertia comes from measured torso motion, then springs and gravity settle it.
export class ClothSimulation {
  constructor(geometry) {
    this.geometry = geometry;
    this.cage = createClothCage(geometry);
    this.world = new World({ gravity: new Vec3(0, -2.8, 0) });
    this.world.broadphase = new SAPBroadphase(this.world);
    this.world.solver.iterations = 8;
    this.world.solver.tolerance = 0.001;
    this.world.defaultContactMaterial.friction = 0.08;
    this.world.defaultContactMaterial.restitution = 0;
    this.targets = this.cage.nodes.map((p) => p.clone());
    this.particles = this.cage.nodes.map((p) => {
      const pinned = p.y > 0.36 && Math.abs(p.x) < 0.3;
      const body = new Body({ mass: pinned ? 0 : 0.008, shape: new Particle(),
        position: new Vec3(p.x, p.y, p.z), linearDamping: 0.35,
        collisionFilterGroup: 1, collisionFilterMask: 2 });
      this.world.addBody(body);
      return body;
    });
    this.springs = this.particles.map((body, i) => {
      const anchor = new Body({ mass: 0, position: body.position.clone() });
      // Body-support tethers prevent a webcam-fitted garment collapsing into a sheet.
      return new Spring(body, anchor, { restLength: 0, stiffness: this.cage.nodes[i].y > 0.2 ? 2.4 : 0.85, damping: 0.035 });
    });
    this.links = this.cage.edges.map(([a, b]) => {
      const link = new DistanceConstraint(this.particles[a], this.particles[b], this.cage.nodes[a].distanceTo(this.cage.nodes[b]), 12);
      link.collideConnected = false;
      this.world.addConstraint(link);
      return link;
    });
    // Coarse torso collision proxy; no expensive or unavailable body scan required.
    this.torso = new Body({ mass: 0, collisionFilterGroup: 2, collisionFilterMask: 1 });
    [0.19, -0.05, -0.29].forEach((y) => this.torso.addShape(new Sphere(0.165), new Vec3(0, y, 0)));
    this.world.addBody(this.torso);
    this.previousMatrix = null;
    this.accumulator = 0;
    this.enabled = true;
    this.v = new THREE.Vector3();
    this.deltaMatrix = new THREE.Matrix4();
    this.rotation = new THREE.Quaternion();
    this.offsets = new Float32Array(this.particles.length * 3);
    geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
  }

  reset() {
    this.previousMatrix = null;
    this.accumulator = 0;
    this.particles.forEach((p, i) => { p.position.copy(this.targets[i]); p.velocity.setZero(); p.force.setZero(); });
  }

  step(dt, matrix, leftArm = -1.38, rightArm = -1.38, enabled = true) {
    this.cage.nodes.forEach((p, i) => sleeveTarget(p.x, p.y, p.z, leftArm, rightArm, this.targets[i]));
    const reset = !this.previousMatrix || dt > 0.15 || this.enabled !== enabled;
    this.enabled = enabled;
    if (reset) this.reset();
    if (this.previousMatrix && enabled) {
      this.deltaMatrix.copy(matrix).invert().multiply(this.previousMatrix);
      // Preserve particle world positions while the tracked body moves underneath.
      this.particles.forEach((p) => {
        if (!p.mass) return;
        this.v.copy(p.position).applyMatrix4(this.deltaMatrix);
        p.position.copy(this.v);
      });
    }
    this.previousMatrix = matrix.clone();
    this.springs.forEach((s, i) => s.bodyB.position.copy(this.targets[i]));
    this.links.forEach((link, i) => {
      const [a, b] = this.cage.edges[i];
      link.distance = this.targets[a].distanceTo(this.targets[b]);
    });
    this.particles.forEach((p, i) => { if (!p.mass || !enabled) { p.position.copy(this.targets[i]); p.velocity.setZero(); } });
    if (enabled) {
      matrix.decompose(this.v, this.rotation, new THREE.Vector3());
      this.v.set(0, -2.8, 0).applyQuaternion(this.rotation.invert());
      this.world.gravity.copy(this.v);
      this.accumulator = Math.min(this.accumulator + Math.max(dt, 0), 3 / 60);
      while (this.accumulator >= 1 / 60) {
        this.springs.forEach((s, i) => { if (this.particles[i].mass) s.applyForce(); });
        this.world.step(1 / 60);
        this.accumulator -= 1 / 60;
        this.limitStretch();
      }
    }
    this.writeGeometry(leftArm, rightArm);
  }

  limitStretch() {
    this.particles.forEach((p, i) => {
      this.v.copy(p.position).sub(this.targets[i]);
      const distance = this.v.length();
      if (!Number.isFinite(distance)) { p.position.copy(this.targets[i]); p.velocity.setZero(); return; }
      // Safety envelope for tracking jumps; ordinary motion stays inside it.
      if (distance > 0.11) {
        this.v.multiplyScalar(0.11 / distance).add(this.targets[i]);
        p.position.copy(this.v); p.velocity.scale(0.35, p.velocity);
      }
    });
  }

  writeGeometry(leftArm, rightArm) {
    const { rest, bindings, weights } = this.cage;
    this.particles.forEach((p, i) => {
      this.offsets[i * 3] = p.position.x - this.targets[i].x;
      this.offsets[i * 3 + 1] = p.position.y - this.targets[i].y;
      this.offsets[i * 3 + 2] = p.position.z - this.targets[i].z;
    });
    const positions = this.geometry.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      sleeveTarget(rest[i * 3], rest[i * 3 + 1], rest[i * 3 + 2], leftArm, rightArm, this.v);
      for (let j = 0; j < 4; j++) {
        const n = bindings[i * 4 + j] * 3, weight = weights[i * 4 + j];
        this.v.x += this.offsets[n] * weight; this.v.y += this.offsets[n + 1] * weight; this.v.z += this.offsets[n + 2] * weight;
      }
      positions.setXYZ(i, this.v.x, this.v.y, this.v.z);
    }
    positions.needsUpdate = true;
    this.geometry.computeVertexNormals();
  }

  dispose() {
    [...this.world.constraints].forEach((c) => this.world.removeConstraint(c));
    [...this.world.bodies].forEach((b) => this.world.removeBody(b));
    this.particles = []; this.springs = []; this.links = [];
  }
}