import * as THREE from "three";

// Share object-cover coordinates between the video, cloth, skeleton and captures.
export function videoCover(videoW, videoH, width, height) {
  const scale = Math.max(width / videoW, height / videoH);
  return { width: videoW * scale, height: videoH * scale,
    x: (width - videoW * scale) / 2, y: (height - videoH * scale) / 2 };
}

export function videoPoint(p, videoW, videoH, width, height) {
  const cover = videoCover(videoW, videoH, width, height);
  return [(1 - p.x) * cover.width + cover.x, p.y * cover.height + cover.y];
}

const visible = (p) => p && [p.x, p.y].every(Number.isFinite) && (p.visibility ?? 1) > 0.45;
const clamp = THREE.MathUtils.clamp;
export const angleDelta = (a, b) => Math.atan2(Math.sin(b - a), Math.cos(b - a));

export function fitFromPose(lm, world, videoW, videoH, width, height, fit = 1, view = "auto") {
  if (!lm || ![11, 12, 23, 24].every((i) => visible(lm[i]))) return null;
  const point = (i) => {
    const [x, y] = videoPoint(lm[i], videoW, videoH, width, height);
    return new THREE.Vector2((x - width / 2) * 2 / height, 1 - y * 2 / height);
  };
  const ls = point(11), rs = point(12), lh = point(23), rh = point(24);
  const shoulder = ls.clone().add(rs).multiplyScalar(0.5);
  const hip = lh.clone().add(rh).multiplyScalar(0.5);
  const up = shoulder.clone().sub(hip);
  const torsoH = up.length(), shoulderW = ls.distanceTo(rs);
  if (torsoH < 0.08 || shoulderW < 0.05) return null;
  // Torso-up is sign-stable in ordinary standing poses, unlike the shoulder atan2.
  const rz = clamp(Math.atan2(-up.x, up.y), -0.65, 0.65);
  let ry = 0, rx = 0;
  if (world && [11, 12, 23, 24].every((i) => world[i] &&
    [world[i].x, world[i].y, world[i].z].every(Number.isFinite))) {
    const a = world[11], b = world[12], c = world[23], d = world[24];
    ry = clamp(Math.atan2(a.z - b.z, Math.abs(a.x - b.x) + 1e-5), -0.85, 0.85);
    // MediaPipe Y points down: hip Y minus shoulder Y is positive while standing.
    rx = clamp(Math.atan2((a.z + b.z - c.z - d.z) / 2,
      Math.abs((c.y + d.y - a.y - b.y) / 2) + 1e-5), -0.35, 0.35);
  }
  if (view !== "auto") ry = view === "back" ? Math.PI : 0;
  const sy = torsoH / 0.88 * fit;
  const sx = shoulderW / (0.49 * Math.max(Math.cos(ry), 0.72)) * fit;
  const sleeveAngle = (shoulderId, elbowId) => {
    if (!visible(lm[elbowId])) return -1.38; // Relaxed arms, never require a T-pose.
    const start = point(shoulderId), arm = point(elbowId).sub(start).rotateAround(new THREE.Vector2(), -rz);
    const side = start.x < shoulder.x ? -1 : 1;
    return clamp(Math.atan2(arm.y, Math.max(arm.x * side, 0.015)), -1.5, 0.25);
  };
  const leftId = ls.x < rs.x ? 11 : 12, rightId = leftId === 11 ? 12 : 11;
  return { px: shoulder.x + Math.sin(rz) * sy * 0.38,
    py: shoulder.y - Math.cos(rz) * sy * 0.38, sx, sy, rz, ry, rx,
    leftArm: sleeveAngle(leftId, leftId + 2), rightArm: sleeveAngle(rightId, rightId + 2) };
}