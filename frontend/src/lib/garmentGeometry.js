export function normalizeGarmentGeometry(geometry, center, height) {
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.scale(1 / height, 1 / height, 1 / height);
  geometry.rotateY(Math.PI);
  return geometry;
}