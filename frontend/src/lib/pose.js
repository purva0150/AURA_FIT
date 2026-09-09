import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

// Loads MediaPipe pose landmarker (browser-side). Model file is fetched from Google's CDN.
export async function createPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  const pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  return pose;
}

export { DrawingUtils };

// Compute an affine matrix that maps four garment anchors (top-left shoulder, top-right shoulder,
// bottom-left hip, bottom-right hip in image px) to the four detected keypoints (video px).
// Returns { a,b,c,d,e,f } for CanvasRenderingContext2D.setTransform.
// Uses a least-squares 2D affine solve.
export function affineFromCorners(src, dst) {
  // src, dst: [[x,y],[x,y],[x,y],[x,y]]  →  4 points each
  // Solve [a b c ; d e f] such that dst = src * M
  // Build linear system: for each pair we get 2 equations.
  const n = src.length;
  const A = [];
  const B = [];
  for (let i = 0; i < n; i++) {
    const [sx, sy] = src[i];
    const [dx, dy] = dst[i];
    A.push([sx, sy, 1, 0, 0, 0]);
    A.push([0, 0, 0, sx, sy, 1]);
    B.push(dx);
    B.push(dy);
  }
  // Solve normal equations (A^T A) x = A^T B, size 6x6
  const AtA = Array.from({ length: 6 }, () => new Array(6).fill(0));
  const AtB = new Array(6).fill(0);
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < 6; j++) {
      AtB[j] += A[i][j] * B[i];
      for (let k = 0; k < 6; k++) AtA[j][k] += A[i][j] * A[i][k];
    }
  }
  const x = solve6(AtA, AtB);
  if (!x) return null;
  return { a: x[0], c: x[1], e: x[2], b: x[3], d: x[4], f: x[5] };
}

// Small 6x6 Gaussian solver
function solve6(M, y) {
  const n = 6;
  const A = M.map((row, i) => [...row, y[i]]);
  for (let i = 0; i < n; i++) {
    // find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    if (Math.abs(A[i][i]) < 1e-9) return null;
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / A[i][i];
      for (let j = i; j <= n; j++) A[k][j] -= f * A[i][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = A[i][n];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  return x;
}
