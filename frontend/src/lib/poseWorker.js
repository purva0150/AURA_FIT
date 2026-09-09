/* global globalThis */
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

let pose = null;
const workerScope = globalThis;
workerScope.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      const vision = await FilesetResolver.forVisionTasks(data.wasmRoot);
      pose = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: data.modelUrl, delegate: "CPU" },
        runningMode: "VIDEO", numPoses: 1,
        minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5, outputSegmentationMasks: false,
      });
      workerScope.postMessage({ type: "ready" });
    } else if (data.type === "frame") {
      const result = pose.detectForVideo(data.bitmap, data.timestamp);
      workerScope.postMessage({ type: "result", landmarks: result.landmarks[0] || null,
        worldLandmarks: result.worldLandmarks[0] || null });
    } else if (data.type === "close") {
      pose?.close(); workerScope.close();
    }
  } catch (err) {
    workerScope.postMessage({ type: "error", message: err?.message || String(err) });
  } finally { data.bitmap?.close(); }
};