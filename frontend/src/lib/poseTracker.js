// One in-flight frame: synchronous WASM stays off the render/UI thread.
export function createPoseTracker(onResult, onError) {
  if (!process.env.REACT_APP_POSE_WASM_URL) throw new Error("REACT_APP_POSE_WASM_URL is required");
  const worker = new Worker(new URL("./poseWorker.js", import.meta.url), { type: "module", name: "aura-pose" });
  let closed = false, busy = false, ready = false, lastSubmitted = 0, timeout;
  const tracker = {
    ready: new Promise((resolve, reject) => {
      const fail = (message) => {
        clearTimeout(timeout); busy = false;
        if (!ready) reject(new Error(message)); else onError(message);
      };
      timeout = setTimeout(() => fail("Body tracking took too long to load."), 45000);
      worker.onmessage = ({ data }) => {
        if (closed) return;
        if (data.type === "ready") { clearTimeout(timeout); ready = true; resolve(); }
        else if (data.type === "result") { busy = false; onResult(data); }
        else if (data.type === "error") fail(data.message);
      };
      worker.onerror = (event) => fail(event.message || "Body tracking unavailable.");
    }),
    async request(video, timestamp) {
      if (closed || busy || !ready || timestamp - lastSubmitted < 50) return;
      busy = true; lastSubmitted = timestamp;
      let bitmap;
      try {
        const width = Math.min(video.videoWidth, 640);
        bitmap = await createImageBitmap(video, { resizeWidth: width,
          resizeHeight: Math.round(width * video.videoHeight / video.videoWidth), resizeQuality: "medium" });
        if (closed) { bitmap.close(); return; }
        worker.postMessage({ type: "frame", bitmap, timestamp }, [bitmap]);
      } catch (err) { bitmap?.close(); busy = false; if (!closed) onError(err.message); }
    },
    close() { closed = true; clearTimeout(timeout); worker.terminate(); },
  };
  worker.postMessage({ type: "init",
    wasmRoot: process.env.REACT_APP_POSE_WASM_URL,
    modelUrl: new URL("/models/pose_landmarker_lite.float16.task", window.location.origin).href });
  return tracker;
}