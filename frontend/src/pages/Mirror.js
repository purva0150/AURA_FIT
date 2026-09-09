import React, { useEffect, useRef, useState, useCallback } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, X, Download, Home as HomeIcon, Copy, Check, QrCode as QrIcon, Wifi, WifiOff } from "lucide-react";
import { fetchGarments, getSession, updateSession, qrUrl } from "../api";
import { createPoseTracker } from "../lib/poseTracker";
import { TryOnScene } from "../lib/tryOnScene";
import { videoCover, videoPoint } from "../lib/poseFit";
import { ClothToggle } from "../components/ClothToggle";

const FIT_SCALE = { fitted: 0.94, regular: 1.0, relaxed: 1.10 };
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

export default function Mirror() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const token = params.get("s");

  const videoRef = useRef(null);
  const stageRef = useRef(null);         // wrapper div sized to viewport
  const overlayCanvasRef = useRef(null); // 2D skeleton overlay
  const threeCanvasRef = useRef(null);   // Three.js WebGL canvas
  const rafRef = useRef(null);
  const lastVideoTs = useRef(-1);
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const sceneRef = useRef(null);
  const loadedGarmentIdRef = useRef(null);
  const lastPoseRef = useRef({ landmarks: null, worldLandmarks: null, at: 0 });
  const frameTimeRef = useRef(null);
  const captureRequestRef = useRef(null);
  const stateRef = useRef({ selected_garment: null, fit_mode: "regular", show_skeleton: true });
  const fpsRef = useRef({ frames: 0, last: performance.now(), value: 0 });

  const [garments, setGarments] = useState([]);
  const [session, setSession] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [copied, setCopied] = useState(false);
  const [fps, setFps] = useState(0);
  const [poseCount, setPoseCount] = useState(0);
  const [showQrOverlay, setShowQrOverlay] = useState(true);
  const [modelStatus, setModelStatus] = useState("idle");
  const [sceneReady, setSceneReady] = useState(false);
  const [controlError, setControlError] = useState(null);

  const acceptSession = useCallback((s) => {
    if ((s.version ?? 0) < (stateRef.current.version ?? 0)) return;
    stateRef.current = s; setSession(s);
  }, []);
  const applyPatch = async (patch) => {
    try { acceptSession(await updateSession(token, { ...patch, source: "mirror" })); setControlError(null); }
    catch { setControlError("Could not update the mirror. Please try again."); }
  };

  useEffect(() => { if (!token) nav("/"); }, [token, nav]);

  useEffect(() => {
    if (!token) return;
    fetchGarments().then(setGarments).catch(() => setControlError("Wardrobe unavailable."));
    getSession(token).then(acceptSession).catch(() => nav("/"));
  }, [token, nav, acceptSession]);

  useEffect(() => {
    if (!token) return;
    const iv = setInterval(async () => {
      try { acceptSession(await getSession(token)); } catch { /* Keep last confirmed state during a transient polling failure. */ }
    }, 900);
    return () => clearInterval(iv);
  }, [token, acceptSession]);

  // Boot camera + MediaPipe + Three.js scene
  useEffect(() => {
    let cancelled = false;
    lastVideoTs.current = -1;
    async function boot() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        const v = videoRef.current;
        v.srcObject = stream;
        await v.play();
        if (cancelled) return;

        // Three.js scene
        const scene = new TryOnScene(threeCanvasRef.current);
        sceneRef.current = scene;
        resizeAll();

        // MediaPipe
        setModelStatus("loading");
        const pose = createPoseTracker((result) => {
          if (!cancelled) lastPoseRef.current = { ...result, at: performance.now() };
        }, (message) => { if (!cancelled) { setControlError(message); setModelStatus("tracking-error"); } });
        poseRef.current = pose;
        await pose.ready;
        if (cancelled) { pose.close(); return; }
        setModelStatus("ready");
        setSceneReady(true);
        loop();
      } catch (err) {
        if (!cancelled) { setCameraError(err?.message || String(err)); setModelStatus("error"); }
      }
    }
    const resizeAll = () => {
      const stage = stageRef.current; if (!stage) return;
      const rect = stage.getBoundingClientRect();
      const w = Math.floor(rect.width), h = Math.floor(rect.height);
      if (overlayCanvasRef.current) { overlayCanvasRef.current.width = w; overlayCanvasRef.current.height = h; }
      if (sceneRef.current) sceneRef.current.resize(w, h);
    };
    boot();
    window.addEventListener("resize", resizeAll);
    const observer = new ResizeObserver(resizeAll);
    if (stageRef.current) observer.observe(stageRef.current);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", resizeAll);
      observer.disconnect();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      poseRef.current?.close?.();
      poseRef.current = null;
      sceneRef.current?.dispose();
      sceneRef.current = null;
    };
    // eslint-disable-next-line
  }, []);

  const selectedModel = garments.find((g) => g.id === session?.selected_garment)?.model;
  const selectedTint = garments.find((g) => g.id === session?.selected_garment)?.tint;
  useEffect(() => { sceneRef.current?.setTint(selectedTint || "#ffffff"); }, [selectedTint, sceneReady]);

  // Color changes never rebuild the simulation; async loading cannot unhide a cleared shirt.
  useEffect(() => {
    const scene = sceneRef.current; if (!scene || !sceneReady) return;
    if (!selectedModel) { scene.setVisible(false); setModelStatus("ready"); return; }
    const url = `${BACKEND_URL}${selectedModel}`;
    if (loadedGarmentIdRef.current === url) { setModelStatus("ready"); return; }
    let cancelled = false;
    setModelStatus("loading-shirt");
    scene.loadShirt(url).then(() => {
      if (!cancelled) { loadedGarmentIdRef.current = url; setModelStatus("ready"); }
    }).catch(() => { if (!cancelled) { setModelStatus("shirt-error"); setControlError("The 3D shirt could not load."); } });
    return () => { cancelled = true; scene.cancelLoad(); };
  }, [selectedModel, sceneReady]);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;
    const scene = sceneRef.current;
    if (!video || !overlay || !scene) { rafRef.current = requestAnimationFrame(loop); return; }
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) { rafRef.current = requestAnimationFrame(loop); return; }

    // FPS meter
    fpsRef.current.frames += 1;
    const now = performance.now();
    const dt = frameTimeRef.current === null ? 1 / 60 : (now - frameTimeRef.current) / 1000;
    frameTimeRef.current = now;
    if (now - fpsRef.current.last > 500) {
      fpsRef.current.value = Math.round((fpsRef.current.frames * 1000) / (now - fpsRef.current.last));
      fpsRef.current.frames = 0; fpsRef.current.last = now;
      setFps(fpsRef.current.value);
      setPoseCount(lastPoseRef.current.landmarks?.length || 0);
    }

    if (poseRef.current && video.currentTime !== lastVideoTs.current) {
      lastVideoTs.current = video.currentTime;
      poseRef.current.request(video, now);
    }
    // A worker may take longer than one video frame, especially on software GPUs.
    // Empty detections still hide immediately; only stalled workers get a grace window.
    const { landmarks, worldLandmarks } = now - lastPoseRef.current.at < 1000 ? lastPoseRef.current : { landmarks: null, worldLandmarks: null };

    const st = stateRef.current || {};
    if (st.selected_garment) scene.updateFromPose(landmarks, worldLandmarks, vw, vh,
      FIT_SCALE[st.fit_mode] || 1, dt, st.cloth_enabled !== false, st.view_mode || "auto");
    else scene.setVisible(false);
    scene.render();

    // Skeleton overlay on separate 2D canvas
    const ox = overlay.getContext("2d");
    ox.clearRect(0, 0, overlay.width, overlay.height);
    if (landmarks && st.show_skeleton) drawSkeleton(ox, landmarks, overlay.width, overlay.height, vw, vh);
    if (captureRequestRef.current) {
      const capture = captureRequestRef.current; captureRequestRef.current = null; capture();
    }

    rafRef.current = requestAnimationFrame(loop);
  }, []);

  const drawSkeleton = (cx, lm, w, h, vw, vh) => {
    // Video is CSS-mirrored, so overlay uses (1-x). Overlay canvas already sits above the mirrored video.
    const px = (i) => videoPoint(lm[i], vw, vh, w, h);
    const conn = [
      [11, 12], [11, 23], [12, 24], [23, 24],
      [11, 13], [13, 15],
      [12, 14], [14, 16],
      [23, 25], [25, 27],
      [24, 26], [26, 28],
    ];
    cx.strokeStyle = "rgba(226,241,59,.85)";
    cx.lineWidth = 3; cx.lineCap = "round";
    conn.forEach(([a, b]) => {
      if (!lm[a] || !lm[b]) return;
      const [ax, ay] = px(a), [bx, by] = px(b);
      cx.beginPath(); cx.moveTo(ax, ay); cx.lineTo(bx, by); cx.stroke();
    });
    cx.fillStyle = "#E2F13B";
    [11, 12, 23, 24, 13, 14, 15, 16, 25, 26, 27, 28].forEach((i) => {
      if (!lm[i]) return;
      const [x, y] = px(i);
      cx.beginPath(); cx.arc(x, y, 4, 0, Math.PI * 2); cx.fill();
    });
  };

  const copyLink = async () => {
    if (!session) return;
    const url = `${window.location.origin}/remote?s=${session.token}`;
    try { await navigator.clipboard.writeText(url); setCopied(true); setTimeout(() => setCopied(false), 1500); }
    catch { setControlError("Could not copy the link. Use the pairing QR instead."); }
  };

  const takeSnapshot = () => {
    captureRequestRef.current = () => {
    // Composite: video (mirrored) + Three.js + skeleton overlay
    const video = videoRef.current; const three = threeCanvasRef.current; const overlay = overlayCanvasRef.current;
    if (!video?.videoWidth || !three || !overlay) return;
    const w = three.width, h = three.height;
    const out = document.createElement("canvas"); out.width = w; out.height = h;
    const cx = out.getContext("2d");
    cx.save(); cx.translate(w, 0); cx.scale(-1, 1);
    const cover = videoCover(video.videoWidth, video.videoHeight, w, h);
    cx.drawImage(video, cover.x, cover.y, cover.width, cover.height);
    cx.restore();
    cx.drawImage(three, 0, 0, w, h);
    cx.drawImage(overlay, 0, 0, w, h);
    setSnapshot(out.toDataURL("image/png"));
    };
  };

  const currentGarment = garments.find((g) => g.id === session?.selected_garment);
  const phoneConnected = !!session?.phone_connected;

  return (
    <div className="min-h-screen bg-obsidian text-white relative overflow-hidden">
      {/* Top HUD */}
      <div className="mirror-hud absolute top-0 left-0 right-0 z-30 flex flex-wrap gap-2 items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <button data-testid="back-home-button" onClick={() => nav("/")} className="w-10 h-10 rounded-xl border border-white/10 bg-obsidian/70 backdrop-blur-xl grid place-items-center hover:bg-white/5">
            <HomeIcon className="w-4 h-4" />
          </button>
          <div className="rounded-full border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-acid pulse-dot" /> AURA FIT · 3D Mirror
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div data-testid="fps-counter" className="rounded-lg border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono">FPS {fps}</div>
          <div data-testid="pose-counter" className="rounded-lg border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono">POSE {poseCount}/33</div>
          <div data-testid="model-status" className="rounded-lg border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono uppercase">{modelStatus}</div>
          <div data-testid="cloth-status" role="status" className="rounded-lg border border-acid/30 bg-obsidian/80 px-3 py-1.5 text-[10px] font-mono text-acid">
            {session?.cloth_enabled === false ? "DRAPE OFF" : sceneRef.current?.tracking && poseCount && session?.selected_garment ? "DRAPE LIVE" : "DRAPE WAITING"}
          </div>
          <div data-testid="connection-status-pill" className={`rounded-full border px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest backdrop-blur-xl flex items-center gap-2 ${phoneConnected ? "border-acid/60 bg-acid/10 text-acid" : "border-white/10 bg-obsidian/70 text-muted"}`}>
            {phoneConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {phoneConnected ? (session?.phone_name || "Phone paired") : "Waiting for phone"}
          </div>
        </div>
      </div>

      {/* Stage: video (mirrored via CSS) + three.js canvas + skeleton overlay stacked */}
      <div ref={stageRef} className="mirror-stage relative w-full h-screen">
        <video
          ref={videoRef}
          playsInline muted
          className="absolute inset-0 z-0 w-full h-full object-cover bg-black"
          style={{ transform: "scaleX(-1)" }}
          data-testid="webcam-video"
        />
        <canvas ref={threeCanvasRef} data-testid="webcam-ar-canvas" className="absolute inset-0 z-[1] w-full h-full pointer-events-none" />
        <canvas ref={overlayCanvasRef} data-testid="skeleton-overlay-canvas" className="absolute inset-0 z-[2] w-full h-full pointer-events-none" />
        <div className="pointer-events-none absolute inset-0 z-[3] bg-gradient-to-b from-black/40 via-transparent to-black/60" />

        {cameraError && (
          <div data-testid="camera-error" role="alert" className="absolute inset-0 z-[4] grid place-items-center p-6">
            <div className="max-w-md rounded-3xl border border-white/10 bg-panel p-8 text-center">
              <Camera className="w-8 h-8 text-acid mx-auto mb-4" />
              <div className="font-display font-bold text-xl mb-2">Camera unavailable</div>
              <div className="text-sm text-muted">{cameraError}</div>
              <div className="text-xs text-muted mt-3">Grant camera permission and reload.</div>
            </div>
          </div>
        )}
        {controlError && <div data-testid="mirror-control-error" role="alert" className="absolute left-4 right-4 top-40 z-30 rounded-lg bg-red-950/90 p-3 text-sm">{controlError}</div>}

        <AnimatePresence>
          {showQrOverlay && session && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: .95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: .95 }}
              className="mirror-qr absolute top-24 right-6 z-20 rounded-3xl border border-white/10 bg-obsidian/85 backdrop-blur-xxl p-5 w-[280px] max-w-[calc(100%-2rem)] shadow-panel"
              data-testid="qr-pairing-panel"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-widest text-acid">PAIR PHONE</div>
                  <div className="font-display font-bold mt-1">Scan to control</div>
                </div>
                <button data-testid="qr-panel-dismiss" onClick={() => setShowQrOverlay(false)} className="w-8 h-8 grid place-items-center rounded-lg hover:bg-white/5"><X className="w-4 h-4" /></button>
              </div>
              <div className="rounded-2xl bg-obsidian border border-acid/20 p-2 grid place-items-center">
                <img src={qrUrl(session.token)} alt="Pair QR" className="w-full aspect-square rounded-lg" />
              </div>
              <div data-testid="pairing-code-display" className="mt-3 rounded-lg border border-white/10 bg-white/5 px-3 py-2 flex items-center justify-between font-mono text-xs">
                <span className="truncate">{session.token}</span>
                <button onClick={copyLink} className="text-acid hover:text-acidhover ml-2 flex items-center gap-1" data-testid="copy-remote-link">
                  {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />} {copied ? "Copied" : "Link"}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom control bar */}
        <div className="absolute left-0 right-0 bottom-0 z-20 p-4">
          <div className="mirror-controls mx-auto max-w-6xl rounded-3xl border border-white/10 bg-obsidian/80 backdrop-blur-xxl p-3 flex flex-wrap items-center gap-2">
            <div className="hidden md:flex items-center gap-3 pr-3 border-r border-white/10 min-w-[220px]">
              <div className="w-11 h-11 rounded-xl grid place-items-center overflow-hidden" style={{ background: currentGarment?.tint || "rgba(255,255,255,.08)" }}>
                <span className="text-[10px] font-mono">3D</span>
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase tracking-widest text-acid">NOW WEARING</div>
                <div data-testid="current-garment-name" className="text-sm font-medium truncate max-w-[160px]">{currentGarment?.name || "— nothing —"}</div>
              </div>
            </div>

            <div className="mirror-wardrobe flex-1 min-w-0 flex items-center gap-2 overflow-x-auto">
              {garments.map((g) => (
                <button
                  key={g.id}
                  data-testid={`quick-garment-${g.id}`}
                  onClick={() => applyPatch({ selected_garment: g.id })}
                  className={`flex-none w-12 h-12 rounded-xl border transition grid place-items-center text-[10px] font-mono ${session?.selected_garment === g.id ? "border-acid shadow-glow" : "border-white/10 hover:border-white/30"}`}
                  style={{ background: g.tint || "#fff", color: pickReadableText(g.tint) }}
                  title={g.name}
                >
                  {g.id.slice(0, 2).toUpperCase()}
                </button>
              ))}
            </div>

            <ClothToggle device="mirror" compact enabled={session?.cloth_enabled !== false} onChange={(cloth_enabled) => applyPatch({ cloth_enabled })} />
            <button data-testid="remove-garment-button" onClick={() => applyPatch({ clear_garment: true })} className="rounded-xl border border-white/10 px-3 py-2 text-xs text-white/80 hover:bg-white/5">Clear</button>
            <button data-testid="skeleton-toggle-button" onClick={() => applyPatch({ show_skeleton: !session?.show_skeleton })} className={`rounded-xl border px-3 py-2 text-xs transition ${session?.show_skeleton ? "border-acid/60 bg-acid/10 text-acid" : "border-white/10 text-white/80 hover:bg-white/5"}`}>Skeleton</button>
            <button data-testid="show-qr-button" onClick={() => setShowQrOverlay((s) => !s)} className="rounded-xl border border-white/10 px-3 py-2 text-xs text-white/80 hover:bg-white/5 flex items-center gap-1"><QrIcon className="w-3.5 h-3.5" /> QR</button>
            <button data-testid="take-snapshot-button" onClick={takeSnapshot} disabled={!sceneReady || !!cameraError} className="rounded-xl bg-acid text-obsidian px-4 py-2 text-xs font-semibold hover:bg-acidhover disabled:opacity-50 flex items-center gap-1"><Camera className="w-3.5 h-3.5" /> Capture</button>
          </div>
        </div>

        <AnimatePresence>
          {snapshot && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 z-40 bg-obsidian/85 backdrop-blur-xl grid place-items-center p-6" data-testid="snapshot-modal">
              <motion.div initial={{ scale: .9 }} animate={{ scale: 1 }} className="max-w-3xl w-full rounded-3xl border border-white/10 bg-panel p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="text-[10px] font-mono uppercase tracking-widest text-acid">SNAPSHOT</div>
                    <div className="font-display font-bold text-xl mt-1">Your look</div>
                  </div>
                  <button data-testid="close-snapshot" onClick={() => setSnapshot(null)} className="w-10 h-10 rounded-xl border border-white/10 grid place-items-center hover:bg-white/5"><X className="w-4 h-4" /></button>
                </div>
                <img src={snapshot} alt="snapshot" className="w-full rounded-xl border border-white/10" />
                <div className="mt-4 flex gap-3 justify-end">
                  <a data-testid="download-snapshot" href={snapshot} download={`aura-fit-${Date.now()}.png`} className="rounded-full bg-acid text-obsidian px-5 py-2.5 text-sm font-semibold hover:bg-acidhover inline-flex items-center gap-2"><Download className="w-4 h-4" /> Download</a>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

// Pick readable text color (white/dark) for a background tint
function pickReadableText(hex) {
  if (!hex) return "#111";
  const c = hex.replace("#", "");
  const r = parseInt(c.substr(0, 2), 16), g = parseInt(c.substr(2, 2), 16), b = parseInt(c.substr(4, 2), 16);
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return lum > 0.6 ? "#111318" : "#ffffff";
}
