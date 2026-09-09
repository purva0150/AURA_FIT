import React, { useEffect, useRef, useState, useCallback } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, X, Download, Home as HomeIcon, Copy, Check, QrCode as QrIcon, Wifi, WifiOff } from "lucide-react";
import { fetchGarments, getSession, updateSession, qrUrl } from "../api";
import { createPoseLandmarker, affineFromCorners } from "../lib/pose";

const FIT_SCALE = { fitted: 0.94, regular: 1.0, relaxed: 1.08 };

export default function Mirror() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const token = params.get("s");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const lastVideoTs = useRef(-1);
  const poseRef = useRef(null);
  const streamRef = useRef(null);
  const garmentImgCache = useRef({});
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

  // Redirect if no token
  useEffect(() => {
    if (!token) nav("/");
  }, [token, nav]);

  // Initial load: garments + session
  useEffect(() => {
    if (!token) return;
    fetchGarments().then(setGarments);
    getSession(token).then((s) => { setSession(s); stateRef.current = s; }).catch(() => nav("/"));
  }, [token, nav]);

  // Poll session for phone updates
  useEffect(() => {
    if (!token) return;
    const iv = setInterval(async () => {
      try {
        const s = await getSession(token);
        setSession(s);
        stateRef.current = s;
      } catch {}
    }, 900);
    return () => clearInterval(iv);
  }, [token]);

  // Preload garment images (with tint variants)
  useEffect(() => {
    garments.forEach((g) => {
      if (garmentImgCache.current[g.id]) return;
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = `${process.env.REACT_APP_BACKEND_URL}${g.image}`;
      garmentImgCache.current[g.id] = { img, ready: false, tintedCanvas: null };
      img.onload = () => {
        garmentImgCache.current[g.id].ready = true;
        // pre-render tinted variant if needed
        if (g.tint) {
          const c = document.createElement("canvas");
          c.width = img.naturalWidth; c.height = img.naturalHeight;
          const cx = c.getContext("2d");
          cx.drawImage(img, 0, 0);
          cx.globalCompositeOperation = "multiply";
          cx.fillStyle = g.tint;
          cx.fillRect(0, 0, c.width, c.height);
          // preserve original alpha
          cx.globalCompositeOperation = "destination-in";
          cx.drawImage(img, 0, 0);
          garmentImgCache.current[g.id].tintedCanvas = c;
        }
      };
    });
  }, [garments]);

  // Setup camera + mediapipe
  useEffect(() => {
    let cancelled = false;
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
        const pose = await createPoseLandmarker();
        poseRef.current = pose;
        loop();
      } catch (err) {
        setCameraError(err?.message || String(err));
      }
    }
    boot();
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      poseRef.current?.close?.();
    };
    // eslint-disable-next-line
  }, []);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) { rafRef.current = requestAnimationFrame(loop); return; }
    const cx = canvas.getContext("2d");
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) { rafRef.current = requestAnimationFrame(loop); return; }
    if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }

    // FPS
    fpsRef.current.frames += 1;
    const now = performance.now();
    if (now - fpsRef.current.last > 500) {
      fpsRef.current.value = Math.round((fpsRef.current.frames * 1000) / (now - fpsRef.current.last));
      fpsRef.current.frames = 0; fpsRef.current.last = now;
      setFps(fpsRef.current.value);
    }

    // Mirror flip: draw video horizontally flipped
    cx.save();
    cx.translate(vw, 0); cx.scale(-1, 1);
    cx.drawImage(video, 0, 0, vw, vh);
    cx.restore();

    // Pose detect
    let landmarks = null;
    if (poseRef.current && video.currentTime !== lastVideoTs.current) {
      lastVideoTs.current = video.currentTime;
      const res = poseRef.current.detectForVideo(video, now);
      if (res?.landmarks?.length) landmarks = res.landmarks[0];
    }
    setPoseCount(landmarks ? landmarks.length : 0);

    // Because we mirror-flipped, landmark X must also be flipped for overlay
    const st = stateRef.current || {};
    const gId = st.selected_garment;
    const g = garments.find((x) => x.id === gId);
    const cache = g ? garmentImgCache.current[g.id] : null;

    if (landmarks && g && cache?.ready) {
      const LS = landmarks[11], RS = landmarks[12], LH = landmarks[23], RH = landmarks[24];
      if (LS && RS && LH && RH) {
        const scale = FIT_SCALE[st.fit_mode || "regular"] || 1;
        // Get mirrored (flipped) points in canvas coords
        const p = (lm) => [(1 - lm.x) * vw, lm.y * vh];
        // Left/right shoulder swap because of mirror
        const l_sh = p(RS), r_sh = p(LS), l_hip = p(RH), r_hip = p(LH);

        // Anchors from garment metadata
        const a = g.anchors;
        // Note: In garment JSON, left_shoulder is on the LEFT side of the image (the model's right).
        // We treat them as raw image points and map to detected points 1:1.
        const src = [a.right_shoulder, a.left_shoulder, a.right_hip, a.left_hip];
        // Apply fit scale by widening/narrowing hip-shoulder line about centre
        const centerX = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4;
        const centerY = (l_sh[1] + r_sh[1] + l_hip[1] + r_hip[1]) / 4;
        const applyScale = (pt) => [centerX + (pt[0] - centerX) * scale, centerY + (pt[1] - centerY) * scale];
        const dst = [applyScale(r_sh), applyScale(l_sh), applyScale(r_hip), applyScale(l_hip)];

        const M = affineFromCorners(src, dst);
        if (M) {
          const source = cache.tintedCanvas || cache.img;
          cx.save();
          cx.globalAlpha = 0.98;
          cx.setTransform(M.a, M.b, M.c, M.d, M.e, M.f);
          cx.drawImage(source, 0, 0);
          cx.setTransform(1, 0, 0, 1, 0, 0);
          cx.restore();
        }
      }
    }

    // Skeleton overlay
    if (landmarks && st.show_skeleton) drawSkeleton(cx, landmarks, vw, vh);

    rafRef.current = requestAnimationFrame(loop);
  }, [garments]);

  const drawSkeleton = (cx, lm, w, h) => {
    const px = (i) => [(1 - lm[i].x) * w, lm[i].y * h];
    const conn = [
      [11, 12], [11, 23], [12, 24], [23, 24],
      [11, 13], [13, 15],
      [12, 14], [14, 16],
      [23, 25], [25, 27], [27, 29], [29, 31],
      [24, 26], [26, 28], [28, 30], [30, 32],
    ];
    cx.save();
    cx.strokeStyle = "rgba(226,241,59,.85)";
    cx.lineWidth = 3;
    cx.lineCap = "round";
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
    cx.restore();
  };

  const copyLink = async () => {
    if (!session) return;
    const url = `${window.location.origin}/remote?s=${session.token}`;
    await navigator.clipboard.writeText(url);
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };

  const takeSnapshot = () => {
    const canvas = canvasRef.current; if (!canvas) return;
    const url = canvas.toDataURL("image/png");
    setSnapshot(url);
  };

  const currentGarment = garments.find((g) => g.id === session?.selected_garment);
  const phoneConnected = !!session?.phone_connected;

  return (
    <div className="min-h-screen bg-obsidian text-white relative overflow-hidden">
      {/* Top HUD */}
      <div className="absolute top-0 left-0 right-0 z-30 flex items-center justify-between px-5 py-4">
        <div className="flex items-center gap-3">
          <button data-testid="back-home-button" onClick={() => nav("/")} className="w-10 h-10 rounded-xl border border-white/10 bg-obsidian/70 backdrop-blur-xl grid place-items-center hover:bg-white/5">
            <HomeIcon className="w-4 h-4" />
          </button>
          <div className="rounded-full border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-acid pulse-dot" /> AURA FIT · Mirror
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div data-testid="fps-counter" className="rounded-lg border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono">FPS {fps}</div>
          <div data-testid="pose-counter" className="rounded-lg border border-white/10 bg-obsidian/70 backdrop-blur-xl px-3 py-1.5 text-[10px] font-mono">POSE {poseCount}/33</div>
          <div data-testid="connection-status-pill" className={`rounded-full border px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest backdrop-blur-xl flex items-center gap-2 ${phoneConnected ? "border-acid/60 bg-acid/10 text-acid" : "border-white/10 bg-obsidian/70 text-muted"}`}>
            {phoneConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {phoneConnected ? (session?.phone_name || "Phone paired") : "Waiting for phone"}
          </div>
        </div>
      </div>

      {/* Video + canvas stage */}
      <div className="relative w-full h-screen">
        <video ref={videoRef} playsInline muted className="hidden" />
        <canvas data-testid="webcam-ar-canvas" ref={canvasRef} className="absolute inset-0 w-full h-full object-cover bg-black" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black/50 via-transparent to-black/60" />

        {cameraError && (
          <div className="absolute inset-0 grid place-items-center p-6">
            <div className="max-w-md rounded-3xl border border-white/10 bg-panel p-8 text-center">
              <Camera className="w-8 h-8 text-acid mx-auto mb-4" />
              <div className="font-display font-bold text-xl mb-2">Camera unavailable</div>
              <div className="text-sm text-muted">{cameraError}</div>
              <div className="text-xs text-muted mt-3">Grant camera permission and reload.</div>
            </div>
          </div>
        )}

        {/* QR Pairing Panel (floating, dismissible) */}
        <AnimatePresence>
          {showQrOverlay && session && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: .95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: .95 }}
              className="absolute top-24 right-6 z-20 rounded-3xl border border-white/10 bg-obsidian/85 backdrop-blur-xxl p-5 w-[280px] shadow-panel"
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
              <p className="text-[11px] text-muted mt-3 leading-relaxed">Camera & tracking stay on this laptop. Nothing is uploaded.</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom control bar */}
        <div className="absolute left-0 right-0 bottom-0 z-20 p-4">
          <div className="mx-auto max-w-5xl rounded-3xl border border-white/10 bg-obsidian/80 backdrop-blur-xxl p-4 flex items-center gap-3">
            {/* Current garment badge */}
            <div className="hidden md:flex items-center gap-3 pr-3 border-r border-white/10 min-w-[220px]">
              <div className="w-11 h-11 rounded-xl bg-white/10 grid place-items-center overflow-hidden" style={{ background: currentGarment?.tint || "rgba(255,255,255,.08)" }}>
                {currentGarment && (
                  <img src={`${process.env.REACT_APP_BACKEND_URL}${currentGarment.image}`} className="w-full h-full object-contain mix-blend-multiply" alt="" />
                )}
              </div>
              <div>
                <div className="text-[10px] font-mono uppercase tracking-widest text-acid">NOW WEARING</div>
                <div className="text-sm font-medium truncate max-w-[160px]">{currentGarment?.name || "— nothing —"}</div>
              </div>
            </div>

            {/* Quick swap thumbnails */}
            <div className="flex-1 flex items-center gap-2 overflow-x-auto">
              {garments.map((g) => (
                <button
                  key={g.id}
                  data-testid={`quick-garment-${g.id}`}
                  onClick={() => updateSession(token, { selected_garment: g.id, source: "mirror" }).then(setSession)}
                  className={`flex-none w-12 h-12 rounded-xl border overflow-hidden grid place-items-center transition ${session?.selected_garment === g.id ? "border-acid shadow-glow" : "border-white/10 hover:border-white/30"}`}
                  style={{ background: g.tint || "#fff" }}
                  title={g.name}
                >
                  <img src={`${process.env.REACT_APP_BACKEND_URL}${g.image}`} alt="" className="w-full h-full object-contain mix-blend-multiply" />
                </button>
              ))}
            </div>

            <button
              data-testid="remove-garment-button"
              onClick={() => updateSession(token, { clear_garment: true, source: "mirror" }).then(setSession)}
              className="rounded-xl border border-white/10 px-3 py-2 text-xs text-white/80 hover:bg-white/5"
            >
              Clear
            </button>
            <button
              data-testid="skeleton-toggle-button"
              onClick={() => updateSession(token, { show_skeleton: !session?.show_skeleton, source: "mirror" }).then(setSession)}
              className={`rounded-xl border px-3 py-2 text-xs transition ${session?.show_skeleton ? "border-acid/60 bg-acid/10 text-acid" : "border-white/10 text-white/80 hover:bg-white/5"}`}
            >
              Skeleton
            </button>
            <button
              data-testid="show-qr-button"
              onClick={() => setShowQrOverlay((s) => !s)}
              className="rounded-xl border border-white/10 px-3 py-2 text-xs text-white/80 hover:bg-white/5 flex items-center gap-1"
            >
              <QrIcon className="w-3.5 h-3.5" /> QR
            </button>
            <button
              data-testid="take-snapshot-button"
              onClick={takeSnapshot}
              className="rounded-xl bg-acid text-obsidian px-4 py-2 text-xs font-semibold hover:bg-acidhover flex items-center gap-1"
            >
              <Camera className="w-3.5 h-3.5" /> Capture
            </button>
          </div>
        </div>

        {/* Snapshot modal */}
        <AnimatePresence>
          {snapshot && (
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="absolute inset-0 z-40 bg-obsidian/85 backdrop-blur-xl grid place-items-center p-6"
              data-testid="snapshot-modal"
            >
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
