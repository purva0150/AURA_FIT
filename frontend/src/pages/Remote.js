import React, { useEffect, useMemo, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Wifi, Shirt, Sparkles, Circle, X, ChevronLeft, ChevronRight } from "lucide-react";
import { fetchGarments, getSession, updateSession, heartbeat } from "../api";

const FITS = [
  { key: "fitted", label: "Fitted", copy: "Snug" },
  { key: "regular", label: "Regular", copy: "Standard" },
  { key: "relaxed", label: "Relaxed", copy: "Loose" },
];
const VIEWS = [
  { key: "auto", label: "Auto" },
  { key: "front", label: "Front" },
  { key: "back", label: "Back" },
];

export default function Remote() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const token = params.get("s");
  const [garments, setGarments] = useState([]);
  const [session, setSession] = useState(null);
  const [error, setError] = useState(null);
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => { if (!token) nav("/"); }, [token, nav]);

  useEffect(() => {
    if (!token) return;
    fetchGarments().then(setGarments);
    getSession(token).then(setSession).catch(() => setError("Session expired. Ask the laptop to show a new QR."));
  }, [token]);

  // Heartbeat every 3s (keeps phone_connected true)
  useEffect(() => {
    if (!token) return;
    const name = /iPhone|iPad|iPod/i.test(navigator.userAgent) ? "iPhone" :
      /Android/i.test(navigator.userAgent) ? "Android phone" : "Mobile controller";
    const beat = () => heartbeat(token, name).then(setSession).catch(() => {});
    beat();
    const iv = setInterval(beat, 3000);
    return () => clearInterval(iv);
  }, [token]);

  const applyPatch = async (patch) => {
    try { setSession(await updateSession(token, { ...patch, source: "phone" })); } catch {}
  };

  const currentIdx = useMemo(() => {
    if (!session?.selected_garment) return -1;
    return garments.findIndex((g) => g.id === session.selected_garment);
  }, [garments, session]);

  const swipe = (dir) => {
    if (!garments.length) return;
    const start = currentIdx >= 0 ? currentIdx : activeIdx;
    const next = (start + dir + garments.length) % garments.length;
    setActiveIdx(next);
    applyPatch({ selected_garment: garments[next].id });
  };

  const cardG = garments[activeIdx] || (currentIdx >= 0 ? garments[currentIdx] : garments[0]);

  if (error) {
    return (
      <div className="min-h-screen bg-obsidian grid place-items-center p-6 text-center">
        <div className="max-w-sm rounded-3xl border border-white/10 bg-panel p-8">
          <div className="w-10 h-10 rounded-xl bg-acid text-obsidian grid place-items-center font-display font-black mx-auto mb-3">AF</div>
          <div className="font-display font-bold text-2xl">Session expired</div>
          <p className="text-muted mt-2 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-obsidian text-white pb-24">
      {/* Header */}
      <div className="px-5 pt-6 pb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl bg-acid text-obsidian grid place-items-center font-display font-black">AF</div>
          <div className="leading-tight">
            <div className="font-display font-bold text-sm">AURA FIT · Remote</div>
            <div className="text-[10px] uppercase tracking-widest text-muted font-mono">Session {token?.slice(0,6)}</div>
          </div>
        </div>
        <div data-testid="phone-connection-pill" className="rounded-full border border-acid/40 bg-acid/10 px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-acid flex items-center gap-1.5">
          <Wifi className="w-3 h-3" /> Paired
        </div>
      </div>

      {/* Intro */}
      <div className="px-5">
        <div className="text-[10px] font-mono uppercase tracking-widest text-acid">Your Remote Wardrobe</div>
        <h1 className="font-display font-black text-4xl tracking-tight leading-[1] mt-2">What do you want <span className="italic text-acid">to try?</span></h1>
        <p className="text-muted text-sm mt-3">Every tap reflects instantly on the mirror.</p>
      </div>

      {/* Hero card / swipe stack */}
      <div className="px-5 mt-6">
        <motion.div
          key={cardG?.id || "empty"}
          initial={{ opacity: 0, scale: .96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative rounded-3xl border border-white/10 bg-surface overflow-hidden"
          data-testid="remote-hero-card"
        >
          <div className="aspect-[4/5] grid place-items-center relative" style={{ background: cardG?.tint ? `linear-gradient(160deg, ${cardG.tint}22, #0a0a10)` : "linear-gradient(160deg, #1a1d2b, #0a0a10)" }}>
            {cardG && (
              <div className="relative w-3/5 aspect-square grid place-items-center rounded-3xl"
                   style={{ background: `radial-gradient(circle at 30% 30%, ${cardG.tint}, ${cardG.tint}66 60%, transparent 80%)`,
                            boxShadow: `0 40px 90px -20px ${cardG.tint}66` }}>
                <svg viewBox="0 0 100 110" className="w-4/5 h-4/5">
                  <path fill={cardG.tint} d="M20 20 L38 12 L50 22 L62 12 L80 20 L86 40 L74 44 L74 96 L26 96 L26 44 L14 40 Z" stroke="rgba(255,255,255,.35)" strokeWidth="1.2"/>
                  <path fill="rgba(0,0,0,.12)" d="M14 40 L26 44 L26 96 L20 96 Z"/>
                </svg>
                <div className="absolute bottom-3 right-3 text-[9px] font-mono uppercase tracking-widest px-2 py-1 rounded-full bg-black/50 text-white border border-white/20">GLB · 3D</div>
              </div>
            )}
            <div className="absolute top-3 left-3 text-[10px] font-mono uppercase tracking-widest px-2 py-1 rounded-full bg-obsidian/70 border border-white/10">
              {cardG?.category || "—"}
            </div>
            <button data-testid="swipe-prev" onClick={() => swipe(-1)} className="absolute left-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-obsidian/70 border border-white/10 grid place-items-center backdrop-blur-xl">
              <ChevronLeft className="w-5 h-5" />
            </button>
            <button data-testid="swipe-next" onClick={() => swipe(1)} className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-obsidian/70 border border-white/10 grid place-items-center backdrop-blur-xl">
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
          <div className="p-4 flex items-center justify-between">
            <div>
              <div className="font-display font-bold text-lg leading-tight">{cardG?.name || "Pick something"}</div>
              <div className="text-[11px] font-mono uppercase tracking-widest text-muted mt-0.5">{cardG?.color || "—"}</div>
            </div>
            <button
              data-testid="apply-current-garment"
              onClick={() => cardG && applyPatch({ selected_garment: cardG.id })}
              className={`rounded-full px-5 py-2.5 text-sm font-semibold transition ${session?.selected_garment === cardG?.id ? "bg-white/10 text-white" : "bg-acid text-obsidian hover:bg-acidhover"}`}
            >
              {session?.selected_garment === cardG?.id ? "Wearing" : "Try on"}
            </button>
          </div>
        </motion.div>
      </div>

      {/* Wardrobe grid */}
      <div className="px-5 mt-8">
        <div className="flex items-baseline justify-between mb-3">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-acid">Wardrobe</div>
            <div className="font-display font-bold text-xl">All pieces</div>
          </div>
          <div className="text-xs text-muted">{garments.length} items</div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          {garments.map((g, i) => (
            <button
              key={g.id}
              data-testid="garment-card-item"
              onClick={() => { setActiveIdx(i); applyPatch({ selected_garment: g.id }); }}
              className={`rounded-2xl border p-2 text-left transition ${session?.selected_garment === g.id ? "border-acid shadow-glow" : "border-white/10 bg-surface/80 hover:border-white/30"}`}
            >
              <div className="aspect-square rounded-xl grid place-items-center overflow-hidden relative"
                   style={{ background: `linear-gradient(140deg, ${g.tint}, ${g.tint}77)` }}>
                <svg viewBox="0 0 100 110" className="w-3/5 h-3/5">
                  <path fill="rgba(255,255,255,.92)" d="M20 20 L38 12 L50 22 L62 12 L80 20 L86 40 L74 44 L74 96 L26 96 L26 44 L14 40 Z" stroke="rgba(0,0,0,.12)" strokeWidth="1"/>
                </svg>
              </div>
              <div className="text-[11px] font-medium mt-2 truncate">{g.name}</div>
              <div className="text-[9px] font-mono uppercase tracking-widest text-muted mt-0.5 truncate">{g.color}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="px-5 mt-8 space-y-4">
        <div className="rounded-2xl border border-white/10 bg-surface/80 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Shirt className="w-4 h-4 text-acid" />
            <div>
              <div className="text-sm font-semibold">Garment fit</div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted">Chest & waist drape</div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {FITS.map((f) => (
              <button
                key={f.key}
                data-testid={`fit-control-${f.key}`}
                onClick={() => applyPatch({ fit_mode: f.key })}
                className={`rounded-xl border py-3 text-center transition ${session?.fit_mode === f.key ? "border-acid bg-acid/10 text-acid" : "border-white/10 text-white/80 hover:bg-white/5"}`}
              >
                <div className="text-sm font-semibold">{f.label}</div>
                <div className="text-[10px] font-mono uppercase tracking-widest opacity-70 mt-0.5">{f.copy}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-surface/80 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles className="w-4 h-4 text-acid" />
            <div>
              <div className="text-sm font-semibold">Body view</div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted">Try Back if you turn around</div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {VIEWS.map((v) => (
              <button
                key={v.key}
                data-testid={`angle-view-${v.key}`}
                onClick={() => applyPatch({ view_mode: v.key })}
                className={`rounded-xl border py-2.5 text-sm transition ${session?.view_mode === v.key ? "border-acid bg-acid/10 text-acid font-semibold" : "border-white/10 text-white/80 hover:bg-white/5"}`}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-surface/80 p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Circle className="w-4 h-4 text-acid" />
            <div>
              <div className="text-sm font-semibold">Skeleton overlay</div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted">Helpful for positioning</div>
            </div>
          </div>
          <button
            data-testid="skeleton-toggle-phone"
            onClick={() => applyPatch({ show_skeleton: !session?.show_skeleton })}
            className={`relative w-12 h-7 rounded-full border transition ${session?.show_skeleton ? "bg-acid border-acid" : "bg-white/10 border-white/20"}`}
          >
            <span className={`absolute top-0.5 w-6 h-6 rounded-full bg-white transition-all ${session?.show_skeleton ? "left-[22px] bg-obsidian" : "left-0.5"}`} />
          </button>
        </div>

        <button
          data-testid="remove-garment-phone"
          onClick={() => applyPatch({ clear_garment: true })}
          className="w-full rounded-2xl border border-white/10 py-4 text-sm text-white/80 hover:bg-white/5 flex items-center justify-center gap-2"
        >
          <X className="w-4 h-4" /> Remove current garment
        </button>
      </div>

      <div className="px-5 mt-8 text-center text-[10px] font-mono uppercase tracking-widest text-dim">
        Camera runs on the laptop · nothing uploaded
      </div>
    </div>
  );
}
