import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowUpRight, ScanLine, Sparkles, Zap, Camera, Smartphone } from "lucide-react";
import { createSession, fetchGarments } from "../api";

const features = [
  { icon: ScanLine, title: "33-Point Body Skeleton", copy: "Real-time MediaPipe pose tracking runs entirely in your browser — no upload, no lag." },
  { icon: Smartphone, title: "QR-Paired Remote", copy: "Scan once from your phone and swipe outfits without ever touching the mirror." },
  { icon: Sparkles, title: "Adaptive Cloth Warp", copy: "Garments snap to shoulders & hips with fitted / regular / relaxed drape modes." },
  { icon: Zap, title: "Instant Snapshot", copy: "Capture and share your favourite fit — right from the runway." },
];

export default function Home() {
  const nav = useNavigate();
  const [garments, setGarments] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchGarments().then(setGarments).catch(() => {});
  }, []);

  const startMirror = async () => {
    setLoading(true);
    try {
      const { session } = await createSession();
      nav(`/mirror?s=${session.token}`);
    } catch (e) {
      alert("Failed to create session — try again.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-obsidian text-white relative overflow-hidden grain">
      {/* Nav */}
      <nav className="relative z-20 flex items-center justify-between px-6 lg:px-12 py-6">
        <div data-testid="nav-brand-logo" className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-acid text-obsidian grid place-items-center font-display font-black text-lg">AF</div>
          <div className="leading-tight">
            <div className="font-display font-bold tracking-tight">AURA FIT</div>
            <div className="text-[10px] uppercase tracking-[0.25em] text-muted font-mono">Fitting Room · 01</div>
          </div>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm text-muted">
          <a href="#features" className="hover:text-white transition">Capabilities</a>
          <a href="#wardrobe" className="hover:text-white transition">Wardrobe</a>
          <a href="#how" className="hover:text-white transition">How it works</a>
        </div>
        <button
          data-testid="launch-mirror-button-nav"
          onClick={startMirror}
          disabled={loading}
          className="hidden sm:inline-flex items-center gap-2 rounded-full bg-acid text-obsidian px-5 py-2.5 font-semibold text-sm hover:bg-acidhover transition disabled:opacity-60"
        >
          {loading ? "Booting…" : "Launch Mirror"} <ArrowUpRight className="w-4 h-4" />
        </button>
      </nav>

      {/* Hero */}
      <section className="relative z-10 px-6 lg:px-12 pt-6 lg:pt-10 pb-24 grid lg:grid-cols-12 gap-10 items-center">
        <div className="lg:col-span-7">
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .7 }}
            className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] font-mono uppercase tracking-widest text-muted mb-6"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-acid pulse-dot" /> Real-time AR · WebGL Canvas
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .05, duration: .7 }}
            className="font-display font-black text-5xl md:text-6xl lg:text-7xl leading-[.92] tracking-tight"
          >
            The runway now <br className="hidden md:block" />
            <span className="italic text-acid">fits inside</span> your laptop.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .12, duration: .7 }}
            className="mt-6 max-w-xl text-lg text-muted leading-relaxed"
          >
            Body-tracked virtual fitting room. Your laptop becomes a mirror. Your phone becomes the remote — swipe garments, adjust the drape, capture the look.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .2, duration: .7 }}
            className="mt-10 flex flex-wrap items-center gap-4"
          >
            <button
              data-testid="launch-mirror-button"
              onClick={startMirror}
              disabled={loading}
              className="group inline-flex items-center gap-3 rounded-full bg-acid text-obsidian px-7 py-4 font-semibold hover:bg-acidhover transition disabled:opacity-60"
            >
              <Camera className="w-5 h-5" /> {loading ? "Booting camera…" : "Launch Laptop Mirror"}
              <ArrowUpRight className="w-4 h-4 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition" />
            </button>
            <a
              href="#how"
              data-testid="how-it-works-link"
              className="inline-flex items-center gap-2 rounded-full border border-white/15 px-6 py-4 text-sm text-white/80 hover:bg-white/5 transition"
            >
              How the QR pairing works
            </a>
          </motion.div>

          <div className="mt-14 grid grid-cols-3 gap-6 max-w-lg">
            {[
              ["60 FPS", "canvas render loop"],
              ["33 pts", "pose landmarks"],
              ["<200 ms", "phone-to-mirror sync"],
            ].map(([n, l]) => (
              <div key={l}>
                <div className="font-display font-bold text-2xl md:text-3xl">{n}</div>
                <div className="text-xs text-muted mt-1">{l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Hero visual */}
        <motion.div
          initial={{ opacity: 0, scale: .96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .8 }}
          className="lg:col-span-5 relative"
        >
          <div className="relative aspect-[4/5] rounded-3xl overflow-hidden border border-white/10 bg-panel">
            <img
              src="https://images.unsplash.com/photo-1659522761084-79196b64abe4?crop=entropy&cs=srgb&fm=jpg&w=900&q=85"
              alt="Model wearing tracked garment"
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-obsidian/70" />
            {/* HUD */}
            <div className="absolute top-4 left-4 right-4 flex items-start justify-between">
              <div className="rounded-full bg-obsidian/70 backdrop-blur-xl border border-white/10 px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-acid pulse-dot" /> Tracking · 33/33
              </div>
              <div className="rounded-lg bg-obsidian/70 backdrop-blur-xl border border-white/10 px-3 py-1.5 text-[10px] font-mono">FPS 61</div>
            </div>
            <div className="absolute left-4 right-4 bottom-4 rounded-2xl bg-obsidian/70 backdrop-blur-xl border border-white/10 p-4">
              <div className="text-[10px] font-mono uppercase tracking-widest text-acid">NOW WEARING</div>
              <div className="font-display font-semibold mt-1">AURA Oversized Heavy Tee · Essential White</div>
            </div>
            {/* Skeleton overlay */}
            <svg className="absolute inset-0 w-full h-full skeleton-pulse" viewBox="0 0 300 400" fill="none" stroke="#E2F13B" strokeWidth="1.2" opacity=".85">
              <circle cx="150" cy="70" r="12" />
              <line x1="150" y1="82" x2="150" y2="180" />
              <line x1="150" y1="110" x2="95"  y2="150" />
              <line x1="150" y1="110" x2="205" y2="150" />
              <line x1="95"  y1="150" x2="80"  y2="215" />
              <line x1="205" y1="150" x2="220" y2="215" />
              <line x1="150" y1="180" x2="125" y2="260" />
              <line x1="150" y1="180" x2="175" y2="260" />
              <line x1="125" y1="260" x2="120" y2="340" />
              <line x1="175" y1="260" x2="180" y2="340" />
            </svg>
          </div>
          <div className="absolute -bottom-6 -left-6 rounded-2xl bg-surface border border-white/10 p-3 shadow-panel">
            <div className="text-[10px] font-mono uppercase tracking-widest text-muted">Phone Remote</div>
            <div className="mt-1 font-display font-semibold text-sm flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-acid pulse-dot" /> Sofia's iPhone · paired
            </div>
          </div>
        </motion.div>
      </section>

      {/* Marquee */}
      <div className="relative border-y border-white/10 bg-panel/60 overflow-hidden">
        <div className="marquee-track flex whitespace-nowrap py-4 gap-14 text-2xl md:text-3xl font-display font-bold text-white/40 uppercase tracking-tight">
          {Array.from({ length: 2 }).flatMap((_, i) => [
            <span key={`a${i}`}>Fit / Track / Swipe</span>,
            <span key={`b${i}`} className="text-acid">✦</span>,
            <span key={`c${i}`}>Editorial AR Runway</span>,
            <span key={`d${i}`} className="text-acid">✦</span>,
            <span key={`e${i}`}>Phone-Paired Wardrobe</span>,
            <span key={`f${i}`} className="text-acid">✦</span>,
            <span key={`g${i}`}>60FPS Body Skeleton</span>,
            <span key={`h${i}`} className="text-acid">✦</span>,
          ])}
        </div>
      </div>

      {/* Features */}
      <section id="features" className="px-6 lg:px-12 py-24">
        <div className="max-w-2xl mb-14">
          <div className="text-[11px] font-mono uppercase tracking-widest text-acid mb-3">CAPABILITIES</div>
          <h2 className="font-display font-bold text-4xl md:text-5xl tracking-tight">Everything a fitting room does — <span className="italic text-acid">without the mirror.</span></h2>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
          {features.map(({ icon: Icon, title, copy }, i) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }} transition={{ delay: i * .06 }}
              className="group rounded-3xl border border-white/10 bg-surface/60 backdrop-blur-xl p-6 hover:border-acid/50 transition"
            >
              <Icon className="w-6 h-6 text-acid" />
              <div className="mt-6 font-display font-semibold text-xl">{title}</div>
              <p className="mt-2 text-sm text-muted leading-relaxed">{copy}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Wardrobe preview */}
      <section id="wardrobe" className="px-6 lg:px-12 py-16">
        <div className="flex items-end justify-between mb-10">
          <div>
            <div className="text-[11px] font-mono uppercase tracking-widest text-acid mb-3">DROP · SS26</div>
            <h2 className="font-display font-bold text-4xl md:text-5xl tracking-tight">Starter wardrobe</h2>
          </div>
          <div className="text-sm text-muted hidden md:block">{garments.length} pieces ready to fit</div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
          {garments.map((g) => (
            <div key={g.id} data-testid="wardrobe-preview-card" className="group rounded-2xl border border-white/10 bg-surface/60 p-3 hover:border-acid/50 transition">
              <div className="aspect-square rounded-xl grid place-items-center overflow-hidden relative"
                   style={{ background: `linear-gradient(140deg, ${g.tint || "#eee"}, ${g.tint ? g.tint + "88" : "#ccc"})` }}>
                <svg viewBox="0 0 100 110" className="w-3/5 h-3/5 opacity-90">
                  <path fill="rgba(255,255,255,.9)" d="M20 20 L38 12 L50 22 L62 12 L80 20 L86 40 L74 44 L74 96 L26 96 L26 44 L14 40 Z" stroke="rgba(0,0,0,.15)" strokeWidth="1"/>
                </svg>
                <div className="absolute bottom-1 right-1 text-[8px] font-mono uppercase tracking-widest px-1.5 py-0.5 rounded bg-black/40 text-white">3D</div>
              </div>
              <div className="mt-3 text-sm font-medium truncate">{g.name}</div>
              <div className="text-[10px] font-mono uppercase tracking-widest text-muted mt-1">{g.color}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="px-6 lg:px-12 py-24">
        <div className="max-w-2xl mb-14">
          <div className="text-[11px] font-mono uppercase tracking-widest text-acid mb-3">HOW IT WORKS</div>
          <h2 className="font-display font-bold text-4xl md:text-5xl tracking-tight">Laptop mirror. Phone remote. Zero install.</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-5">
          {[
            ["01", "Launch the mirror", "Open AURA FIT on your laptop, allow camera access. Pose landmarks are extracted locally in your browser."],
            ["02", "Scan the QR", "A QR code appears on the mirror. Scan it with your phone — instantly become the remote wardrobe."],
            ["03", "Swipe & fit", "Tap a garment on your phone. Watch it snap to your body on the laptop in real time. Change fit, capture, share."],
          ].map(([n, t, c]) => (
            <div key={n} className="rounded-3xl border border-white/10 bg-surface/60 p-8">
              <div className="font-mono text-xs text-acid tracking-widest">STEP {n}</div>
              <div className="mt-3 font-display font-semibold text-2xl">{t}</div>
              <p className="mt-3 text-sm text-muted leading-relaxed">{c}</p>
            </div>
          ))}
        </div>
        <div className="mt-14 flex flex-wrap items-center justify-between gap-6 rounded-3xl border border-acid/30 bg-acid/5 p-8">
          <div>
            <div className="font-display font-bold text-3xl">Ready when you are.</div>
            <div className="text-sm text-muted mt-1">Camera stays on your device. Nothing is uploaded.</div>
          </div>
          <button
            data-testid="launch-mirror-button-cta"
            onClick={startMirror}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-full bg-acid text-obsidian px-6 py-3 font-semibold hover:bg-acidhover transition disabled:opacity-60"
          >
            <Camera className="w-4 h-4" /> {loading ? "Booting…" : "Launch Mirror Now"}
          </button>
        </div>
      </section>

      <footer className="px-6 lg:px-12 py-8 border-t border-white/10 flex flex-wrap items-center justify-between text-xs text-muted font-mono uppercase tracking-widest">
        <div>© 2026 · AURA FIT · Studio Build</div>
        <div>Made with MediaPipe · React · Canvas 2D</div>
      </footer>
    </div>
  );
}
