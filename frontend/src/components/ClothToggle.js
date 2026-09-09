import React from "react";
import { Waves } from "lucide-react";

export const ClothToggle = ({ enabled, onChange, device, compact = false }) => (
  <button
    type="button"
    role="switch"
    aria-checked={enabled}
    aria-label="Soft-body drape"
    title="Soft-body drape"
    data-testid={`cloth-toggle-${device}`}
    onClick={() => onChange(!enabled)}
    className={`flex items-center gap-2 rounded-xl border transition-colors hover:border-acid/60 ${compact ? "px-3 py-2 text-xs" : "w-full p-4 text-sm"} ${enabled ? "border-acid/50 bg-acid/10 text-acid" : "border-white/10 bg-white/5 text-white/80"}`}
  >
    <Waves className="w-4 h-4 shrink-0" />
    <span className="flex-1 text-left whitespace-nowrap">{compact ? "Drape" : "Soft-body drape"}</span>
    <span aria-hidden="true" className={`relative shrink-0 rounded-full border ${compact ? "w-7 h-4" : "w-10 h-6"} ${enabled ? "border-acid bg-acid" : "border-white/20 bg-white/10"}`}>
      <span className={`absolute rounded-full transition-transform ${compact ? "w-2.5 h-2.5 top-[2px] left-[2px]" : "w-4 h-4 top-[3px] left-[3px]"} ${enabled ? `bg-obsidian ${compact ? "translate-x-3" : "translate-x-4"}` : "bg-white translate-x-0"}`} />
    </span>
  </button>
);