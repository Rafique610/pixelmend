import React from 'react';
import { Network, Crown, Binary } from 'lucide-react';

interface GatingWeightsProps {
  routingWeights: Record<string, number>;
  dominantExpert: string;
  dominantWeight: number;
  entropy: number;
}

const EXPERT_META: Record<
  string,
  { label: string; barColor: string; textColor: string; glowBorder: string }
> = {
  identity_clean: {
    label: 'Identity Pass-through (Clean)',
    barColor: 'bg-emerald-500',
    textColor: 'text-emerald-400',
    glowBorder: 'border-emerald-500/80 bg-emerald-950/40 ring-1 ring-emerald-500/40',
  },
  expert_salt: {
    label: 'Salt & Pepper Specialist',
    barColor: 'bg-amber-500',
    textColor: 'text-amber-400',
    glowBorder: 'border-amber-500/80 bg-amber-950/40 ring-1 ring-amber-500/40',
  },
  expert_blur: {
    label: 'Gaussian Blur Specialist',
    barColor: 'bg-sky-500',
    textColor: 'text-sky-400',
    glowBorder: 'border-sky-500/80 bg-sky-950/40 ring-1 ring-sky-500/40',
  },
  expert_occlusion: {
    label: 'Rectangular Occlusion Specialist',
    barColor: 'bg-rose-500',
    textColor: 'text-rose-400',
    glowBorder: 'border-rose-500/80 bg-rose-950/40 ring-1 ring-rose-500/40',
  },
};

export const GatingWeights: React.FC<GatingWeightsProps> = ({
  routingWeights,
  dominantExpert,
  entropy,
}) => {
  return (
    <div className="p-4 rounded-xl border border-slate-700/80 bg-slate-800/60 space-y-3.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-purple-400" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Continuous Gating Weights (MoE)
          </h3>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-slate-900 border border-slate-700 text-[10px] font-mono text-purple-300">
          <Binary className="w-3 h-3 text-purple-400" />
          <span>Entropy: {entropy} bits</span>
        </div>
      </div>

      <div className="space-y-2.5">
        {Object.entries(routingWeights).map(([expertKey, weight]) => {
          const meta = EXPERT_META[expertKey] || {
            label: expertKey,
            barColor: 'bg-purple-500',
            textColor: 'text-purple-400',
            glowBorder: 'border-purple-500/80 bg-purple-950/40',
          };
          const isDominant = expertKey === dominantExpert;
          const percentage = (weight * 100).toFixed(1);

          return (
            <div
              key={expertKey}
              className={`p-2.5 rounded-lg border transition-all ${
                isDominant ? meta.glowBorder : 'border-slate-700/50 bg-slate-900/40'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <div className="flex items-center gap-1.5 font-medium">
                  {isDominant && <Crown className="w-3.5 h-3.5 text-purple-400 shrink-0" />}
                  <span className={isDominant ? meta.textColor : 'text-slate-300'}>
                    {meta.label}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold text-slate-200">
                    {percentage}%
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    (w = {weight.toFixed(4)})
                  </span>
                </div>
              </div>
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${meta.barColor}`}
                  style={{ width: `${Math.max(Number(percentage), 1)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
