import React from 'react';
import { Layers, Sparkles } from 'lucide-react';

interface ExpertContributionProps {
  dominantExpert: string;
  dominantWeight: number;
}

export const ExpertContribution: React.FC<ExpertContributionProps> = ({
  dominantExpert,
  dominantWeight,
}) => {
  const formattedExpertName = dominantExpert
    .replace('expert_', '')
    .replace('identity_', '')
    .replace('_', ' ')
    .toUpperCase();

  return (
    <div className="p-4 rounded-xl border border-slate-700/80 bg-slate-800/60 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-purple-400" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Convex Composite Blending Mechanics
          </h3>
        </div>
        <span className="text-[10px] text-purple-300 font-mono bg-purple-950/60 border border-purple-800/60 px-2 py-0.5 rounded-full">
          τ = 2.526 Soft Temperature
        </span>
      </div>

      <div className="p-3 rounded-lg border border-slate-700 bg-slate-900/60 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">Composite Equation:</span>
          <span className="font-mono text-purple-300 font-medium">
            x̂ = ∑ wᵢ · Eᵢ(x),  ∑ wᵢ = 1.0
          </span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Rather than an all-or-nothing hard routing split, Soft MoE evaluates all 4 expert branches in a unified computation graph, softly weighting their pixel representations by continuous gate confidence coefficients.
        </p>
      </div>

      <div className="flex items-center justify-between p-2.5 rounded-lg border border-purple-800/60 bg-purple-950/30 text-xs">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <span className="text-slate-300 font-medium">Dominant Branch Contribution:</span>
        </div>
        <span className="font-mono font-bold text-purple-300">
          {formattedExpertName} ({(dominantWeight * 100).toFixed(1)}%)
        </span>
      </div>
    </div>
  );
};
