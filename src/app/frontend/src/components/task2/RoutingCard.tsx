import React from 'react';
import { GitFork, ShieldAlert, Zap, ArrowRight, Settings2 } from 'lucide-react';

interface RoutingCardProps {
  predictedClass: string;
  confidence: number;
  selectedExpert: string;
  forceExpert?: string | null;
  onForceExpertChange?: (val: string | null) => void;
}

export const RoutingCard: React.FC<RoutingCardProps> = ({
  predictedClass,
  confidence,
  selectedExpert,
  forceExpert,
  onForceExpertChange,
}) => {
  const isBypass = selectedExpert.toLowerCase().includes('identity') || selectedExpert.toLowerCase().includes('bypass');

  return (
    <div className="p-4 rounded-xl border border-slate-700/80 bg-slate-800/60 space-y-3.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitFork className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Stage 2: Specialist Routing Decision
          </h3>
        </div>
        {onForceExpertChange && (
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Settings2 className="w-3.5 h-3.5 text-slate-500" />
            <select
              value={forceExpert || ''}
              onChange={(e) => onForceExpertChange(e.target.value ? e.target.value : null)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-[11px] text-slate-300 focus:ring-1 focus:ring-amber-500 focus:outline-none"
            >
              <option value="">Auto (Classifier)</option>
              <option value="clean">Force Clean Bypass</option>
              <option value="blur">Force Blur Specialist</option>
              <option value="salt">Force Salt Specialist</option>
              <option value="occlusion">Force Occlusion Specialist</option>
            </select>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-3 rounded-lg border border-slate-700 bg-slate-900/60">
        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          <div className="p-2 rounded-lg bg-amber-950/60 border border-amber-800/60 text-amber-400 shrink-0">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div>
            <p className="text-[11px] text-slate-400 font-medium">Predicted Category</p>
            <p className="text-sm font-bold text-slate-100 capitalize">
              {predictedClass.replace(/_/g, ' ')}{' '}
              <span className="text-xs font-mono font-normal text-amber-400">
                ({(confidence * 100).toFixed(1)}%)
              </span>
            </p>
          </div>
        </div>

        <ArrowRight className="hidden sm:block w-4 h-4 text-slate-600 shrink-0" />

        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          <div
            className={`p-2 rounded-lg border shrink-0 ${
              isBypass
                ? 'bg-emerald-950/60 border-emerald-800/60 text-emerald-400'
                : 'bg-amber-950/60 border-amber-800/60 text-amber-400'
            }`}
          >
            <Zap className="w-4 h-4" />
          </div>
          <div>
            <p className="text-[11px] text-slate-400 font-medium">Active Specialist</p>
            <p className="text-sm font-bold text-slate-100">
              {isBypass ? 'Identity Bypass (0 ms)' : selectedExpert}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
