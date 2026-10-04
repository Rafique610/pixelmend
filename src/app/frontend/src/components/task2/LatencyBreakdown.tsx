import React from 'react';
import { Clock, Cpu, Gauge } from 'lucide-react';
import type { LatencyBreakdown as LatencyType } from '../../types';

interface LatencyBreakdownProps {
  latency: LatencyType;
}

export const LatencyBreakdown: React.FC<LatencyBreakdownProps> = ({ latency }) => {
  const isZeroSpec = latency.specialist_ms === 0.0;

  return (
    <div className="p-4 rounded-xl border border-slate-700/80 bg-slate-800/60 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Execution Latency Decomposition
          </h3>
        </div>
        <span className="text-[10px] text-slate-400 font-mono">CPU Execution</span>
      </div>

      <div className="grid grid-cols-3 gap-2.5">
        <div className="p-2.5 rounded-lg border border-slate-700 bg-slate-900/60 flex flex-col">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span>Stage 1 (Classifier)</span>
          </div>
          <span className="text-base font-bold font-mono text-slate-100 mt-1">
            {latency.classifier_ms}{' '}
            <span className="text-xs font-normal text-slate-400">ms</span>
          </span>
        </div>

        <div className="p-2.5 rounded-lg border border-slate-700 bg-slate-900/60 flex flex-col">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <Gauge className={`w-3.5 h-3.5 ${isZeroSpec ? 'text-emerald-400' : 'text-amber-400'}`} />
            <span>Stage 2 (Specialist)</span>
          </div>
          <span
            className={`text-base font-bold font-mono mt-1 ${
              isZeroSpec ? 'text-emerald-400' : 'text-slate-100'
            }`}
          >
            {latency.specialist_ms}{' '}
            <span className="text-xs font-normal text-slate-400">ms</span>
          </span>
        </div>

        <div className="p-2.5 rounded-lg border border-amber-800/50 bg-amber-950/30 flex flex-col">
          <div className="flex items-center gap-1.5 text-[11px] text-amber-300">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            <span>Total End-to-End</span>
          </div>
          <span className="text-base font-bold font-mono text-amber-300 mt-1">
            {latency.total_ms}{' '}
            <span className="text-xs font-normal text-amber-400/80">ms</span>
          </span>
        </div>
      </div>
    </div>
  );
};
