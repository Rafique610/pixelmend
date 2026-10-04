import React from 'react';
import { Cpu, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';
import type { HealthResponse } from '../../types';

interface NavbarProps {
  health: HealthResponse | null;
  healthLoading: boolean;
  workspaceTitle: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  health,
  healthLoading,
  workspaceTitle,
}) => {
  const isHealthy = health?.status === 'ok';
  const provider = health?.provider || 'CPUExecutionProvider';
  const modelsCount = health?.models_loaded
    ? Object.values(health.models_loaded).filter(Boolean).length
    : 0;

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-6 flex items-center justify-between z-10 sticky top-0">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-cyan-950/60 border border-cyan-800/60 text-cyan-400">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-base font-bold text-slate-100 flex items-center gap-2">
            PixelMend <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono font-normal">v1.0</span>
          </h1>
          <p className="text-xs text-slate-400">{workspaceTitle}</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Execution Provider Badge */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-mono text-[11px]">{provider}</span>
        </div>

        {/* Backend Status Pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1 rounded-lg border text-xs font-medium transition-colors ${
            healthLoading
              ? 'bg-slate-800 border-slate-700 text-slate-400'
              : isHealthy
              ? 'bg-emerald-950/50 border-emerald-800/60 text-emerald-300'
              : 'bg-amber-950/50 border-amber-800/60 text-amber-300'
          }`}
        >
          {healthLoading ? (
            <div className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" />
          ) : isHealthy ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          ) : (
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
          )}
          <span>
            {healthLoading
              ? 'Connecting...'
              : isHealthy
              ? `7 Models Ready (${modelsCount}/7)`
              : 'Backend Degraded'}
          </span>
        </div>
      </div>
    </header>
  );
};
