import React, { useState } from 'react';
import { Activity } from 'lucide-react';
import { ImagePanel } from '../shared/ImagePanel';

interface ErrorMapViewerProps {
  errorMap: string;
  cleanErrorMap?: string | null;
}

export const ErrorMapViewer: React.FC<ErrorMapViewerProps> = ({
  errorMap,
  cleanErrorMap,
}) => {
  const [activeTab, setActiveTab] = useState<'corrupted' | 'clean'>('corrupted');
  const displayMap = activeTab === 'clean' && cleanErrorMap ? cleanErrorMap : errorMap;

  return (
    <div className="space-y-3 max-w-xl mx-auto">
      {cleanErrorMap && (
        <div className="flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <button
            type="button"
            onClick={() => setActiveTab('corrupted')}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
              activeTab === 'corrupted'
                ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-800/80'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            |Restored − Corrupted| Heatmap
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('clean')}
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
              activeTab === 'clean'
                ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-800/80'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            |Restored − Clean| Ground Truth Error
          </button>
        </div>
      )}

      <ImagePanel
        title="Residual Error Heatmap"
        subtitle={
          activeTab === 'clean' && cleanErrorMap
            ? 'Absolute reconstruction error vs ground truth clean photo'
            : 'Pixel-level restoration difference (|Restored − Input|)'
        }
        imageUrl={displayMap}
        badge="Turbo Colormap"
        downloadFilename="pixelmend-residual-heatmap.png"
      />

      {/* Turbo Colormap Color Gradient Legend */}
      <div className="p-3 rounded-xl border border-slate-800 bg-slate-900/60 space-y-1.5">
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span className="flex items-center gap-1 font-medium">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            Residual Intensity Legend:
          </span>
          <span className="font-mono text-[10px]">0.0 (Min) → 1.0 (Max Error)</span>
        </div>
        {/* Turbo gradient representation */}
        <div className="h-3 rounded-full w-full bg-gradient-to-r from-[#30123b] via-[#28bbec] via-[#a2fc3c] via-[#fb8022] to-[#7a0403] shadow-inner" />
        <div className="flex justify-between text-[10px] text-slate-400 font-mono">
          <span>0.0 (No diff)</span>
          <span>0.25</span>
          <span>0.50</span>
          <span>0.75</span>
          <span>1.0 (Max diff)</span>
        </div>
      </div>
    </div>
  );
};
