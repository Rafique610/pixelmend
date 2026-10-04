import React from 'react';
import type { CorruptionType } from '../../types';

interface CorruptionControlsProps {
  applyCorruption: boolean;
  onToggleApply: (apply: boolean) => void;
  corruptionType: CorruptionType;
  onChangeType: (type: CorruptionType) => void;
  severity: number;
  onChangeSeverity: (severity: number) => void;
}

const SEVERITY_HINTS: Record<CorruptionType, Record<number, string>> = {
  clean: { 1: 'Clean (Identity pass)', 2: 'Clean', 3: 'Clean' },
  gaussian_blur: {
    1: 'Kernel 3×3, σ = 0.7 (Mild blur)',
    2: 'Kernel 5×5, σ = 1.5 (Standard test)',
    3: 'Kernel 7×7, σ = 2.5 (Severe blur)',
  },
  salt_and_pepper: {
    1: 'Impulse noise p = 0.03 (Mild)',
    2: 'Impulse noise p = 0.08 (Standard test)',
    3: 'Impulse noise p = 0.15 (Severe)',
  },
  rectangular_occlusion: {
    1: '1 Box covering ~10% area',
    2: '2 Boxes covering ~20% area',
    3: '3 Boxes covering ~35% area',
  },
};

export const CorruptionControls: React.FC<CorruptionControlsProps> = ({
  applyCorruption,
  onToggleApply,
  corruptionType,
  onChangeType,
  severity,
  onChangeSeverity,
}) => {
  return (
    <div className="p-4 rounded-xl border border-slate-700/80 bg-slate-800/60 space-y-3.5">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={applyCorruption}
            onChange={(e) => onToggleApply(e.target.checked)}
            className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-cyan-500 focus:ring-cyan-500/30"
          />
          <span className="text-xs font-semibold text-slate-200">
            Apply Synthetic Corruption (Studio Mode)
          </span>
        </label>
        <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded-full border border-cyan-800/40">
          Assignment Benchmark
        </span>
      </div>

      {applyCorruption ? (
        <div className="space-y-3 pt-1 border-t border-slate-700/60">
          {/* Corruption Type Selector */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-medium text-slate-400">Degradation Type:</span>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: 'gaussian_blur', label: 'Blur' },
                { id: 'salt_and_pepper', label: 'Salt & Pepper' },
                { id: 'rectangular_occlusion', label: 'Occlusion' },
              ].map((type) => (
                <button
                  key={type.id}
                  type="button"
                  onClick={() => onChangeType(type.id as CorruptionType)}
                  className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                    corruptionType === type.id
                      ? 'border-cyan-500 bg-cyan-950/50 text-cyan-300'
                      : 'border-slate-700 bg-slate-800/80 text-slate-400 hover:text-slate-200 hover:border-slate-600'
                  }`}
                >
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          {/* Severity Selector */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-medium text-slate-400">Severity Level:</span>
              <span className="font-mono text-cyan-400 font-semibold">Level {severity}</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[1, 2, 3].map((lvl) => (
                <button
                  key={lvl}
                  type="button"
                  onClick={() => onChangeSeverity(lvl)}
                  className={`py-1 rounded-lg border text-xs font-medium transition-all ${
                    severity === lvl
                      ? 'border-cyan-500 bg-cyan-950/50 text-cyan-300 font-bold'
                      : 'border-slate-700 bg-slate-800/80 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {lvl === 1 ? '1 (Mild)' : lvl === 2 ? '2 (Medium)' : '3 (Severe)'}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-400/90 italic mt-1">
              {SEVERITY_HINTS[corruptionType]?.[severity] || ''}
            </p>
          </div>
        </div>
      ) : (
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Direct Mode: The uploaded image will be restored directly as-is without applying server-side corruption.
        </p>
      )}
    </div>
  );
};
