import React from 'react';
import { Clock, Cpu, Zap } from 'lucide-react';

interface MetricBadgeProps {
  label: string;
  value: string | number;
  unit?: string;
  variant?: 'cyan' | 'amber' | 'purple' | 'pink' | 'emerald' | 'slate';
  icon?: 'clock' | 'cpu' | 'zap';
}

const variantStyles = {
  cyan: 'bg-cyan-950/60 text-cyan-300 border-cyan-800/60',
  amber: 'bg-amber-950/60 text-amber-300 border-amber-800/60',
  purple: 'bg-purple-950/60 text-purple-300 border-purple-800/60',
  pink: 'bg-pink-950/60 text-pink-300 border-pink-800/60',
  emerald: 'bg-emerald-950/60 text-emerald-300 border-emerald-800/60',
  slate: 'bg-slate-800/80 text-slate-300 border-slate-700',
};

export const MetricBadge: React.FC<MetricBadgeProps> = ({
  label,
  value,
  unit,
  variant = 'slate',
  icon = 'clock',
}) => {
  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium ${variantStyles[variant]}`}
    >
      {icon === 'clock' && <Clock className="w-3.5 h-3.5 opacity-80" />}
      {icon === 'cpu' && <Cpu className="w-3.5 h-3.5 opacity-80" />}
      {icon === 'zap' && <Zap className="w-3.5 h-3.5 opacity-80" />}
      <span className="text-slate-400">{label}:</span>
      <span className="font-mono font-semibold text-slate-100">
        {value}
        {unit && <span className="text-xs font-normal opacity-70 ml-0.5">{unit}</span>}
      </span>
    </div>
  );
};
