import React from 'react';
import { BarChart3, CheckCircle2 } from 'lucide-react';

interface ClassifierProbabilitiesProps {
  probabilities: Record<string, number>;
  predictedClass: string;
}

const CLASS_CONFIG: Record<
  string,
  { label: string; barColor: string; textColor: string; bgLight: string }
> = {
  clean: {
    label: 'Clean (No Distortion)',
    barColor: 'bg-emerald-500',
    textColor: 'text-emerald-400',
    bgLight: 'bg-emerald-950/40 border-emerald-800/60',
  },
  salt_and_pepper: {
    label: 'Salt & Pepper Noise',
    barColor: 'bg-amber-500',
    textColor: 'text-amber-400',
    bgLight: 'bg-amber-950/40 border-amber-800/60',
  },
  gaussian_blur: {
    label: 'Gaussian Blur',
    barColor: 'bg-sky-500',
    textColor: 'text-sky-400',
    bgLight: 'bg-sky-950/40 border-sky-800/60',
  },
  rectangular_occlusion: {
    label: 'Rectangular Occlusion',
    barColor: 'bg-rose-500',
    textColor: 'text-rose-400',
    bgLight: 'bg-rose-950/40 border-rose-800/60',
  },
};

export const ClassifierProbabilities: React.FC<ClassifierProbabilitiesProps> = ({
  probabilities,
  predictedClass,
}) => {
  return (
    <div className="p-4 rounded-xl border border-slate-700/80 bg-slate-800/60 space-y-3.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-amber-400" />
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
            Stage 1: Classifier Probabilities
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-400">Softmax ∑ = 100%</span>
      </div>

      <div className="space-y-2.5">
        {Object.entries(probabilities).map(([className, prob]) => {
          const config = CLASS_CONFIG[className] || {
            label: className,
            barColor: 'bg-slate-500',
            textColor: 'text-slate-400',
            bgLight: 'bg-slate-900 border-slate-700',
          };
          const isWinner = className === predictedClass;
          const percentage = (prob * 100).toFixed(1);

          return (
            <div
              key={className}
              className={`p-2.5 rounded-lg border transition-all ${
                isWinner ? `${config.bgLight} ring-1 ring-amber-500/30` : 'border-slate-700/50 bg-slate-900/40'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <div className="flex items-center gap-1.5 font-medium">
                  {isWinner && <CheckCircle2 className="w-3.5 h-3.5 text-amber-400" />}
                  <span className={isWinner ? config.textColor : 'text-slate-300'}>
                    {config.label}
                  </span>
                </div>
                <span className="font-mono text-xs font-semibold text-slate-200">
                  {percentage}%
                </span>
              </div>
              {/* Progress bar track */}
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${config.barColor}`}
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
