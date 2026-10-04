import React from 'react';
import { Sparkles, Brush, Layers, CheckCircle2 } from 'lucide-react';

interface StyleOption {
  id: 1 | 2 | 3;
  name: string;
  badge: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  tone: string;
}

const STYLES: StyleOption[] = [
  {
    id: 1,
    name: 'Clean Contour',
    badge: 'Style 1',
    description: 'Crisp pencil outlines with minimalist, delicate line art.',
    icon: Sparkles,
    tone: 'from-pink-500/20 to-pink-500/5 border-pink-500/40 text-pink-300',
  },
  {
    id: 2,
    name: 'Cross-Hatch',
    badge: 'Style 2',
    description: 'Dense multi-directional strokes producing textured artistic shading.',
    icon: Brush,
    tone: 'from-purple-500/20 to-purple-500/5 border-purple-500/40 text-purple-300',
  },
  {
    id: 3,
    name: 'Tonal Shading',
    badge: 'Style 3',
    description: 'Continuous graphite gradients with rich contrast and deep shadows.',
    icon: Layers,
    tone: 'from-amber-500/20 to-amber-500/5 border-amber-500/40 text-amber-300',
  },
];

interface StyleSelectorProps {
  selectedStyle: number;
  onSelectStyle: (styleId: 1 | 2 | 3) => void;
  disabled?: boolean;
}

export const StyleSelector: React.FC<StyleSelectorProps> = ({
  selectedStyle,
  onSelectStyle,
  disabled = false,
}) => {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
        <span>FS2K Sketch Conditioning Style</span>
        <span className="text-slate-500 text-[11px]">FiLM Bottleneck Modulation</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {STYLES.map((style) => {
          const isSelected = selectedStyle === style.id;
          const Icon = style.icon;
          return (
            <button
              key={style.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelectStyle(style.id)}
              className={`text-left p-3.5 rounded-xl border transition-all relative overflow-hidden group ${
                isSelected
                  ? `bg-gradient-to-b ${style.tone} shadow-lg ring-1 ring-pink-500/50`
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className={`text-[10px] font-bold tracking-wider px-2 py-0.5 rounded-full ${
                    isSelected ? 'bg-pink-500/30 text-pink-200' : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {style.badge}
                </span>
                {isSelected ? (
                  <CheckCircle2 className="w-4 h-4 text-pink-400" />
                ) : (
                  <Icon className="w-4 h-4 text-slate-500 group-hover:text-slate-300 transition-colors" />
                )}
              </div>
              <h4 className="text-xs font-bold text-slate-100">{style.name}</h4>
              <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                {style.description}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
};
