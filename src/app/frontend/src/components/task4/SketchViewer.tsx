import React from 'react';
import { Download, Sparkles, Clock, Check } from 'lucide-react';

interface SketchViewerProps {
  originalImage: string;
  sketchImage: string;
  selectedStyle: number;
  styleDescription: string;
  inferenceTimeMs: number;
}

export const SketchViewer: React.FC<SketchViewerProps> = ({
  originalImage,
  sketchImage,
  selectedStyle,
  styleDescription,
  inferenceTimeMs,
}) => {
  const [downloaded, setDownloaded] = React.useState(false);

  const handleDownload = () => {
    const filename = `sketch_style_${selectedStyle}_${Date.now()}.png`;
    const link = document.createElement('a');
    link.href = sketchImage;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setDownloaded(true);
    setTimeout(() => setDownloaded(false), 2000);
  };

  return (
    <div className="space-y-4 p-5 rounded-2xl border border-slate-800 bg-slate-900/70">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-pink-500/10 text-pink-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100">Synthesized Sketch Result</h3>
            <p className="text-[11px] text-slate-400">{styleDescription}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
            <Clock className="w-3.5 h-3.5 text-pink-400" />
            {inferenceTimeMs.toFixed(1)} ms
          </span>
          <button
            type="button"
            onClick={handleDownload}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-pink-600 hover:bg-pink-500 text-white text-xs font-semibold shadow-md transition-all active:scale-95"
          >
            {downloaded ? (
              <>
                <Check className="w-3.5 h-3.5 text-white" />
                <span>Saved!</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Download PNG</span>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Source Face */}
        <div className="space-y-1.5">
          <span className="text-[11px] font-semibold text-slate-400 tracking-wide uppercase">
            Input Portrait (128x128)
          </span>
          <div className="aspect-square rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
            <img
              src={originalImage}
              alt="Source Face"
              className="w-full h-full object-contain"
            />
          </div>
        </div>

        {/* Synthesized Sketch */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-pink-400 tracking-wide uppercase">
              cGAN Synthesis (Style {selectedStyle})
            </span>
          </div>
          <div className="aspect-square rounded-xl overflow-hidden bg-slate-950 border border-pink-500/30 ring-1 ring-pink-500/20 flex items-center justify-center">
            <img
              src={sketchImage}
              alt="Synthesized Sketch"
              className="w-full h-full object-contain"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
