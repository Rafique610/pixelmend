import React from 'react';
import { Download, Eye } from 'lucide-react';

interface ImagePanelProps {
  title: string;
  imageUrl?: string | null;
  subtitle?: string;
  badge?: string;
  allowDownload?: boolean;
  downloadFilename?: string;
  aspectSquare?: boolean;
}

export const ImagePanel: React.FC<ImagePanelProps> = ({
  title,
  imageUrl,
  subtitle,
  badge,
  allowDownload = true,
  downloadFilename = 'pixelmend-output.png',
  aspectSquare = true,
}) => {
  const handleDownload = () => {
    if (!imageUrl) return;
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = downloadFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="flex flex-col rounded-xl border border-slate-700/80 bg-slate-800/60 overflow-hidden shadow-sm">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/80 bg-slate-800/90">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">{title}</h3>
            {badge && (
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 font-medium">
                {badge}
              </span>
            )}
          </div>
          {subtitle && <p className="text-[11px] text-slate-400 mt-0.5">{subtitle}</p>}
        </div>

        {imageUrl && allowDownload && (
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-700/80 hover:bg-slate-700 text-slate-200 hover:text-white transition-colors"
            title="Download image"
          >
            <Download className="w-3.5 h-3.5" />
            Save
          </button>
        )}
      </div>

      <div
        className={`relative flex items-center justify-center p-3 bg-slate-950/60 ${
          aspectSquare ? 'aspect-square w-full max-h-[320px]' : 'min-h-[200px]'
        }`}
      >
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={title}
            className="w-full h-full object-contain rounded-lg border border-slate-800 shadow-md transition-transform"
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 gap-2">
            <Eye className="w-8 h-8 opacity-40" />
            <span className="text-xs">No image generated</span>
          </div>
        )}
      </div>
    </div>
  );
};
