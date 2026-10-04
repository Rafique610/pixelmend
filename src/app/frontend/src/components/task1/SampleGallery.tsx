import React from 'react';
import { Images } from 'lucide-react';

interface SampleGalleryProps {
  onSelectSample: (file: File, previewUrl: string) => void;
  selectedSample?: string | null;
}

const SAMPLES = [
  { id: 'pet_sample1', name: 'Abyssinian #1', path: '/samples/pet_sample1.jpg' },
  { id: 'pet_sample2', name: 'Abyssinian #2', path: '/samples/pet_sample2.jpg' },
  { id: 'pet_sample3', name: 'Abyssinian #3', path: '/samples/pet_sample3.jpg' },
];

export const SampleGallery: React.FC<SampleGalleryProps> = ({
  onSelectSample,
  selectedSample,
}) => {
  const handleSampleClick = async (sample: typeof SAMPLES[0]) => {
    try {
      const response = await fetch(sample.path);
      const blob = await response.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: 'image/jpeg' });
      onSelectSample(file, sample.path);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
        <Images className="w-3.5 h-3.5 text-cyan-400" />
        <span>Or Choose a Clean Preset Sample</span>
      </div>
      <div className="grid grid-cols-3 gap-2.5">
        {SAMPLES.map((sample) => {
          const isSelected = selectedSample === sample.path;
          return (
            <button
              key={sample.id}
              type="button"
              onClick={() => handleSampleClick(sample)}
              className={`flex flex-col items-center p-2 rounded-xl border text-center transition-all ${
                isSelected
                  ? 'border-cyan-500 bg-cyan-950/40 ring-1 ring-cyan-500/50'
                  : 'border-slate-700/80 bg-slate-800/40 hover:bg-slate-800/80 hover:border-slate-600'
              }`}
            >
              <img
                src={sample.path}
                alt={sample.name}
                className="w-14 h-14 rounded-lg object-cover border border-slate-700/60 shadow-sm"
              />
              <span className="text-[11px] font-medium text-slate-300 mt-1.5 truncate w-full">
                {sample.name}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
