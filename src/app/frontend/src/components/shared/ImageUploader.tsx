import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  onClear?: () => void;
  previewUrl?: string | null;
  label?: string;
  maxSizeBytes?: number;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelected,
  onClear,
  previewUrl,
  label = 'Upload Image (JPEG, PNG, WEBP)',
  maxSizeBytes = 10 * 1024 * 1024,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    setError(null);
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Please upload a valid image (JPEG, PNG, or WEBP).');
      return;
    }
    if (file.size > maxSizeBytes) {
      setError(`File is too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Limit is 10MB.`);
      return;
    }
    onImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full space-y-2">
      {previewUrl ? (
        <div className="relative group rounded-xl overflow-hidden border border-slate-700 bg-slate-900 aspect-square max-w-[280px] mx-auto flex items-center justify-center">
          <img
            src={previewUrl}
            alt="Upload Preview"
            className="w-full h-full object-contain"
          />
          {onClear && (
            <button
              onClick={onClear}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-900/80 hover:bg-red-900/80 text-slate-300 hover:text-white transition-colors"
              title="Remove image"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      ) : (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
            isDragging
              ? 'border-cyan-400 bg-cyan-950/20'
              : 'border-slate-700 hover:border-slate-500 bg-slate-800/40 hover:bg-slate-800/70'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) handleFile(e.target.files[0]);
            }}
          />
          <div className="p-3 rounded-full bg-slate-800 text-slate-300 mb-3 group-hover:scale-105 transition-transform">
            {isDragging ? <UploadCloud className="w-6 h-6 text-cyan-400" /> : <ImageIcon className="w-6 h-6" />}
          </div>
          <p className="text-sm font-medium text-slate-200">{label}</p>
          <p className="text-xs text-slate-400 mt-1">Drag & drop or click to browse (max 10MB)</p>
        </div>
      )}
      {error && <p className="text-xs text-red-400 font-medium">{error}</p>}
    </div>
  );
};
