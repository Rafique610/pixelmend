import React, { useState } from 'react';
import { Palette, Sparkles, Camera, Upload, RotateCcw, User } from 'lucide-react';
import { ImageUploader } from '../components/shared/ImageUploader';
import { ErrorAlert } from '../components/shared/ErrorAlert';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { StyleSelector } from '../components/task4/StyleSelector';
import { SketchViewer } from '../components/task4/SketchViewer';
import { WebcamCapture } from '../components/task4/WebcamCapture';
import { api } from '../services/api';
import type { SketchGenerateResponse } from '../types';

export const FaceToSketch: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedStyle, setSelectedStyle] = useState<1 | 2 | 3>(1);
  const [inputMode, setInputMode] = useState<'upload' | 'webcam'>('upload');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SketchGenerateResponse | null>(null);

  const handleSelectFile = (f: File) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleWebcamCapture = (f: File, url: string) => {
    setFile(f);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch('/samples/face_sample.jpg');
      const blob = await res.blob();
      setFile(new File([blob], 'face_sample.jpg', { type: 'image/jpeg' }));
      setPreviewUrl('/samples/face_sample.jpg');
      setResult(null);
      setError(null);
    } catch {
      setError('Unable to load preset face sample.');
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleGenerate = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('style', String(selectedStyle));
      setResult(await api.generateSketch(formData));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sketch synthesis failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-pink-950/60 border border-pink-800/60 text-pink-400">
            <Palette className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-100">Face-to-Sketch Synthesis Workspace</h2>
            <p className="text-xs text-slate-400 mt-0.5">Conditional GAN generator conditioned on FS2K sketch styles via FiLM modulation.</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={handleLoadSample} className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/70 hover:bg-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition-colors">
            <User className="w-3.5 h-3.5 text-pink-400" /> Preset Face Sample
          </button>
          {file && (
            <button type="button" onClick={handleClear} className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200 text-xs flex items-center gap-1 transition-colors">
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          )}
        </div>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} />}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center rounded-lg bg-slate-900/90 p-1 border border-slate-800 text-xs">
            <button type="button" onClick={() => setInputMode('upload')} className={`flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 font-medium transition-all ${inputMode === 'upload' ? 'bg-pink-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}>
              <Upload className="w-3.5 h-3.5" /> Photo Upload
            </button>
            <button type="button" onClick={() => setInputMode('webcam')} className={`flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 font-medium transition-all ${inputMode === 'webcam' ? 'bg-pink-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}>
              <Camera className="w-3.5 h-3.5" /> Live Webcam
            </button>
          </div>

          {inputMode === 'upload' ? (
            <ImageUploader onImageSelected={handleSelectFile} onClear={handleClear} previewUrl={previewUrl} label="Upload Portrait Photo" />
          ) : (
            <WebcamCapture onCapture={handleWebcamCapture} />
          )}

          <StyleSelector selectedStyle={selectedStyle} onSelectStyle={setSelectedStyle} disabled={loading} />

          <button type="button" disabled={!file || loading} onClick={handleGenerate} className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 hover:from-pink-500 hover:to-rose-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-bold flex items-center justify-center gap-2 shadow-lg shadow-pink-900/20 transition-all active:scale-[0.99]">
            {loading ? <><LoadingSpinner size="sm" /><span>Synthesizing Sketch...</span></> : <><Sparkles className="w-4 h-4" /><span>Generate Sketch (Style {selectedStyle})</span></>}
          </button>
        </div>

        <div className="lg:col-span-7">
          {result ? (
            <SketchViewer originalImage={result.original_image} sketchImage={result.sketch_image} selectedStyle={result.selected_style} styleDescription={result.style_description} inferenceTimeMs={result.inference_time_ms} />
          ) : (
            <div className="h-full min-h-[360px] rounded-2xl border border-dashed border-slate-800 bg-slate-900/30 flex flex-col items-center justify-center p-8 text-center">
              <Palette className="w-12 h-12 text-slate-700 mb-3" />
              <h3 className="text-sm font-semibold text-slate-300">Awaiting Portrait Input</h3>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">Upload a portrait photo or capture a live webcam snapshot, choose a sketch style, and synthesize your sketch.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FaceToSketch;
