import React, { useState } from 'react';
import { Sparkles, Wand2 } from 'lucide-react';
import { ImageUploader } from '../components/shared/ImageUploader';
import { ImagePanel } from '../components/shared/ImagePanel';
import { MetricBadge } from '../components/shared/MetricBadge';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { ErrorAlert } from '../components/shared/ErrorAlert';
import { CorruptionControls } from '../components/task1/CorruptionControls';
import { SampleGallery } from '../components/task1/SampleGallery';
import { ErrorMapViewer } from '../components/task1/ErrorMapViewer';
import { api } from '../services/api';
import type { CorruptionType, UniversalRestoreResponse } from '../types';

export const UniversalRestoration: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [applyCorruption, setApplyCorruption] = useState(false);
  const [corruptionType, setCorruptionType] = useState<CorruptionType>('gaussian_blur');
  const [severity, setSeverity] = useState(2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UniversalRestoreResponse | null>(null);

  const handleSelectSample = (sampleFile: File, url: string) => {
    setFile(sampleFile);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleRestore = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('apply_corruption', String(applyCorruption));
      formData.append('corruption_type', corruptionType);
      formData.append('severity', String(severity));
      const res = await api.restoreUniversal(formData);
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Universal restoration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Workspace Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-cyan-950/60 border border-cyan-800/60 text-cyan-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-bold text-slate-100">Universal Restoration Workspace</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Task 1: Single Convolutional Autoencoder restoring corrupted photos without category labels.
          </p>
        </div>
        {result && (
          <div className="flex items-center gap-2">
            <MetricBadge label="Inference" value={result.inference_time_ms} unit="ms" variant="cyan" />
            <MetricBadge label="Model" value="task1_universal_ae.onnx" variant="slate" icon="cpu" />
          </div>
        )}
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError(null)} onRetry={handleRestore} />}

      {/* Grid: Controls Left, Results Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Control Panel (5 columns) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 space-y-4">
            <ImageUploader
              onImageSelected={(f) => {
                setFile(f);
                setPreviewUrl(URL.createObjectURL(f));
                setResult(null);
              }}
              onClear={handleClear}
              previewUrl={previewUrl}
            />
            <SampleGallery onSelectSample={handleSelectSample} selectedSample={previewUrl} />
            <CorruptionControls
              applyCorruption={applyCorruption}
              onToggleApply={setApplyCorruption}
              corruptionType={corruptionType}
              onChangeType={setCorruptionType}
              severity={severity}
              onChangeSeverity={setSeverity}
            />
            <button
              onClick={handleRestore}
              disabled={!file || loading}
              className={`w-full py-2.5 px-4 rounded-xl font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
                !file || loading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20 active:scale-[0.99]'
              }`}
            >
              <Wand2 className="w-4 h-4" />
              {loading ? 'Restoring...' : 'Restore Image'}
            </button>
          </div>
        </div>

        {/* Right Output Panel (7 columns) */}
        <div className="lg:col-span-7 space-y-4">
          {loading ? (
            <div className="p-12 rounded-xl border border-slate-800 bg-slate-900/40">
              <LoadingSpinner label="Running Task 1 Autoencoder Inference..." variant="cyan" size="lg" />
            </div>
          ) : result ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <ImagePanel
                  title="Input Image"
                  subtitle={result.corruption_applied ? `Corrupted: ${result.corruption_applied.type}` : 'Direct Upload'}
                  imageUrl={result.corrupted_image}
                  badge="Corrupted"
                  downloadFilename="pixelmend-input.png"
                />
                <ImagePanel
                  title="Reconstructed Output"
                  subtitle="Universal Autoencoder"
                  imageUrl={result.restored_image}
                  badge="Restored"
                  downloadFilename="pixelmend-restored.png"
                />
              </div>
              <ErrorMapViewer errorMap={result.error_map} cleanErrorMap={result.clean_error_map} />
            </div>
          ) : (
            <div className="p-12 rounded-xl border border-dashed border-slate-800 bg-slate-900/20 text-center text-slate-500">
              <Sparkles className="w-8 h-8 mx-auto opacity-30 mb-2 text-cyan-400" />
              <p className="text-xs">Upload an image or choose a sample to execute Universal Autoencoder restoration.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UniversalRestoration;
